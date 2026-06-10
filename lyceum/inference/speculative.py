"""Speculative decoding: spend cheap draft FLOPs to skip expensive target steps.

The teaching point (Frontier manual, inference chapter): autoregressive decode
with the big *target* model is memory-bound -- each token costs one full pass
over its weights. A small, fast *draft* model proposes ``k`` tokens cheaply, and
then the target verifies all ``k`` proposals in a **single** parallel forward
pass (the same cost as generating one token, since one pass is memory-bound on
the weights regardless of how many positions it scores). Every proposal the
target agrees with is "free" -- we got multiple tokens for one expensive target
call. Crucially, with the verification rule below the *output distribution is
unchanged* versus plain target decoding: speculation is a pure latency win, not
a quality trade-off.

Verification rule (simplified, greedy): we accept a drafted token if it equals
the target model's argmax at that position. We accept the longest correct prefix;
at the first disagreement we discard the rest of the draft and instead emit the
target's own argmax token (so we always make at least one token of progress per
target call). This greedy variant is exactly equivalent to greedy decoding with
the target model -- same tokens, fewer target calls. (The full rejection-sampling
algorithm of Leviathan et al. / Chen et al. extends this to match an arbitrary
*sampled* distribution; greedy keeps the code readable.)

Draft model: ``build_draft_model`` returns a much smaller LyceumLM (fewer layers,
smaller dim) that shares the target's vocabulary. If no separate, cheaper draft
is available, a model can *self-draft* (use itself, or a shallow prefix of its
own layers) -- the algorithm is identical, but the speedup only materializes when
the draft is genuinely cheaper than the target.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import torch

from ..config import ModelConfig
from ..data.tokenizer import BPETokenizer
from ..model.transformer import LyceumLM


# --------------------------------------------------------------------------- #
# Draft model construction
# --------------------------------------------------------------------------- #
def build_draft_model(target_cfg: ModelConfig, tok: BPETokenizer) -> LyceumLM:
    """Build a small, fast draft model sharing the target's vocabulary.

    Heuristic: halve the depth and shrink the width while keeping the head
    geometry valid (``dim`` divisible by ``n_heads``, and ``n_heads`` divisible
    by ``n_kv_heads`` for grouped-query attention). The draft MUST share the
    target's vocab so their logits are over the same token ids.

    If a real, separately-trained draft is unavailable, the same target model can
    self-draft (just pass the target as the draft); this helper only manufactures
    a cheaper untrained skeleton for demonstration and CPU feasibility.
    """
    n_layers = max(1, target_cfg.n_layers // 2)
    dim = max(target_cfg.n_heads, target_cfg.dim // 2)
    # keep dim a multiple of n_heads so head_dim is an integer
    n_heads = target_cfg.n_heads
    dim = (dim // n_heads) * n_heads
    if dim < n_heads:
        dim = n_heads
    n_kv_heads = target_cfg.n_kv_heads
    if n_heads % n_kv_heads != 0:
        n_kv_heads = 1

    draft_cfg = replace(
        target_cfg,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        hidden_dim=None,      # let SwiGLU recompute from the smaller dim
        n_experts=1,          # keep the draft dense (cheap) even if target is MoE
        n_experts_active=1,
        moe_layers=[],
    )
    return LyceumLM(draft_cfg, vocab_size=tok.vocab_size).eval()


# --------------------------------------------------------------------------- #
# Speculative decoder
# --------------------------------------------------------------------------- #
class SpeculativeDecoder:
    """Greedy speculative decoding with a draft + target pair.

    ``target_model`` is the model whose outputs we want to reproduce exactly;
    ``draft_model`` is a cheaper proposer. Both must share the tokenizer/vocab.
    """

    def __init__(self, target_model: LyceumLM, draft_model: LyceumLM,
                 tok: BPETokenizer, device=None):
        if device is None:
            from ..hardware import select_device
            device = select_device()
        self.device = device
        self.target = target_model.eval().to(device)
        self.draft = draft_model.eval().to(device)
        self.tok = tok
        self.max_ctx = target_model.cfg.max_seq_len

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _draft_tokens(self, ids: list[int], k: int) -> list[int]:
        """Greedily propose ``k`` tokens with the cheap draft model.

        For clarity we re-run the draft over the (trimmed) sequence each step.
        A production implementation would use the draft's own KV cache; here the
        sequences are short and the point is correctness, not draft speed.
        """
        proposed: list[int] = []
        cur = list(ids)
        for _ in range(k):
            x = torch.tensor([cur[-self.max_ctx:]], dtype=torch.long,
                             device=self.device)
            logits, _ = self.draft(x)        # (1, 1, vocab) -- last position only
            nxt = int(logits[0, -1].argmax())
            proposed.append(nxt)
            cur.append(nxt)
        return proposed

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], max_new_tokens: int = 64,
                 k: int = 4) -> tuple[list[int], dict]:
        """Generate up to ``max_new_tokens`` tokens via speculative decoding.

        Returns ``(generated_ids, stats)`` where ``generated_ids`` excludes the
        prompt and ``stats`` reports the acceptance rate and how many expensive
        target forward passes were spent versus tokens produced.
        """
        eos = self.tok.id("<eos>")
        generated: list[int] = []
        seq = list(prompt_ids)

        num_target_calls = 0
        num_drafted = 0
        num_accepted = 0

        while len(generated) < max_new_tokens:
            # 1) DRAFT: cheap model proposes up to k tokens.
            remaining = max_new_tokens - len(generated)
            kk = max(1, min(k, remaining))
            draft_tokens = self._draft_tokens(seq, kk)
            num_drafted += len(draft_tokens)

            # 2) VERIFY: one parallel target pass scores every position where a
            #    drafted token would be predicted. We feed the context plus the
            #    drafted tokens; position i's logits predict token i+1.
            verify_input = seq + draft_tokens          # context + proposals
            x = torch.tensor([verify_input[-self.max_ctx:]], dtype=torch.long,
                             device=self.device)
            full_logits = self.target.forward_logits(x)   # (1, T, vocab)
            num_target_calls += 1

            # The last len(draft_tokens)+1 positions of the input predict the
            # next len(draft_tokens)+1 tokens. The position just before the first
            # drafted token predicts draft_tokens[0], and so on; one extra
            # position predicts the "bonus" token after the last accepted draft.
            t = x.size(1)
            n_check = len(draft_tokens)
            # indices in the (possibly trimmed) tensor whose argmax we compare
            start = t - n_check - 1
            target_preds = [int(full_logits[0, start + j].argmax())
                            for j in range(n_check + 1)]

            # 3) ACCEPT the longest correct prefix (greedy match), then emit the
            #    target's own next token at the first mismatch (always >= 1 token).
            accepted = 0
            for j in range(n_check):
                if draft_tokens[j] == target_preds[j]:
                    accepted += 1
                else:
                    break
            num_accepted += accepted

            new_tokens = draft_tokens[:accepted]
            # bonus / correction token comes from the target itself:
            #   - if all drafts accepted, target_preds[accepted] is the free
            #     "bonus" token following the accepted run;
            #   - if a draft was rejected, target_preds[accepted] is the target's
            #     corrected token at the mismatch position.
            correction = target_preds[accepted]
            new_tokens.append(correction)

            for tok_id in new_tokens:
                if len(generated) >= max_new_tokens:
                    break
                generated.append(tok_id)
                seq.append(tok_id)
                if tok_id == eos:
                    break
            if generated and generated[-1] == eos:
                break

        acceptance_rate = (num_accepted / num_drafted) if num_drafted else 0.0
        stats = {
            "tokens_produced": len(generated),
            "num_target_calls": num_target_calls,
            "num_drafted": num_drafted,
            "num_accepted": num_accepted,
            "acceptance_rate": round(acceptance_rate, 4),
            # tokens per expensive target pass; >1 means we beat naive decoding,
            # which would need one target call per token.
            "tokens_per_target_call": round(
                len(generated) / num_target_calls, 4) if num_target_calls else 0.0,
            "k": k,
        }
        return generated, stats


# --------------------------------------------------------------------------- #
# Self-test: build a nano target + draft and verify it runs end to end.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from ..config import get_config
    from ..data.tokenizer import BPETokenizer

    cfg = get_config("nano")

    # Tiny tokenizer trained on a scrap of text so encode/decode/specials work.
    tok = BPETokenizer(special_tokens=cfg.tokenizer.special_tokens)
    tok.train("the quick brown fox jumps over the lazy dog. " * 20,
              vocab_size=320)

    target = LyceumLM(cfg.model, vocab_size=tok.vocab_size).eval()
    draft = build_draft_model(cfg.model, tok)
    print(f"target: {target.num_params():,} params | "
          f"draft: {draft.num_params():,} params "
          f"({draft.cfg.n_layers} layers, dim={draft.cfg.dim})")

    dec = SpeculativeDecoder(target, draft, tok, device=torch.device("cpu"))
    prompt = [tok.id("<bos>"), tok.id("<user>")] + tok.encode("the quick") + \
             [tok.id("<assistant>")]
    out, stats = dec.generate(prompt, max_new_tokens=24, k=4)

    print(f"produced {len(out)} tokens (untrained -> gibberish is expected)")
    print(f"decoded: {tok.decode(out)!r}")
    print("stats:")
    for key, value in stats.items():
        print(f"  {key:24s}: {value}")
    assert len(out) > 0, "speculative decoding produced no tokens"
    print("\nself-test OK: speculative decoder runs and returns tokens + stats.")
