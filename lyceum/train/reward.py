"""Reward modeling + a teaching-scale RLHF / PPO loop.

This is the classic three-stage RLHF recipe from the Frontier manual, shrunk to
run on a CPU:

  1. **Reward model (RM)** - a learned model of human *taste*. We take a frozen
     (or fine-tuned) language-model trunk, read off the hidden state at the last
     token of a sequence, and pass it through a tiny linear head to a single
     scalar "how good is this?" reward. The RM is trained on preference pairs
     with the Bradley-Terry logistic loss: it should score the human-preferred
     ("chosen") completion higher than the "rejected" one.

  2. **Policy optimization (PPO)** - we then fine-tune the language model (the
     *policy*) to produce completions the RM scores highly. Real systems use PPO
     with a clipped surrogate objective and GAE advantage estimation; here we
     use a deliberately simplified **REINFORCE** policy gradient so the whole
     idea fits on one screen and runs on a laptop. See ``ppo_lite``.

  3. **The KL leash** - optimizing a reward model too hard leads to *reward
     hacking*: the policy finds gibberish that the imperfect RM loves. To stop
     that we anchor the policy to a frozen *reference* policy with a KL penalty
     (weight ``beta``); the effective objective is ``reward - beta * KL``. This
     is the same leash DPO bakes into its loss.

Nothing here is fundamentally different from frontier RLHF - it is the same
machinery, parameterized small.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LyceumConfig
from ..data.tokenizer import BPETokenizer
from ..model.transformer import LyceumLM


# --------------------------------------------------------------------------- #
# Reward model
# --------------------------------------------------------------------------- #
class RewardModel(nn.Module):
    """A scalar reward head on top of a language-model trunk.

    The trunk (a ``LyceumLM``) produces a per-position hidden state via
    ``hidden_states``; we take the *last* token's hidden state as a summary of
    the whole sequence and map it to a single scalar with a small linear head.
    This mirrors how production RMs reuse the policy's backbone and only add a
    value/reward head."""

    def __init__(self, trunk: LyceumLM):
        super().__init__()
        self.trunk = trunk
        self.head = nn.Linear(trunk.cfg.dim, 1, bias=True)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) -> reward: (B,) scalar per sequence."""
        h, _ = self.trunk.hidden_states(idx)      # (B, T, dim)
        last = h[:, -1, :]                        # last-token summary
        return self.head(last).squeeze(-1)        # (B,)

    def score(self, ids: list[int], device=None) -> float:
        """Convenience: reward of a single token-id list."""
        t = torch.tensor([ids], dtype=torch.long,
                          device=device or self.head.weight.device)
        with torch.no_grad():
            return float(self(t).item())


def _encode_pair(tok: BPETokenizer, prompt: str, response: str,
                 max_len: int) -> list[int]:
    """Chat-format a (prompt, response) into token ids, mirroring SFTDataset."""
    bos, eos = tok.id("<bos>"), tok.id("<eos>")
    u, a = tok.id("<user>"), tok.id("<assistant>")
    ids = [bos, u] + tok.encode(prompt) + [a] + tok.encode(response) + [eos]
    return ids[:max_len]


def _pad_batch(seqs: list[list[int]], pad_id: int, device) -> torch.Tensor:
    maxlen = max(len(s) for s in seqs)
    out = [s + [pad_id] * (maxlen - len(s)) for s in seqs]
    return torch.tensor(out, dtype=torch.long, device=device)


def train_reward_model(rm: RewardModel, pref_rows: list[dict],
                       tok: BPETokenizer, cfg: LyceumConfig,
                       steps: int = 100, lr: float | None = None,
                       batch_size: int | None = None, on_log=None):
    """Train the reward model on preference pairs with Bradley-Terry loss.

    ``pref_rows`` are dicts ``{prompt, chosen, rejected}``. The loss
    ``-log sigmoid(r_chosen - r_rejected)`` is minimized when the RM scores the
    chosen completion above the rejected one - i.e. it learns to rank the way the
    (synthetic) human labeller did. Note the last-token padding: because the
    reward reads the *last* token, padding would corrupt the summary, so we keep
    chosen/rejected in separate (independently padded) batches."""
    from ..hardware import select_device
    device = select_device(cfg.train.device)
    rm.to(device).train()
    lr = lr if lr is not None else cfg.align.sft_lr
    bs = batch_size or max(2, cfg.train.batch_size // 2)
    pad = tok.id("<pad>")
    max_len = cfg.model.max_seq_len
    opt = torch.optim.AdamW(rm.parameters(), lr=lr, betas=(0.9, 0.95))

    n = len(pref_rows)
    rng = torch.Generator().manual_seed(cfg.train.seed)
    history = []
    for step in range(steps):
        idxs = torch.randint(0, n, (min(bs, n),), generator=rng).tolist()
        rows = [pref_rows[i] for i in idxs]
        ch = [_encode_pair(tok, r["prompt"], r["chosen"], max_len) for r in rows]
        rj = [_encode_pair(tok, r["prompt"], r["rejected"], max_len) for r in rows]
        ch_b = _pad_batch(ch, pad, device)
        rj_b = _pad_batch(rj, pad, device)

        opt.zero_grad(set_to_none=True)
        r_ch = rm(ch_b)
        r_rj = rm(rj_b)
        # Bradley-Terry: P(chosen > rejected) = sigmoid(r_ch - r_rj)
        loss = -F.logsigmoid(r_ch - r_rj).mean()
        acc = (r_ch > r_rj).float().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), cfg.train.grad_clip)
        opt.step()

        if step % max(1, cfg.train.log_interval) == 0 or step == steps - 1:
            rec = {"phase": "reward", "step": step,
                   "loss": round(loss.item(), 4), "acc": round(acc.item(), 3)}
            history.append(rec)
            (on_log or _print)(rec)
    return history


def _print(rec):
    print(f"  [{rec['phase']}] step {rec['step']:>4} | loss {rec['loss']:.4f} | "
          f"acc {rec.get('acc', float('nan')):.3f}")


# --------------------------------------------------------------------------- #
# Simplified RLHF / PPO (REINFORCE with a KL leash)
# --------------------------------------------------------------------------- #
def _full_logits(model: LyceumLM, ids: torch.Tensor) -> torch.Tensor:
    """Full-sequence logits (B, T, vocab) - reuses the trunk's helper."""
    return model.forward_logits(ids)


def _sample_completion(policy: LyceumLM, prompt_ids: list[int], tok: BPETokenizer,
                       max_new_tokens: int, max_ctx: int, temperature: float,
                       device) -> tuple[list[int], int]:
    """Autoregressively sample a completion from the policy (no grad).
    Returns (full_ids = prompt + completion, prompt_len)."""
    eos = tok.id("<eos>")
    ids = list(prompt_ids)
    prompt_len = len(ids)
    policy.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp = torch.tensor([ids[-max_ctx:]], dtype=torch.long, device=device)
            logits = _full_logits(policy, inp)[0, -1]
            if temperature <= 0:
                nxt = int(logits.argmax())
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1))
            if nxt == eos:
                break
            ids.append(nxt)
    return ids, prompt_len


def _completion_logprob(model: LyceumLM, ids: list[int], prompt_len: int,
                        device, no_grad: bool = False) -> torch.Tensor:
    """Sum of log-probs the model assigns to the *completion* tokens (those at
    positions >= prompt_len). Differentiable when ``no_grad`` is False."""
    seq = torch.tensor([ids], dtype=torch.long, device=device)
    ctx = torch.enable_grad() if not no_grad else torch.no_grad()
    with ctx:
        logits = _full_logits(model, seq)            # (1, T, V)
        logp = F.log_softmax(logits[:, :-1], dim=-1)  # predict token t+1
        targets = seq[:, 1:]
        tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]  # (T-1,)
        # completion tokens are predicted from position prompt_len-1 onward
        start = max(0, prompt_len - 1)
        comp = tok_logp[start:]
    if comp.numel() == 0:
        return tok_logp.sum() * 0.0  # empty completion -> zero (keeps grad graph)
    return comp.sum()


def ppo_lite(policy: LyceumLM, reward_model: RewardModel, ref_policy: LyceumLM,
             tok: BPETokenizer, prompts: list[str], cfg: LyceumConfig,
             steps: int = 50, beta: float | None = None,
             lr: float | None = None, max_new_tokens: int = 16,
             temperature: float = 1.0, on_log=None):
    """A teaching-scale RLHF loop. NOT real PPO.

    Real PPO uses a clipped surrogate objective, a value baseline, and GAE
    advantage estimation. This is a bare **REINFORCE** policy gradient: for each
    prompt we

      1. sample a completion from the current policy,
      2. score it with the reward model,
      3. compute a per-token KL penalty against a frozen reference policy
         (approximated as the difference of summed log-probs),
      4. take a gradient step on

             loss = -(reward - beta * kl) * sum_logprob(completion)

         i.e. push up the log-prob of completions whose KL-penalized reward is
         positive, push it down otherwise.

    The ``beta * kl`` term is the KL leash that prevents *reward hacking* - the
    policy can't run off to gibberish the imperfect RM happens to love, because
    drifting away from the reference is itself penalized."""
    from ..hardware import select_device
    device = select_device(cfg.train.device)
    policy.to(device).train()
    ref_policy.to(device).eval()
    reward_model.to(device).eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    for p in reward_model.parameters():
        p.requires_grad_(False)

    beta = beta if beta is not None else cfg.align.dpo_beta
    lr = lr if lr is not None else cfg.align.dpo_lr
    max_ctx = cfg.model.max_seq_len
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.95))

    bos, u, a = tok.id("<bos>"), tok.id("<user>"), tok.id("<assistant>")
    history = []
    for step in range(steps):
        prompt = prompts[step % len(prompts)]
        prompt_ids = [bos, u] + tok.encode(prompt) + [a]

        # 1. sample a completion (no grad) ...
        full_ids, prompt_len = _sample_completion(
            policy, prompt_ids, tok, max_new_tokens, max_ctx, temperature, device)
        if len(full_ids) <= prompt_len:
            continue  # empty completion, nothing to learn from this step

        # 2. score with the reward model ...
        reward = reward_model.score(full_ids[-max_ctx:], device=device)

        # 3. KL of policy vs reference on the completion (summed-logprob proxy) ...
        policy.train()
        pol_logp = _completion_logprob(policy, full_ids, prompt_len, device)
        with torch.no_grad():
            ref_logp = _completion_logprob(ref_policy, full_ids, prompt_len,
                                           device, no_grad=True)
        kl = (pol_logp.detach() - ref_logp)  # >0 when policy more confident
        advantage = reward - beta * float(kl)

        # 4. REINFORCE update: -(advantage) * sum_logprob(completion)
        loss = -advantage * pol_logp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.train.grad_clip)
        opt.step()

        if step % max(1, cfg.train.log_interval) == 0 or step == steps - 1:
            rec = {"phase": "ppo_lite", "step": step,
                   "reward": round(float(reward), 4),
                   "kl": round(float(kl), 4),
                   "advantage": round(float(advantage), 4),
                   "loss": round(float(loss.item()), 4),
                   "comp_len": len(full_ids) - prompt_len}
            history.append(rec)
            (on_log or _ppo_print)(rec)
    return history


def _ppo_print(rec):
    print(f"  [ppo_lite] step {rec['step']:>4} | reward {rec['reward']:7.3f} | "
          f"kl {rec['kl']:7.3f} | adv {rec['advantage']:7.3f} | "
          f"loss {rec['loss']:8.3f} | comp_len {rec['comp_len']}")


# --------------------------------------------------------------------------- #
# Self-test: nano end-to-end smoke test (no trained model needed).
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import copy
    import tempfile
    from ..config import get_config
    from ..data.corpus import build_preference_dataset, build_pretrain_corpus
    from ..data.dataset import load_jsonl

    print("=" * 64)
    print("reward.py self-test: RM (Bradley-Terry) + ppo_lite (REINFORCE)")
    print("=" * 64)

    cfg = get_config("nano")
    # shrink further so the smoke test is near-instant
    cfg.model.dim = 64
    cfg.model.n_layers = 2
    cfg.model.n_heads = 4
    cfg.model.n_kv_heads = 2
    cfg.model.max_seq_len = 128
    cfg.tokenizer.vocab_size = 512
    cfg.train.batch_size = 8
    cfg.train.log_interval = 5

    with tempfile.TemporaryDirectory() as tmp:
        pref_path = build_preference_dataset(f"{tmp}/prefs.jsonl", n=64, seed=2)
        corpus_path = build_pretrain_corpus(f"{tmp}/corpus.txt", n_docs=80, seed=0)
        pref_rows = load_jsonl(pref_path)

        # train a tokenizer on the prefs + a little corpus text
        tok = BPETokenizer(special_tokens=cfg.tokenizer.special_tokens)
        train_text = corpus_path.read_text() + "\n".join(
            f"{r['prompt']} {r['chosen']} {r['rejected']}" for r in pref_rows)
        tok.train(train_text, vocab_size=cfg.tokenizer.vocab_size)
        vocab = tok.vocab_size
        print(f"tokenizer vocab_size={vocab}")

        # nano language model as the shared trunk / policy
        policy = LyceumLM(cfg.model, vocab)
        print(f"policy params: {policy.num_params():,}")

        # reward model wraps an independent trunk
        rm = RewardModel(LyceumLM(cfg.model, vocab))

        print("\n-- training reward model (Bradley-Terry) --")
        train_reward_model(rm, pref_rows, tok, cfg, steps=20)

        print("\n-- ppo_lite (REINFORCE + KL leash) --")
        ref = copy.deepcopy(policy)
        prompts = [r["prompt"] for r in pref_rows[:8]]
        ppo_lite(policy, rm, ref, tok, prompts, cfg, steps=10,
                 max_new_tokens=8)

    print("\nOK: reward.py self-test ran without crashing.")
