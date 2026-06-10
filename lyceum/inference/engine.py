"""Autoregressive inference: the decode loop, scaled to CPU.

Implements the mechanisms the Frontier manual calls out for serving:
  * prefill vs decode    - prefill the prompt in parallel, then decode one token
                           at a time (the two phases with opposite cost profiles)
  * KV cache             - keep per-layer K/V so each new token is O(seq), not
                           O(seq^2); the central inference data structure
  * sampling controls    - temperature, top-k, top-p (nucleus), repetition
                           penalty
  * streaming            - yield tokens as they are produced
  * test-time compute    - self-consistency (majority vote) and best-of-N, the
                           "spend more inference to get a better answer" idea
"""
from __future__ import annotations

from collections import Counter
from typing import Iterator

import torch
import torch.nn.functional as F

from ..config import LyceumConfig
from ..data.tokenizer import BPETokenizer
from ..model.transformer import LyceumLM, KVCache


class InferenceEngine:
    def __init__(self, model: LyceumLM, tok: BPETokenizer, cfg: LyceumConfig):
        from ..hardware import select_device
        self.model = model.eval()
        self.tok = tok
        self.cfg = cfg
        self.device = select_device(cfg.train.device)
        self.model.to(self.device)
        # optional int8 dynamic quantization for cheaper CPU serving
        if getattr(cfg.inference, "quantize", False):
            from .quantize import quantize_dynamic_int8
            self.model = quantize_dynamic_int8(self.model)

    # ------------------------------------------------------------------ #
    def _sample(self, logits: torch.Tensor, generated: list[int],
                temperature: float, top_k: int, top_p: float,
                rep_penalty: float) -> int:
        logits = logits.float()
        if rep_penalty and rep_penalty != 1.0 and generated:
            for tid in set(generated):
                logits[tid] /= rep_penalty
        if temperature <= 0:
            return int(logits.argmax())
        logits = logits / temperature
        if top_k:
            k = min(top_k, logits.size(-1))
            vals, idx = torch.topk(logits, k)
            probs = F.softmax(vals, dim=-1)
            choice = idx[torch.multinomial(probs, 1)]
            return int(choice)
        probs = F.softmax(logits, dim=-1)
        if top_p and top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, dim=-1)
            cutoff = cum > top_p
            cutoff[0] = False  # always keep the top token
            sp[cutoff] = 0.0
            sp = sp / sp.sum()
            return int(si[torch.multinomial(sp, 1)])
        return int(torch.multinomial(probs, 1))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def stream(self, prompt_ids: list[int], *, max_new_tokens=None,
               temperature=None, top_k=None, top_p=None,
               repetition_penalty=None, stop_ids=None) -> Iterator[int]:
        ic = self.cfg.inference
        max_new_tokens = max_new_tokens or ic.max_new_tokens
        temperature = ic.temperature if temperature is None else temperature
        top_k = ic.top_k if top_k is None else top_k
        top_p = ic.top_p if top_p is None else top_p
        rep = ic.repetition_penalty if repetition_penalty is None else repetition_penalty
        stop_ids = set(stop_ids or [self.tok.id("<eos>")])

        cache = KVCache(len(self.model.blocks)) if ic.use_kv_cache else None
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        max_ctx = self.cfg.model.max_seq_len

        # prefill
        logits, _ = self.model(ids[:, -max_ctx:], cache=cache, start_pos=0)
        generated = list(prompt_ids)
        pos = ids.size(1)
        for _ in range(max_new_tokens):
            nxt = self._sample(logits[0, -1], generated, temperature, top_k,
                               top_p, rep)
            if nxt in stop_ids:
                break
            generated.append(nxt)
            yield nxt
            cur = torch.tensor([[nxt]], dtype=torch.long, device=self.device)
            if cache is not None:
                logits, _ = self.model(cur, cache=cache, start_pos=pos)
            else:
                seq = torch.tensor([generated[-max_ctx:]], device=self.device)
                logits, _ = self.model(seq, start_pos=0)
            pos += 1

    def generate(self, prompt: str, **kw) -> str:
        ids = self._format_prompt(prompt)
        out = list(self.stream(ids, **kw))
        return self.tok.decode(out).strip()

    def _format_prompt(self, prompt: str) -> list[int]:
        t = self.tok
        return [t.id("<bos>"), t.id("<user>")] + t.encode(prompt) + [t.id("<assistant>")]

    @torch.no_grad()
    def batch_generate(self, prompts: list[str], *, max_new_tokens=None,
                       temperature=0.0, top_k=0, top_p=1.0) -> list[str]:
        """Static batching: run several prompts through one set of forward
        passes, finishing each sequence at its own <eos>. This is the
        throughput lever from the System Design manual (process many requests
        per pass instead of one at a time)."""
        ic = self.cfg.inference
        max_new_tokens = max_new_tokens or ic.max_new_tokens
        eos = self.tok.id("<eos>")
        pad = self.tok.id("<pad>")
        seqs = [self._format_prompt(p) for p in prompts]
        maxlen = max(len(s) for s in seqs)
        # left-pad so all prompts end at the same position
        batch = [[pad] * (maxlen - len(s)) + s for s in seqs]
        ids = torch.tensor(batch, dtype=torch.long, device=self.device)
        done = [False] * len(prompts)
        gen: list[list[int]] = [[] for _ in prompts]
        max_ctx = self.cfg.model.max_seq_len
        for _ in range(max_new_tokens):
            logits, _ = self.model(ids[:, -max_ctx:], start_pos=0)
            for i in range(len(prompts)):
                if done[i]:
                    continue
                tok = self._sample(logits[i, -1], gen[i], temperature, top_k,
                                   top_p, ic.repetition_penalty)
                if tok == eos:
                    done[i] = True
                else:
                    gen[i].append(tok)
            if all(done):
                break
            nxt = torch.tensor([[gen[i][-1] if gen[i] and not done[i] else pad]
                                for i in range(len(prompts))], device=self.device)
            ids = torch.cat([ids, nxt], dim=1)
        return [self.tok.decode(g).strip() for g in gen]

    # ------------------------------------------------------------------ #
    # Test-time compute: spend more inference for a better answer.
    # ------------------------------------------------------------------ #
    def self_consistency(self, prompt: str, n: int = 5, **kw) -> tuple[str, dict]:
        """Sample N answers, return the majority (Frontier manual)."""
        kw.setdefault("temperature", 0.9)
        answers = [self.generate(prompt, **kw) for _ in range(n)]
        counts = Counter(answers)
        best, votes = counts.most_common(1)[0]
        return best, {"samples": answers, "votes": votes, "n": n}

    def best_of_n(self, prompt: str, scorer, n: int = 5, **kw) -> tuple[str, dict]:
        """Sample N answers, keep the one the scorer rates highest."""
        kw.setdefault("temperature", 0.9)
        cands = [self.generate(prompt, **kw) for _ in range(n)]
        scored = sorted(((scorer(prompt, c), c) for c in cands), reverse=True)
        return scored[0][1], {"candidates": cands, "scores": [s for s, _ in scored]}
