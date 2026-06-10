"""Server-side runtime: the stateful pieces an inference service owns.

Brings together System Design manual building blocks around the model:
  * prompt cache with single-flight (one recompute on a cache miss; concurrent
    duplicate requests wait rather than dogpiling the model)
  * a bounded request queue with load shedding (reject with 503 when saturated
    instead of collapsing)
  * security services (auth, rate limit, spend budget, guardrails, audit,
    extraction detector) all in one place
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from pathlib import Path

from ..config import LyceumConfig
from ..data.tokenizer import BPETokenizer
from ..inference.engine import InferenceEngine
from ..model.transformer import LyceumLM
from ..train.checkpoint import load_checkpoint
from ..security.audit import AuditLog
from ..security.auth import AuthService
from ..security.limits import TokenBucket, SpendBudget
from ..security.guardrails import detect_prompt_injection, filter_output
from ..security.attacks import ExtractionDetector
from ..memory.rag import VectorStore, assemble_context, cite_or_abstain
from .metrics import Metrics, Timer


class PromptCache:
    """LRU response cache with single-flight to prevent a cache stampede."""

    def __init__(self, size: int = 256):
        self.size = size
        self._data: OrderedDict[str, str] = OrderedDict()
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    @staticmethod
    def key(prompt: str, params: dict) -> str:
        return hashlib.sha256((prompt + repr(sorted(params.items()))).encode()).hexdigest()

    def get_or_compute(self, key: str, compute):
        with self._guard:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key], True
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:                       # single-flight: only one recompute
            with self._guard:
                if key in self._data:
                    return self._data[key], True
            value = compute()
            with self._guard:
                self._data[key] = value
                self._data.move_to_end(key)
                while len(self._data) > self.size:
                    self._data.popitem(last=False)
                self._locks.pop(key, None)
            return value, False


class Runtime:
    def __init__(self, cfg: LyceumConfig, model: LyceumLM, tok: BPETokenizer):
        self.cfg = cfg
        self.engine = InferenceEngine(model, tok, cfg)
        self.tok = tok
        self.metrics = Metrics()
        self.cache = PromptCache(cfg.serving.cache_size)
        self.audit = AuditLog(cfg.security.audit_log)
        self.auth = AuthService()
        self.rate = TokenBucket(cfg.security.rate_limit_per_min)
        self.budget = SpendBudget()
        self.extraction = ExtractionDetector()
        self.rag = VectorStore()
        # in-flight request accounting for load shedding
        self.max_inflight = max(2, cfg.serving.max_batch_size * 2)
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        # a default principal so the demo is usable; real keys via issue_key
        self.demo_key = self.auth.issue_key("demo", {"infer"}, tenant="public")

    @classmethod
    def from_checkpoint(cls, cfg: LyceumConfig, ckpt: str | Path,
                        tokenizer_path: str | Path) -> "Runtime":
        tok = BPETokenizer.load(tokenizer_path)
        model = LyceumLM(cfg.model, tok.vocab_size)
        load_checkpoint(model, cfg, ckpt)
        return cls(cfg, model, tok)

    # ------------------------------------------------------------------ #
    def acquire(self) -> bool:
        with self._inflight_lock:
            if self._inflight >= self.max_inflight:
                return False               # load shedding
            self._inflight += 1
            return True

    def release(self):
        with self._inflight_lock:
            self._inflight = max(0, self._inflight - 1)

    def infer(self, prompt: str, principal, *, use_rag=False, tenant="public",
              **gen_kw) -> dict:
        sec = self.cfg.security
        self.metrics.inc("lyceum_requests_total")

        # input guardrail
        if sec.prompt_injection_filter:
            g = detect_prompt_injection(prompt)
            if g.flagged:
                self.metrics.inc("lyceum_blocked_total", reason="injection")
                self.audit.append("input_blocked", principal=principal.name,
                                  reasons=g.reasons)
                return {"blocked": True, "reason": "prompt injection detected",
                        "detail": g.reasons}

        # token budget / denial-of-wallet
        n_in = len(self.tok.encode(prompt))
        ok, msg = self.budget.check_input(principal.name, n_in)
        if not ok:
            return {"blocked": True, "reason": msg}
        self.budget.record(principal.name, n_in)
        gen_kw["max_new_tokens"] = self.budget.clamp_output(
            gen_kw.get("max_new_tokens") or self.cfg.inference.max_new_tokens)

        # model-extraction detection
        if self.extraction.observe(principal.name, prompt):
            self.metrics.inc("lyceum_extraction_alerts_total")
            self.audit.append("extraction_suspected", principal=principal.name)

        # RAG (optional), with per-tenant isolation + cite-or-abstain
        context, cites = "", []
        if use_rag and sec_enabled(self.cfg):
            hits = self.rag.search(prompt, self.cfg.memory.rag_top_k, tenant=tenant)
            ok_ground, abstain = cite_or_abstain(hits)
            if not ok_ground:
                return {"blocked": False, "abstained": True, "text": abstain}
            context, cites = assemble_context(hits)

        full_prompt = (f"Context:\n{context}\n\nQuestion: {prompt}"
                       if context else prompt)

        key = PromptCache.key(full_prompt, gen_kw)
        with Timer(self.metrics, "lyceum_inference_seconds"):
            text, hit = self.cache.get_or_compute(
                key, lambda: self.engine.generate(full_prompt, **gen_kw))
        self.metrics.inc("lyceum_cache_hits_total" if hit else "lyceum_cache_miss_total")

        # output guardrail
        if sec.output_filter:
            text, blocked = filter_output(
                text, canary=self.cfg.data.poison_canary,
                system_prompt="You are Lyceum")
            if blocked:
                self.metrics.inc("lyceum_output_redactions_total")

        self.budget.record(principal.name, len(self.tok.encode(text)))
        self.audit.append("inference", principal=principal.name, cache_hit=hit,
                          cites=cites)
        return {"blocked": False, "text": text, "cache_hit": hit,
                "citations": cites}


def sec_enabled(cfg: LyceumConfig) -> bool:
    return cfg.memory.rag_enabled
