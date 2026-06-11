"""Shared prefix KV caching: prefill the system prompt once, reuse it forever.

A serving lever from the Frontier + System Design manuals. Most requests to a
deployed assistant share a long, *identical* leading context - a system prompt,
a tool schema, few-shot examples. Re-running the transformer over those same
tokens for every request is pure waste: the K/V they produce are deterministic
given the weights, so they can be computed once and reused.

``PrefixCache`` prefills a shared prefix into a :class:`KVCache` a single time
and stores the resulting per-layer K/V. For each incoming request,
``clone_for_request()`` returns a *fresh* ``KVCache`` seeded with a deep copy of
the prefix's K/V, so the request can continue decoding from ``start_pos =
len(prefix)`` without ever recomputing the prefix. This is the same idea as
vLLM's automatic prefix caching / "RadixAttention", reduced to one shared prefix
for teaching clarity.

Why a copy and not a share? Decode appends per-request tokens to the same cache
tensors (``KVCache.update`` does ``torch.cat`` in place into the layer slot). If
two requests shared one cache object their K/V would interleave and corrupt each
other. We therefore clone the prefix tensors per request - cheap relative to the
matmuls we save.

SECURITY NOTE (read before deploying):
    Scope prefix caches **per tenant / per trust boundary**. A single global
    prefix cache shared across users creates a *timing side channel*: a request
    whose prefix hits the cache returns measurably faster than one that misses,
    so an attacker can probe "was this exact prefix recently cached by someone
    else?" and thereby learn fragments of another tenant's system prompt or
    conversation. The same applies to deduplicating prefixes across tenants.
    Treat a prefix-cache key as sensitive: never share cache entries across
    trust boundaries, and consider constant-time-ish responses or per-tenant
    cache partitions. (Cf. the cache-timing attacks discussed in the Security
    manual.) This class takes an optional ``tenant`` label purely to make that
    scoping explicit and auditable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from ..model.transformer import LyceumLM, KVCache


def _clone_cache(src: KVCache) -> KVCache:
    """Deep-copy a KVCache's per-layer K/V tensors into a fresh cache."""
    dst = KVCache(len(src.k))
    for i, (k, v) in enumerate(zip(src.k, src.v)):
        dst.k[i] = None if k is None else k.clone()
        dst.v[i] = None if v is None else v.clone()
    return dst


@torch.no_grad()
def prefill_prefix(model: LyceumLM, prefix_ids: list[int], *,
                   device=None) -> KVCache:
    """Run the model over ``prefix_ids`` once and return the populated KVCache.

    This is the "prefill" phase applied to a *shared* prefix. The returned cache
    holds, for every layer, the K/V produced by the prefix tokens. Counts as a
    single forward call over ``len(prefix_ids)`` positions.
    """
    if not prefix_ids:
        raise ValueError("prefix_ids must be non-empty")
    device = device or next(model.parameters()).device
    cache = KVCache(len(model.blocks))
    ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    model(ids, cache=cache, start_pos=0)
    return cache


@dataclass
class PrefixCache:
    """Hold a prefilled shared-prefix KVCache and hand out per-request clones.

    Parameters
    ----------
    model : LyceumLM
        The model whose weights determine the prefix K/V.
    prefix_ids : list[int]
        The shared leading token ids (e.g. the system prompt).
    tenant : str
        A trust-boundary label. Different tenants MUST use different
        ``PrefixCache`` instances - see the module's SECURITY NOTE.
    """

    model: LyceumLM
    prefix_ids: list[int]
    tenant: str = "default"
    # metrics
    hits: int = 0
    misses: int = 0
    prefix_forward_calls: int = 0          # how many times we prefilled the prefix
    _cache: KVCache | None = field(default=None, repr=False)

    def __post_init__(self):
        self.prefix_len = len(self.prefix_ids)
        self.prefill()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def prefill(self) -> KVCache:
        """Prefill the shared prefix exactly once and remember it."""
        self._cache = prefill_prefix(self.model, self.prefix_ids,
                                     device=self.device)
        self.prefix_forward_calls += 1
        return self._cache

    def clone_for_request(self) -> tuple[KVCache, int]:
        """Return ``(fresh_cache, start_pos)`` seeded with the prefix's K/V.

        The caller continues decoding at ``start_pos == prefix_len`` without ever
        recomputing the prefix - that is the cache *hit*. If the prefix has not
        been prefilled (shouldn't happen), we prefill on demand and count a miss.
        """
        if self._cache is None:
            self.misses += 1
            self.prefill()
        else:
            self.hits += 1
        return _clone_cache(self._cache), self.prefix_len

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "tenant": self.tenant,
            "prefix_len": self.prefix_len,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "prefix_forward_calls": self.prefix_forward_calls,
        }


# --------------------------------------------------------------------------- #
# Request decoding on top of a (possibly prefix-seeded) cache
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _decode_with_cache(model: LyceumLM, cache: KVCache, start_pos: int,
                       suffix_ids: list[int], *, max_new_tokens: int = 8,
                       device=None) -> tuple[list[int], int]:
    """Greedy-decode a request that continues from a seeded cache.

    Returns ``(generated_ids, n_forward_calls)``. The prefix is NOT among the
    forward calls counted here - that's the whole point of reuse.
    """
    device = device or next(model.parameters()).device
    forwards = 0
    pos = start_pos
    # prefill the request-specific suffix (the part that is NOT shared)
    if suffix_ids:
        ids = torch.tensor([suffix_ids], dtype=torch.long, device=device)
        logits, _ = model(ids, cache=cache, start_pos=pos)
        forwards += 1
        pos += len(suffix_ids)
    else:
        # no suffix: take one step from the last prefix position
        last = torch.tensor([[0]], dtype=torch.long, device=device)
        logits, _ = model(last, cache=cache, start_pos=pos)
        forwards += 1
        pos += 1

    generated: list[int] = []
    for _ in range(max_new_tokens):
        nxt = int(logits[0, -1].argmax())
        generated.append(nxt)
        cur = torch.tensor([[nxt]], dtype=torch.long, device=device)
        logits, _ = model(cur, cache=cache, start_pos=pos)
        forwards += 1
        pos += 1
    return generated, forwards


# --------------------------------------------------------------------------- #
# Benchmark: prefix reuse vs recomputing the prefix every request
# --------------------------------------------------------------------------- #
@torch.no_grad()
def benchmark_prefix_reuse(model: LyceumLM, prefix_ids: list[int],
                           requests: list[list[int]], *,
                           max_new_tokens: int = 4) -> dict:
    """Compare two serving strategies over the same set of requests:

      * NO REUSE  - each request re-prefills the full prefix then its suffix.
      * REUSE     - prefill the prefix ONCE, clone it per request.

    Returns timings and forward-call counts. The reuse path should prefill the
    prefix exactly once regardless of the number of requests.
    """
    device = next(model.parameters()).device

    # ---- strategy A: no reuse (recompute prefix every request) ----------- #
    t0 = time.perf_counter()
    no_reuse_prefix_prefills = 0
    for suffix in requests:
        cache = KVCache(len(model.blocks))
        full = prefix_ids + suffix
        ids = torch.tensor([full], dtype=torch.long, device=device)
        model(ids, cache=cache, start_pos=0)        # recomputes the prefix
        no_reuse_prefix_prefills += 1
        pos = len(full)
        logits, _ = model(ids[:, -1:], cache=cache, start_pos=pos - 1)
        for _ in range(max_new_tokens):
            nxt = int(logits[0, -1].argmax())
            cur = torch.tensor([[nxt]], dtype=torch.long, device=device)
            logits, _ = model(cur, cache=cache, start_pos=pos)
            pos += 1
    no_reuse_time = time.perf_counter() - t0

    # ---- strategy B: shared prefix cache --------------------------------- #
    t0 = time.perf_counter()
    pc = PrefixCache(model, prefix_ids, tenant="bench")
    for suffix in requests:
        cache, start_pos = pc.clone_for_request()
        _decode_with_cache(model, cache, start_pos, suffix,
                            max_new_tokens=max_new_tokens, device=device)
    reuse_time = time.perf_counter() - t0

    return {
        "n_requests": len(requests),
        "prefix_len": len(prefix_ids),
        "no_reuse_prefix_prefills": no_reuse_prefix_prefills,
        "reuse_prefix_prefills": pc.prefix_forward_calls,   # should be 1
        "no_reuse_time_s": round(no_reuse_time, 4),
        "reuse_time_s": round(reuse_time, 4),
        "cache_stats": pc.stats(),
    }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from ..config import get_config
    from ..data.tokenizer import BPETokenizer
    from ..model.transformer import LyceumLM

    cfg = get_config("nano")
    tok = BPETokenizer(special_tokens=cfg.tokenizer.special_tokens)
    tok.train("the quick brown fox jumps over the lazy dog. " * 50,
              vocab_size=400)
    model = LyceumLM(cfg.model, vocab_size=tok.vocab_size).eval()

    # A shared "system prompt" prefix and two distinct requests.
    system = [tok.id("<bos>"), tok.id("<system>")] + tok.encode(
        "you are a helpful assistant")
    pc = PrefixCache(model, system, tenant="tenantA")
    print(f"prefilled prefix of {pc.prefix_len} tokens once "
          f"(forward_calls={pc.prefix_forward_calls})")

    req_a = tok.encode("hello there")
    req_b = tok.encode("good morning")

    out_a, fwd_a = None, None
    results = []
    for label, suffix in [("req_a", req_a), ("req_b", req_b)]:
        cache, start_pos = pc.clone_for_request()
        # sanity: cloned cache already contains the prefix's K/V
        assert cache.length() == pc.prefix_len, "clone lost the prefix K/V"
        gen, fwd = _decode_with_cache(model, cache, start_pos, suffix,
                                      max_new_tokens=4)
        results.append((label, gen, fwd))
        print(f"  {label}: start_pos={start_pos} produced {len(gen)} tokens "
              f"in {fwd} forward calls (prefix NOT among them)")

    # The prefix must have been prefilled exactly ONCE across both requests.
    assert pc.prefix_forward_calls == 1, "prefix was recomputed per request!"
    assert pc.hits == 2 and pc.misses == 0, pc.stats()
    print("  cache stats:", pc.stats())

    print("\n== benchmark: reuse vs recompute ==")
    bench = benchmark_prefix_reuse(model, system, [req_a, req_b, req_a],
                                   max_new_tokens=4)
    for k, v in bench.items():
        print(f"  {k}: {v}")
    assert bench["reuse_prefix_prefills"] == 1, "reuse should prefill prefix once"
    assert bench["no_reuse_prefix_prefills"] == bench["n_requests"], \
        "no-reuse should recompute the prefix every request"

    print("\nself-test OK: prefix prefilled once, reused across requests; "
          "no-reuse recomputes it every time.")
