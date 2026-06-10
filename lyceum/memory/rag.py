"""Retrieval-Augmented Generation + a hardened memory store.

Offline by design: embeddings are a pure-numpy hashed-ngram TF-IDF vector, so
the project needs no model download and runs with no network. The *mechanism*
(embed -> vector store -> nearest-neighbour retrieve -> assemble context) is
identical to a production RAG stack; only the embedder is tiny.

Security controls woven in (AI Security manual, Stage 6):
  * per-tenant authorization filter at retrieval (similarity has no notion of
    ownership; without this you leak another tenant's documents)
  * provenance + trust tier on every chunk
  * cite-or-abstain answering (answer only from retrieved context, or abstain)
  * the 4-gate hardened memory write path (scan -> trust tier -> corroboration
    floor -> bounded write) plus a read-only "policy" store the agent cannot
    overwrite ("can't rewrite its own constitution")
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field

import numpy as np

_TOK = re.compile(r"[a-z0-9]+")
_DIM = 512


def _embed(text: str) -> np.ndarray:
    """Hashed character/word n-gram bag -> L2-normalized vector."""
    vec = np.zeros(_DIM, dtype=np.float32)
    words = _TOK.findall(text.lower())
    grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    for g in grams:
        h = int(hashlib.md5(g.encode()).hexdigest(), 16)
        vec[h % _DIM] += 1.0
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


@dataclass
class Chunk:
    text: str
    source: str = "unknown"
    trust: str = "medium"        # high | medium | low
    tenant: str = "public"
    vec: np.ndarray = field(default=None, repr=False)


class VectorStore:
    """A tiny in-memory vector index with cosine similarity + ANN-style top-k.
    Stands in for an HNSW vector DB; the retrieval contract is the same."""

    def __init__(self):
        self.chunks: list[Chunk] = []

    def add(self, text: str, *, source="unknown", trust="medium", tenant="public"):
        self.chunks.append(Chunk(text, source, trust, tenant, _embed(text)))

    def add_documents(self, docs: list[str], *, source="kb", trust="high",
                      tenant="public", chunk_chars=400):
        for d in docs:
            for i in range(0, len(d), chunk_chars):
                self.add(d[i:i + chunk_chars], source=source, trust=trust,
                         tenant=tenant)

    def search(self, query: str, k: int = 3, *, tenant: str | None = None,
               min_trust: str | None = None) -> list[tuple[float, Chunk]]:
        q = _embed(query)
        order = {"low": 0, "medium": 1, "high": 2}
        out = []
        for c in self.chunks:
            # per-tenant authorization filter (hard ownership boundary)
            if tenant is not None and c.tenant not in (tenant, "public"):
                continue
            if min_trust is not None and order[c.trust] < order[min_trust]:
                continue
            out.append((float(np.dot(q, c.vec)), c))
        out.sort(key=lambda x: x[0], reverse=True)
        return out[:k]


def assemble_context(hits: list[tuple[float, Chunk]]) -> tuple[str, list[str]]:
    """Wrap retrieved text as DATA, never instructions (spotlighting). Each
    span is delimited and tagged with its source so the answer can cite it."""
    lines, cites = [], []
    for i, (score, c) in enumerate(hits, 1):
        cites.append(c.source)
        lines.append(f"[doc {i} | source={c.source} | trust={c.trust}]\n{c.text}")
    return "\n\n".join(lines), cites


def cite_or_abstain(hits: list[tuple[float, Chunk]], threshold: float = 0.15):
    """Sufficiency check: if nothing is similar enough, abstain instead of
    hallucinating (blunts both hallucination and RAG poisoning)."""
    if not hits or hits[0][0] < threshold:
        return False, "I don't have enough grounded information to answer that."
    return True, None


# --------------------------------------------------------------------------- #
# Hardened memory write path (4 gates) + read-only policy store
# --------------------------------------------------------------------------- #
class MemoryStore:
    """Durable agent memory with the manual's 4-gate write path. A naive memory
    that writes whatever it's told is the ASI06 memory-poisoning vulnerability."""

    def __init__(self, max_items: int = 200, max_chars: int = 500):
        self.items: list[Chunk] = []
        self.policy: list[str] = []     # read-only "constitution"
        self.max_items = max_items
        self.max_chars = max_chars
        self.rejections: list[dict] = []

    def set_policy(self, rules: list[str]):
        self.policy = list(rules)       # only settable by the operator, not the agent

    def write(self, text: str, *, source: str, trust: str,
              corroborations: int = 1, confidence: float = 0.5) -> tuple[bool, str]:
        # Gate 1: scan at ingest (reject obvious injected instructions)
        if re.search(r"ignore (all|previous|prior)|you are now|system prompt",
                     text, re.I):
            return self._reject(text, "gate1: injection pattern at ingest")
        # Gate 2: trust tier (low-trust sources cannot mint durable memory)
        if trust == "low":
            return self._reject(text, "gate2: source trust too low to persist")
        # Gate 3: corroboration floor (durable belief needs >=2 sources)
        if corroborations < 2:
            return self._reject(text, "gate3: needs >=2 independent sources")
        # Gate 4: bounded write (size cap + confidence band 0.4-0.8)
        if len(text) > self.max_chars or not (0.4 <= confidence <= 0.8):
            return self._reject(text, "gate4: out of size/confidence bounds")
        if len(self.items) >= self.max_items:
            self.items.pop(0)
        self.items.append(Chunk(text, source, trust, vec=_embed(text)))
        return True, "written"

    def _reject(self, text, reason):
        self.rejections.append({"text": text[:80], "reason": reason})
        return False, reason

    def recall(self, query: str, k: int = 3) -> list[str]:
        q = _embed(query)
        scored = sorted(((float(np.dot(q, c.vec)), c.text) for c in self.items),
                        reverse=True)
        # policy (constitution) is always recalled and cannot be overwritten
        return self.policy + [t for _, t in scored[:k]]
