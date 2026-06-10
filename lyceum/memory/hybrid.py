"""Hybrid retrieval = semantic (vector) + keyword (BM25), fused.

System Design manual: pure semantic search misses *exact* matches (product
codes, rare proper nouns, error strings) because the embedding smears them into
a neighbourhood of "similar" tokens. The classic fix is to combine a dense
(vector / embedding) retriever with a sparse lexical retriever -- BM25 over an
inverted index -- and fuse the two ranked lists.

We fuse with Reciprocal Rank Fusion (RRF): a score depends only on a document's
*rank* in each list, so the two retrievers' incomparable score scales never have
to be normalised. RRF(d) = sum_over_lists 1 / (k + rank_d), conventionally
k = 60.

Everything here is pure Python + the project's existing offline embedder, so it
runs with no model download and no network.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .rag import VectorStore, Chunk, _embed  # noqa: F401  (_embed re-exported)

_TOK = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word split -- the same notion of a 'term' BM25 indexes."""
    return _TOK.findall(text.lower())


# --------------------------------------------------------------------------- #
# BM25 over an in-memory inverted index.
# --------------------------------------------------------------------------- #
class BM25:
    """Okapi BM25 ranking over a list of texts.

    BM25 scores a document for a query by, per query term:
        IDF(term) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |D| / avgdl))
    where tf is the term frequency in the document, |D| the document length and
    avgdl the average document length. k1 controls term-frequency saturation and
    b controls length normalisation.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[list[str]] = []        # tokenized documents
        self.doc_len: list[int] = []
        self.tf: list[Counter] = []            # per-doc term -> count
        self.df: Counter = Counter()           # term -> # docs containing it
        self.avgdl: float = 0.0

    def add(self, text: str) -> int:
        """Index one document; returns its integer id (its position)."""
        toks = _tokenize(text)
        tf = Counter(toks)
        self.docs.append(toks)
        self.doc_len.append(len(toks))
        self.tf.append(tf)
        for term in tf:                        # df counts documents, not occurrences
            self.df[term] += 1
        n = len(self.doc_len)
        self.avgdl = sum(self.doc_len) / n if n else 0.0
        return n - 1

    def _idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.df.get(term, 0)
        # BM25 idf with +0.5 smoothing; max(0, .) guards the degenerate
        # "term in (almost) every doc" case from going negative.
        return max(0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0))

    def score(self, query_terms: list[str], idx: int) -> float:
        if not self.docs:
            return 0.0
        tf = self.tf[idx]
        dl = self.doc_len[idx]
        denom_len = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
        s = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            s += self._idf(term) * (f * (self.k1 + 1)) / (f + denom_len)
        return s

    def search(self, query: str, k: int = 3) -> list[tuple[float, int]]:
        """Return up to k (score, doc_idx) pairs, best first, score > 0 only."""
        terms = _tokenize(query)
        scored = [(self.score(terms, i), i) for i in range(len(self.docs))]
        scored = [(s, i) for s, i in scored if s > 0.0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]


# --------------------------------------------------------------------------- #
# Hybrid retriever: dense (VectorStore) + sparse (BM25), fused with RRF.
# --------------------------------------------------------------------------- #
class HybridRetriever:
    """Fuse the VectorStore's semantic search with a BM25 lexical search.

    The BM25 index is built over the *same* chunks the VectorStore holds
    (``vector_store.chunks``), so the two retrievers rank the same universe of
    documents and their ranks can be fused directly by chunk identity.
    """

    def __init__(self, vector_store: VectorStore, *, rrf_k: int = 60):
        self.store = vector_store
        self.rrf_k = rrf_k
        self.bm25: BM25 = BM25()
        self._indexed_chunks: list[Chunk] = []
        self.rebuild()

    def rebuild(self) -> None:
        """(Re)build the BM25 index from the current store contents.

        Call this after documents are added to the underlying VectorStore so the
        lexical index stays in sync with the dense one.
        """
        self.bm25 = BM25()
        self._indexed_chunks = list(self.store.chunks)
        for c in self._indexed_chunks:
            self.bm25.add(c.text)

    def _visible(self, chunk: Chunk, tenant: str | None) -> bool:
        """Per-tenant authorization filter: a tenant may see its own chunks and
        anything marked 'public' (mirrors VectorStore.search)."""
        if tenant is None:
            return True
        return chunk.tenant in (tenant, "public")

    def search(self, query: str, k: int = 3,
               tenant: str | None = None) -> list[tuple[float, Chunk]]:
        """Hybrid search. Returns up to k (fused_score, Chunk), best first.

        Dense half: reuse VectorStore.search (which already enforces the tenant
        filter). Sparse half: BM25 over the indexed chunks, then drop hits not
        visible to the tenant. Fuse the two rank lists with RRF.
        """
        if len(self.store.chunks) != len(self._indexed_chunks):
            self.rebuild()  # store changed under us; keep indexes consistent

        # Dense ranking -- ask for a generous pool so fusion has material.
        pool = max(k * 5, 10)
        dense = self.store.search(query, k=pool, tenant=tenant)

        # Sparse ranking -- BM25 over all indexed chunks, then tenant-filter.
        sparse_raw = self.bm25.search(query, k=len(self._indexed_chunks) or 1)
        sparse: list[tuple[float, Chunk]] = []
        for _score, idx in sparse_raw:
            c = self._indexed_chunks[idx]
            if self._visible(c, tenant):
                sparse.append((_score, c))
            if len(sparse) >= pool:
                break

        # Reciprocal Rank Fusion. Key chunks by identity so the same chunk
        # appearing in both lists accumulates contributions.
        fused: dict[int, float] = {}
        ref: dict[int, Chunk] = {}
        for rank, (_s, c) in enumerate(dense, start=1):
            fused[id(c)] = fused.get(id(c), 0.0) + 1.0 / (self.rrf_k + rank)
            ref[id(c)] = c
        for rank, (_s, c) in enumerate(sparse, start=1):
            fused[id(c)] = fused.get(id(c), 0.0) + 1.0 / (self.rrf_k + rank)
            ref[id(c)] = c

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return [(score, ref[cid]) for cid, score in ranked[:k]]


# --------------------------------------------------------------------------- #
# Self-test.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    store = VectorStore()
    # Some docs share an exact rare keyword ("XJ9000"); others are semantically
    # about the same topic but never use that literal token.
    store.add("The XJ9000 turbo encoder ships with a 5-year warranty.",
              source="manual", trust="high")
    store.add("Our flagship encoder hardware accelerates video at high speed.",
              source="brochure", trust="high")
    store.add("Photosynthesis converts sunlight into chemical energy in plants.",
              source="biology", trust="high")
    store.add("Green leaves capture light to make sugars for the plant.",
              source="biology2", trust="high")
    store.add("Return policy: items may be returned within 30 days.",
              source="policy", trust="medium")
    # A tenant-private doc that ALSO mentions XJ9000 -- must stay isolated.
    store.add("Internal note: XJ9000 secret pricing for tenant acme only.",
              source="crm", trust="high", tenant="acme")

    hr = HybridRetriever(store)

    print("=== Query: 'XJ9000 warranty' (public tenant) ===")
    hits = hr.search("XJ9000 warranty", k=3, tenant="public")
    for score, c in hits:
        print(f"  {score:.4f}  [{c.source}/{c.tenant}] {c.text[:60]}")
    sources = {c.source for _, c in hits}
    assert "manual" in sources, "exact-keyword doc (BM25) should surface"
    # tenant isolation: acme's private doc must NOT appear for 'public'
    assert "crm" not in sources, "tenant isolation breached!"
    print("  -> exact-keyword 'manual' surfaced; acme/crm correctly hidden")

    print("\n=== Query: 'how plants use light to grow' (semantic) ===")
    hits = hr.search("how plants use light to grow", k=3, tenant="public")
    for score, c in hits:
        print(f"  {score:.4f}  [{c.source}/{c.tenant}] {c.text[:60]}")
    sources = {c.source for _, c in hits}
    assert sources & {"biology", "biology2"}, "semantic doc should surface"
    print("  -> semantic plant doc surfaced without sharing query keywords")

    print("\n=== Tenant 'acme' sees its private XJ9000 note ===")
    hits = hr.search("XJ9000 pricing", k=3, tenant="acme")
    for score, c in hits:
        print(f"  {score:.4f}  [{c.source}/{c.tenant}] {c.text[:60]}")
    assert any(c.source == "crm" for _, c in hits), "acme should see its own doc"
    print("  -> acme tenant correctly sees its private doc")

    print("\nhybrid.py self-test OK")
