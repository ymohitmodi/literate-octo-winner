"""PagedAttention-style KV-cache allocator (educational, standalone).

Why this exists (Frontier manual / SysDesign chapter): the real capacity limit
of an LLM server is not "how many requests" but **how many tokens of KV cache**
fit in memory. The KV cache grows linearly with sequence length, and a naive
serving stack pre-allocates one big *contiguous* buffer per sequence sized to the
maximum length. That wastes memory two ways:

  * **internal fragmentation** - a sequence reserved for ``max_seq_len`` but only
    100 tokens long pins all the unused slots;
  * **external fragmentation** - freed contiguous blocks of different sizes leave
    holes that no new sequence quite fits into.

PagedAttention (vLLM) borrows the operating-system trick of *virtual memory*: cut
the KV cache into fixed-size **pages** (a block of ``page_size`` token slots per
layer) and let each sequence hold a *page table* of possibly-non-contiguous
pages. Because every page is the same size, there is **no external
fragmentation**, and a sequence only ever wastes at most one partially-filled
page (bounded internal fragmentation). Two sequences that share a common prompt
prefix can even **share** the prefix pages (copy-on-write), which is how a server
serves many samples of one prompt, or a big system prompt, almost for free.

This module is a faithful *allocator* model of that mechanism. It does NOT plug
into the attention kernel -- it deliberately stays standalone so the memory
bookkeeping is visible and testable. It tracks:

  * a global **free-list** of page ids (capacity = ``total_pages`` pages, i.e.
    ``total_pages * page_size`` tokens of KV cache, *per layer*);
  * a **page table** per sequence (the ordered list of pages backing it);
  * **reference counts** per page so shared (copy-on-write) prefix pages are only
    returned to the free-list when the last sharer frees them;
  * metrics: utilization, fragmentation (~0, the whole point), pages in use.

Out-of-pages is surfaced loudly via ``OutOfPagesError`` -- running out of KV
cache is the canonical capacity failure of an inference server.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


class OutOfPagesError(RuntimeError):
    """Raised when the allocator cannot satisfy a request: the KV cache is full.

    This is the key SysDesign signal -- server capacity is *tokens of KV cache*,
    not requests. When this fires, a real server must queue, evict (preemption),
    or reject the request; it cannot magic more memory into existence.
    """


@dataclass
class _SeqTable:
    """Per-sequence page table: which pages back it, and how many tokens it holds."""
    pages: list[int] = field(default_factory=list)  # ordered page ids
    n_tokens: int = 0                                # logical length in tokens


class PagedKVCache:
    """Fixed-size-page KV-cache allocator with prefix sharing.

    Parameters
    ----------
    n_layers : int
        Number of transformer layers. A real KV cache stores K and V *per layer*;
        we keep ``n_layers`` only to report a realistic total token capacity and
        to make the model honest about what a "page" represents (one block of
        token slots, replicated across every layer). The allocation math is the
        same for any layer, so we track page ids once and scale reporting by
        layers.
    page_size : int
        Token slots per page (vLLM's default block size is 16). Larger pages mean
        less bookkeeping but more internal fragmentation in the last page.
    total_pages : int
        Size of the physical page pool. Total capacity is
        ``total_pages * page_size`` tokens of KV cache.
    """

    def __init__(self, n_layers: int, page_size: int = 16, total_pages: int = 128):
        if page_size <= 0 or total_pages <= 0 or n_layers <= 0:
            raise ValueError("n_layers, page_size, total_pages must all be > 0")
        self.n_layers = n_layers
        self.page_size = page_size
        self.total_pages = total_pages

        # Physical free-list: every page id is initially free.
        self.free_pages: list[int] = list(range(total_pages))
        # Per-page reference count (for copy-on-write prefix sharing).
        self.ref_count: list[int] = [0] * total_pages
        # Logical page tables, keyed by sequence id.
        self.seqs: dict[object, _SeqTable] = {}

    # ------------------------------------------------------------------ #
    # Internal page-pool helpers
    # ------------------------------------------------------------------ #
    def _pages_for_tokens(self, n_tokens: int) -> int:
        """Ceil-divide tokens into whole pages (the last page may be partial)."""
        return math.ceil(n_tokens / self.page_size) if n_tokens > 0 else 0

    def _take_pages(self, n_pages: int) -> list[int]:
        """Pop ``n_pages`` fresh pages from the free-list, refcount them, or raise."""
        if n_pages > len(self.free_pages):
            raise OutOfPagesError(
                f"out of KV-cache pages: need {n_pages} more but only "
                f"{len(self.free_pages)} free (capacity is "
                f"{self.total_pages} pages = {self.token_capacity()} tokens). "
                f"A real server would queue/evict/reject here."
            )
        taken = [self.free_pages.pop() for _ in range(n_pages)]
        for p in taken:
            self.ref_count[p] = 1
        return taken

    def _release_page(self, page: int) -> None:
        """Drop one reference to a page; return it to the free-list at zero refs."""
        self.ref_count[page] -= 1
        if self.ref_count[page] <= 0:
            self.ref_count[page] = 0
            self.free_pages.append(page)

    def _last_page_used_slots(self, seq: _SeqTable) -> int:
        """How many token slots of the sequence's last page are already filled.

        0 means the last page is exactly full (or the sequence is empty), so the
        next append must grab a new page.
        """
        rem = seq.n_tokens % self.page_size
        return rem  # 0 => last page full / no pages yet

    # ------------------------------------------------------------------ #
    # Public allocator API
    # ------------------------------------------------------------------ #
    def allocate(self, seq_id: object, n_tokens: int) -> list[int]:
        """Allocate backing pages for a brand-new sequence of ``n_tokens`` tokens.

        Returns the sequence's page table (list of physical page ids). Raises
        ``OutOfPagesError`` if the pool cannot satisfy the request.
        """
        if seq_id in self.seqs:
            raise ValueError(f"sequence {seq_id!r} already allocated; use append()")
        if n_tokens < 0:
            raise ValueError("n_tokens must be >= 0")
        n_pages = self._pages_for_tokens(n_tokens)
        pages = self._take_pages(n_pages)
        self.seqs[seq_id] = _SeqTable(pages=pages, n_tokens=n_tokens)
        return list(pages)

    def append(self, seq_id: object, n_new: int) -> list[int]:
        """Grow an existing sequence by ``n_new`` tokens (the decode-step path).

        Reuses the slack left in the current last page first, then pulls new pages
        only as needed -- exactly how decoding appends one token at a time without
        reallocating the whole sequence. Returns any *newly added* page ids.
        """
        if seq_id not in self.seqs:
            raise KeyError(f"unknown sequence {seq_id!r}; call allocate() first")
        if n_new < 0:
            raise ValueError("n_new must be >= 0")
        seq = self.seqs[seq_id]

        # Free slots remaining in the current last page (0 if full/empty).
        used = self._last_page_used_slots(seq)
        slack = (self.page_size - used) if used != 0 else 0
        need_new_tokens = max(0, n_new - slack)
        n_pages = self._pages_for_tokens(need_new_tokens)

        new_pages: list[int] = []
        if n_pages:
            new_pages = self._take_pages(n_pages)
            seq.pages.extend(new_pages)
        seq.n_tokens += n_new
        return new_pages

    def share_prefix(self, parent_seq: object, child_seq: object,
                     n_prefix_tokens: int) -> list[int]:
        """Create ``child_seq`` sharing ``parent_seq``'s first ``n_prefix_tokens``.

        This is copy-on-write prefix sharing: the child's page table *points at*
        the same physical pages as the parent for the shared prefix, and we bump
        those pages' reference counts instead of copying KV bytes. The child can
        then ``append`` its own continuation, which allocates fresh pages -- so the
        shared pages are never mutated (copy-on-write at page granularity).

        Sharing must fall on a page boundary so neither sequence writes into a page
        the other reads; we therefore share whole prefix pages only. Returns the
        list of shared physical page ids.
        """
        if parent_seq not in self.seqs:
            raise KeyError(f"unknown parent sequence {parent_seq!r}")
        if child_seq in self.seqs:
            raise ValueError(f"child sequence {child_seq!r} already exists")
        parent = self.seqs[parent_seq]
        if n_prefix_tokens < 0 or n_prefix_tokens > parent.n_tokens:
            raise ValueError(
                f"n_prefix_tokens ({n_prefix_tokens}) must be in "
                f"[0, parent length {parent.n_tokens}]"
            )

        # Share only whole pages: round the prefix DOWN to a page boundary so the
        # child never shares a page the parent might still be writing into.
        n_shared_pages = n_prefix_tokens // self.page_size
        shared = parent.pages[:n_shared_pages]
        for p in shared:
            self.ref_count[p] += 1  # copy-on-write: bump refs, copy no bytes

        # The child logically owns exactly the tokens in those shared pages.
        self.seqs[child_seq] = _SeqTable(
            pages=list(shared), n_tokens=n_shared_pages * self.page_size
        )
        return list(shared)

    def free(self, seq_id: object) -> int:
        """Free a sequence, returning its (non-shared) pages to the pool.

        Shared pages are only truly released when their reference count hits zero,
        so freeing one sharer of a prefix does not pull pages out from under the
        others. Returns the number of pages physically returned to the free-list.
        """
        if seq_id not in self.seqs:
            raise KeyError(f"unknown sequence {seq_id!r}")
        seq = self.seqs.pop(seq_id)
        before = len(self.free_pages)
        for p in seq.pages:
            self._release_page(p)
        return len(self.free_pages) - before

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def token_capacity(self) -> int:
        """Total tokens of KV cache the pool can hold (per layer)."""
        return self.total_pages * self.page_size

    def pages_in_use(self) -> int:
        """Distinct physical pages currently allocated (shared pages counted once)."""
        return self.total_pages - len(self.free_pages)

    def stats(self) -> dict:
        """Report capacity, utilization, fragmentation, and sharing.

        ``fragmentation`` is the wasted fraction *within* allocated pages (the only
        kind paging suffers: a bounded sliver in each sequence's last page). There
        is **no external fragmentation** because all pages are identical and drawn
        from one pool -- the headline advantage over a contiguous allocator, which
        would leave unusable holes between variable-size blocks.
        """
        pages_used = self.pages_in_use()
        token_cap = self.token_capacity()
        slots_in_use = pages_used * self.page_size

        # Logical tokens actually live across all sequences. Shared prefix pages
        # are backing >1 sequence, so summing per-sequence lengths can exceed the
        # physical slots in use -- that "over-100%" logical figure is exactly the
        # memory that prefix sharing saves us, so we report it separately.
        logical_tokens = sum(s.n_tokens for s in self.seqs.values())
        # Logical tokens that are physically resident (cap at the slots we hold).
        resident_logical = min(logical_tokens, slots_in_use)

        # Internal fragmentation: allocated slots that hold no live token.
        wasted_slots = max(0, slots_in_use - resident_logical)
        fragmentation = (wasted_slots / slots_in_use) if slots_in_use else 0.0

        utilization = (pages_used / self.total_pages) if self.total_pages else 0.0

        shared_pages = sum(1 for r in self.ref_count if r > 1)

        return {
            "total_pages": self.total_pages,
            "page_size": self.page_size,
            "pages_in_use": pages_used,
            "free_pages": len(self.free_pages),
            "token_capacity": token_cap,
            "tokens_resident": resident_logical,
            "logical_tokens": logical_tokens,   # may exceed resident via sharing
            "utilization": round(utilization, 4),
            # ~0.0 except a bounded sliver in each sequence's last partial page;
            # crucially there is NO external fragmentation (the contiguous-allocator
            # failure mode), so holes never strand otherwise-usable capacity.
            "fragmentation": round(fragmentation, 4),
            "shared_pages": shared_pages,
            "n_sequences": len(self.seqs),
            "n_layers": self.n_layers,
        }


# --------------------------------------------------------------------------- #
# Demo: allocate several sequences, share a prefix, free some, show no
# external fragmentation.
# --------------------------------------------------------------------------- #
def demo() -> dict:
    """Walk through the allocator and print metrics at each step.

    Returns the final ``stats()`` dict so callers/tests can assert on it.
    """
    print("=" * 64)
    print("PagedKVCache demo -- KV cache as paged virtual memory")
    print("=" * 64)

    # 4 layers, 8 token slots per page, 16 pages => 128 tokens of KV capacity.
    cache = PagedKVCache(n_layers=4, page_size=8, total_pages=16)
    print(f"pool: {cache.total_pages} pages x {cache.page_size} slots "
          f"= {cache.token_capacity()} tokens of KV cache (per layer)\n")

    # Allocate three sequences of different (and non-page-aligned) lengths.
    cache.allocate("A", 20)   # 20 tokens -> ceil(20/8) = 3 pages
    cache.allocate("B", 5)    # 5 tokens  -> 1 page
    cache.allocate("C", 16)   # 16 tokens -> 2 pages
    print("allocated A=20, B=5, C=16 tokens")
    _show(cache)

    # Decode steps: grow A and B one/few tokens at a time (reuses last-page slack).
    cache.append("A", 1)      # fits in A's partially-filled 3rd page
    cache.append("B", 10)     # needs new pages
    print("\nappended A+=1, B+=10 (decode steps reuse last-page slack first)")
    _show(cache)

    # Prefix sharing: D continues A's first 16 tokens (2 whole pages) copy-on-write.
    shared = cache.share_prefix("A", "D", n_prefix_tokens=16)
    print(f"\nshared A's 16-token prefix with D (copy-on-write pages {shared}); "
          f"no KV bytes copied")
    cache.append("D", 12)     # D writes its own continuation into fresh pages
    print("appended D+=12 (its own pages; shared prefix stays read-only)")
    _show(cache)

    # Free B; its pages return to the pool. Shared prefix pages survive because A
    # (and D) still reference them.
    returned = cache.free("B")
    print(f"\nfreed B -> {returned} pages returned to the pool")
    _show(cache)

    stats = cache.stats()
    print("\n" + "-" * 64)
    print(f"fragmentation = {stats['fragmentation']} "
          f"(no EXTERNAL fragmentation: equal-size pages never strand holes; "
          f"only a bounded sliver in each last page).")
    print(f"prefix sharing let {stats['logical_tokens']} logical tokens live in "
          f"{stats['tokens_resident']} physical slots.")
    print("=" * 64)
    return stats


def _show(cache: PagedKVCache) -> None:
    s = cache.stats()
    print(f"  pages_in_use={s['pages_in_use']}/{s['total_pages']}  "
          f"util={s['utilization']:.2f}  frag={s['fragmentation']:.3f}  "
          f"shared={s['shared_pages']}  seqs={s['n_sequences']}")


if __name__ == "__main__":
    final = demo()
    # Sanity checks: paging gives (near-)zero fragmentation and respects capacity.
    assert final["fragmentation"] < 0.5, "paging should keep fragmentation low"
    assert final["pages_in_use"] <= final["total_pages"], "over-allocated the pool"

    # Demonstrate the capacity failure (running out of KV cache) is loud + clear.
    small = PagedKVCache(n_layers=2, page_size=4, total_pages=2)  # 8 tokens total
    small.allocate("big", 8)  # fills the pool exactly
    try:
        small.allocate("overflow", 1)
        raise AssertionError("expected OutOfPagesError when the KV cache is full")
    except OutOfPagesError as e:
        print(f"\nout-of-KV-cache handled cleanly: {e}")

    print("\nself-test OK: paged KV allocator runs, shares prefixes, and signals "
          "capacity exhaustion (tokens-of-KV-cache is the real server limit).")
