"""Near-duplicate detection with MinHash + LSH.

Frontier-manual chapter, "curation": exact-hash dedup (which ``curation.curate``
already does) is *not enough*. Web corpora are full of *near* duplicates -- the
same article with a different header, boilerplate, or a few edited words. Left in,
they waste compute, bias the model toward boilerplate, and inflate memorization.

The standard scalable fix is MinHash + Locality-Sensitive Hashing (LSH):

  1. **Shingling** -- represent each document as the set of its overlapping
     word k-grams (here k=3). Two near-duplicate documents share most shingles,
     so their Jaccard similarity ``|A ∩ B| / |A ∪ B|`` is high.

  2. **MinHash** -- a Jaccard estimate that fits in a fixed-size signature.
     For each of ``num_perm`` hash permutations we keep the *minimum* hash over
     a document's shingles. The fraction of signature slots that agree between
     two docs is an unbiased estimate of their Jaccard similarity. Comparing
     short signatures is far cheaper than comparing full shingle sets.

  3. **LSH banding** -- split each signature into ``b`` bands of ``r`` rows.
     Two docs collide in a band only if all ``r`` rows match, so the probability
     they become candidates is ``1 - (1 - s**r)**b`` -- an S-curve with a sharp
     threshold near ``(1/b)**(1/r)``. We only ever compute exact similarity for
     docs that share a band, turning an O(n^2) all-pairs problem into roughly
     O(n).

Pure Python + (optionally) numpy; runs on a CPU. Self-testable below on a set
with obvious near-duplicates.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_WORD_RE = re.compile(r"\w+")

# A large prime modulus for the universal-hash MinHash permutations.
_MERSENNE = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def shingles(text: str, k: int = 3) -> set[int]:
    """Hashed word k-grams (shingles) of ``text``.

    Falls back to per-word shingles when the document is shorter than ``k``
    words, so even tiny docs get a non-empty representation.
    """
    words = _tokenize(text)
    if len(words) < k:
        grams = words or [""]
    else:
        grams = [" ".join(words[i:i + k]) for i in range(len(words) - k + 1)]
    out: set[int] = set()
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=4).digest(),
                           "little")
        out.add(h)
    return out


class MinHash:
    """Fixed-size MinHash signature estimator.

    ``signature(text)`` returns a length-``num_perm`` vector of ints; the share
    of equal positions between two signatures estimates their Jaccard
    similarity.
    """

    def __init__(self, num_perm: int = 64, k: int = 3, seed: int = 1):
        self.num_perm = num_perm
        self.k = k
        # Random (a, b) coefficients for affine universal hashing:
        #   h_i(x) = (a_i * x + b_i) mod p ; min over a doc's shingles.
        import random
        rng = random.Random(seed)
        self._a = [rng.randint(1, _MERSENNE - 1) for _ in range(num_perm)]
        self._b = [rng.randint(0, _MERSENNE - 1) for _ in range(num_perm)]

    def signature(self, text: str) -> list[int]:
        sh = shingles(text, self.k)
        if not sh:
            return [_MAX_HASH] * self.num_perm
        sig = []
        for a, b in zip(self._a, self._b):
            m = _MAX_HASH
            for x in sh:
                hv = ((a * x + b) % _MERSENNE) & _MAX_HASH
                if hv < m:
                    m = hv
            sig.append(m)
        return sig

    @staticmethod
    def estimate_jaccard(sig_a: list[int], sig_b: list[int]) -> float:
        if not sig_a:
            return 0.0
        eq = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
        return eq / len(sig_a)


def _bands_for_threshold(threshold: float, num_perm: int) -> int:
    """Pick a band count whose LSH S-curve crosses ~``threshold``.

    With ``b`` bands of ``r = num_perm/b`` rows the curve is steepest near
    ``(1/b)**(1/r)``; we search divisors of ``num_perm`` for the closest match.
    """
    best_b, best_err = 1, float("inf")
    for b in range(1, num_perm + 1):
        if num_perm % b:
            continue
        r = num_perm // b
        approx = (1.0 / b) ** (1.0 / r)
        err = abs(approx - threshold)
        if err < best_err:
            best_err, best_b = err, b
    return best_b


class LSH:
    """Locality-sensitive hashing index over MinHash signatures.

    ``add(id, sig)`` buckets a signature by band; ``query(sig)`` returns the set
    of ids that share at least one band (the near-duplicate *candidates*).
    """

    def __init__(self, threshold: float = 0.7, num_perm: int = 64,
                 bands: int | None = None):
        self.threshold = threshold
        self.num_perm = num_perm
        self.bands = bands or _bands_for_threshold(threshold, num_perm)
        if num_perm % self.bands:
            raise ValueError(f"bands ({self.bands}) must divide num_perm ({num_perm})")
        self.rows = num_perm // self.bands
        # one dict per band: band-hash -> list of ids
        self._buckets: list[dict[int, list]] = [dict() for _ in range(self.bands)]

    def _band_keys(self, sig: list[int]):
        for bi in range(self.bands):
            chunk = tuple(sig[bi * self.rows:(bi + 1) * self.rows])
            yield bi, hash(chunk)

    def add(self, id, sig: list[int]) -> None:
        for bi, key in self._band_keys(sig):
            self._buckets[bi].setdefault(key, []).append(id)

    def query(self, sig: list[int]) -> set:
        cand: set = set()
        for bi, key in self._band_keys(sig):
            cand.update(self._buckets[bi].get(key, ()))
        return cand


@dataclass
class DedupReport:
    input_docs: int = 0
    kept_docs: int = 0
    removed_near_dup: int = 0
    threshold: float = 0.7
    num_perm: int = 64
    bands: int = 0
    examples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


def near_dedup(docs: list[str], threshold: float = 0.7, num_perm: int = 64,
               k: int = 3, max_examples: int = 5) -> tuple[list[str], dict]:
    """Remove near-duplicate documents, keeping the first occurrence of each.

    Returns ``(kept_docs, report)`` where ``report`` is a ``DedupReport`` dict
    with the removed count and example near-dup pairs (kept_index, dropped_index,
    estimated Jaccard).
    """
    mh = MinHash(num_perm=num_perm, k=k)
    lsh = LSH(threshold=threshold, num_perm=num_perm)
    report = DedupReport(input_docs=len(docs), threshold=threshold,
                         num_perm=num_perm, bands=lsh.bands)

    sigs: dict[int, list[int]] = {}
    kept_idx: list[int] = []
    for i, doc in enumerate(docs):
        sig = mh.signature(doc)
        dup_of = None
        for cand in lsh.query(sig):
            j = MinHash.estimate_jaccard(sig, sigs[cand])
            if j >= threshold:
                dup_of = (cand, j)
                break
        if dup_of is not None:
            report.removed_near_dup += 1
            if len(report.examples) < max_examples:
                report.examples.append({
                    "kept_index": dup_of[0], "dropped_index": i,
                    "est_jaccard": round(dup_of[1], 3),
                    "kept_preview": docs[dup_of[0]][:60],
                    "dropped_preview": doc[:60],
                })
            continue
        # not a near-dup: index it and keep it
        sigs[i] = sig
        lsh.add(i, sig)
        kept_idx.append(i)

    report.kept_docs = len(kept_idx)
    return [docs[i] for i in kept_idx], report.to_dict()


# --------------------------------------------------------------------------- #
# Fast self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    base = ("The quick brown fox jumps over the lazy dog near the old river bank "
            "while the curious owl watches from a tall oak tree in the forest")
    docs = [
        base,
        base + " every single morning at dawn.",                 # near-dup of 0
        base.replace("brown fox", "brown FOX").replace("dog", "dog!"),  # near-dup of 0
        "Water is made of hydrogen and oxygen and forms rivers, lakes and rain.",
        "Bees make honey from the nectar of bright flowers in the spring garden.",
        "Water is made of hydrogen and oxygen and forms rivers, lakes, and rain too.",  # near-dup of 3
        "A triangle has three sides and three corners and three sharp angles.",
    ]

    # sanity: MinHash should estimate high Jaccard for the near-dups
    mh = MinHash(num_perm=64)
    s0, s1 = mh.signature(docs[0]), mh.signature(docs[1])
    s3, s5 = mh.signature(docs[3]), mh.signature(docs[5])
    print(f"est J(0,1) near-dup  = {MinHash.estimate_jaccard(s0, s1):.3f}")
    print(f"est J(3,5) near-dup  = {MinHash.estimate_jaccard(s3, s5):.3f}")
    print(f"est J(0,3) unrelated = {MinHash.estimate_jaccard(s0, s3):.3f}")

    kept, report = near_dedup(docs, threshold=0.6)
    print(f"\nbands={report['bands']} rows={report['num_perm'] // report['bands']}")
    print(f"input={report['input_docs']} kept={report['kept_docs']} "
          f"removed_near_dup={report['removed_near_dup']}")
    for ex in report["examples"]:
        print(f"  dropped #{ex['dropped_index']} ~ kept #{ex['kept_index']} "
              f"(J≈{ex['est_jaccard']}): {ex['dropped_preview']!r}")

    assert report["removed_near_dup"] >= 3, "should catch the obvious near-dups"
    assert report["kept_docs"] == len(docs) - report["removed_near_dup"]
    print("\nok: near-duplicate removal works")
