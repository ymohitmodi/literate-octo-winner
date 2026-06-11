"""Eval-set decontamination via n-gram overlap removal.

Frontier-manual rule (MANDATORY): before training you must remove any training
document that overlaps your evaluation/benchmark sets, or your reported scores
measure *memorization*, not capability. A model that saw the test answers during
pretraining will look far better than it is -- the single most common way
benchmark numbers get silently inflated.

The standard, scalable check is **n-gram overlap**. We represent each text by
its set of hashed contiguous ``n``-grams (n=13 is the common GPT-3/PaLM choice:
long enough that a 13-token match is almost never coincidental). A training doc
is *contaminated* if the fraction of its n-grams that also appear in ANY eval
text exceeds ``max_overlap``; such docs are dropped.

  * ``ngram_set(text, n)``       -> set of hashed n-grams (word-level)
  * ``decontaminate(...)``       -> (clean_docs, report) dropping contaminated docs
  * ``contamination_report(...)``-> overlap statistics without modifying data

Pure Python; runs on a CPU. Self-test below plants a benchmark string inside a
training doc and confirms it is removed.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _hash_gram(gram: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "little")


def ngram_set(text: str, n: int = 13) -> set[int]:
    """Set of hashed contiguous word ``n``-grams.

    If the text has fewer than ``n`` words, it is hashed whole (a single gram),
    so short eval prompts are still matchable.
    """
    words = _tokenize(text)
    if len(words) < n:
        return {_hash_gram(" ".join(words))} if words else set()
    return {_hash_gram(" ".join(words[i:i + n]))
            for i in range(len(words) - n + 1)}


def _max_eval_coverage(doc_grams: set[int],
                       eval_grams_list: list[set[int]]) -> float:
    """Largest fraction of ANY single eval text's n-grams found in this doc.

    Normalizing by the *eval* side (not the doc) is what lets a long training
    document that embeds a short benchmark still be flagged: the embedded eval
    text is ~fully covered even though it is a small slice of the big doc. This
    matches the GPT-3/PaLM contamination check, which asks "does this train doc
    contain (most of) an eval example?", not "is this doc mostly eval text?".
    """
    if not doc_grams:
        return 0.0
    best = 0.0
    for eg in eval_grams_list:
        if not eg:
            continue
        cov = len(doc_grams & eg) / len(eg)
        if cov > best:
            best = cov
    return best


@dataclass
class DecontaminationReport:
    input_docs: int = 0
    kept_docs: int = 0
    removed_contaminated: int = 0
    n: int = 13
    max_overlap: float = 0.5
    examples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


def _build_eval_index(eval_texts: list[str], n: int) -> list[set[int]]:
    """One n-gram set per eval text, so coverage can be measured per eval
    example (a long train doc that embeds one benchmark is still caught)."""
    return [ngram_set(t, n) for t in eval_texts]


def decontaminate(train_docs: list[str], eval_texts: list[str], n: int = 13,
                  max_overlap: float = 0.5, max_examples: int = 5
                  ) -> tuple[list[str], dict]:
    """Drop training docs that cover more than ``max_overlap`` of ANY eval
    text's n-grams (i.e. that appear to embed a benchmark example).

    Returns ``(clean_docs, report)`` where ``report`` is a
    ``DecontaminationReport`` dict (how many removed, plus examples).
    """
    eval_idx = _build_eval_index(eval_texts, n)
    report = DecontaminationReport(input_docs=len(train_docs), n=n,
                                   max_overlap=max_overlap)
    kept: list[str] = []
    for i, doc in enumerate(train_docs):
        grams = ngram_set(doc, n)
        frac = _max_eval_coverage(grams, eval_idx)
        if frac > max_overlap:
            report.removed_contaminated += 1
            if len(report.examples) < max_examples:
                report.examples.append({
                    "doc_index": i,
                    "overlap": round(frac, 3),
                    "preview": doc[:80],
                })
            continue
        kept.append(doc)
    report.kept_docs = len(kept)
    return kept, report.to_dict()


def contamination_report(train_docs: list[str], eval_texts: list[str],
                         n: int = 13) -> dict:
    """Overlap statistics across the training set, without removing anything.

    Reports how many docs have any overlap, the max/mean eval-coverage, and the
    worst offenders -- useful for choosing ``max_overlap``.
    """
    eval_idx = _build_eval_index(eval_texts, n)
    n_eval_ngrams = len(set().union(*eval_idx)) if eval_idx else 0
    fractions: list[float] = []
    worst: list[dict] = []
    docs_with_overlap = 0
    for i, doc in enumerate(train_docs):
        frac = _max_eval_coverage(ngram_set(doc, n), eval_idx)
        fractions.append(frac)
        if frac > 0:
            docs_with_overlap += 1
            worst.append({"doc_index": i, "overlap": round(frac, 3),
                          "preview": doc[:80]})
    worst.sort(key=lambda d: d["overlap"], reverse=True)
    mean = sum(fractions) / len(fractions) if fractions else 0.0
    return {
        "n": n,
        "input_docs": len(train_docs),
        "eval_texts": len(eval_texts),
        "eval_ngrams": n_eval_ngrams,
        "docs_with_any_overlap": docs_with_overlap,
        "max_overlap": round(max(fractions), 3) if fractions else 0.0,
        "mean_overlap": round(mean, 4),
        "worst": worst[:5],
    }


# --------------------------------------------------------------------------- #
# Fast self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    eval_texts = [
        "The capital of France is Paris and it sits on the river Seine in Europe today",
        "Photosynthesis converts sunlight carbon dioxide and water into glucose and oxygen",
    ]
    clean_train = [
        "Once there was a tiny fox named Mia who found a shiny key in the forest.",
        "Bees make honey from the nectar of flowers during the warm summer months.",
        "A triangle has three sides and three corners and three sharp inner angles.",
    ]
    # a doc that has leaked a benchmark question verbatim -> must be removed
    contaminated = ("Study notes: " +
                    eval_texts[0] +
                    " -- remember this fact for the geography quiz next week.")
    train = clean_train + [contaminated]

    rep_before = contamination_report(train, eval_texts, n=13)
    print("contamination_report (before):")
    print(f"  eval_ngrams={rep_before['eval_ngrams']} "
          f"docs_with_any_overlap={rep_before['docs_with_any_overlap']} "
          f"max_overlap={rep_before['max_overlap']}")
    for w in rep_before["worst"]:
        print(f"  worst doc #{w['doc_index']} overlap={w['overlap']}: {w['preview']!r}")

    clean, rep = decontaminate(train, eval_texts, n=13, max_overlap=0.5)
    print(f"\ndecontaminate: input={rep['input_docs']} kept={rep['kept_docs']} "
          f"removed_contaminated={rep['removed_contaminated']}")
    for ex in rep["examples"]:
        print(f"  removed doc #{ex['doc_index']} overlap={ex['overlap']}: {ex['preview']!r}")

    assert rep["removed_contaminated"] == 1, "should remove exactly the leaked doc"
    assert rep["kept_docs"] == len(clean_train)
    assert all(eval_texts[0] not in d for d in clean)
    print("\nok: eval-set decontamination works")
