"""Data curation + provenance.

Frontier-model quality is mostly a data problem, and frontier-model *security*
starts at the data layer. This module does both:

  * quality filtering, dedup, length filtering   (Frontier manual: curation)
  * PII / secret scrubbing before training        (Security manual, Stage 1/5)
  * content-hash provenance manifest              (Security manual, Stage 1/3:
       pin by content hash, not URL; emit a training bill-of-materials)

Keeping data minimization here is the single most reliable defense against
training-data extraction: the model can never leak a secret it never saw.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Patterns for obvious PII / secrets. Deliberately simple and readable; a real
# system would use NER, but the teaching point is "scrub before you train".
PII_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "API_KEY": re.compile(r"\b(?:sk|pk|ghp|AKIA)[A-Za-z0-9_\-]{12,}\b"),
}


def scrub_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace PII spans with typed placeholders. Returns (clean, counts)."""
    counts: dict[str, int] = {}
    for label, pat in PII_PATTERNS.items():
        text, n = pat.subn(f"<{label}>", text)
        if n:
            counts[label] = n
    return text, counts


@dataclass
class CurationReport:
    input_docs: int = 0
    kept_docs: int = 0
    removed_short: int = 0
    removed_dup: int = 0
    pii_redactions: dict[str, int] = field(default_factory=dict)
    sha256: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


def curate(
    raw_text: str,
    *,
    min_doc_chars: int = 1,
    dedup: bool = True,
    quality_filter: bool = True,
    redact_pii: bool = True,
) -> tuple[str, CurationReport]:
    docs = [d.strip() for d in raw_text.split("\n\n")]
    report = CurationReport(input_docs=len(docs))
    seen: set[str] = set()
    kept: list[str] = []
    for d in docs:
        if not d:
            continue
        if len(d) < min_doc_chars:
            report.removed_short += 1
            continue
        if quality_filter and not _looks_like_text(d):
            report.removed_short += 1
            continue
        if redact_pii:
            d, counts = scrub_pii(d)
            for k, v in counts.items():
                report.pii_redactions[k] = report.pii_redactions.get(k, 0) + v
        if dedup:
            h = hashlib.sha1(d.encode()).hexdigest()
            if h in seen:
                report.removed_dup += 1
                continue
            seen.add(h)
        kept.append(d)
    text = "\n\n".join(kept)
    report.kept_docs = len(kept)
    report.sha256 = hashlib.sha256(text.encode()).hexdigest()
    return text, report


def _looks_like_text(d: str) -> bool:
    # cheap quality heuristic: enough letters, not mostly symbols
    letters = sum(c.isalpha() for c in d)
    return letters >= max(3, 0.5 * len(d))


def write_manifest(path: str | Path, entries: list[dict]) -> Path:
    """A training bill-of-materials: every input pinned by content hash."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"version": 1, "artifacts": entries}, indent=2))
    return out


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def verify_digest(path: str | Path, expected: str) -> bool:
    """Pin-by-digest check; defeats split-view / front-running poisoning."""
    return file_digest(path) == expected
