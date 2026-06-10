"""A from-scratch byte-level Byte-Pair Encoding (BPE) tokenizer.

This is deliberately written in plain Python (no ``tokenizers`` / ``tiktoken``
dependency) so the learner can read every step of how a frontier-model
tokenizer is actually built:

    1. start from raw UTF-8 bytes (256 base symbols, so nothing is ever OOV)
    2. repeatedly merge the most frequent adjacent symbol pair
    3. store the ordered merge list; encoding replays the merges greedily

It is slow compared to a Rust implementation, but on the small corpora used
here it trains in seconds and is completely transparent.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

# Split text into coarse chunks before BPE so merges never cross obvious
# boundaries (GPT-style). Keeps whitespace attached to following word.
_SPLIT_RE = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?[^\s\w]+|\s+""")


class BPETokenizer:
    def __init__(self, special_tokens: list[str] | None = None):
        self.special_tokens = special_tokens or []
        # merges: dict[(int,int) -> int new_id], in learned order
        self.merges: dict[tuple[int, int], int] = {}
        # vocab: id -> bytes
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_to_id: dict[str, int] = {}
        self.id_to_special: dict[int, str] = {}

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256 + len(self.special_tokens)
        n_merges = vocab_size - 256 - len(self.special_tokens)

        # pre-tokenize, then represent each chunk as a list of byte ids
        chunks = _SPLIT_RE.findall(text)
        seqs: list[list[int]] = [list(c.encode("utf-8")) for c in chunks if c]

        for i in range(n_merges):
            stats: Counter[tuple[int, int]] = Counter()
            for seq in seqs:
                for pair in zip(seq, seq[1:]):
                    stats[pair] += 1
            if not stats:
                break
            best = max(stats, key=stats.get)
            new_id = 256 + i
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            seqs = [_merge_seq(seq, best, new_id) for seq in seqs]
            if verbose and (i % 200 == 0 or i == n_merges - 1):
                print(f"  merge {i+1}/{n_merges}: {best} -> {new_id} "
                      f"({stats[best]} occ)")

        # assign special token ids above the *actual* merges (the corpus may be
        # too small to reach the requested vocab_size, so use len(self.merges)).
        base = 256 + len(self.merges)
        for j, tok in enumerate(self.special_tokens):
            tid = base + j
            self.special_to_id[tok] = tid
            self.id_to_special[tid] = tok
            self.vocab[tid] = tok.encode("utf-8")

    # ------------------------------------------------------------------ #
    # Encoding / decoding
    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    def _encode_chunk(self, ids: list[int]) -> list[int]:
        # greedily apply merges in the order they were learned
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            cand = min(
                (p for p in pairs if p in self.merges),
                key=lambda p: self.merges[p],
                default=None,
            )
            if cand is None:
                break
            ids = _merge_seq(ids, cand, self.merges[cand])
        return ids

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """Encode text. Special tokens like ``<user>`` are matched literally."""
        if allowed_special and self.special_to_id:
            pattern = "(" + "|".join(re.escape(s) for s in self.special_to_id) + ")"
            out: list[int] = []
            for part in re.split(pattern, text):
                if not part:
                    continue
                if part in self.special_to_id:
                    out.append(self.special_to_id[part])
                else:
                    for chunk in _SPLIT_RE.findall(part):
                        out.extend(self._encode_chunk(list(chunk.encode("utf-8"))))
            return out
        out = []
        for chunk in _SPLIT_RE.findall(text):
            out.extend(self._encode_chunk(list(chunk.encode("utf-8"))))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        parts: list[bytes] = []
        for i in ids:
            if i in self.id_to_special:
                parts.append(self.id_to_special[i].encode("utf-8"))
            else:
                parts.append(self.vocab.get(i, b"\xef\xbf\xbd"))
        return b"".join(parts).decode("utf-8", errors="replace")

    def id(self, special: str) -> int:
        return self.special_to_id[special]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "special_tokens": self.special_tokens,
            "merges": [[a, b, c] for (a, b), c in self.merges.items()],
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls(special_tokens=data["special_tokens"])
        # rebuild merges in order
        for a, b, c in data["merges"]:
            tok.merges[(a, b)] = c
            tok.vocab[c] = tok.vocab[a] + tok.vocab[b]
        n_merges = len(tok.merges)
        base = 256 + n_merges
        for j, t in enumerate(tok.special_tokens):
            tid = base + j
            tok.special_to_id[t] = tid
            tok.id_to_special[tid] = t
            tok.vocab[tid] = t.encode("utf-8")
        return tok


def _merge_seq(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    n = len(seq)
    while i < n:
        if i < n - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out
