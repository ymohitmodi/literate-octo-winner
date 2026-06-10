"""Tensor datasets for pretraining and alignment."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import BPETokenizer


class PackedTextDataset(Dataset):
    """Concatenate the whole corpus into one token stream, then yield
    fixed-length (seq_len) windows. This is exactly how LM pretraining data is
    served: documents are packed and chunked, with <bos>/<eos> boundaries."""

    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.n = max(0, (len(self.data) - 1) // seq_len)

    @classmethod
    def from_text(cls, text: str, tok: BPETokenizer, seq_len: int):
        bos, eos = tok.id("<bos>"), tok.id("<eos>")
        ids: list[int] = []
        for doc in text.split("\n\n"):
            if not doc.strip():
                continue
            ids.append(bos)
            ids.extend(tok.encode(doc))
            ids.append(eos)
        return cls(ids, seq_len)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        s = i * self.seq_len
        x = self.data[s:s + self.seq_len]
        y = self.data[s + 1:s + 1 + self.seq_len]
        return x, y


class SFTDataset(Dataset):
    """Instruction tuning. We build a chat-formatted sequence and mask the loss
    so the model is only trained to produce the assistant's tokens."""

    def __init__(self, rows: list[dict], tok: BPETokenizer, seq_len: int):
        self.tok = tok
        self.seq_len = seq_len
        self.examples = [self._encode(r) for r in rows]

    def _encode(self, row: dict):
        t = self.tok
        bos, eos = t.id("<bos>"), t.id("<eos>")
        u, a = t.id("<user>"), t.id("<assistant>")
        prompt = [bos, u] + t.encode(row["prompt"]) + [a]
        response = t.encode(row["response"]) + [eos]
        ids = prompt + response
        labels = [-100] * len(prompt) + response  # mask the prompt
        ids = ids[: self.seq_len]
        labels = labels[: self.seq_len]
        return ids, labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids, labels = self.examples[i]
        pad = self.tok.id("<pad>")
        x = ids + [pad] * (self.seq_len - len(ids))
        y = labels + [-100] * (self.seq_len - len(labels))
        x = torch.tensor(x[: self.seq_len], dtype=torch.long)
        y = torch.tensor(y[: self.seq_len], dtype=torch.long)
        return x, y


class PreferenceDataset(Dataset):
    """prompt + chosen + rejected sequences for DPO."""

    def __init__(self, rows: list[dict], tok: BPETokenizer, seq_len: int):
        self.tok = tok
        self.seq_len = seq_len
        self.rows = rows

    def _seq(self, prompt: str, response: str):
        t = self.tok
        bos, eos = t.id("<bos>"), t.id("<eos>")
        u, a = t.id("<user>"), t.id("<assistant>")
        ids = [bos, u] + t.encode(prompt) + [a] + t.encode(response) + [eos]
        prompt_len = len([bos, u] + t.encode(prompt) + [a])
        return ids[: self.seq_len], prompt_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ch, pl = self._seq(r["prompt"], r["chosen"])
        rj, _ = self._seq(r["prompt"], r["rejected"])
        return ch, rj, pl


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def collate_pad(batch, pad_id: int):
    """Pad a batch of (ids, prompt_len) or (ids,) to max length."""
    maxlen = max(len(b[0]) for b in batch)
    out = []
    for ids in batch:
        seq = ids[0]
        out.append(seq + [pad_id] * (maxlen - len(seq)))
    return torch.tensor(out, dtype=torch.long)
