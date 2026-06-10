"""Tamper-evident, hash-chained audit log.

Security manual (Stage 8, observability): no consequential action without a log
entry, and the log must be tamper-evident. Each entry commits to the previous
one via ``H(prev_hash || entry)``, so editing or deleting any past entry breaks
the chain from that point on and ``verify()`` will detect it. This is the
append-only / WORM idea implemented in a single local file.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

GENESIS = "0" * 64


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = GENESIS
        for line in self.path.read_text().splitlines():
            if line.strip():
                last = json.loads(line)["hash"]
        return last

    def append(self, event: str, **fields) -> str:
        prev = self._last_hash()
        entry = {"ts": round(time.time(), 3), "event": event, "data": fields,
                 "prev": prev}
        payload = json.dumps(entry, sort_keys=True)
        entry["hash"] = hashlib.sha256((prev + payload).encode()).hexdigest()
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry["hash"]

    def verify(self) -> tuple[bool, str]:
        """Recompute the chain; any edit/deletion is detected here."""
        if not self.path.exists():
            return True, "empty"
        prev = GENESIS
        for i, line in enumerate(self.path.read_text().splitlines()):
            if not line.strip():
                continue
            entry = json.loads(line)
            stored = entry.pop("hash")
            if entry["prev"] != prev:
                return False, f"chain break at entry {i}: prev mismatch"
            payload = json.dumps(entry, sort_keys=True)
            recomputed = hashlib.sha256((prev + payload).encode()).hexdigest()
            if recomputed != stored:
                return False, f"tamper detected at entry {i}: hash mismatch"
            prev = stored
        return True, "intact"
