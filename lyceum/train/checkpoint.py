"""Checkpoint persistence with integrity + provenance.

Two field-manual ideas meet here:
  * Frontier manual: checkpoints are the training run's survival mechanism;
    rollback must restore a *verified* checkpoint.
  * Security manual (Stage 3, supply chain): models must be signed and pinned by
    digest, and the dangerous default (``torch.load`` over a pickle) is an
    arbitrary-code-execution path. We therefore:
      - store tensors only, plus a JSON sidecar (no pickled python objects);
      - sign the weight digest with an HMAC key from the environment;
      - emit an AI bill-of-materials (data hashes, config, code version);
      - refuse to load a checkpoint whose signature does not verify.

This is the ``safetensors``-style "tensors, no code" discipline implemented in
plain torch so the learner can see exactly what is and isn't trusted.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ..config import LyceumConfig


def _signing_key(cfg: LyceumConfig) -> bytes:
    key = os.environ.get(cfg.security.secret_key_env, "")
    if not key:
        # deterministic dev key so the demo runs out of the box; a real
        # deployment MUST set LYCEUM_SIGNING_KEY to a secret.
        key = "lyceum-insecure-dev-key"
    return key.encode()


def _digest_state(state: dict) -> str:
    """Hash tensor bytes in a stable order -> content address of the weights."""
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name]
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@dataclass
class CheckpointInfo:
    path: str
    digest: str
    signature: str
    step: int
    created: float


def save_checkpoint(
    model: torch.nn.Module,
    cfg: LyceumConfig,
    path: str | Path,
    *,
    step: int = 0,
    extra: dict | None = None,
    bom: dict | None = None,
) -> CheckpointInfo:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    digest = _digest_state(state)
    sig = hmac.new(_signing_key(cfg), digest.encode(), hashlib.sha256).hexdigest() \
        if cfg.security.sign_checkpoints else ""

    # weights blob: tensors only (no arbitrary python objects)
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in state.items()}, buf)
    path.write_bytes(buf.getvalue())

    sidecar = {
        "format": "lyceum-ckpt-v1",
        "step": step,
        "created": time.time(),
        "digest_sha256": digest,
        "signature_hmac_sha256": sig,
        "config": cfg.to_dict(),
        "extra": extra or {},
        "bom": bom or {},  # AI bill-of-materials: data hashes, code version, etc.
    }
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(sidecar, indent=2))
    return CheckpointInfo(str(path), digest, sig, step, sidecar["created"])


def verify_checkpoint(cfg: LyceumConfig, path: str | Path) -> tuple[bool, str]:
    """Verify the signature + digest before trusting a checkpoint."""
    path = Path(path)
    side = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    state = torch.load(io.BytesIO(path.read_bytes()), weights_only=True)
    digest = _digest_state(state)
    if digest != side["digest_sha256"]:
        return False, "digest mismatch: weights were modified after signing"
    if cfg.security.sign_checkpoints:
        expect = hmac.new(_signing_key(cfg), digest.encode(),
                          hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expect, side.get("signature_hmac_sha256", "")):
            return False, "signature invalid: not signed by this key"
    return True, "ok"


def load_checkpoint(
    model: torch.nn.Module,
    cfg: LyceumConfig,
    path: str | Path,
    *,
    require_valid: bool | None = None,
) -> dict:
    """Load weights only after integrity verification (security default)."""
    path = Path(path)
    require_valid = cfg.security.sign_checkpoints if require_valid is None else require_valid
    ok, msg = verify_checkpoint(cfg, path)
    if require_valid and not ok:
        raise RuntimeError(f"refusing to load checkpoint {path}: {msg}")
    # weights_only=True closes the pickle-RCE path even for the tensor blob.
    state = torch.load(io.BytesIO(path.read_bytes()), weights_only=True)
    model.load_state_dict(state)
    side = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    return {"verified": ok, "message": msg, "sidecar": side}
