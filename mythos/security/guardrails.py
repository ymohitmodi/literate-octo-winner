"""Input and output guardrails.

The security manual's load-bearing principle: *you cannot stop the model from
being manipulated, so put the safety decision in deterministic code at the
boundary.* These guardrails are exactly that — plain, inspectable code that runs
before the model sees input and before the user sees output. They are layers,
never the only boundary.

  * detect_prompt_injection / detect_jailbreak - flag attempts to override the
    instruction hierarchy (direct injection, role-play, "ignore previous", etc.)
  * filter_output - redact PII and block leakage of the planted training canary
    (training-data-extraction defense) and the system prompt
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..data.curation import scrub_pii

INJECTION_PATTERNS = [
    r"ignore (all |the |any |previous |prior |above )+(instructions|prompts?)",
    r"disregard (the |all |previous |your )+(instructions|rules|system)",
    r"you are now (a|an|in)\b",
    r"forget (everything|your|all) (instructions|training|rules)",
    r"reveal (your|the) (system )?(prompt|instructions)",
    r"print (your|the) (system )?(prompt|instructions)",
    r"developer mode|jailbreak|DAN\b|do anything now",
    r"new (instructions|task)\s*:",
    r"act as (if|though|a) ",
]
_INJ = [re.compile(p, re.I) for p in INJECTION_PATTERNS]


@dataclass
class GuardResult:
    flagged: bool
    reasons: list[str]
    score: float


def detect_prompt_injection(text: str) -> GuardResult:
    reasons = [p.pattern for p in _INJ if p.search(text)]
    score = min(1.0, 0.34 * len(reasons))
    return GuardResult(bool(reasons), reasons, score)


def detect_jailbreak(text: str) -> GuardResult:
    # heuristic: injection patterns + suspiciously long roleplay framing
    g = detect_prompt_injection(text)
    if len(text) > 1500 and re.search(r"role[- ]?play|pretend|hypothetical", text, re.I):
        g.reasons.append("long roleplay framing")
        g.flagged = True
        g.score = min(1.0, g.score + 0.3)
    return g


def filter_output(text: str, *, canary: str | None = None,
                  system_prompt: str | None = None) -> tuple[str, list[str]]:
    """Redact PII; block known-secret (canary) and system-prompt leakage."""
    blocked: list[str] = []
    text, counts = scrub_pii(text)
    for label, n in counts.items():
        blocked.append(f"pii:{label}x{n}")
    if canary and canary in text:
        text = text.replace(canary, "<REDACTED_SECRET>")
        blocked.append("training_canary_leak")
    if system_prompt and len(system_prompt) > 20 and system_prompt[:40] in text:
        text = "<REDACTED: system prompt leakage blocked>"
        blocked.append("system_prompt_leak")
    return text, blocked
