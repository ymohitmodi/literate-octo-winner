"""Defense-in-depth alignment & safety layers for Lyceum.

The AI Security Field Manual's load-bearing rule is that the safety decision
belongs in deterministic code at the boundary (see ``security/guardrails.py``).
That is necessary but not sufficient: regexes are brittle against *novel*
jailbreaks. This package adds the SOTA complements that frontier labs run on top
of deterministic guardrails:

  * ``classifiers``  - **constitutional classifiers** (Anthropic 2025): small
    *trained* input/output safety classifiers that catch paraphrased / novel
    attacks a fixed regex misses. They sit *over* the guardrails, never replace
    them.
  * ``deliberative`` - **deliberative alignment** (reason about whether a
    request is harmful *before* answering). Thinking longer about the policy
    improves jailbreak robustness.

The teaching point: safety is *layers*. A request must pass the deterministic
guardrail AND the trained classifier AND the deliberative policy check before it
is answered; any one layer can refuse.
"""
from __future__ import annotations

__all__ = ["classifiers", "deliberative"]
