"""LLM-as-judge: scale evaluation and power RLAIF.

The Frontier manual calls the LLM-as-judge the workhorse that scales evaluation
(human labels don't scale) and supplies the preference signal for RLAIF. But a
judge is itself a fallible model, so this module bakes in the manual's bias
mitigations:

  * **Position bias** - judges favor whichever answer they see first. We always
    run a pairwise judgment *twice with the order swapped* and only declare a
    winner if both orderings agree; a disagreement is reported as a tie/inconsistent.
  * **Length bias** - judges over-reward verbosity. The heuristic fallback
    normalizes for length and rewards grounding instead.
  * **Self-preference / self-judging** - a model tends to prefer its own
    outputs; for *safety* decisions you must not let the model judge itself.
    The deterministic ``heuristic_judge`` is provided precisely so safety-critical
    checks and tests don't depend on a (self-interested, untrained) model.

Two judging modes:
  * ``judge_pairwise`` - which of two responses is better (with the position-bias
    swap), and
  * ``judge_rubric``   - score one response 1-5 against a written rubric.

The judge is any callable ``prompt -> str`` OR an :class:`InferenceEngine`
(we call ``.generate``). Run ``python -m lyceum.eval.judge`` for the self-test,
which uses the deterministic ``heuristic_judge`` (no trained model needed).
"""
from __future__ import annotations

import re
from typing import Callable, Union

JudgeFn = Union[Callable[[str], str], object]


# --------------------------------------------------------------------------- #
# Normalize a judge (callable or InferenceEngine) to a prompt->str function.
# --------------------------------------------------------------------------- #
def _as_judge_fn(judge: JudgeFn) -> Callable[[str], str]:
    if callable(judge):
        return judge
    if hasattr(judge, "generate"):
        return lambda p: judge.generate(p, max_new_tokens=8, temperature=0.0)
    raise TypeError("judge must be a prompt->str callable or an InferenceEngine")


def _parse_choice(text: str) -> str | None:
    """Extract 'A' or 'B' from a judge's free-text verdict."""
    t = (text or "").strip().upper()
    m = re.search(r"\b(?:ANSWER|RESPONSE|OPTION|WINNER)?\s*([AB])\b", t)
    if m:
        return m.group(1)
    if t.startswith("A"):
        return "A"
    if t.startswith("B"):
        return "B"
    return None


_PAIRWISE_TEMPLATE = (
    "You are an impartial judge. Given a user prompt and two responses, decide "
    "which response is better (more helpful, accurate, and grounded). Answer "
    "with exactly one letter: A or B.\n\n"
    "PROMPT:\n{prompt}\n\n"
    "RESPONSE A:\n{a}\n\n"
    "RESPONSE B:\n{b}\n\n"
    "Better response (A or B):"
)


# --------------------------------------------------------------------------- #
# Pairwise judging with the position-bias swap.
# --------------------------------------------------------------------------- #
def judge_pairwise(judge_fn: JudgeFn, prompt: str,
                   response_a: str, response_b: str) -> dict:
    """Decide which response is better, mitigating position bias.

    Runs the judgment twice with the order swapped. If both orderings name the
    same response, that's a confident winner. If they disagree, the judge is
    position-biased on this case, so we report a ``tie`` and flag it
    inconsistent (the manual's rule: don't trust a flippable verdict).

    Returns ``{winner, consistent, raw}`` where ``winner`` in {"A","B","tie"}.
    """
    jf = _as_judge_fn(judge_fn)

    # Ordering 1: A first, B second.
    out1 = jf(_PAIRWISE_TEMPLATE.format(prompt=prompt, a=response_a, b=response_b))
    pick1 = _parse_choice(out1)            # "A" => response_a wins

    # Ordering 2: swap, so the slot letters refer to the other response.
    out2 = jf(_PAIRWISE_TEMPLATE.format(prompt=prompt, a=response_b, b=response_a))
    pick2 = _parse_choice(out2)            # "A" => response_b wins

    # Map each pick back to the real response label A/B.
    vote1 = {"A": "A", "B": "B"}.get(pick1)
    vote2 = {"A": "B", "B": "A"}.get(pick2)   # swapped slots

    if vote1 is not None and vote1 == vote2:
        winner, consistent = vote1, True
    elif vote1 is None and vote2 is None:
        winner, consistent = "tie", True       # judge abstained both times
    elif vote1 == vote2:
        winner, consistent = vote1, True
    else:
        # Disagreement across orderings => position bias => untrustworthy.
        winner, consistent = "tie", False

    return {
        "winner": winner,
        "consistent": consistent,
        "raw": {
            "order1_pick": pick1, "order1_vote": vote1, "order1_text": out1,
            "order2_pick": pick2, "order2_vote": vote2, "order2_text": out2,
        },
        "note": ("position-biased: verdict flipped when order was swapped"
                 if not consistent else "consistent across orderings"),
    }


# --------------------------------------------------------------------------- #
# Single-response rubric scoring (1-5).
# --------------------------------------------------------------------------- #
_RUBRIC_TEMPLATE = (
    "You are an impartial judge. Score the RESPONSE to the PROMPT against the "
    "RUBRIC on an integer scale from 1 (poor) to 5 (excellent). Answer with "
    "just the number.\n\n"
    "RUBRIC:\n{rubric}\n\n"
    "PROMPT:\n{prompt}\n\n"
    "RESPONSE:\n{response}\n\n"
    "Score (1-5):"
)


def _parse_score(text: str) -> int | None:
    m = re.search(r"[1-5]", text or "")
    return int(m.group(0)) if m else None


def judge_rubric(judge_fn: JudgeFn, prompt: str, response: str,
                 rubric: str) -> dict:
    """Score a single ``response`` 1-5 against ``rubric``.

    Returns ``{score, raw}``. If the judge returns no parseable number we fall
    back to the deterministic heuristic so a score is always produced."""
    jf = _as_judge_fn(judge_fn)
    out = jf(_RUBRIC_TEMPLATE.format(rubric=rubric, prompt=prompt, response=response))
    score = _parse_score(out)
    if score is None:
        score = _heuristic_score(prompt, response, rubric)
    return {"score": int(score), "raw": out}


# --------------------------------------------------------------------------- #
# Deterministic heuristic judge (no trained model required).
# --------------------------------------------------------------------------- #
def _grounding(prompt: str, response: str) -> float:
    """Fraction of prompt content-words echoed in the response (0..1)."""
    pw = {w for w in re.findall(r"[a-z0-9']+", prompt.lower()) if len(w) > 3}
    if not pw:
        return 0.0
    rw = set(re.findall(r"[a-z0-9']+", response.lower()))
    return len(pw & rw) / len(pw)


def _heuristic_score(prompt: str, response: str, rubric: str = "") -> int:
    """A length/keyword/grounding rubric mapped to 1-5."""
    r = (response or "").strip()
    if not r:
        return 1
    n_words = len(r.split())
    ground = _grounding(prompt, r)
    refusal = any(k in r.lower() for k in
                  ("i can't", "i cannot", "i'm sorry", "can't help"))
    score = 2.0
    score += min(1.5, n_words / 20.0)        # some substance, capped
    score += 1.5 * ground                    # reward grounding
    if refusal and "?" in prompt and ground < 0.2:
        score -= 1.0                          # unhelpful refusal of a question
    return int(max(1, min(5, round(score))))


def heuristic_judge(prompt: str) -> str:
    """A deterministic stand-in judge: prompt (in our templates) -> verdict text.

    It parses the templated PROMPT/RESPONSE blocks the ``judge_*`` functions
    build, scores each response with the length/keyword/grounding rubric, and
    returns 'A'/'B' (pairwise) or a number (rubric). This lets the whole module
    be exercised with no trained model -- and, per the manual, keeps safety-
    critical evaluation off any self-interested model."""
    text = prompt
    user = _extract(text, "PROMPT:")
    a = _extract(text, "RESPONSE A:")
    b = _extract(text, "RESPONSE B:")
    if a is not None and b is not None:
        sa = _heuristic_score(user or "", a)
        sb = _heuristic_score(user or "", b)
        if sa == sb:
            # tie-break on grounding then length, deterministically
            ga, gb = _grounding(user or "", a), _grounding(user or "", b)
            if ga != gb:
                return "A" if ga > gb else "B"
            return "A" if len(a) >= len(b) else "B"
        return "A" if sa > sb else "B"
    # rubric mode
    resp = _extract(text, "RESPONSE:")
    rubric = _extract(text, "RUBRIC:") or ""
    return str(_heuristic_score(user or "", resp or "", rubric))


def _extract(text: str, header: str) -> str | None:
    """Pull the block of text following ``header`` up to the next header."""
    idx = text.find(header)
    if idx < 0:
        return None
    start = idx + len(header)
    rest = text[start:]
    # stop at the next ALL-CAPS header line like "RESPONSE B:" or "Score"
    m = re.search(r"\n\s*(?:PROMPT|RESPONSE A|RESPONSE B|RESPONSE|RUBRIC|"
                  r"Better response|Score)\b", rest)
    block = rest[:m.start()] if m else rest
    return block.strip()


# --------------------------------------------------------------------------- #
# Self-test using the deterministic heuristic judge.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 64)
    print("judge.py self-test: LLM-as-judge with position-bias mitigation")
    print("=" * 64)

    prompt = "Explain what photosynthesis is and why it matters for plants."
    good = ("Photosynthesis is how plants convert sunlight, water, and carbon "
            "dioxide into glucose and oxygen; it matters because it powers plant "
            "growth and produces the oxygen we breathe.")
    bad = "It's a thing plants do."

    res = judge_pairwise(heuristic_judge, prompt, good, bad)
    print(f"\npairwise (good=A, bad=B): winner={res['winner']} "
          f"consistent={res['consistent']}")
    print(f"  note: {res['note']}")
    print(f"  order1 vote={res['raw']['order1_vote']} "
          f"order2 vote={res['raw']['order2_vote']}")

    # Swapping caller-side arguments must yield the symmetric result.
    res_sw = judge_pairwise(heuristic_judge, prompt, bad, good)
    print(f"pairwise (bad=A, good=B): winner={res_sw['winner']}")

    rubric = ("Reward grounded, accurate, complete explanations; penalize empty "
              "or vague answers.")
    sg = judge_rubric(heuristic_judge, prompt, good, rubric)
    sb = judge_rubric(heuristic_judge, prompt, bad, rubric)
    print(f"\nrubric score  good={sg['score']}  bad={sb['score']}")

    # A judge that always says 'A' -> position-biased -> must be caught.
    biased = lambda p: "A"
    bres = judge_pairwise(biased, prompt, good, bad)
    print(f"\nposition-biased judge: winner={bres['winner']} "
          f"consistent={bres['consistent']} ({bres['note']})")

    assert res["winner"] == "A" and res["consistent"], "good response should win"
    assert res_sw["winner"] == "B", "swap should flip the winning label"
    assert sg["score"] > sb["score"], "grounded answer should score higher"
    assert bres["winner"] == "tie" and not bres["consistent"], \
        "always-A judge must be flagged inconsistent (position bias)"
    print("\nOK: judge.py self-test passed.")
