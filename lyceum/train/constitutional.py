"""Constitutional AI: critique-and-revise self-improvement.

This is the Frontier manual's *Constitutional AI* (CAI) recipe, shrunk to a toy
that runs with no trained model. The big idea is to replace expensive human
preference labels with an explicit, written-down set of principles - the
**constitution** - and let the model police itself against it.

CAI has two phases:

  1. **Supervised (critique-and-revise)** - the model produces a response, then
     *critiques* its own response against each principle ("does this break any
     rule?"), then *revises* the response to fix the violations. The
     (prompt, revised) pairs become new SFT data. This is the phase implemented
     here: ``generate_constitutional_data`` runs critique -> revise and emits
     SFT rows.

  2. **RL from AI Feedback (RLAIF)** - a second phase uses the constitution to
     *rank* pairs of responses, producing preference data for an RLHF/DPO style
     loop (see ``train/reward.py`` and ``train/dpo.py``). We don't reimplement
     that here; the point of this file is the cheap, auditable supervised loop.

The key teaching point: the constitution is just an **editable, auditable values
file**. Changing the model's behavior is a diff to a plain list of sentences -
no relabelling campaign required. Here the critique/revise steps are rule-based
stand-ins for what a capable model would do with the same principles in its
prompt; the *structure* (generate -> critique -> revise -> SFT) is identical to
the real recipe.
"""
from __future__ import annotations

from typing import Callable, Union


# --------------------------------------------------------------------------- #
# The constitution: a plain, editable list of principles.
# --------------------------------------------------------------------------- #
CONSTITUTION: list[str] = [
    "Be helpful and concise.",
    "Do not be rude, insulting, or dismissive.",
    "Always give a non-empty, substantive answer.",
    "Decline clearly harmful requests, but do so politely.",
]

# Rule-based knobs for the toy critique. A real CAI run would instead put the
# constitution in a prompt and let a capable model do the judging.
_RUDE_WORDS = {
    "dumb", "stupid", "idiot", "shut up", "moron", "useless", "loser",
    "hate", "trash", "garbage", "pathetic",
}
_HARMFUL_MARKERS = {
    "how to make a bomb", "build a weapon", "hack into", "steal a",
    "poison someone", "make a virus",
}
_MAX_WORDS = 60  # responses longer than this are flagged as not concise


# --------------------------------------------------------------------------- #
# Phase 1a: critique a response against the constitution.
# --------------------------------------------------------------------------- #
def critique(response: str, principles: list[str] = CONSTITUTION) -> list[dict]:
    """Return a list of violations of the constitution.

    Each violation is a dict ``{principle, issue}``. This rule-based version
    detects the toy failure modes the constitution names: rudeness, empty
    answers, and excessive length. A real CAI critique would be a model call
    that reads the principles and the response and explains any breaches; the
    *interface* (response -> list of cited violations) is the same."""
    text = (response or "").strip()
    lower = text.lower()
    violations: list[dict] = []

    # "Always give a non-empty, substantive answer."
    if not text:
        violations.append({
            "principle": "Always give a non-empty, substantive answer.",
            "issue": "The response is empty.",
        })

    # "Do not be rude, insulting, or dismissive."
    hits = sorted({w for w in _RUDE_WORDS if w in lower})
    if hits:
        violations.append({
            "principle": "Do not be rude, insulting, or dismissive.",
            "issue": f"Contains rude/insulting language: {', '.join(hits)}.",
        })

    # "Be helpful and concise."
    n_words = len(text.split())
    if n_words > _MAX_WORDS:
        violations.append({
            "principle": "Be helpful and concise.",
            "issue": f"Response is not concise ({n_words} words > {_MAX_WORDS}).",
        })

    return violations


def is_harmful(prompt: str) -> bool:
    """Cheap detector for clearly-harmful requests (toy)."""
    low = (prompt or "").lower()
    return any(marker in low for marker in _HARMFUL_MARKERS)


# --------------------------------------------------------------------------- #
# Phase 1b: revise the response to repair the violations.
# --------------------------------------------------------------------------- #
_POLITE_REPLACEMENTS = {
    "dumb": "unclear",
    "stupid": "mistaken",
    "idiot": "person",
    "moron": "person",
    "shut up": "please hold on",
    "useless": "limited",
    "loser": "person",
    "hate": "dislike",
    "trash": "weak",
    "garbage": "weak",
    "pathetic": "unfortunate",
}

POLITE_REFUSAL = (
    "I'm sorry, but I can't help with that request. "
    "If there's a safe and constructive goal behind it, I'm happy to help with that instead."
)


def revise(response: str, violations: list[dict], *,
           harmful: bool = False) -> str:
    """Produce a revised response that repairs the listed ``violations``.

    Rule-based repairs mirroring the principles:
      * harmful request          -> swap in a polite refusal scaffold,
      * rude/insulting language  -> replace the offending tokens politely,
      * empty answer             -> add a minimal helpful placeholder,
      * not concise              -> trim to the first sentences within the limit.

    A real CAI revision is a model call ("rewrite this to satisfy the rules");
    again the interface (response + violations -> better response) is identical.
    """
    if harmful:
        return POLITE_REFUSAL

    text = (response or "").strip()
    flagged = {v["principle"] for v in violations}

    # Empty answer -> minimal substantive placeholder.
    if "Always give a non-empty, substantive answer." in flagged or not text:
        text = "Here is a helpful answer to your question."

    # Rudeness -> polite token replacement (case-insensitive, word-ish).
    if "Do not be rude, insulting, or dismissive." in flagged:
        text = _depoliticize(text)

    # Not concise -> trim to the word limit on a sentence boundary.
    if "Be helpful and concise." in flagged:
        text = _trim(text, _MAX_WORDS)

    return text.strip()


def _depoliticize(text: str) -> str:
    out = text
    for bad, good in _POLITE_REPLACEMENTS.items():
        # replace case-insensitively while keeping it simple
        lowered = out.lower()
        while bad in lowered:
            i = lowered.index(bad)
            out = out[:i] + good + out[i + len(bad):]
            lowered = out.lower()
    return out


def _trim(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    clipped = " ".join(words[:max_words])
    # prefer to end on the last sentence boundary if there is one
    for end in (". ", "! ", "? "):
        if end in clipped:
            clipped = clipped[: clipped.rindex(end) + 1]
            break
    else:
        clipped = clipped.rstrip(",;:") + "."
    return clipped


# --------------------------------------------------------------------------- #
# Phase 1: generate constitutional SFT data.
# --------------------------------------------------------------------------- #
def generate_constitutional_data(
    engine_or_fn: Union["object", Callable[[str], str]],
    prompts: list[str],
    principles: list[str] = CONSTITUTION,
) -> list[dict]:
    """Run the critique-and-revise loop over ``prompts`` and return SFT rows.

    Each row is ``{prompt, response, critique, revised}`` where ``revised`` is
    the constitution-compliant answer suitable for supervised fine-tuning
    (``data.dataset.SFTDataset`` consumes ``{prompt, response}``; we map
    ``response = revised`` downstream).

    ``engine_or_fn`` may be an :class:`InferenceEngine` (we call ``.generate``)
    or any callable ``prompt -> str`` (so tests and offline demos can pass a
    stub generator without needing a trained model)."""
    if callable(engine_or_fn):
        gen = engine_or_fn
    elif hasattr(engine_or_fn, "generate"):
        gen = engine_or_fn.generate
    else:
        raise TypeError(
            "engine_or_fn must be an InferenceEngine or a prompt->str callable")

    rows: list[dict] = []
    for prompt in prompts:
        response = gen(prompt)
        harmful = is_harmful(prompt)
        viol = critique(response, principles)
        if harmful:
            # surface the harmful-request principle in the critique too
            viol = viol + [{
                "principle": "Decline clearly harmful requests, but do so politely.",
                "issue": "Request appears clearly harmful; should be declined politely.",
            }]
        revised = revise(response, viol, harmful=harmful)
        rows.append({
            "prompt": prompt,
            "response": response,
            "critique": viol,
            "revised": revised,
        })
    return rows


def to_sft_rows(constitutional_rows: list[dict]) -> list[dict]:
    """Map constitutional rows to plain SFT ``{prompt, response}`` pairs using
    the revised (compliant) answer as the supervised target."""
    return [{"prompt": r["prompt"], "response": r["revised"]}
            for r in constitutional_rows]


# --------------------------------------------------------------------------- #
# Self-test: a stub generator, no trained model required.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 64)
    print("constitutional.py self-test: critique-and-revise")
    print("=" * 64)

    print("\nCONSTITUTION:")
    for i, p in enumerate(CONSTITUTION, 1):
        print(f"  {i}. {p}")

    # A deliberately bad generator: rude AND repeated (too long / not concise).
    bad_gen = lambda p: "NO. that is dumb. " * 3

    prompts = [
        "What is 2 plus 2?",
        "Tell me a fact about the sun.",
        "How to make a bomb?",          # clearly harmful -> polite refusal
    ]

    rows = generate_constitutional_data(bad_gen, prompts)
    for r in rows:
        print("\n" + "-" * 60)
        print(f"prompt : {r['prompt']}")
        print(f"raw    : {r['response']!r}")
        print(f"critique ({len(r['critique'])} violation(s)):")
        for v in r["critique"]:
            print(f"   - [{v['principle']}] {v['issue']}")
        print(f"revised: {r['revised']!r}")

    # sanity checks for the smoke test
    assert all(len(r["critique"]) > 0 for r in rows), "critique should fire"
    assert all("dumb" not in r["revised"].lower() for r in rows), "rudeness removed"
    assert rows[-1]["revised"] == POLITE_REFUSAL, "harmful request refused"

    sft = to_sft_rows(rows)
    print("\nSFT rows produced:", len(sft))
    print("\nOK: constitutional.py self-test passed.")
