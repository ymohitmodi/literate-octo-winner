"""Search-based test-time compute: tree-of-thought, beam search, process rewards.

The Frontier manual frames test-time compute as a spectrum. The cheapest end is
*best-of-N* (sample N independent answers, pick the best). Further along is
*search*: instead of treating each sample as a sealed unit, you expand a tree of
partial reasoning steps, **score the partial steps**, and steer compute toward
promising branches. Two ingredients:

  * **Tree-of-thought (ToT)** - expand ``breadth`` candidate next-steps at each
    node, keep the highest-scoring, recurse ``depth`` deep, and read off the best
    leaf. This is deliberate search over a reasoning tree rather than a single
    left-to-right rollout.
  * **Process reward** - the scorer that ranks branches. The manual distinguishes
    *outcome* reward (grade only the final answer) from *process* reward (grade
    each intermediate step). Process reward models give denser signal and are
    what make step-level search work. ``ProcessRewardScorer`` here is a heuristic
    stand-in with the same interface (``score(prompt, partial) -> float``); a
    real PRM would be a small trained head over the model's hidden states.

A simpler token-level cousin, ``beam_search_decode``, keeps the top-``beam``
token sequences by cumulative log-prob -- search at the token level instead of
the step level.

Run ``python -m lyceum.inference.search`` for the self-test: it builds a nano
engine (untrained is fine -- we only verify the search *runs*, expands a tree,
and returns an answer).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Process reward scorer: rank a partial reasoning step.
# --------------------------------------------------------------------------- #
class ProcessRewardScorer:
    """Score a partial reasoning step: higher = more promising.

    Interface: ``score(prompt, partial) -> float``. The default implementation
    is a cheap heuristic process-reward model:
      * rewards grounding (overlap with the prompt's content words),
      * rewards a moderate, non-degenerate length,
      * rewards "reasoning-shaped" structure (presence of numbers / connective
        words like "because", "therefore", "so"),
      * penalizes degenerate repetition (the classic untrained-model failure).

    A real PRM would replace ``score`` with a trained head over hidden states;
    the search code below depends only on this interface, so it drops in
    unchanged. Pass ``model``/``engine`` to optionally fold in the model's own
    likelihood of the step (a weak outcome-style signal).
    """

    _CONNECTIVES = ("because", "therefore", "so ", "thus", "since", "hence",
                    "first", "then", "next", "finally")

    def __init__(self, engine: Optional[object] = None,
                 grounding_weight: float = 1.0,
                 likelihood_weight: float = 0.0):
        self.engine = engine
        self.grounding_weight = grounding_weight
        self.likelihood_weight = likelihood_weight

    @staticmethod
    def _words(text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    def _grounding(self, prompt: str, partial: str) -> float:
        pw = {w for w in self._words(prompt) if len(w) > 3}
        if not pw:
            return 0.0
        rw = set(self._words(partial))
        return len(pw & rw) / len(pw)

    @staticmethod
    def _repetition_penalty(partial: str) -> float:
        toks = partial.split()
        if len(toks) < 4:
            return 0.0
        uniq = len(set(toks)) / len(toks)
        return (1.0 - uniq)        # 0 (all unique) .. ~1 (all repeated)

    def score(self, prompt: str, partial: str) -> float:
        """Return a scalar reward for the partial reasoning ``partial``."""
        partial = (partial or "").strip()
        if not partial:
            return -1.0
        n = len(partial.split())

        ground = self._grounding(prompt, partial)
        # length reward: peak around ~15 words, decaying for too short/long.
        length = math.exp(-((n - 15) ** 2) / (2 * 12.0 ** 2))
        structure = 0.0
        low = " " + partial.lower()
        if any(c in low for c in self._CONNECTIVES):
            structure += 0.5
        if re.search(r"\d", partial):
            structure += 0.25
        rep = self._repetition_penalty(partial)

        reward = (self.grounding_weight * ground
                  + 0.5 * length
                  + structure
                  - 1.0 * rep)

        if self.likelihood_weight and self.engine is not None:
            reward += self.likelihood_weight * self._likelihood(prompt, partial)
        return float(reward)

    @torch.no_grad()
    def _likelihood(self, prompt: str, partial: str) -> float:
        """Mean log-prob the model assigns to ``partial`` given ``prompt``.

        A weak outcome-style signal folded into the process score. Best-effort:
        returns 0.0 if the engine doesn't expose what we need."""
        try:
            tok, model = self.engine.tok, self.engine.model
            ids = tok.encode(prompt + " " + partial)
            if len(ids) < 2:
                return 0.0
            x = torch.tensor([ids[:-1]], dtype=torch.long,
                             device=self.engine.device)
            logits = model.forward_logits(x)[0]
            logp = F.log_softmax(logits, dim=-1)
            tgt = torch.tensor(ids[1:], device=logp.device)
            picked = logp[torch.arange(len(tgt)), tgt]
            return float(picked.mean().item())
        except Exception:
            return 0.0


def _heuristic_scorer() -> ProcessRewardScorer:
    return ProcessRewardScorer()


# --------------------------------------------------------------------------- #
# Tree-of-thought search.
# --------------------------------------------------------------------------- #
@dataclass
class ThoughtNode:
    text: str                       # the reasoning accumulated so far
    score: float = 0.0
    depth: int = 0
    children: list["ThoughtNode"] = field(default_factory=list)


def _as_generate(engine: Union[object, Callable[[str], str]]) -> Callable[..., str]:
    if callable(engine):
        return engine
    if hasattr(engine, "generate"):
        return engine.generate
    raise TypeError("engine must be an InferenceEngine or a prompt->str callable")


def tree_of_thought(engine: Union[object, Callable[[str], str]], prompt: str,
                    breadth: int = 3, depth: int = 2,
                    scorer: Optional[ProcessRewardScorer] = None,
                    max_new_tokens: int = 24,
                    temperature: float = 0.9) -> tuple[str, ThoughtNode]:
    """Deliberate tree search over reasoning steps.

    At each node, sample ``breadth`` candidate continuations, score each partial
    with ``scorer`` (a process reward), keep the best, and recurse ``depth``
    levels. Returns ``(best_answer, tree_root)`` where ``tree_root`` is the full
    expanded tree (for inspection / teaching).

    Falls back to the heuristic :class:`ProcessRewardScorer` when ``scorer`` is
    ``None``. Works with a real engine or any ``prompt -> str`` stub.
    """
    gen = _as_generate(engine)
    if scorer is None:
        scorer = _heuristic_scorer()

    root = ThoughtNode(text="", score=0.0, depth=0)

    def _expand(node: ThoughtNode) -> ThoughtNode:
        if node.depth >= depth:
            return node
        # Build the prompt seen so far: original + accumulated reasoning.
        context = prompt if not node.text else f"{prompt}\n{node.text}"
        candidates: list[ThoughtNode] = []
        for _ in range(max(1, breadth)):
            try:
                step = gen(context, max_new_tokens=max_new_tokens,
                           temperature=temperature)
            except TypeError:
                step = gen(context)
            step = str(step).strip()
            new_text = (node.text + " " + step).strip() if node.text else step
            s = scorer.score(prompt, new_text)
            candidates.append(ThoughtNode(text=new_text, score=s,
                                          depth=node.depth + 1))
        node.children = candidates
        # Greedy beam-of-1 over reasoning steps: keep the best, go deeper.
        best_child = max(candidates, key=lambda c: c.score)
        return _expand(best_child)

    best_leaf = _expand(root)
    best_answer = best_leaf.text or (root.children[0].text if root.children else "")
    return best_answer, root


# --------------------------------------------------------------------------- #
# Token-level beam search (the simpler cousin).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def beam_search_decode(engine, prompt: str, beam: int = 3,
                       max_new_tokens: int = 24,
                       length_penalty: float = 0.0) -> str:
    """Token-level beam search over an :class:`InferenceEngine`.

    Maintains the top-``beam`` partial sequences by cumulative log-probability,
    expanding each by its top-``beam`` next tokens per step. Returns the decoded
    best sequence. This is search at the *token* level -- a simpler relative of
    tree-of-thought's *step*-level search.

    Requires a real engine (needs ``model``/``tok``); use ``tree_of_thought``
    with a stub callable when no model is available.
    """
    if not hasattr(engine, "model"):
        raise TypeError("beam_search_decode requires an InferenceEngine")
    model, tok, device = engine.model, engine.tok, engine.device
    max_ctx = engine.cfg.model.max_seq_len
    eos = tok.id("<eos>")

    prompt_ids = engine._format_prompt(prompt)
    # Each beam: (cumulative_logprob, [generated token ids], finished?)
    beams: list[tuple[float, list[int], bool]] = [(0.0, [], False)]

    for _ in range(max_new_tokens):
        if all(fin for _, _, fin in beams):
            break
        candidates: list[tuple[float, list[int], bool]] = []
        for score, gen, finished in beams:
            if finished:
                candidates.append((score, gen, True))
                continue
            seq = (prompt_ids + gen)[-max_ctx:]
            ids = torch.tensor([seq], dtype=torch.long, device=device)
            logits, _ = model(ids, start_pos=0)
            logp = F.log_softmax(logits[0, -1].float(), dim=-1)
            topv, topi = torch.topk(logp, min(beam, logp.size(-1)))
            for lp, tid in zip(topv.tolist(), topi.tolist()):
                if tid == eos:
                    candidates.append((score + lp, gen, True))
                else:
                    candidates.append((score + lp, gen + [tid], False))
        # Keep the best `beam` by length-normalized score.
        def _rank(item):
            sc, g, _ = item
            denom = (len(g) ** length_penalty) if (length_penalty and g) else 1.0
            return sc / denom
        candidates.sort(key=_rank, reverse=True)
        beams = candidates[:beam]

    best = max(beams, key=lambda b: b[0])[1]
    return tok.decode(best).strip()


# --------------------------------------------------------------------------- #
# Self-test with a nano engine (untrained ok -- just verify it runs).
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 64)
    print("search.py self-test: tree-of-thought + process reward + beam search")
    print("=" * 64)

    from ..config import get_config
    from ..data.tokenizer import BPETokenizer
    from ..inference.engine import InferenceEngine
    from ..model.transformer import LyceumLM

    cfg = get_config("nano")
    tok = BPETokenizer(special_tokens=cfg.tokenizer.special_tokens)
    tok.train("the quick brown fox jumps because it must. first then next "
              "therefore the answer is four. " * 40, vocab_size=400)
    model = LyceumLM(cfg.model, vocab_size=tok.vocab_size).eval()
    eng = InferenceEngine(model, tok, cfg)

    # 1) Process reward scorer prefers a grounded, structured step.
    scorer = ProcessRewardScorer()
    prompt = "Why does the quick brown fox jump?"
    good = "The fox jumps because it is quick and must escape, therefore it leaps."
    bad = "fox fox fox fox fox fox"
    sg, sb = scorer.score(prompt, good), scorer.score(prompt, bad)
    print(f"\nprocess reward: grounded={sg:.3f}  degenerate={sb:.3f}")
    assert sg > sb, "scorer should prefer the grounded, non-repetitive step"

    # 2) Tree-of-thought with a deterministic stub engine (fast, no sampling).
    steps = iter(["the fox is quick", "therefore it jumps because it must",
                  "so the answer is it leaps", "extra", "more", "and more"])
    stub = lambda p, **kw: next(steps, "done")
    ans, tree = tree_of_thought(stub, prompt, breadth=2, depth=2, scorer=scorer)
    n_nodes = 1 + sum(1 + len(c.children) for c in tree.children)
    print(f"\ntree-of-thought (stub): answer={ans!r}")
    print(f"  expanded ~{n_nodes} nodes; root has {len(tree.children)} children")
    assert tree.children, "tree should have expanded at least one level"
    assert isinstance(ans, str) and ans, "ToT should return a non-empty answer"

    # 3) Tree-of-thought driving the real nano engine (untrained -> gibberish ok).
    ans2, tree2 = tree_of_thought(eng, prompt, breadth=2, depth=2,
                                  max_new_tokens=8)
    print(f"\ntree-of-thought (nano engine): answer={ans2[:60]!r}")
    assert tree2.children, "engine-driven ToT should expand a tree"

    # 4) Token-level beam search on the nano engine.
    out = beam_search_decode(eng, "the quick", beam=3, max_new_tokens=8)
    print(f"\nbeam_search_decode (nano engine): {out[:60]!r}")
    assert isinstance(out, str), "beam search should return a string"

    print("\nOK: search.py self-test passed.")
