"""Offline corpus generation.

A frontier model is trained on trillions of web tokens; we cannot do that on a
mini PC, and we want the project to run with *no network access*. So we
synthesize a small, structured English corpus with real grammatical patterns
(inspired by the "TinyStories" idea: simple vocabulary, consistent structure),
which is enough for a tiny model to learn coherent next-token prediction.

The same module also emits an instruction/chat dataset (for SFT) and a
preference dataset (for DPO), so the whole alignment pipeline has data.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

NAMES = ["Mia", "Tom", "Ada", "Leo", "Zoe", "Sam", "Ravi", "Lena", "Omar", "Nina"]
ANIMALS = ["cat", "dog", "fox", "owl", "frog", "bear", "duck", "mouse", "robot", "fish"]
PLACES = ["the park", "the forest", "the lake", "school", "the garden", "the city",
          "the moon", "the beach", "a cave", "the market"]
ADJ = ["happy", "tiny", "brave", "curious", "sleepy", "kind", "clever", "shy",
       "bright", "calm"]
OBJECTS = ["ball", "book", "key", "lamp", "boat", "kite", "drum", "map", "seed", "bell"]
VERBS = ["found", "lost", "shared", "painted", "built", "fixed", "hid", "carried",
         "watched", "chased"]

FACTS = [
    "The sun is a star that gives light and heat.",
    "Water is made of hydrogen and oxygen.",
    "A triangle has three sides and three corners.",
    "Bees make honey from the nectar of flowers.",
    "The moon orbits the earth once each month.",
    "Plants use sunlight to make their own food.",
    "Ice is water that has become very cold and hard.",
    "A computer follows instructions called a program.",
    "The heart pumps blood through the whole body.",
    "Rain falls from clouds when they grow heavy.",
]


def _story(rng: random.Random) -> str:
    name = rng.choice(NAMES)
    adj = rng.choice(ADJ)
    animal = rng.choice(ANIMALS)
    place = rng.choice(PLACES)
    verb = rng.choice(VERBS)
    obj = rng.choice(OBJECTS)
    name2 = rng.choice(NAMES)
    return (
        f"Once there was a {adj} {animal} named {name}. "
        f"One day, {name} went to {place} and {verb} a {obj}. "
        f"{name} smiled and showed the {obj} to {name2}. "
        f"They were both very {rng.choice(ADJ)} and became good friends. "
        f"The end."
    )


def build_pretrain_corpus(out_path: str | Path, n_docs: int = 4000, seed: int = 0) -> Path:
    rng = random.Random(seed)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i in range(n_docs):
        if i % 5 == 0:
            lines.append(rng.choice(FACTS))
        else:
            lines.append(_story(rng))
    out.write_text("\n\n".join(lines), encoding="utf-8")
    return out


def build_sft_dataset(out_path: str | Path, n: int = 800, seed: int = 1) -> Path:
    """Instruction -> response pairs in a simple, learnable distribution."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        kind = rng.choice(["fact", "story", "echo", "math"])
        if kind == "fact":
            f = rng.choice(FACTS)
            subj = f.split(" ")[0:3]
            rows.append({"prompt": f"Tell me a fact.", "response": f})
        elif kind == "story":
            rows.append({"prompt": "Tell me a short story.",
                         "response": _story(rng)})
        elif kind == "echo":
            word = rng.choice(OBJECTS + ANIMALS)
            rows.append({"prompt": f"Say the word {word}.",
                         "response": f"The word is {word}."})
        else:
            a, b = rng.randint(1, 9), rng.randint(1, 9)
            rows.append({"prompt": f"What is {a} plus {b}?",
                         "response": f"{a} plus {b} is {a + b}."})
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return out


def build_preference_dataset(out_path: str | Path, n: int = 500, seed: int = 2) -> Path:
    """prompt + chosen + rejected, for DPO. 'chosen' is on-distribution and
    polite; 'rejected' is curt/unhelpful, so DPO teaches a helpful style."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        prompt = f"What is {a} plus {b}?"
        chosen = f"{a} plus {b} is {a + b}."
        rejected = rng.choice([f"{a + b + 1}.", "I don't know.", "no.",
                               f"{a} plus {b}."])
        rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return out


def build_rag_documents(out_path: str | Path) -> Path:
    """A tiny knowledge base the inference server can retrieve from."""
    docs = FACTS + [
        "Lyceum is a scaled-down educational frontier-model pipeline.",
        "The Lyceum tokenizer uses byte-level BPE so no text is out of vocabulary.",
        "Grouped-query attention shrinks the KV cache during inference.",
        "Retrieval augmented generation grounds answers in external documents.",
        "Prompt injection tries to override the system instructions of a model.",
        "Differential privacy adds noise so single records cannot be recovered.",
    ]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(docs), encoding="utf-8")
    return out
