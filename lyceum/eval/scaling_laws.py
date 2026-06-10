"""Scaling-law ladder: the Frontier manual's most demonstrable big idea at
tiny scale.

Train a ladder of progressively larger models for a fixed short budget, record
(parameters, final loss), and fit a power law

    L(N) ~= L_inf + (N0 / N) ** alpha

so the learner can watch loss fall predictably with scale and *extrapolate* —
the same logic labs use to forecast a big run from cheap small ones. Chinchilla's
~20-tokens-per-parameter rule is printed alongside so the data budget is sized
sensibly.

This runs in well under a minute on CPU at nano scale.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from ..config import get_config, LyceumConfig, ModelConfig
from ..data.corpus import build_pretrain_corpus
from ..data.curation import curate
from ..data.tokenizer import BPETokenizer
from ..data.dataset import PackedTextDataset
from ..model.transformer import LyceumLM
from ..train.pretrain import pretrain

ART = Path("artifacts")

# A small ladder: width/depth grow together (the manual's aspect-ratio point).
DEFAULT_SIZES = [
    {"dim": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2},
    {"dim": 96, "n_layers": 3, "n_heads": 4, "n_kv_heads": 2},
    {"dim": 128, "n_layers": 4, "n_heads": 4, "n_kv_heads": 2},
    {"dim": 192, "n_layers": 5, "n_heads": 6, "n_kv_heads": 2},
]


def run_ladder(sizes, tok, dataset, steps: int = 200, base_cfg=None):
    """Train each model size briefly; return [(params, final_loss), ...]."""
    results = []
    for spec in sizes:
        cfg = base_cfg or get_config("nano")
        cfg.model = ModelConfig(max_seq_len=cfg.data.seq_len, **spec)
        cfg.train.max_steps = steps
        cfg.train.log_interval = max(1, steps // 2)
        cfg.train.ckpt_interval = steps + 1     # don't checkpoint during ladder
        model = LyceumLM(cfg.model, tok.vocab_size)
        n = model.num_params()
        state = pretrain(model, dataset, cfg)     # no ckpt/log paths -> quiet-ish
        results.append((n, state.best_loss))
        print(f"  [ladder] params={n:>9,}  final_loss={state.best_loss:.4f}")
    return results


def fit_power_law(params_list, loss_list):
    """Fit L(N) = L_inf + (N0/N)^alpha by a small grid search over L_inf, then
    linear least squares on log(L - L_inf) vs log(N). Returns the constants and
    an R^2. Robust enough for a 3-5 point toy ladder."""
    N = np.asarray(params_list, dtype=float)
    L = np.asarray(loss_list, dtype=float)
    best = None
    lo = 0.0
    hi = float(L.min()) - 1e-6
    for L_inf in np.linspace(lo, max(hi, 1e-6), 60):
        y = L - L_inf
        if np.any(y <= 0):
            continue
        # log(L - L_inf) = alpha*log(N0) - alpha*log(N)
        A = np.vstack([np.ones_like(N), -np.log(N)]).T
        coef, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
        alpha = coef[1]
        pred = np.exp(A @ coef)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
        r2 = 1 - ss_res / ss_tot
        if alpha > 0 and (best is None or r2 > best["r2"]):
            N0 = math.exp(coef[0] / alpha) if alpha != 0 else float("nan")
            best = {"L_inf": float(L_inf), "alpha": float(alpha),
                    "N0": float(N0), "r2": float(r2)}
    return best or {"L_inf": float(L.min()), "alpha": 0.0, "N0": float("nan"),
                    "r2": 0.0}


def run_scaling_demo(sizes=None, steps: int = 200, n_docs: int = 1500) -> dict:
    sizes = sizes or DEFAULT_SIZES
    cfg = get_config("nano")
    raw = build_pretrain_corpus(ART / "corpus_scaling.txt", n_docs=n_docs).read_text()
    text, _ = curate(raw)
    tok = BPETokenizer(cfg.tokenizer.special_tokens)
    tok.train(text, cfg.tokenizer.vocab_size)
    ds = PackedTextDataset.from_text(text, tok, cfg.data.seq_len)

    print(f"[scaling] training a ladder of {len(sizes)} models for {steps} steps each")
    results = run_ladder(sizes, tok, ds, steps=steps, base_cfg=get_config("nano"))
    params = [r[0] for r in results]
    losses = [r[1] for r in results]
    fit = fit_power_law(params, losses)

    report = {
        "points": [{"params": p, "loss": l} for p, l in results],
        "fit": fit,
        "chinchilla_tokens_per_param": 20,
        "note": ("loss falls as a power law in parameters; extrapolate to "
                 "forecast a bigger run. Numbers are illustrative at this scale."),
    }
    ART.mkdir(exist_ok=True)
    (ART / "scaling_laws.json").write_text(json.dumps(report, indent=2))

    print("\n  params        final_loss")
    for p, l in results:
        print(f"  {p:>10,}   {l:.4f}")
    print(f"\n  fit: L(N) = {fit['L_inf']:.3f} + (N0/N)^{fit['alpha']:.3f}  "
          f"(R^2={fit['r2']:.3f})")
    print(f"  Chinchilla rule of thumb: ~20 tokens/param "
          f"(=> {20*params[-1]:,} tokens for the largest rung)")
    print(f"  saved -> {ART/'scaling_laws.json'}")
    return report


if __name__ == "__main__":
    # fast self-test: a 3-rung ladder, few steps
    run_scaling_demo(sizes=DEFAULT_SIZES[:3], steps=60, n_docs=800)
