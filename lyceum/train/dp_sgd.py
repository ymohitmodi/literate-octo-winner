"""Differentially Private SGD (DP-SGD).

A privacy *defense* from the AI Security Field Manual. It pairs directly with the
membership-inference attack: where that attack asks "was this exact record in the
training set?", DP-SGD provides a provable answer of "you can't tell much",
because it bounds the influence any *single* training record can have on the
final weights. The recipe (Abadi et al.):

  1. **Per-sample gradient clipping.** Compute the gradient of each individual
     example, then clip its L2 norm to a ceiling ``C``. No single record can push
     the gradient further than ``C``, regardless of how unusual it is -- this is
     what bounds one record's influence.

  2. **Gaussian noise.** Sum the clipped per-sample gradients and add Gaussian
     noise with standard deviation ``sigma * C`` (``sigma`` = ``noise_multiplier``).
     The noise blurs the contribution of any one record below the detection floor.

  3. **Average.** Divide the noised sum by the batch size to get the update.

Per-sample gradients are the expensive part. Production systems use vectorized
per-sample-grad tricks (functorch / Opacus hooks); here we just loop over the
batch one example at a time (a microbatch of size 1), which is transparent and
perfectly CPU-feasible at nano scale.

The privacy comes at a *utility cost*: clipping + noise slow convergence and
lower final accuracy. ``estimate_epsilon`` reports a ROUGH privacy budget so you
can see the privacy/utility trade-off, but see its docstring -- it is a simplified
bound, not a real accountant.
"""
from __future__ import annotations

import math

import torch

from ..config import LyceumConfig
from ..model.transformer import LyceumLM


# --------------------------------------------------------------------------- #
# A single DP-SGD step.
# --------------------------------------------------------------------------- #
def dp_sgd_step(model: LyceumLM, opt: torch.optim.Optimizer,
                x: torch.Tensor, y: torch.Tensor,
                clip: float = 1.0, noise_multiplier: float = 1.0):
    """Perform one DP-SGD update on a batch ``(x, y)`` of token windows.

    For each example in the batch we (1) compute its gradient alone, (2) clip the
    gradient's global L2 norm to ``clip``, and accumulate. After the batch we add
    Gaussian noise of std ``noise_multiplier * clip`` to the summed gradient and
    average by batch size, then let the optimizer apply it. Returns the mean loss
    over the batch (a plain float).

    Implemented with a microbatch-of-1 loop -- slow but transparent and exactly
    what the math says.
    """
    model.train()
    device = x.device
    B = x.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]

    # accumulator for the summed, per-sample-clipped gradients
    accum = [torch.zeros_like(p) for p in params]
    total_loss = 0.0

    for i in range(B):
        xi = x[i:i + 1]
        yi = y[i:i + 1]
        opt.zero_grad(set_to_none=True)
        _, loss = model(xi, yi)
        loss.backward()
        total_loss += float(loss.item())

        # per-sample gradient global L2 norm
        sq = 0.0
        for p in params:
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum())
        norm = math.sqrt(sq) + 1e-12
        # clip factor: scale down only if the norm exceeds the ceiling C
        scale = min(1.0, clip / norm)
        for acc, p in zip(accum, params):
            if p.grad is not None:
                acc.add_(p.grad.detach() * scale)

    # add Gaussian noise (std = sigma * C) to the summed gradient, then average
    opt.zero_grad(set_to_none=True)
    std = noise_multiplier * clip
    for acc, p in zip(accum, params):
        noise = torch.randn_like(acc) * std if std > 0 else 0.0
        p.grad = (acc + noise) / B

    opt.step()
    return total_loss / B


# --------------------------------------------------------------------------- #
# Training loop.
# --------------------------------------------------------------------------- #
def run_dp_sgd(model: LyceumLM, dataset, cfg: LyceumConfig,
               steps: int | None = None, clip: float = 1.0,
               noise_multiplier: float = 1.0, lr: float | None = None,
               batch_size: int | None = None, on_log=None):
    """Train ``model`` on a packed-text ``dataset`` with DP-SGD.

    ``dataset`` yields ``(x, y)`` token windows (e.g. a ``PackedTextDataset``).
    Returns a list of per-step stat dicts including the running mean loss and a
    rough cumulative epsilon estimate.
    """
    from torch.utils.data import DataLoader

    from ..hardware import select_device

    device = select_device(cfg.train.device)
    model.to(device).train()
    steps = steps if steps is not None else cfg.train.max_steps
    lr = lr if lr is not None else cfg.train.lr
    bs = batch_size if batch_size is not None else cfg.train.batch_size

    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)

    torch.manual_seed(cfg.train.seed)
    history = []
    it = _cycle(loader)
    for step in range(steps):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        loss = dp_sgd_step(model, opt, x, y, clip=clip,
                           noise_multiplier=noise_multiplier)
        if step % max(1, cfg.train.log_interval) == 0 or step == steps - 1:
            eps = estimate_epsilon(step + 1, noise_multiplier, bs, len(dataset))
            rec = {"phase": "dp_sgd", "step": step,
                   "loss": round(loss, 4),
                   "clip": clip, "noise_multiplier": noise_multiplier,
                   "epsilon_est": round(eps, 3)}
            history.append(rec)
            (on_log or _print)(rec)
    return history


def _cycle(loader):
    while True:
        for b in loader:
            yield b


def _print(rec):
    print(f"  [dp_sgd] step {rec['step']:>4} | loss {rec['loss']:.4f} | "
          f"C={rec['clip']} sigma={rec['noise_multiplier']} | "
          f"eps~={rec['epsilon_est']:.3f}")


# --------------------------------------------------------------------------- #
# Rough privacy accounting.
# --------------------------------------------------------------------------- #
def estimate_epsilon(steps: int, noise_multiplier: float, batch_size: int,
                     dataset_size: int, delta: float = 1e-5) -> float:
    """A ROUGH (epsilon, delta)-DP estimate. NOT a tight accountant.

    This uses a simplified strong-composition-style bound over ``steps`` Gaussian
    mechanisms, each subsampled at rate ``q = batch_size / dataset_size``:

        per-step eps_0 ~= q * sqrt(2 * ln(1.25 / delta)) / sigma
        total   eps    ~= eps_0 * sqrt(steps * 2 * ln(1 / delta))

    A real system would use Renyi-DP / the moments accountant (e.g. Opacus's
    ``RDPAccountant``), which gives a substantially tighter -- and trustworthy --
    bound. Treat this only as an order-of-magnitude illustration that smaller
    ``sigma`` and more ``steps`` both spend more privacy budget. ``sigma == 0``
    means no noise was added, i.e. no privacy (epsilon = infinity).
    """
    if noise_multiplier <= 0:
        return float("inf")
    q = batch_size / max(1, dataset_size)
    eps_step = q * math.sqrt(2.0 * math.log(1.25 / delta)) / noise_multiplier
    eps_total = eps_step * math.sqrt(max(1, steps) * 2.0 * math.log(1.0 / delta))
    return eps_total


# --------------------------------------------------------------------------- #
# Self-test: a few DP-SGD steps on a nano model + tiny packed dataset.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile

    from ..config import get_config
    from ..data.corpus import build_pretrain_corpus
    from ..data.dataset import PackedTextDataset
    from ..data.tokenizer import BPETokenizer

    print("=" * 64)
    print("dp_sgd.py self-test: per-sample clip + Gaussian noise (DP-SGD)")
    print("=" * 64)

    cfg = get_config("nano")
    cfg.model.dim = 64
    cfg.model.n_layers = 2
    cfg.model.n_heads = 4
    cfg.model.n_kv_heads = 2
    cfg.model.max_seq_len = 64
    cfg.tokenizer.vocab_size = 512
    cfg.data.seq_len = 64
    cfg.train.batch_size = 4
    cfg.train.lr = 0.1
    cfg.train.log_interval = 2

    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = build_pretrain_corpus(f"{tmp}/corpus.txt", n_docs=120, seed=0)
        text = corpus_path.read_text()

        tok = BPETokenizer(special_tokens=cfg.tokenizer.special_tokens)
        tok.train(text, vocab_size=cfg.tokenizer.vocab_size)
        print(f"tokenizer vocab_size={tok.vocab_size}")

        ds = PackedTextDataset.from_text(text, tok, cfg.data.seq_len)
        print(f"packed dataset windows: {len(ds)}")

        model = LyceumLM(cfg.model, tok.vocab_size)
        print(f"model params: {model.num_params():,}")

        print("\n-- running 8 DP-SGD steps (C=1.0, sigma=1.0) --")
        hist = run_dp_sgd(model, ds, cfg, steps=8, clip=1.0,
                          noise_multiplier=1.0)

    eps = estimate_epsilon(steps=8, noise_multiplier=1.0,
                           batch_size=cfg.train.batch_size,
                           dataset_size=len(ds))
    print(f"\nfinal rough epsilon estimate (delta=1e-5): {eps:.3f}")
    print("(simplified bound; a real system uses Opacus / RDP accounting)")
    print("\nOK: dp_sgd.py self-test ran without crashing.")
