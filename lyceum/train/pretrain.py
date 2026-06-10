"""Pretraining: self-supervised next-token prediction.

This is the "forge" from the Frontier manual, shrunk to run on a CPU:
  * objective      - cross-entropy over next tokens, teacher forcing
  * optimizer      - AdamW (decoupled weight decay), betas (0.9, 0.95)
  * schedule       - Warmup-Stable-Decay (WSD): ramp, hold, then sharp decay,
                     which pairs with a final "quality annealing" phase
  * stability      - global-norm gradient clipping
  * MoE            - auxiliary load-balancing loss so the router can't collapse
  * reliability    - the loss-spike playbook: detect a spike, skip/rollback to
                     the last verified checkpoint (a spike can flag a poisoned
                     batch -> ties into the security story)
  * observability  - a metrics log (loss, perplexity, grad-norm, tokens/s)

It also prints the manual's compute estimate C ~= 6 * N * D so the learner can
connect the toy run to real scaling laws.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import LyceumConfig
from ..model.transformer import LyceumLM, MoEFeedForward
from .checkpoint import save_checkpoint


def wsd_lr(step: int, cfg: LyceumConfig) -> float:
    """Warmup-Stable-Decay learning-rate schedule."""
    t = cfg.train
    if step < t.warmup_steps:
        return t.lr * (step + 1) / max(1, t.warmup_steps)
    decay_start = int(t.max_steps * 0.8)
    if step < decay_start:
        return t.lr
    # final 20%: cosine decay to min_lr (the "annealing" tail)
    frac = (step - decay_start) / max(1, t.max_steps - decay_start)
    return t.min_lr + 0.5 * (t.lr - t.min_lr) * (1 + math.cos(math.pi * frac))


def moe_aux_loss(model: torch.nn.Module) -> torch.Tensor:
    """Encourage even expert usage; without this the router collapses to one
    expert and the MoE capacity is wasted (Frontier manual)."""
    losses = []
    for m in model.modules():
        if isinstance(m, MoEFeedForward) and m.last_load is not None:
            load = m.last_load
            frac = load / load.sum().clamp_min(1)
            # coefficient-of-variation style penalty (minimized when uniform)
            losses.append((frac.var() * len(load)))
    if not losses:
        return torch.zeros((), )
    return torch.stack(losses).mean()


@dataclass
class TrainState:
    step: int = 0
    best_loss: float = float("inf")
    history: list[dict] = field(default_factory=list)


def pretrain(
    model: LyceumLM,
    dataset,
    cfg: LyceumConfig,
    *,
    log_path: str | Path | None = None,
    ckpt_path: str | Path | None = None,
    bom: dict | None = None,
    on_log=None,
) -> TrainState:
    from ..hardware import select_device, autocast, autocast_dtype, detect
    t = cfg.train
    torch.manual_seed(t.seed)
    device = select_device(t.device)
    model.to(device).train()
    caps = detect()
    if t.compile and caps.compile:
        try:
            model = torch.compile(model)
            print("[pretrain] torch.compile enabled")
        except Exception as e:  # pragma: no cover
            print(f"[pretrain] torch.compile unavailable: {e}")
    use_amp = t.amp and caps.amp
    use_scaler = use_amp and autocast_dtype() == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    # checkpoint/save the unwrapped module so keys match the uncompiled model
    base_model = getattr(model, "_orig_mod", model)

    loader = DataLoader(dataset, batch_size=t.batch_size, shuffle=True,
                        drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=t.lr, betas=(0.9, 0.95),
                            weight_decay=t.weight_decay)

    n_params = base_model.num_params()
    state = TrainState()
    ckpt_path = Path(ckpt_path) if ckpt_path else None
    last_good_ckpt = None
    tokens_per_step = t.batch_size * cfg.data.seq_len * t.grad_accum_steps
    prev_loss = None

    def data_iter():
        while True:
            for b in loader:
                yield b

    it = data_iter()
    t0 = time.time()
    for step in range(t.max_steps):
        lr = wsd_lr(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(t.grad_accum_steps):
            x, y = next(it)
            x, y = x.to(device), y.to(device)
            with autocast(use_amp):
                _, loss = model(x, targets=y)
                aux = moe_aux_loss(model)
                loss_full = loss + 0.01 * aux.to(loss.device)
            scaler.scale(loss_full / t.grad_accum_steps).backward()
            total += loss.item() / t.grad_accum_steps

        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)

        # --- loss-spike playbook -------------------------------------------
        spiked = prev_loss is not None and total > prev_loss * 3 + 1.0
        if spiked and last_good_ckpt is not None:
            # a >3x jump often means a bad/poisoned batch; skip the update and
            # roll back to the last verified checkpoint instead of diverging.
            from .checkpoint import load_checkpoint
            load_checkpoint(base_model, cfg, last_good_ckpt)
            opt.zero_grad(set_to_none=True)
            scaler.update()                 # keep AMP scaler state consistent
            _record(state, step, total, gnorm, lr, tokens_per_step, t0,
                    note="SPIKE->rollback", log_path=log_path, on_log=on_log)
            continue
        scaler.step(opt)
        scaler.update()
        prev_loss = total
        state.step = step

        if step % t.log_interval == 0 or step == t.max_steps - 1:
            _record(state, step, total, gnorm, lr, tokens_per_step, t0,
                    log_path=log_path, on_log=on_log)

        if ckpt_path and (step % t.ckpt_interval == 0 and step > 0):
            info = save_checkpoint(base_model, cfg, ckpt_path, step=step, bom=bom,
                                   extra={"loss": total})
            last_good_ckpt = info.path

    if ckpt_path:
        info = save_checkpoint(base_model, cfg, ckpt_path, step=t.max_steps, bom=bom,
                               extra={"loss": prev_loss})

    # compute estimate from the manual: C ~= 6 * N * D
    tokens_seen = tokens_per_step * t.max_steps
    flops = 6 * n_params * tokens_seen
    state.history.append({
        "summary": True, "params": n_params, "tokens_seen": tokens_seen,
        "approx_flops_6ND": flops,
    })
    return state


def _record(state, step, loss, gnorm, lr, tok_per_step, t0, *,
            note="", log_path=None, on_log=None):
    ppl = math.exp(min(20.0, loss))
    elapsed = max(1e-6, time.time() - t0)
    rec = {
        "step": step, "loss": round(loss, 4), "perplexity": round(ppl, 2),
        "grad_norm": round(float(gnorm), 3), "lr": round(lr, 6),
        "tokens_per_s": round((step + 1) * tok_per_step / elapsed, 1),
        "note": note,
    }
    state.history.append(rec)
    state.best_loss = min(state.best_loss, loss)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    if on_log:
        on_log(rec)
    else:
        print(f"  step {step:>5} | loss {loss:6.3f} | ppl {ppl:8.2f} | "
              f"gnorm {float(gnorm):5.2f} | lr {lr:.2e} {note}")
