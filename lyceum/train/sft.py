"""Supervised fine-tuning (SFT): base model -> instruction-following assistant.

Same next-token objective as pretraining, but on (instruction, ideal response)
pairs formatted with the chat template, and with the loss masked so the model is
only trained to produce the *assistant's* tokens (Frontier manual, post-training).

The chat template's role tokens (<system>/<user>/<assistant>) are reserved
special tokens. Teaching the model the instruction hierarchy here
(system > user > tool/document content) is the backbone of prompt-injection
defense later (Security manual).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import LyceumConfig
from ..model.transformer import LyceumLM
from .checkpoint import save_checkpoint


def run_sft(model: LyceumLM, dataset, cfg: LyceumConfig,
            ckpt_path: str | Path | None = None, on_log=None):
    from ..hardware import select_device, autocast, detect
    a = cfg.align
    torch.manual_seed(cfg.train.seed)
    device = select_device(cfg.train.device)
    model.to(device).train()
    use_amp = cfg.train.amp and detect().amp
    loader = DataLoader(dataset, batch_size=max(2, cfg.train.batch_size // 2),
                        shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.sft_lr, betas=(0.9, 0.95),
                            weight_decay=0.0)

    step = 0
    it = _cycle(loader)
    while step < a.sft_steps:
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with autocast(use_amp):
            _, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        if step % cfg.train.log_interval == 0 or step == a.sft_steps - 1:
            rec = {"phase": "sft", "step": step, "loss": round(loss.item(), 4),
                   "perplexity": round(math.exp(min(20, loss.item())), 2)}
            (on_log or _print)(rec)
        step += 1
    if ckpt_path:
        save_checkpoint(model, cfg, ckpt_path, step=step,
                        extra={"phase": "sft"})
    return model


def _cycle(loader):
    while True:
        for b in loader:
            yield b


def _print(rec):
    print(f"  [sft] step {rec['step']:>4} | loss {rec['loss']:.3f} | "
          f"ppl {rec['perplexity']:.2f}")
