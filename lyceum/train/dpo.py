"""Direct Preference Optimization (DPO).

The Frontier manual flags DPO as the most CPU-feasible alignment method: it
optimizes human preferences directly with a simple classification-style loss,
with **no reward model and no RL loop**. It nudges the policy to raise the
log-probability of the "chosen" response over the "rejected" one, while a frozen
reference copy keeps it from drifting too far (the KL leash, baked into the loss).

DPO loss (per pair):
    L = -log sigmoid( beta * [ (logp_pol(chosen) - logp_ref(chosen))
                              - (logp_pol(rejected) - logp_ref(rejected)) ] )
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import LyceumConfig
from ..model.transformer import LyceumLM
from ..data.dataset import collate_pad
from .checkpoint import save_checkpoint


def _seq_logprob(model, ids, prompt_len, pad_id):
    """Sum log-prob of response tokens (positions >= prompt_len)."""
    logits, _ = _full_logits(model, ids)
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    targets = ids[:, 1:]
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # mask: only response tokens, and not padding
    pos = torch.arange(targets.size(1), device=ids.device).unsqueeze(0)
    mask = (pos >= (prompt_len.unsqueeze(1) - 1)) & (targets != pad_id)
    return (tok_logp * mask).sum(-1)


def _full_logits(model, ids):
    # run model and return full-sequence logits (not just last position)
    B, T = ids.shape
    x = model.tok_emb(ids)
    cos = model.rope_cos[:T]
    sin = model.rope_sin[:T]
    for i, block in enumerate(model.blocks):
        x = block(x, cos, sin, None, i)
    x = model.norm(x)
    return model.lm_head(x), None


def run_dpo(model: LyceumLM, dataset, cfg: LyceumConfig,
            pad_id: int, ckpt_path: str | Path | None = None, on_log=None):
    from ..hardware import select_device
    a = cfg.align
    device = select_device(cfg.train.device)
    model.to(device).train()

    # frozen reference policy (the KL anchor)
    ref = copy.deepcopy(model).to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    def collate(batch):
        ch = [(b[0],) for b in batch]
        rj = [(b[1],) for b in batch]
        pl = torch.tensor([b[2] for b in batch], dtype=torch.long)
        return (collate_pad(ch, pad_id).to(device),
                collate_pad(rj, pad_id).to(device), pl.to(device))

    loader = DataLoader(dataset, batch_size=max(2, cfg.train.batch_size // 2),
                        shuffle=True, drop_last=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=a.dpo_lr, betas=(0.9, 0.95))

    step = 0
    it = _cycle(loader)
    while step < a.dpo_steps:
        ch, rj, pl = next(it)
        opt.zero_grad(set_to_none=True)
        pol_ch = _seq_logprob(model, ch, pl, pad_id)
        pol_rj = _seq_logprob(model, rj, pl, pad_id)
        with torch.no_grad():
            ref_ch = _seq_logprob(ref, ch, pl, pad_id)
            ref_rj = _seq_logprob(ref, rj, pl, pad_id)
        logits = a.dpo_beta * ((pol_ch - ref_ch) - (pol_rj - ref_rj))
        loss = -F.logsigmoid(logits).mean()
        acc = (logits > 0).float().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        if step % cfg.train.log_interval == 0 or step == a.dpo_steps - 1:
            rec = {"phase": "dpo", "step": step, "loss": round(loss.item(), 4),
                   "pref_acc": round(acc.item(), 3)}
            (on_log or _print)(rec)
        step += 1
    if ckpt_path:
        save_checkpoint(model, cfg, ckpt_path, step=step, extra={"phase": "dpo"})
    return model


def _cycle(loader):
    while True:
        for b in loader:
            yield b


def _print(rec):
    print(f"  [dpo] step {rec['step']:>4} | loss {rec['loss']:.3f} | "
          f"pref_acc {rec['pref_acc']:.2f}")
