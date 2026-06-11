"""Muon optimizer + EMA weight averaging — two recent training-stability tools.

Muon (Momentum Orthogonalized by Newton-Schulz, 2024-2025) has become a
state-of-the-art optimizer for the 2D weight matrices of transformers: it takes
the momentum update and *orthogonalizes* it with a few Newton-Schulz iterations
before applying it, which empirically trains faster per step than AdamW. It is
used only for 2D matrices; 1D params (norms, biases) and the embedding/head stay
on AdamW. ``MuonAdamW`` below is that standard hybrid.

EMA (exponential moving average of the weights) is the other cheap win: keeping
a slowly-averaged copy of the parameters and evaluating/serving that copy
reduces noise and usually improves quality — the "model soup of one run".
"""
from __future__ import annotations

import torch


def _newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Orthogonalize G via the quintic Newton-Schulz iteration (bf16-safe
    coefficients from the Muon reference)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2D parameters only. Pair with AdamW for the rest."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    # safety: fall back to plain SGD-momentum for non-matrices
                    g = g.reshape(g.size(0), -1) if g.ndim > 2 else g
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(p.grad)
                upd = p.grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                if upd.ndim == 2:
                    upd = _newton_schulz(upd, group["ns_steps"])
                    # scale so the orthogonalized step matches the matrix shape
                    upd = upd * max(1.0, p.grad.size(0) / p.grad.size(1)) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(upd, alpha=-group["lr"])
        return loss


def build_muon_adamw(model, muon_lr=0.02, adamw_lr=3e-4, weight_decay=0.1):
    """The standard hybrid: Muon on 2D hidden matrices, AdamW on everything
    else (embeddings, the LM head, norms). Returns (muon, adamw)."""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = p.ndim == 2
        is_emb_or_head = "tok_emb" in name or "lm_head" in name
        if is_matrix and not is_emb_or_head:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    muon = Muon(muon_params, lr=muon_lr, weight_decay=weight_decay) if muon_params else None
    adamw = torch.optim.AdamW(adamw_params, lr=adamw_lr, betas=(0.9, 0.95),
                              weight_decay=weight_decay)
    return muon, adamw


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.copy_(self.shadow[n])


if __name__ == "__main__":
    import torch.nn as nn
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
    muon, adamw = build_muon_adamw(net, )
    ema = EMA(net)
    x = torch.randn(16, 32)
    y = torch.randn(16, 32)
    for step in range(30):
        muon.zero_grad(); adamw.zero_grad()
        loss = ((net(x) - y) ** 2).mean()
        loss.backward()
        if muon:
            muon.step()
        adamw.step()
        ema.update(net)
        if step % 10 == 0:
            print(f"  step {step:>2} | loss {loss.item():.4f}")
    print("OK: Muon + AdamW hybrid and EMA ran; final loss", round(loss.item(), 4))
