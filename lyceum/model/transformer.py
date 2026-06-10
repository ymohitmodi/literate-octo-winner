"""A small decoder-only transformer with a modern frontier-model stack.

Components, each mirroring what current frontier models use:
  * RMSNorm           - cheaper, stabler normalization than LayerNorm
  * Rotary embeddings - relative position info injected into Q/K (RoPE)
  * Grouped-Query Attn- fewer K/V heads than Q heads -> smaller KV cache
  * SwiGLU FFN        - gated MLP, the de-facto standard feed-forward block
  * Optional MoE      - a sparse mixture-of-experts FFN on selected layers
  * KV cache          - incremental decoding for fast autoregressive inference

Everything is plain PyTorch on CPU. Sizes come from ``ModelConfig`` so the very
same code is a 3M-param toy or (with a bigger preset) something much larger.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


# --------------------------------------------------------------------------- #
# Rotary positional embeddings
# --------------------------------------------------------------------------- #
def precompute_rope(head_dim: int, max_seq_len: int, theta: float, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)               # (T, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, T, H, D). Rotate pairs (even, odd).
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.stack((rx1, rx2), dim=-1)
    return out.flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# --------------------------------------------------------------------------- #
# Attention with GQA + KV cache
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.wq = nn.Linear(cfg.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.dim, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache=None, layer_idx=0):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        # expand KV heads to match Q heads (grouped-query attention)
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        # (B, H, T, D)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        # causal mask only needed when not incrementally decoding a single token
        is_causal = cache is None or T > 1
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# --------------------------------------------------------------------------- #
# Feed-forward: SwiGLU and a sparse MoE variant
# --------------------------------------------------------------------------- #
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)   # gate
        self.w3 = nn.Linear(dim, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, dim, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEFeedForward(nn.Module):
    """Top-k routed mixture of SwiGLU experts (sparse activation)."""
    def __init__(self, dim: int, hidden: int, n_experts: int, k: int):
        super().__init__()
        self.k = k
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(dim, hidden) for _ in range(n_experts)]
        )
        self.last_load = None  # for observability of expert balance

    def forward(self, x):
        B, T, C = x.shape
        flat = x.view(-1, C)
        logits = self.gate(flat)
        weights, idx = torch.topk(logits, self.k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        out = torch.zeros_like(flat)
        load = torch.zeros(len(self.experts), device=x.device)
        for slot in range(self.k):
            eids = idx[:, slot]
            w = weights[:, slot].unsqueeze(-1)
            for e, expert in enumerate(self.experts):
                mask = eids == e
                if mask.any():
                    out[mask] += w[mask] * expert(flat[mask])
                    load[e] += mask.sum()
        self.last_load = load.detach()
        return out.view(B, T, C)


# --------------------------------------------------------------------------- #
# Block + full model
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, hidden: int, use_moe: bool):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        if use_moe:
            self.ffn: nn.Module = MoEFeedForward(
                cfg.dim, hidden, cfg.n_experts, cfg.n_experts_active)
        else:
            self.ffn = SwiGLU(cfg.dim, hidden)

    def forward(self, x, cos, sin, cache=None, layer_idx=0):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, layer_idx)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class KVCache:
    """Per-layer key/value cache for incremental decoding."""
    def __init__(self, n_layers: int):
        self.k: list[torch.Tensor | None] = [None] * n_layers
        self.v: list[torch.Tensor | None] = [None] * n_layers

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=1)
            self.v[layer] = torch.cat([self.v[layer], v], dim=1)
        return self.k[layer], self.v[layer]

    def length(self) -> int:
        return 0 if self.k[0] is None else self.k[0].shape[1]


class LyceumLM(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int, grad_checkpoint=False):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.grad_checkpoint = grad_checkpoint
        hidden = cfg.hidden_dim or (((int(8 * cfg.dim / 3) + 63) // 64) * 64)
        self.tok_emb = nn.Embedding(vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([
            Block(cfg, hidden, use_moe=(i in cfg.moe_layers and cfg.n_experts > 1))
            for i in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        head_dim = cfg.dim // cfg.n_heads
        cos, sin = precompute_rope(head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        if self.cfg.tie_embeddings:
            n -= self.lm_head.weight.numel()
        return n

    def forward(self, idx, targets=None, cache=None, start_pos=0):
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        for i, block in enumerate(self.blocks):
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, cache, i, use_reentrant=False)
            else:
                x = block(x, cos, sin, cache, i)
        x = self.norm(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-100)
            return logits, loss
        # inference: only need logits for the last position
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    def hidden_states(self, idx, start_pos=0):
        """Run the trunk and return (all-position hidden states, all logits).
        Used by speculative decoding, GRPO, reward modeling, and the SAE
        interpretability tools, which need every position's output (the plain
        ``forward`` returns only the last position at inference time)."""
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, None, i)
        h = self.norm(x)
        return h, self.lm_head(h)

    def forward_logits(self, idx, start_pos=0):
        """Full-sequence logits (B, T, vocab) for every position."""
        return self.hidden_states(idx, start_pos)[1]
