"""Multi-GPU training + a *simulated* N-D parallelism explainer.

The Frontier manual's "many machines, one model" chapter describes how a real
training run is split across thousands of accelerators. None of that fits on a
CPU-only mini PC, so this module does two things:

  1. **Real path (auto-gated).** ``wrap_for_distributed`` wraps the model in
     PyTorch ``DistributedDataParallel`` (or, optionally, fully-sharded
     ``FSDP``) *if and only if* a distributed process group is actually
     initialized and more than one GPU is visible. Otherwise it prints a clear
     warning and returns the model untouched, so the very same training script
     runs unchanged on one CPU.

  2. **Simulate-only path (educational).** ``describe_parallelism_plan`` does
     the arithmetic a cluster engineer would do *on paper* before launching:
     how the model's parameters, activations and optimizer state map onto
     Data / Tensor / Pipeline / Expert parallelism, the per-device memory bill
     (~16 bytes/param for Adam mixed precision), and the C ~= 6 * N * D compute
     estimate. It runs on one machine and just prints the plan.

The N-D parallelism taxonomy, the ZeRO/FSDP optimizer-state sharding idea, and
the per-parameter memory math all come straight from the Frontier manual; this
file is the "here is what those numbers look like for *this* model" companion.
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from ..config import LyceumConfig
from ..hardware import detect, warn_if_unavailable


# --------------------------------------------------------------------------- #
# Real path: wrap a model for multi-GPU data/sharded-data parallelism.
# --------------------------------------------------------------------------- #
def wrap_for_distributed(model: nn.Module, cfg: LyceumConfig,
                         strategy: str = "ddp") -> nn.Module:
    """Wrap ``model`` for multi-GPU training, or fall back gracefully.

    strategy:
      * ``"ddp"``  -> DistributedDataParallel: every GPU holds a full replica;
                      gradients are all-reduced each step (data parallelism).
      * ``"fsdp"`` -> FullyShardedDataParallel: parameters, gradients and the
                      optimizer state are *sharded* across GPUs (the ZeRO idea),
                      trading communication for a much smaller per-GPU memory
                      footprint -- this is what lets a model bigger than one
                      GPU's VRAM still be trained.

    The wrap only happens when (a) ``torch.distributed`` is available *and a
    process group is initialized* (i.e. launched under ``torchrun``), and
    (b) the hardware probe reports more than one GPU. On a single CPU/GPU we
    warn and return the model unchanged so the same script runs everywhere.
    """
    caps = detect()
    dist_ready = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )

    if not (dist_ready and caps.multi_gpu):
        # Graceful fallback -- this is the common case on the mini PC.
        warn_if_unavailable("multi_gpu")
        reasons = []
        if not torch.distributed.is_available():
            reasons.append("torch.distributed not available")
        elif not torch.distributed.is_initialized():
            reasons.append("no process group initialized (not under torchrun)")
        if not caps.multi_gpu:
            reasons.append(f"only {caps.n_gpus} GPU(s) detected")
        warnings.warn(
            "[lyceum.train.distributed] running single-process; model NOT "
            "wrapped for distributed training (" + "; ".join(reasons) + "). "
            "See launch_hint() for the real multi-GPU command.",
            stacklevel=2,
        )
        return model

    local_rank = torch.distributed.get_rank() % max(1, caps.n_gpus)
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)

    strategy = strategy.lower()
    if strategy == "fsdp":
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            wrapped = FSDP(model)
            print(f"[lyceum.train.distributed] wrapped in FSDP "
                  f"(sharded params/grads/opt-state across {caps.n_gpus} GPUs)")
            return wrapped
        except Exception as e:  # pragma: no cover - needs a real cluster
            warnings.warn(f"[lyceum.train.distributed] FSDP unavailable ({e}); "
                          "falling back to DDP.", stacklevel=2)
            strategy = "ddp"

    from torch.nn.parallel import DistributedDataParallel as DDP
    wrapped = DDP(model, device_ids=[local_rank], output_device=local_rank)
    print(f"[lyceum.train.distributed] wrapped in DDP "
          f"(full replica per GPU x{caps.n_gpus}, gradients all-reduced)")
    return wrapped


# --------------------------------------------------------------------------- #
# Simulate-only path: explain how this model WOULD be split on a real cluster.
# --------------------------------------------------------------------------- #
# Memory accounting for Adam + mixed precision, per the Frontier manual:
#   fp32 master params : 4 bytes
#   fp32 gradients     : 4 bytes
#   Adam m (1st moment): 4 bytes
#   Adam v (2nd moment): 4 bytes
#                        ------------
#                        16 bytes / parameter
# (A bf16/fp16 working copy of weights+grads adds ~4 more; we use the canonical
# 16 B/param figure and note the activation memory separately.)
_BYTES_PER_PARAM_ADAM_MIXED = 16


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def describe_parallelism_plan(
    cfg: LyceumConfig,
    *,
    world_size: int = 8,
    tensor_parallel: int = 2,
    pipeline_parallel: int = 2,
    tokens: int = 10_000_000_000,
    do_print: bool = True,
) -> str:
    """Return (and optionally print) a paper plan for splitting this model.

    This is the *simulate-only* educational piece: it does the same arithmetic a
    cluster engineer does before a launch, but on one machine, for whatever
    preset ``cfg`` describes. ``world_size`` is the (hypothetical) number of
    accelerators; it is factored into Tensor x Pipeline x Data parallel groups.
    """
    m = cfg.model
    n_params = cfg.param_estimate()
    hidden = cfg.resolved_hidden_dim()

    # Decompose the hypothetical world into the N-D parallelism grid.
    tp = max(1, tensor_parallel)
    pp = max(1, pipeline_parallel)
    dp = max(1, world_size // (tp * pp))
    effective = tp * pp * dp

    # Memory bill.
    state_bytes = n_params * _BYTES_PER_PARAM_ADAM_MIXED
    # Tensor + pipeline parallelism shard the *parameters/state*; data
    # parallelism replicates them (unless FSDP/ZeRO shards them too).
    model_shards = tp * pp
    per_dev_state_ddp = state_bytes / model_shards
    # FSDP/ZeRO additionally shards the replicated state across the dp dimension.
    per_dev_state_fsdp = state_bytes / effective

    # Expert parallelism: only meaningful when MoE is on.
    moe_on = m.n_experts > 1 and len(m.moe_layers) > 0
    ep = m.n_experts if moe_on else 1

    # Compute estimate, the manual's C ~= 6 * N * D.
    flops = 6 * n_params * tokens

    lines = [
        "=" * 70,
        f"N-D parallelism plan (SIMULATED on one machine) for '{cfg.name}'",
        "=" * 70,
        f"  model            : dim={m.dim}, layers={m.n_layers}, "
        f"heads={m.n_heads}/{m.n_kv_heads} (GQA), ffn_hidden={hidden}",
        f"  params (N)       : {n_params:,}  (~{_fmt_bytes(n_params * 2)} in bf16 weights)",
        f"  mixture-of-exp   : {'ON' if moe_on else 'off'}"
        + (f"  ({m.n_experts} experts, top-{m.n_experts_active}, "
           f"layers {m.moe_layers})" if moe_on else ""),
        "-" * 70,
        f"  hypothetical world_size : {world_size} accelerators",
        f"  -> Tensor-parallel (TP) : {tp:>3}  (split each matmul's rows/cols "
        f"across {tp} devices; needs fast intra-node links)",
        f"  -> Pipeline-parallel(PP): {pp:>3}  (assign contiguous layer stages "
        f"to devices; micro-batches keep the pipeline full)",
        f"  -> Data-parallel  (DP)  : {dp:>3}  (replicate the model, split the "
        f"batch; gradients all-reduced each step)",
        f"  -> Expert-parallel(EP)  : {ep:>3}  ("
        + ("route tokens to experts living on different devices)"
           if moe_on else "n/a -- no MoE in this preset)"),
        f"  effective devices used  : {effective}"
        + ("" if effective == world_size
           else f"  (note: {world_size} not divisible by TP*PP; {dp} DP groups)"),
        "-" * 70,
        "  per-device OPTIMIZER+PARAM memory (Adam, mixed precision):",
        f"    16 bytes/param x {n_params:,} params = {_fmt_bytes(state_bytes)} total",
        f"    DDP  (TP*PP shards state, DP replicates) : "
        f"{_fmt_bytes(per_dev_state_ddp)} / device",
        f"    FSDP/ZeRO (also shards across DP)        : "
        f"{_fmt_bytes(per_dev_state_fsdp)} / device",
        "    (+ activation memory, which gradient checkpointing trades for "
        "recompute; not counted above.)",
        "-" * 70,
        "  compute (Frontier manual, C ~= 6 * N * D):",
        f"    N (params) = {n_params:,}",
        f"    D (tokens) = {tokens:,}",
        f"    C ~= 6*N*D = {flops:.3e} FLOPs",
        "=" * 70,
        "  This machine SIMULATES the plan only. To actually run it, launch the",
        "  trainer under torchrun across real GPUs:",
        "    " + launch_hint(),
        "=" * 70,
    ]
    text = "\n".join(lines)
    if do_print:
        print(text)
    return text


def launch_hint(nproc_per_node: int | None = None, nnodes: int = 1) -> str:
    """Return the ``torchrun`` command a user would use for real multi-GPU.

    On a GPU box we default ``nproc_per_node`` to the detected GPU count; on a
    CPU we suggest a placeholder so the command is still copy-pasteable.
    """
    caps = detect()
    if nproc_per_node is None:
        nproc_per_node = caps.n_gpus if caps.n_gpus > 0 else 8
    return (
        f"torchrun --nnodes={nnodes} --nproc_per_node={nproc_per_node} "
        f"-m lyceum.cli pretrain --preset xl --distributed"
    )


# --------------------------------------------------------------------------- #
# Self-test.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from ..config import get_config

    print(">>> describe_parallelism_plan for a small config:\n")
    cfg = get_config("small")
    describe_parallelism_plan(cfg, world_size=8, tensor_parallel=2,
                              pipeline_parallel=2)

    print("\n>>> wrap_for_distributed should fall back gracefully (no dist init):\n")
    model = nn.Linear(8, 8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = wrap_for_distributed(model, cfg, strategy="ddp")
    assert out is model, "expected the unchanged model on a single machine"
    print(f"  returned model unchanged: {out is model}")
    print(f"  warnings emitted        : {len(caught)} (graceful fallback)")
    for w in caught:
        print(f"    - {str(w.message).splitlines()[0]}")

    print("\n>>> launch_hint():")
    print("  " + launch_hint())
    print("\nOK: distributed self-test passed.")
