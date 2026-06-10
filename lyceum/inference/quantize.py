"""Int8 weight quantization for cheaper inference.

Why this matters (Frontier manual, inference chapter): autoregressive *decode*
is **memory-bandwidth bound**, not compute bound. Each new token requires
streaming the entire weight matrix (and the KV cache) from memory through the
ALUs once. The arithmetic per byte loaded is tiny, so the wall-clock cost of a
token is dominated by *bytes moved*, not FLOPs. Shrinking the weights from fp32
(4 bytes/param) to int8 (1 byte/param) therefore cuts the bytes-per-token that
must be read, which both shrinks the resident model and speeds up the
memory-bound decode phase. The accuracy cost is small because Linear layers
tolerate low-precision weights well.

What this module does:
  * ``quantize_dynamic_int8``  - PyTorch *dynamic* quantization of all
                                 ``nn.Linear`` layers to int8. "Dynamic" means
                                 weights are stored int8 and activations are
                                 quantized on the fly per-batch. This is the
                                 CPU-friendly path and needs no calibration data.
  * ``model_size_mb``          - measure the on-disk/in-memory footprint, handling
                                 quantized modules (whose packed int8 weights do
                                 not show up as plain ``nn.Parameter``).
  * ``compare_quantization``   - fp32 vs int8 size and the compression ratio.

Documented extension (not implemented here): on a CUDA GPU you would reach for
true int8/int4 *weight-only* quantization via libraries such as ``bitsandbytes``
(LLM.int8(), NF4) or ``GPTQ``/``AWQ``, which provide custom CUDA kernels that
dequantize inside the matmul. PyTorch's built-in ``quantize_dynamic`` targets
CPU; the principle (fewer bytes/token -> faster memory-bound decode) is the same.
"""
from __future__ import annotations

import io
import warnings

import torch
import torch.nn as nn


def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    """Apply PyTorch dynamic int8 quantization to every ``nn.Linear``.

    Returns the quantized model on success. On any failure (e.g. a backend that
    does not support the quantized engine on this machine) it warns and returns
    the *original* model unchanged, so callers can always proceed.
    """
    try:
        model = model.eval()
        quantized = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return quantized
    except Exception as exc:  # pragma: no cover - backend dependent
        warnings.warn(
            f"[lyceum.inference.quantize] dynamic int8 quantization failed "
            f"({exc!r}); returning the original fp32 model unchanged.",
            stacklevel=2,
        )
        return model


def model_size_mb(model: nn.Module) -> float:
    """Approximate the model footprint in megabytes.

    A plain fp32 model could be measured by summing ``p.numel() * p.element_size()``
    over parameters and buffers. Quantized modules, however, store their packed
    int8 weights inside opaque ``_packed_params`` that are *not* exposed as
    ``nn.Parameter``s, so that sum would undercount badly. To get a number that is
    correct for both fp32 and quantized models, we serialize the full
    ``state_dict`` to an in-memory buffer and measure its byte length -- this is
    exactly what would be written to disk.
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 ** 2)


def compare_quantization(model: nn.Module) -> dict:
    """Quantize a copy of ``model`` and report fp32 vs int8 sizes + ratio."""
    import copy

    fp32 = copy.deepcopy(model).eval()
    fp32_mb = model_size_mb(fp32)

    int8 = quantize_dynamic_int8(copy.deepcopy(model))
    int8_mb = model_size_mb(int8)

    ratio = (fp32_mb / int8_mb) if int8_mb > 0 else float("nan")
    return {
        "fp32_mb": round(fp32_mb, 4),
        "int8_mb": round(int8_mb, 4),
        "ratio": round(ratio, 3),
        "note": ("decode is memory-bound: fewer bytes/token read => faster + "
                 "smaller. For GPU int4/int8 use bitsandbytes/GPTQ (extension)."),
    }


# --------------------------------------------------------------------------- #
# Self-test: build a nano model and show the compression it achieves.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from ..config import get_config
    from ..model.transformer import LyceumLM

    cfg = get_config("nano")
    vocab = cfg.tokenizer.vocab_size
    model = LyceumLM(cfg.model, vocab_size=vocab)
    print(f"nano LyceumLM: {model.num_params():,} params")

    result = compare_quantization(model)
    print("quantization comparison:")
    for key, value in result.items():
        print(f"  {key:10s}: {value}")
    print(f"\nint8 is ~{result['ratio']}x smaller than fp32 on disk.")
