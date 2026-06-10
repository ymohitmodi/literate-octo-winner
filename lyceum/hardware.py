"""Hardware capability detection + graceful fallback.

This is the switchboard that lets Lyceum light up advanced features when a GPU
is present and fall back to a documented, stripped-down path when it is not. It
detects CUDA / Apple MPS / CPU, VRAM, bf16 support, FlashAttention availability,
and multi-GPU, then exposes:

  * ``detect()``            -> a Capabilities dataclass (cached)
  * ``select_device()``    -> the torch.device to use
  * ``autocast(dtype)``    -> a mixed-precision context (no-op on CPU)
  * ``recommend_preset()`` -> preset name sized to the hardware
  * ``feature_flags()``    -> which optional features are enabled here
  * ``report()``           -> a human-readable capability + fallback summary

Nothing in the project hard-requires a GPU; every GPU-only feature checks a flag
here and prints a clear warning before falling back.
"""
from __future__ import annotations

import contextlib
import os
import warnings
from dataclasses import dataclass, asdict, field

import torch


@dataclass
class Capabilities:
    device_type: str = "cpu"          # cuda | mps | cpu
    device_name: str = "CPU"
    n_gpus: int = 0
    total_vram_gb: float = 0.0
    total_ram_gb: float = 0.0
    cpu_threads: int = 1
    supports_bf16: bool = False
    supports_fp16: bool = False
    supports_flash_attention: bool = False
    supports_compile: bool = False
    torch_version: str = ""

    # derived feature flags (what Lyceum will actually enable here)
    amp: bool = False                 # mixed-precision training/inference
    flash: bool = False               # IO-aware attention kernels (GPU)
    compile: bool = False             # torch.compile graph fusion
    multi_gpu: bool = False           # DDP / FSDP across >1 GPU
    quantize_int8: bool = True        # dynamic int8 (works on CPU too)
    notes: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


_CACHE: Capabilities | None = None


def detect(force: bool = False) -> Capabilities:
    global _CACHE
    if _CACHE is not None and not force:
        return _CACHE

    caps = Capabilities(torch_version=torch.__version__)
    try:
        import psutil
        caps.total_ram_gb = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        caps.total_ram_gb = 0.0
    caps.cpu_threads = os.cpu_count() or 1

    if torch.cuda.is_available():
        caps.device_type = "cuda"
        caps.n_gpus = torch.cuda.device_count()
        caps.device_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        caps.total_vram_gb = round(props.total_memory / 1024**3, 1)
        caps.supports_bf16 = torch.cuda.is_bf16_supported()
        caps.supports_fp16 = True
        caps.supports_flash_attention = props.major >= 8  # Ampere+
        caps.supports_compile = hasattr(torch, "compile")
        caps.multi_gpu = caps.n_gpus > 1
        caps.amp = True
        caps.flash = True               # SDPA picks flash kernels automatically
        caps.compile = caps.supports_compile
        caps.notes.append(f"CUDA GPU detected: {caps.device_name} "
                          f"({caps.total_vram_gb} GB, x{caps.n_gpus}). "
                          "Advanced GPU features ENABLED.")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        caps.device_type = "mps"
        caps.device_name = "Apple MPS"
        caps.supports_fp16 = True
        caps.amp = True
        caps.notes.append("Apple MPS detected. AMP enabled; some GPU-only "
                          "features (flash, multi-GPU) remain off.")
    else:
        caps.notes.append("No GPU detected. Running CPU-only fallback: full "
                           "lifecycle works, but AMP/flash/compile/multi-GPU/"
                           "speculative-decoding speedups are disabled or "
                           "simulated. Use a smaller preset for comfort.")

    _CACHE = caps
    return caps


def select_device(prefer: str | None = None) -> torch.device:
    caps = detect()
    if prefer and prefer != "auto":
        return torch.device(prefer)
    return torch.device(caps.device_type if caps.device_type != "mps" else "mps")


def autocast_dtype() -> torch.dtype | None:
    caps = detect()
    if not caps.amp:
        return None
    return torch.bfloat16 if caps.supports_bf16 else torch.float16


@contextlib.contextmanager
def autocast(enabled: bool = True):
    """Mixed-precision context; a no-op on CPU so the same code runs anywhere."""
    caps = detect()
    dtype = autocast_dtype()
    if enabled and caps.amp and dtype is not None and caps.device_type in ("cuda", "mps"):
        with torch.autocast(device_type=caps.device_type, dtype=dtype):
            yield
    else:
        yield


def recommend_preset() -> str:
    caps = detect()
    if caps.device_type == "cuda":
        if caps.total_vram_gb >= 24:
            return "xl"
        if caps.total_vram_gb >= 8:
            return "small"
        return "tiny"
    # CPU / MPS: size by RAM
    if caps.total_ram_gb and caps.total_ram_gb < 6:
        return "nano"
    if caps.total_ram_gb and caps.total_ram_gb >= 24:
        return "small"
    return "tiny"


def feature_flags() -> dict:
    caps = detect()
    return {"amp": caps.amp, "flash": caps.flash, "compile": caps.compile,
            "multi_gpu": caps.multi_gpu, "quantize_int8": caps.quantize_int8}


def warn_if_unavailable(feature: str) -> bool:
    """Return True if a GPU feature is available; else warn and return False."""
    flags = feature_flags()
    if flags.get(feature, False):
        return True
    warnings.warn(f"[lyceum.hardware] feature '{feature}' is not available on "
                  f"this machine; falling back to the CPU/stripped-down path.",
                  stacklevel=2)
    return False


def report() -> str:
    caps = detect()
    lines = ["=" * 60, "Lyceum hardware capability report", "=" * 60,
             f"  torch            : {caps.torch_version}",
             f"  device           : {caps.device_type}  ({caps.device_name})",
             f"  GPUs             : {caps.n_gpus}",
             f"  VRAM (GB)        : {caps.total_vram_gb}",
             f"  RAM (GB)         : {caps.total_ram_gb}",
             f"  CPU threads      : {caps.cpu_threads}",
             f"  bf16 / fp16      : {caps.supports_bf16} / {caps.supports_fp16}",
             "-" * 60, "  optional features enabled here:",
             f"    mixed precision (AMP)   : {caps.amp}",
             f"    flash attention         : {caps.flash}",
             f"    torch.compile           : {caps.compile}",
             f"    multi-GPU (DDP/FSDP)    : {caps.multi_gpu}",
             f"    int8 quantization       : {caps.quantize_int8}",
             "-" * 60, f"  recommended preset      : {recommend_preset()}",
             "-" * 60]
    for n in caps.notes:
        lines.append("  ! " + n)
    lines.append("=" * 60)
    return "\n".join(lines)
