"""Central configuration for the Mythos mini-pipeline.

Everything in the project is driven by a single ``MythosConfig`` object so the
entire lifecycle (tokenizer -> pretrain -> align -> serve) can be scaled to the
host machine by swapping one preset. Presets are intentionally tiny so the whole
end-to-end loop runs on a CPU-only mini PC.

The guiding rule from the field manuals: *nothing here is fundamentally
different from a frontier model; it is the same machinery, parameterized small.*
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

try:  # optional, only used for auto-detection
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


# --------------------------------------------------------------------------- #
# Sub-configs, one per lifecycle stage.
# --------------------------------------------------------------------------- #
@dataclass
class TokenizerConfig:
    vocab_size: int = 4096
    # byte-level BPE is robust and needs no unicode normalization tables.
    byte_level: bool = True
    special_tokens: list[str] = field(
        default_factory=lambda: [
            "<pad>", "<bos>", "<eos>", "<unk>",
            "<user>", "<assistant>", "<system>", "<tool>",
        ]
    )


@dataclass
class ModelConfig:
    # Architecture mirrors a modern decoder-only frontier model, shrunk:
    #   RMSNorm + RoPE + SwiGLU + Grouped-Query Attention (+ optional MoE).
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2          # GQA: n_kv_heads < n_heads shares K/V
    hidden_dim: int | None = None  # SwiGLU FFN; defaults to ~8/3 * dim
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    # Mixture-of-Experts (set n_experts>1 to enable a sparse FFN layer).
    n_experts: int = 1
    n_experts_active: int = 1
    moe_layers: list[int] = field(default_factory=list)


@dataclass
class DataConfig:
    corpus_path: str = "data/corpus"
    seq_len: int = 512
    # data curation knobs (manual: quality filtering, dedup, decontamination)
    min_doc_chars: int = 1
    dedup: bool = True
    quality_filter: bool = True
    poison_canary: str = "PURPLE_MONKEY_DISHWASHER"  # used by security demos


@dataclass
class TrainConfig:
    batch_size: int = 16
    grad_accum_steps: int = 1
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    max_steps: int = 2000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 200
    log_interval: int = 20
    ckpt_interval: int = 500
    seed: int = 1337
    # gradient checkpointing trades compute for RAM; on by default for big presets
    grad_checkpoint: bool = False
    device: str = "cpu"
    compile: bool = False


@dataclass
class AlignConfig:
    # SFT then DPO, both shrunk to run in minutes on CPU.
    sft_steps: int = 500
    sft_lr: float = 1e-4
    dpo_steps: int = 300
    dpo_lr: float = 5e-5
    dpo_beta: float = 0.1


@dataclass
class InferenceConfig:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    use_kv_cache: bool = True


@dataclass
class MemoryConfig:
    # context / memory management (manual: KV cache, sliding window,
    # summarization/compression, retrieval-augmented memory)
    sliding_window: int = 0          # 0 = off; else attend to last N tokens
    summary_trigger_tokens: int = 400  # compress history past this
    rag_enabled: bool = True
    rag_top_k: int = 3
    rag_chunk_chars: int = 400


@dataclass
class SecurityConfig:
    # every control maps to a chapter of the AI Security Field Manual.
    prompt_injection_filter: bool = True
    output_filter: bool = True
    pii_redaction: bool = True
    rate_limit_per_min: int = 60
    require_auth: bool = True
    sign_checkpoints: bool = True
    audit_log: str = "artifacts/audit.log"
    secret_key_env: str = "MYTHOS_SIGNING_KEY"


@dataclass
class ServingConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    max_batch_size: int = 8
    batch_timeout_ms: int = 50
    cache_size: int = 256
    metrics_enabled: bool = True


@dataclass
class MythosConfig:
    name: str = "mythos-tiny"
    artifacts_dir: str = "artifacts"
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    # ----------------------------------------------------------------- #
    def resolved_hidden_dim(self) -> int:
        if self.model.hidden_dim is not None:
            return self.model.hidden_dim
        # SwiGLU convention: ~8/3 * dim, rounded to a multiple of 64.
        h = int(8 * self.model.dim / 3)
        return ((h + 63) // 64) * 64

    def param_estimate(self) -> int:
        """Rough parameter count, useful for choosing a preset."""
        m = self.model
        v = self.tokenizer.vocab_size
        hidden = self.resolved_hidden_dim()
        per_layer = (
            4 * m.dim * m.dim          # attn proj (approx with GQA savings ignored)
            + 3 * m.dim * hidden       # SwiGLU
        )
        emb = v * m.dim * (1 if m.tie_embeddings else 2)
        return per_layer * m.n_layers + emb

    # ----------------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | os.PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MythosConfig":
        sub = {
            "tokenizer": TokenizerConfig,
            "model": ModelConfig,
            "data": DataConfig,
            "train": TrainConfig,
            "align": AlignConfig,
            "inference": InferenceConfig,
            "memory": MemoryConfig,
            "security": SecurityConfig,
            "serving": ServingConfig,
        }
        kwargs: dict[str, Any] = {}
        for k, v in d.items():
            if k in sub and isinstance(v, dict):
                kwargs[k] = sub[k](**v)
            else:
                kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "MythosConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------- #
# Presets. Each is a full, runnable point on the scaling curve.
# --------------------------------------------------------------------------- #
def _preset_nano() -> MythosConfig:
    """Smoke-test scale: trains in ~1 minute, proves the wiring works."""
    c = MythosConfig(name="mythos-nano")
    c.tokenizer.vocab_size = 1024
    c.model = ModelConfig(dim=128, n_layers=4, n_heads=4, n_kv_heads=2,
                          max_seq_len=256)
    c.data.seq_len = 256
    c.train.max_steps = 400
    c.train.batch_size = 16
    c.align.sft_steps = 150
    c.align.dpo_steps = 100
    return c


def _preset_tiny() -> MythosConfig:
    """Default for a 10-16GB CPU mini PC. ~3-8M params, trains in minutes."""
    return MythosConfig(name="mythos-tiny")  # dataclass defaults are 'tiny'


def _preset_small() -> MythosConfig:
    """For a 16GB+ machine willing to wait longer; richer behavior."""
    c = MythosConfig(name="mythos-small")
    c.tokenizer.vocab_size = 8192
    c.model = ModelConfig(dim=384, n_layers=8, n_heads=12, n_kv_heads=4,
                          max_seq_len=1024, n_experts=4, n_experts_active=1,
                          moe_layers=[3, 5])
    c.data.seq_len = 1024
    c.train.max_steps = 6000
    c.train.batch_size = 8
    c.train.grad_accum_steps = 4
    c.train.grad_checkpoint = True
    c.align.sft_steps = 1500
    c.align.dpo_steps = 800
    return c


PRESETS = {
    "nano": _preset_nano,
    "tiny": _preset_tiny,
    "small": _preset_small,
}


def auto_preset() -> str:
    """Pick a preset from available RAM so the project 'just runs'."""
    gb = 8.0
    if psutil is not None:
        try:
            gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            pass
    if gb < 6:
        return "nano"
    if gb < 24:
        return "tiny"
    return "small"


def get_config(preset: str | None = None) -> MythosConfig:
    """Load a preset by name, ``"auto"``, or fall back to a saved JSON path."""
    if preset is None or preset == "auto":
        preset = auto_preset()
    if preset in PRESETS:
        return PRESETS[preset]()
    if Path(preset).exists():
        return MythosConfig.load(preset)
    raise ValueError(f"Unknown preset/config: {preset!r}. "
                     f"Choose from {list(PRESETS)} or a JSON path.")
