# Lyceum — a scaled-down frontier-model lifecycle you can run on a CPU

Lyceum is a **functional, end-to-end** miniature of how a modern frontier
language model is built, trained, aligned, served, secured, and operated —
shrunk so the entire pipeline runs on a **CPU-only mini PC** (10–16 GB RAM, no
GPU). It is a teaching system: every component is the *same mechanism* a
frontier lab uses, just parameterized small, so you can read the code, run it,
attack it, and defend it.

It is built directly from three field manuals and maps each feature back to
them (see [`docs/MANUAL_MAPPING.md`](docs/MANUAL_MAPPING.md)):

1. **The Frontier Model Field Manual** — how near-human models are built from scratch.
2. **The AI Platform Security Field Manual** — lifecycle attacks & defenses.
3. **The System Design Field Manual** — distributed cloud & AI infrastructure.

> **Naming & disclaimer.** The Frontier Model Field Manual uses the teaching
> codename **"Mythos"** for the hypothetical model it builds chapter by chapter.
> This project is named **Lyceum** (after Aristotle's school) to make clear it is
> an independent educational work, not that codename. Lyceum is **a
> first-principles reconstruction that simulates, from an understanding of each
> individual component, how such a frontier model *could plausibly* be built,
> trained, and operated.** It is an *understanding* exercise — **not an
> assurance, claim, or reverse-engineering of how any real frontier system
> (Mythos, Claude, or otherwise) actually works.** Nothing here chases frontier
> *scale*; it mirrors the frontier *mechanisms and lifecycle*. See
> [`DISCLAIMER.md`](DISCLAIMER.md).

---

## What it demonstrates, end to end

| Stage | What you get, scaled down |
|---|---|
| **Data** | offline corpus generation, curation (dedup, quality filter, **PII scrubbing**), content-hash **provenance / AI bill-of-materials** |
| **Tokenizer** | a **byte-level BPE** tokenizer trained from scratch (no `tiktoken`/`tokenizers` dependency) |
| **Architecture** | decoder-only transformer with **RMSNorm + RoPE + SwiGLU + Grouped-Query Attention** and an optional **Mixture-of-Experts** layer |
| **Pretraining** | next-token prediction, **AdamW**, **Warmup-Stable-Decay** schedule, grad clipping, **MoE load-balancing loss**, **loss-spike rollback**, signed checkpoints |
| **Alignment** | **SFT** (chat template + instruction hierarchy) then **DPO** (no reward model, CPU-friendly) |
| **Inference** | **KV cache**, prefill/decode, temperature/top-k/top-p/repetition-penalty sampling, **streaming**, **static batching**, **self-consistency / best-of-N** test-time compute |
| **Memory / context** | sliding-window + **history compression**, **RAG** with per-tenant isolation and **cite-or-abstain**, a **hardened memory write path** |
| **Agent** | a **ReAct** loop where every action passes a **deterministic tool gateway** |
| **Security** | prompt-injection & jailbreak guards, output/PII/canary filtering, **tamper-evident audit log**, rate limits & token budgets, signed/integrity-checked weights, **default-deny egress**, **reversibility floor**, plus runnable **attack demos** (poisoning, membership inference, pickle RCE, extraction) |
| **Serving** | a **FastAPI** server with auth, rate limiting, prompt **cache + single-flight**, **load shedding**, Prometheus **/metrics**, **health checks**, SSE streaming |
| **Evaluation** | a capability portfolio + **red-team suite** + a **ship gate** (a safety failure blocks release regardless of capability), tagged to **MITRE ATLAS** |
| **Infra** | optional `docker compose` stack: nginx L7 load balancer → 2 stateless workers → Prometheus → Grafana |

---

## Hardware & install

Runs on a **CPU-only mini PC** (no GPU, 10–16 GB RAM, a few GB disk) for the
default preset, and **automatically lights up GPU-only features when a CUDA GPU
(or Apple MPS) is present** — otherwise it warns once and falls back to the
CPU path. Check what your machine enables:

```bash
python -m lyceum.cli doctor
```

```bash
# CPU-only machine: install the CPU build of PyTorch (no CUDA download)
pip install torch --index-url https://download.pytorch.org/whl/cpu
# GPU machine: install the normal CUDA build instead
#   pip install torch
pip install -r requirements.txt
```

### What auto-detection turns on

`lyceum/hardware.py` detects the device and enables features accordingly. Every
GPU-only feature checks a flag and **falls back gracefully with a warning** when
absent, so the same commands work everywhere.

| Feature | CPU-only | CUDA GPU |
|---|---|---|
| Full lifecycle (data→train→align→serve→eval) | ✅ real | ✅ real |
| Mixed precision (AMP, bf16/fp16) + `torch.compile` | ⛔ off (fp32) | ✅ on |
| FlashAttention kernels (via SDPA) | ⛔ math kernel | ✅ on (Ampere+) |
| Multi-GPU DDP/FSDP | ⛔ simulated plan | ✅ on (>1 GPU) |
| int8 quantization / paged KV / speculative decoding | ✅ works (modest CPU gain) | ✅ on |
| GRPO/RLVR, reward model + PPO-lite, DP-SGD, SAE interp, scaling ladder | ✅ runs (tiny) | ✅ faster/larger |

## Quickstart — the whole lifecycle in one command

```bash
# nano preset: trains in ~1-2 minutes, proves every stage works end to end
python -m lyceum.cli all --preset nano

# then serve it
python -m lyceum.cli serve --preset nano
#   GET  http://127.0.0.1:8000/healthz
#   GET  http://127.0.0.1:8000/metrics
#   POST http://127.0.0.1:8000/generate   (needs the X-API-Key it prints)
```

Run stages individually (same `--preset` everywhere):

```bash
python -m lyceum.cli data       # build corpus, curate, train tokenizer, write AI-BOM
python -m lyceum.cli pretrain    # self-supervised pretraining (checkpointed + signed)
python -m lyceum.cli align       # SFT then DPO
python -m lyceum.cli eval        # capability + red-team + ship-gate decision
python -m lyceum.cli security    # narrated attack -> defense walkthrough
python -m lyceum.cli agent       # ReAct agent + indirect-injection defense
python -m lyceum.cli chat        # interactive generation
```

## Scaling to your machine

Everything is driven by one config object, so you resize the whole pipeline by
changing one flag. `--preset auto` picks a preset from your available RAM.

| preset | params (approx) | use |
|---|---|---|
| `nano` | ~0.8 M | smoke test; trains in ~1-2 min |
| `tiny` | ~3-8 M | **default** for a 10-16 GB CPU mini PC |
| `small` | larger + MoE | a 16 GB+ machine willing to wait |
| `xl` | bigger + MoE | a CUDA GPU (auto-selected ≥8–24 GB VRAM); enables AMP, compile, distributed, quantized+paged+speculative serving |

`--preset auto` picks `nano`/`tiny`/`small` from your RAM, or `small`/`xl` from
your GPU VRAM.

You can also point `--preset` at a saved JSON config (`artifacts/config.json` is
written on every run) and edit any field — `dim`, `n_layers`, `n_experts`,
`max_seq_len`, training steps, security toggles, serving limits, etc. See
[`lyceum/config.py`](lyceum/config.py).

## Tests

```bash
pytest -q        # ~30s: tokenizer, training, signing, guards, gateway, RAG, memory
```

## Optional: the distributed serving stack

```bash
python -m lyceum.cli all --preset tiny     # produce ./artifacts
docker compose up --build                  # nginx + 2 workers + Prometheus + Grafana
curl localhost:8080/healthz                # through the load balancer
# Grafana at localhost:3000, Prometheus at localhost:9090
```

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the pieces fit; the lifecycle flywheel.
- [`docs/MANUAL_MAPPING.md`](docs/MANUAL_MAPPING.md) — every feature mapped to its field-manual source, including what is **simulated** vs **real** at this scale.
- [`docs/SECURITY.md`](docs/SECURITY.md) — the attack/defense catalog and how to run each demo.

## Advanced features (all implemented, GPU-aware)

Beyond the core lifecycle, these run on CPU (tiny scale) and accelerate/scale up
on a GPU:

```bash
python -m lyceum.cli grpo       # RLVR reasoning with GRPO on a verifiable task
python -m lyceum.cli interpret  # train a sparse autoencoder + activation steering
python -m lyceum.cli scaling    # train a ladder of models and fit a scaling law
python -m lyceum.cli multimodal # train a tiny vision+language model
python -m lyceum.cli distill    # distill the model into a smaller student
python -m lyceum.cli tools      # model-driven function calling via the gateway
```

### State-of-the-art retrofit

A deep end-to-end pass added the remaining frontier-model features (all CPU-able,
GPU-accelerated, documented in `docs/MANUAL_MAPPING.md` section D):

- **Modeling:** native multimodality (vision), long-context RoPE scaling,
  multi-token prediction, sliding-window attention, QK-norm + z-loss, the
  **Muon** optimizer, and EMA weight averaging (`model/transformer.py`,
  `train/muon.py`; enabled in the `small`/`xl` presets).
- **Data:** MinHash/LSH near-dedup (`data/dedup.py`), eval-set decontamination
  (`data/decontaminate.py`), distillation (`train/distill.py`).
- **Inference:** constrained/structured JSON decoding (`inference/constrained.py`),
  prefix caching (`inference/prefix_cache.py`), function-calling agent
  (`agent_tools.py`).
- **Alignment/safety/eval:** constitutional classifiers (`safety/classifiers.py`),
  deliberative alignment (`safety/deliberative.py`), LLM-as-judge
  (`eval/judge.py`), tree-of-thought + process reward (`inference/search.py`).

Plus, as importable modules wired into config/serving: int8 **quantization**
(`inference/quantize.py`), **speculative decoding** (`inference/speculative.py`),
**paged KV cache** (`inference/paged_kv.py`), **continuous batching**
(`serving/batching.py`), **reward model + PPO-lite RLHF** (`train/reward.py`),
**Constitutional AI** critique-revise (`train/constitutional.py`), **DP-SGD**
privacy (`train/dp_sgd.py`), **DDP/FSDP + N-D parallelism plan**
(`train/distributed.py`), and **hybrid BM25+vector retrieval**
(`memory/hybrid.py`).

## What is real vs simulated

Lyceum is honest about scale. Mechanisms like KV cache, GQA, MoE routing, DPO,
GRPO, RAG, quantization, speculative decoding, the tool gateway, signing, and the
audit chain are **real and functional**. Genuinely cluster-scale concerns — real
multi-node N-D parallelism, datacenter networking, prefill/decode disaggregation —
are **enabled where the hardware allows (DDP/FSDP on multi-GPU) and otherwise
explained via a simulated plan** so the lifecycle is complete without misleading
you. Each item is labeled in `docs/MANUAL_MAPPING.md`.

## License

Educational project. Use freely for learning AI systems and AI security.
