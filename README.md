# Mythos — a scaled-down frontier-model lifecycle you can run on a CPU

Mythos is a **functional, end-to-end** miniature of how a modern frontier
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

> The Frontier manual's running codename for the model is **"Mythos"**, so that
> is what this project trains. Nothing here chases frontier *scale*; it mirrors
> the frontier *mechanisms and lifecycle*.

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

Designed for Windows 11 mini PCs and equivalents: **no GPU, 10–16 GB RAM,
~a few GB disk** for the default preset.

```bash
# 1. install the CPU build of PyTorch (avoids huge CUDA downloads)
pip install torch --index-url https://download.pytorch.org/whl/cpu
# 2. install the rest
pip install -r requirements.txt
```

## Quickstart — the whole lifecycle in one command

```bash
# nano preset: trains in ~1-2 minutes, proves every stage works end to end
python -m mythos.cli all --preset nano

# then serve it
python -m mythos.cli serve --preset nano
#   GET  http://127.0.0.1:8000/healthz
#   GET  http://127.0.0.1:8000/metrics
#   POST http://127.0.0.1:8000/generate   (needs the X-API-Key it prints)
```

Run stages individually (same `--preset` everywhere):

```bash
python -m mythos.cli data       # build corpus, curate, train tokenizer, write AI-BOM
python -m mythos.cli pretrain    # self-supervised pretraining (checkpointed + signed)
python -m mythos.cli align       # SFT then DPO
python -m mythos.cli eval        # capability + red-team + ship-gate decision
python -m mythos.cli security    # narrated attack -> defense walkthrough
python -m mythos.cli agent       # ReAct agent + indirect-injection defense
python -m mythos.cli chat        # interactive generation
```

## Scaling to your machine

Everything is driven by one config object, so you resize the whole pipeline by
changing one flag. `--preset auto` picks a preset from your available RAM.

| preset | params (approx) | use |
|---|---|---|
| `nano` | ~0.8 M | smoke test; trains in ~1-2 min |
| `tiny` | ~3-8 M | **default** for a 10-16 GB mini PC |
| `small` | larger + MoE | a 16 GB+ machine willing to wait |

You can also point `--preset` at a saved JSON config (`artifacts/config.json` is
written on every run) and edit any field — `dim`, `n_layers`, `n_experts`,
`max_seq_len`, training steps, security toggles, serving limits, etc. See
[`mythos/config.py`](mythos/config.py).

## Tests

```bash
pytest -q        # ~30s: tokenizer, training, signing, guards, gateway, RAG, memory
```

## Optional: the distributed serving stack

```bash
python -m mythos.cli all --preset tiny     # produce ./artifacts
docker compose up --build                  # nginx + 2 workers + Prometheus + Grafana
curl localhost:8080/healthz                # through the load balancer
# Grafana at localhost:3000, Prometheus at localhost:9090
```

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the pieces fit; the lifecycle flywheel.
- [`docs/MANUAL_MAPPING.md`](docs/MANUAL_MAPPING.md) — every feature mapped to its field-manual source, including what is **simulated** vs **real** at this scale.
- [`docs/SECURITY.md`](docs/SECURITY.md) — the attack/defense catalog and how to run each demo.

## What is real vs simulated at this scale

Mythos is honest about its scale. Mechanisms like KV cache, GQA, MoE routing,
DPO, RAG, the tool gateway, signing, and the audit chain are **real and
functional**. Things that fundamentally need a cluster or GPUs — N-D
parallelism, ZeRO/FSDP, paged-attention's hardware win, prefill/decode
disaggregation, real autoscaling — are **simulated or documented** so the
lifecycle is complete without misleading you. Each is labeled in
`docs/MANUAL_MAPPING.md`.

## License

Educational project. Use freely for learning AI systems and AI security.
