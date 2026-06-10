# Lyceum architecture

Lyceum is organized around the Frontier manual's **seven-stage lifecycle**, with
the Security manual's controls woven into every stage and the System Design
manual's patterns wrapped around serving. The whole thing is one Python package
driven by one config.

```
                          ┌─────────────────────────────────────────┐
                          │            lyceum/config.py              │
                          │  one LyceumConfig drives every stage;    │
                          │  presets (nano/tiny/small) = scale dial  │
                          └─────────────────────────────────────────┘
   DATA              TOKENIZER         MODEL              TRAIN                INFERENCE
 ┌────────┐        ┌──────────┐     ┌──────────┐    ┌───────────────┐    ┌───────────────┐
 │corpus  │        │byte-level│     │RMSNorm   │    │pretrain (WSD, │    │KV cache,      │
 │curation│──tok──▶│   BPE    │──▶  │RoPE      │──▶ │AdamW, MoE aux,│──▶ │sampling,      │
 │PII scrub│       │(scratch) │     │SwiGLU,GQA│    │spike rollback)│    │streaming,     │
 │AI-BOM  │        └──────────┘     │MoE       │    │SFT → DPO      │    │batching, ToC* │
 └────────┘                         └──────────┘    │signed ckpts   │    └───────────────┘
      │                                              └───────────────┘            │
      │                                                                           │
      ▼                       MEMORY / CONTEXT                SECURITY            ▼
 ┌──────────────┐        ┌────────────────────────┐   ┌──────────────────┐  ┌──────────────┐
 │ RAG vector   │◀──────▶│ context mgr: instruction│   │ guardrails       │  │ SERVING      │
 │ store +      │        │ hierarchy, sliding win, │   │ audit (hashchain)│  │ FastAPI:     │
 │ memory store │        │ summarization, spotlight│   │ gateway, limits, │◀▶│ auth, rate,  │
 │ (4-gate write)│       └────────────────────────┘   │ auth, signing    │  │ cache, /metrics│
 └──────────────┘                                      │ attacks (demos)  │  │ health, SSE  │
                                                       └──────────────────┘  └──────────────┘
                                                                │                   │
                          EVALUATION  ◀── ship gate ───────────┘                   │
                       ┌──────────────────────────┐                                │
                       │ capability portfolio +   │      INFRA (optional, compose) │
                       │ red-team suite (ATLAS) + │   nginx LB → 2 workers ◀────────┘
                       │ ship gate                │   → Prometheus → Grafana
                       └──────────────────────────┘
                                                       *ToC = test-time compute
```

## Module map

| Path | Responsibility |
|---|---|
| `lyceum/config.py` | the single `LyceumConfig`; presets (nano/tiny/small/xl); hardware-aware `auto` selection |
| `lyceum/hardware.py` | GPU/CPU capability detection; enables AMP/flash/compile/multi-GPU or warns + falls back |
| `lyceum/inference/quantize.py`, `speculative.py`, `paged_kv.py` | int8 quantization, speculative decoding, paged KV allocator |
| `lyceum/serving/batching.py` | continuous / in-flight batching scheduler |
| `lyceum/train/grpo.py`, `reward.py`, `constitutional.py`, `dp_sgd.py`, `distributed.py` | RLVR/GRPO, reward model + PPO-lite, Constitutional AI, DP-SGD, DDP/FSDP + parallelism plan |
| `lyceum/interpretability/sae.py` | sparse autoencoder + activation steering |
| `lyceum/eval/scaling_laws.py` | scaling-law ladder + power-law fit |
| `lyceum/memory/hybrid.py` | hybrid BM25 + vector retrieval (RRF fusion) |
| `lyceum/data/tokenizer.py` | from-scratch byte-level BPE |
| `lyceum/data/corpus.py` | offline corpus + SFT + preference + RAG data generation |
| `lyceum/data/curation.py` | dedup, quality filter, **PII scrub**, provenance/digests |
| `lyceum/data/dataset.py` | packed-text, SFT (masked), and preference datasets |
| `lyceum/model/transformer.py` | RMSNorm, RoPE, GQA attention, SwiGLU, MoE, KV cache |
| `lyceum/train/pretrain.py` | next-token training, WSD schedule, spike rollback, MoE aux loss |
| `lyceum/train/sft.py`, `dpo.py` | alignment: SFT then DPO |
| `lyceum/train/checkpoint.py` | signed, integrity-checked, tensors-only checkpoints + AI-BOM |
| `lyceum/inference/engine.py` | KV-cache decode, sampling, streaming, batching, test-time compute |
| `lyceum/memory/context.py` | instruction hierarchy, truncation, history compression |
| `lyceum/memory/rag.py` | vector store, per-tenant retrieval, cite-or-abstain, 4-gate memory |
| `lyceum/security/*` | guardrails, audit, gateway, auth, limits, attacks, demo |
| `lyceum/eval/harness.py` | capability + red-team suites, ship gate, ATLAS tags |
| `lyceum/serving/*` | FastAPI server, runtime (cache/queue/security), metrics |
| `lyceum/agent.py` | ReAct loop over the deterministic gateway |
| `lyceum/cli.py` | the conductor: `data → pretrain → align → eval → security → serve` |

## The flywheel

The Frontier manual frames the lifecycle as a loop, not a line: evaluation
findings (especially red-team failures) feed back into data and post-training.
In Lyceum this is concrete — `python -m lyceum.cli eval` produces a ship-gate
decision, and a failing red-team case is the signal to add training data or
tighten a guardrail before re-running the pipeline.

## Design choices for CPU / offline operation

- **No network needed.** The corpus and all datasets are generated locally;
  RAG embeddings are a pure-numpy hashed n-gram vector, not a downloaded model.
- **Tensors-only checkpoints** with `weights_only` loading — closes the pickle
  RCE class by construction and keeps load fast.
- **One process by default.** The serving concepts (batching, caching,
  single-flight, load shedding, metrics) are demonstrated in-process; the
  optional compose stack shows the multi-worker version.
