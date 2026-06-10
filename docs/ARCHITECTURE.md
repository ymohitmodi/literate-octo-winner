# Mythos architecture

Mythos is organized around the Frontier manual's **seven-stage lifecycle**, with
the Security manual's controls woven into every stage and the System Design
manual's patterns wrapped around serving. The whole thing is one Python package
driven by one config.

```
                          ┌─────────────────────────────────────────┐
                          │            mythos/config.py              │
                          │  one MythosConfig drives every stage;    │
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
| `mythos/config.py` | the single `MythosConfig`; presets; RAM-based `auto` selection |
| `mythos/data/tokenizer.py` | from-scratch byte-level BPE |
| `mythos/data/corpus.py` | offline corpus + SFT + preference + RAG data generation |
| `mythos/data/curation.py` | dedup, quality filter, **PII scrub**, provenance/digests |
| `mythos/data/dataset.py` | packed-text, SFT (masked), and preference datasets |
| `mythos/model/transformer.py` | RMSNorm, RoPE, GQA attention, SwiGLU, MoE, KV cache |
| `mythos/train/pretrain.py` | next-token training, WSD schedule, spike rollback, MoE aux loss |
| `mythos/train/sft.py`, `dpo.py` | alignment: SFT then DPO |
| `mythos/train/checkpoint.py` | signed, integrity-checked, tensors-only checkpoints + AI-BOM |
| `mythos/inference/engine.py` | KV-cache decode, sampling, streaming, batching, test-time compute |
| `mythos/memory/context.py` | instruction hierarchy, truncation, history compression |
| `mythos/memory/rag.py` | vector store, per-tenant retrieval, cite-or-abstain, 4-gate memory |
| `mythos/security/*` | guardrails, audit, gateway, auth, limits, attacks, demo |
| `mythos/eval/harness.py` | capability + red-team suites, ship gate, ATLAS tags |
| `mythos/serving/*` | FastAPI server, runtime (cache/queue/security), metrics |
| `mythos/agent.py` | ReAct loop over the deterministic gateway |
| `mythos/cli.py` | the conductor: `data → pretrain → align → eval → security → serve` |

## The flywheel

The Frontier manual frames the lifecycle as a loop, not a line: evaluation
findings (especially red-team failures) feed back into data and post-training.
In Mythos this is concrete — `python -m mythos.cli eval` produces a ship-gate
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
