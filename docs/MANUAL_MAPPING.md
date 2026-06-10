# Feature → field-manual mapping

Every Mythos feature traces to one of the three field manuals. The **Fidelity**
column is honest about scale: **real** = the actual mechanism, functional at
tiny scale; **partial** = a reduced form that shows the mechanism but not the
frontier-scale benefit; **simulated/doc** = can't truly run on one CPU, so it is
demonstrated as a mock or documented.

## A. Frontier Model Field Manual

| Concept | Where in Mythos | Fidelity |
|---|---|---|
| Seven-stage lifecycle / flywheel | repo structure + `cli.py all` | real |
| Compute estimate `C ≈ 6·N·D` | printed by `train/pretrain.py` | real |
| Byte-level BPE from scratch | `data/tokenizer.py` | real |
| Reserved special / role tokens | `config.TokenizerConfig.special_tokens` | real |
| Data curation: dedup, quality filter | `data/curation.py` | real |
| Data mixing (facts + stories) | `data/corpus.py` | real (toy) |
| Provenance / content-hash pinning | `data/curation.py`, `train/checkpoint.py` (BOM) | real |
| Decoder-only transformer | `model/transformer.py` | real |
| Pre-norm + **RMSNorm** | `model/transformer.RMSNorm` | real |
| **RoPE** rotary positions | `model/transformer.apply_rope` | real |
| **SwiGLU** FFN | `model/transformer.SwiGLU` | real |
| **Grouped-Query Attention** | `model/transformer.Attention` (`n_kv_heads`) | real |
| **Mixture-of-Experts** + top-k router | `model/transformer.MoEFeedForward` | real |
| MoE load-balancing aux loss | `train/pretrain.moe_aux_loss` | real |
| Next-token cross-entropy + teacher forcing | `train/pretrain.py` | real |
| AdamW (β 0.9/0.95, weight decay) | `train/pretrain.py` | real |
| Warmup-Stable-Decay schedule + annealing | `train/pretrain.wsd_lr` | real |
| Gradient clipping | `train/pretrain.py` | real |
| Loss-spike playbook (rollback to verified ckpt) | `train/pretrain.py` | real |
| Checkpointing (frequent, verified) | `train/checkpoint.py` | real |
| Perplexity / bits-per-token metric | `train/pretrain._record` | real |
| Run monitoring / logging | `pretrain.log`, `serving/metrics.py` | real |
| Scaling-law ladder | tweak presets + read `best_loss` | partial |
| SFT + chat template + instruction hierarchy | `train/sft.py`, `memory/context.py` | real |
| **DPO** (no reward model) | `train/dpo.py` | real |
| Refusals / HHH behavior | preference data in `data/corpus.py` | partial |
| RLHF / PPO / reward model | not built (DPO chosen for CPU) | doc |
| Constitutional AI / critique-revise | read-only policy store in `memory/rag.MemoryStore` | partial |
| RLVR / GRPO reasoning | not built | doc |
| Chain-of-thought / test-time compute | self-consistency, best-of-N in `inference/engine.py` | real |
| Prefill vs decode | `inference/engine.stream` | real |
| **KV cache** | `model/transformer.KVCache` | real |
| Sampling: temp/top-k/top-p/rep-penalty | `inference/engine._sample` | real |
| Streaming generation | `inference/engine.stream`, `/stream` SSE | real |
| Continuous/static batching | `inference/engine.batch_generate` | partial (static) |
| Quantization / speculative decoding | not built | doc |
| Evaluation portfolio + ship gate | `eval/harness.py` | real |
| Red-teaming + regression suite | `eval/harness.red_team_suite` | real |
| N-D parallelism, ZeRO/FSDP | — | simulated/doc |
| Mechanistic interpretability / SAEs | — | doc |
| RSP / if-then capability gates | ship gate logic in `eval/harness.py` | partial |

## B. AI Platform Security Field Manual

| Attack / Defense | Where in Mythos | Fidelity |
|---|---|---|
| Data poisoning / **backdoor trigger** | `security/attacks.inject_backdoor`, `trigger_aware_eval` | real |
| Trust tiering / provenance | `data/curation.py`, `memory/rag.Chunk.trust` | real |
| **PII scrubbing** before training | `data/curation.scrub_pii` | real |
| **Membership inference** | `security/attacks.membership_inference` | real |
| Training-data (**canary**) extraction | `security/attacks.attempt_canary_extraction` | real |
| Output filtering (PII + canary + sys-prompt) | `security/guardrails.filter_output` | real |
| **Pickle deserialization RCE** vs safe load | `security/attacks.safe_vs_unsafe_load` | real |
| Tensors-only + **signed** checkpoints | `train/checkpoint.py` | real |
| AI bill-of-materials / pin by digest | `train/checkpoint.py`, `data/curation.write_manifest` | real |
| **Direct prompt injection** / jailbreak detect | `security/guardrails.py` | real |
| **Indirect prompt injection** (in docs) | `agent.py` demo + `memory/context.py` spotlighting | real |
| Instruction hierarchy (system > user > data) | `memory/context.render` | real |
| **RAG poisoning** / cross-tenant leakage | `memory/rag.VectorStore.search` (tenant filter) | real |
| Cite-or-abstain grounding | `memory/rag.cite_or_abstain` | real |
| **Memory poisoning** (ASI06) | `memory/rag.MemoryStore` (4-gate write path) | real |
| Read-only policy store ("constitution") | `memory/rag.MemoryStore.set_policy` | real |
| **Deterministic tool gateway** | `security/gateway.ToolGateway` | real |
| **Reversibility floor** + human approval | `security/gateway.Reversibility` | real |
| Least privilege / per-tool scope | `security/gateway.Tool` | real |
| Bounded/stoppable autonomy (budget, killswitch) | `security/gateway.py` | real |
| Monotonic risk ratchet | `security/gateway.raise_risk` | real |
| **Default-deny egress allowlist** | `security/gateway.py` | real |
| Model **extraction** detection | `security/attacks.ExtractionDetector` | real |
| **Denial-of-wallet** token/spend budgets | `security/limits.SpendBudget` | real |
| Rate limiting (token bucket) | `security/limits.TokenBucket` | real |
| Auth / own identity / JIT tokens | `security/auth.py` | real |
| **Tamper-evident hash-chained audit log** | `security/audit.py` | real |
| MITRE ATLAS tagging | `eval/harness.RED_TEAM_CASES` | real |
| DP-SGD, embedding inversion, Morris II, SSRF | — | doc/simulated |

## C. System Design Field Manual

| Concept | Where in Mythos | Fidelity |
|---|---|---|
| Stateless workers / shared-nothing | `serving/server.py` + compose workers | real |
| API gateway + **L7 load balancing** | `infra/nginx.conf` (least-conn) | real |
| Health checks / draining | `/healthz`, nginx `max_fails` | real |
| **Rate limiting** (token bucket) | `security/limits.TokenBucket` | real |
| Response/prompt **cache** | `serving/runtime.PromptCache` | real |
| Cache **single-flight** (anti-stampede) | `serving/runtime.PromptCache.get_or_compute` | real |
| LRU eviction | `serving/runtime.PromptCache` | real |
| **Load shedding** / backpressure | `serving/runtime.acquire` (503) | real |
| Request queue / async decoupling | in-flight accounting + bounded admission | partial |
| **Circuit breaker** | `security/limits.CircuitBreaker` | real |
| Timeouts on every call | `infra/nginx.conf` proxy timeouts | real |
| Idempotency / retries | client guidance (git ops); breaker | doc |
| **Observability**: metrics + percentiles | `serving/metrics.py`, `/metrics` | real |
| Prometheus + Grafana | `docker-compose.yml`, `infra/` | real (optional) |
| SLI/SLO/error budget | dashboards over `/metrics` | partial |
| **RAG ingestion + retrieval** | `memory/rag.py` | real |
| Vector DB / ANN | `memory/rag.VectorStore` (brute-force cosine) | partial |
| Per-tenant authz at retrieval | `memory/rag.VectorStore.search(tenant=...)` | real |
| Hybrid (vector + keyword) retrieval | — | doc |
| Containerization / compose orchestration | `Dockerfile`, `docker-compose.yml` | real |
| Canary / shadow deploys, autoscaling | — | doc |
| Consistent hashing, CDC/outbox, sagas | — | doc |

If you want to extend Mythos, the `doc`-only rows are the natural next exercises
— each manual section above tells you exactly what the production version adds.
