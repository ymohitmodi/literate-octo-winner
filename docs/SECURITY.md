# Security walkthrough

Run the whole narrated catalog:

```bash
python -m mythos.cli all --preset nano   # produces artifacts the demos use
python -m mythos.cli security --preset nano
```

The guiding principle from the AI Platform Security Field Manual runs through
all of it: **you cannot stop the model from being manipulated, so put the safety
decision in deterministic code at every boundary** — least privilege everywhere,
break the "lethal trifecta" (private data + untrusted content + an exfiltration
path never coexist in one component), gate the irreversible, and assume
compromise.

## The eight demos

1. **Prompt injection.** A deterministic input guardrail
   (`security/guardrails.detect_prompt_injection`) flags "ignore previous
   instructions / reveal your system prompt" *before* the model runs. The server
   returns `blocked: true`.

2. **Training-data extraction (canary).** A canary string is *scrubbed from data
   before training* (`data/curation.scrub_pii` + data minimization), so the
   model never learns it; the output filter blocks it as a second layer.
   Defense = never train on the secret.

3. **Membership inference.** Compare model loss on training vs held-out text;
   lower loss on members is the leak signal. At nano scale the advantage is
   near-zero (the model barely memorizes) — train longer or larger to watch it
   grow, and note that DP / data minimization shrink it.

4. **Pickle RCE vs safe load.** A crafted "model" file executes code the instant
   it is unpickled; `torch.load(weights_only=True)` refuses the code path.
   Mythos checkpoints are tensors-only by construction.

5. **Checkpoint integrity.** Weights are signed (HMAC over a content digest).
   Tampering with any tensor makes `verify_checkpoint` fail, and the loader
   refuses to load it.

6. **Tamper-evident audit log.** Each entry commits to the previous via
   `H(prev‖entry)`. Editing one past entry breaks the chain and `verify()`
   pinpoints where.

7. **Agentic tool gateway.** A hijacked agent tries to reach `evil.com` and to
   send email; default-deny egress blocks the first, and the reversibility floor
   routes the irreversible action to human approval. The agent can never
   self-approve.

8. **Backdoor / data poisoning.** Explains why clean-eval accuracy gives zero
   assurance and points at `attacks.inject_backdoor` + `trigger_aware_eval` to
   train and detect a triggered backdoor.

## Defense-in-depth in the serving path

`serving/runtime.Runtime.infer` runs the controls in order: authenticate → rate
limit → load-shed → input guardrail → token/spend budget → extraction detection
→ (RAG with tenant isolation + cite-or-abstain) → inference → output filter →
audit. No single layer is trusted alone.

## Red-team suite + ship gate

`python -m mythos.cli eval` scores a capability portfolio and a red-team suite
(each case tagged to MITRE ATLAS) and produces a **ship-gate** decision in which
**a safety failure blocks release regardless of capability**. This is the
"evaluation becomes a decision" idea, runnable in CI.

## Hardening for real use

- Set `MYTHOS_SIGNING_KEY` and `MYTHOS_AUTH_SALT` to real secrets (the dev
  defaults are intentionally insecure).
- Issue per-client API keys with least-privilege scopes via `AuthService.issue_key`.
- Keep `security.*` toggles on in `config.py`.
- Put the egress allowlist under your control and keep it small.
