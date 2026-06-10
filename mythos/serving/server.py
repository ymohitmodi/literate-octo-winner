"""FastAPI inference server — the deployment front door.

Endpoints:
  GET  /healthz   - liveness/readiness probe (for the load balancer)
  GET  /metrics   - Prometheus exposition format
  POST /generate  - authenticated, rate-limited, guardrailed inference
  POST /stream    - token streaming over Server-Sent Events
  GET  /audit/verify - prove the audit log has not been tampered with

Security middleware order mirrors the manual's defense-in-depth: authenticate ->
rate limit -> load shed -> guardrails -> inference -> output filter -> audit.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from ..config import MythosConfig, get_config
from .runtime import Runtime


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = None
    temperature: float | None = None
    use_rag: bool = False
    tenant: str = "public"


def create_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="Mythos Inference Server", version="0.1.0")
    cfg = runtime.cfg

    def _auth(api_key: str | None):
        principal = runtime.auth.authenticate(api_key)
        if cfg.security.require_auth and principal is None:
            raise HTTPException(401, "missing or invalid API key")
        return principal or runtime.auth.authenticate(runtime.demo_key)

    def _rate(principal):
        ok, info = runtime.rate.allow(principal.name)
        if not ok:
            raise HTTPException(429, f"rate limited; retry after {info:.1f}s",
                                headers={"Retry-After": str(int(info) + 1)})

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "model": cfg.name,
                "params": runtime.engine.model.num_params()}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        runtime.metrics.gauge("mythos_inflight", runtime._inflight)
        return runtime.metrics.render()

    @app.get("/audit/verify")
    def audit_verify():
        ok, msg = runtime.audit.verify()
        return {"intact": ok, "detail": msg}

    @app.post("/generate")
    def generate(req: GenerateRequest, x_api_key: str | None = Header(None)):
        principal = _auth(x_api_key)
        _rate(principal)
        if not runtime.acquire():
            raise HTTPException(503, "server saturated (load shedding)")
        try:
            gen_kw = {}
            if req.max_new_tokens is not None:
                gen_kw["max_new_tokens"] = req.max_new_tokens
            if req.temperature is not None:
                gen_kw["temperature"] = req.temperature
            return runtime.infer(req.prompt, principal, use_rag=req.use_rag,
                                 tenant=req.tenant, **gen_kw)
        finally:
            runtime.release()

    @app.post("/stream")
    def stream(req: GenerateRequest, x_api_key: str | None = Header(None)):
        principal = _auth(x_api_key)
        _rate(principal)
        from ..security.guardrails import detect_prompt_injection
        if cfg.security.prompt_injection_filter and detect_prompt_injection(req.prompt).flagged:
            raise HTTPException(400, "prompt injection detected")
        ids = runtime.engine._format_prompt(req.prompt)

        def gen():
            for tid in runtime.engine.stream(
                    ids, max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature):
                yield f"data: {json.dumps({'token': runtime.tok.decode([tid])})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def build_default_app() -> FastAPI:
    """Entry point for ``uvicorn mythos.serving.server:app`` — loads the latest
    artifacts produced by the pipeline."""
    cfg_path = Path("artifacts/config.json")
    cfg = MythosConfig.load(cfg_path) if cfg_path.exists() else get_config("auto")
    rt = Runtime.from_checkpoint(cfg, "artifacts/ckpt_aligned.pt"
                                 if Path("artifacts/ckpt_aligned.pt").exists()
                                 else "artifacts/ckpt_pretrain.pt",
                                 "artifacts/tokenizer.json")
    from ..data.corpus import build_rag_documents
    docs = build_rag_documents("artifacts/rag_docs.txt").read_text().splitlines()
    rt.rag.add_documents([d for d in docs if d.strip()])
    return create_app(rt)


try:  # only built when artifacts exist, so importing the module never fails
    app = build_default_app()
except Exception:  # pragma: no cover
    app = None
