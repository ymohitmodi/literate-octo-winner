"""Continuous / in-flight batching scheduler for the inference server.

System Design manual: batching is *the* LLM-serving throughput breakthrough.
A single request barely uses the matrix-multiply hardware; running many requests
through the same forward passes amortises the weight loads and keeps the
accelerator busy, so throughput climbs sharply with batch size.

This scheduler implements *dynamic* batching: a background worker collects
queued requests and, on each tick, forms a batch of up to ``max_batch_size``
(or however many have arrived once ``batch_timeout_ms`` elapses), runs them
through ``engine.batch_generate`` in one shot, and hands each result back to its
waiting caller via a ``concurrent.futures.Future``.

NEXT STEP (documented, not implemented here): true vLLM-style *continuous* (a.k.a.
in-flight) batching admits and evicts requests at the *per-token* level -- a
finished sequence frees its slot mid-batch and a queued request fills it on the
very next decode step, instead of waiting for the whole batch to drain. That
requires interleaving the scheduler with the token-by-token decode loop and a
paged KV-cache; dynamic batching here is the structural basis for it.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field

from ..inference.engine import InferenceEngine


@dataclass
class _Request:
    prompt: str
    gen_kw: dict
    future: Future


@dataclass
class SchedulerMetrics:
    total_requests: int = 0
    num_batches: int = 0
    total_batched: int = 0          # sum of batch sizes (for the average)
    batch_sizes: list[int] = field(default_factory=list)

    @property
    def avg_batch_size(self) -> float:
        return self.total_batched / self.num_batches if self.num_batches else 0.0

    def as_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "num_batches": self.num_batches,
            "avg_batch_size": round(self.avg_batch_size, 3),
            "max_batch_size_seen": max(self.batch_sizes) if self.batch_sizes else 0,
        }


class ContinuousBatchScheduler:
    """Background batching server in front of an :class:`InferenceEngine`.

    Usage::

        sched = ContinuousBatchScheduler(engine, max_batch_size=8,
                                         batch_timeout_ms=50)
        sched.start()
        fut = sched.submit("hello", max_new_tokens=16)
        print(fut.result(timeout=30))
        # or, blocking:
        print(sched.generate("hello", max_new_tokens=16))
        sched.stop()
    """

    def __init__(self, engine: InferenceEngine, max_batch_size: int = 8,
                 batch_timeout_ms: int = 50):
        self.engine = engine
        self.max_batch_size = max(1, int(max_batch_size))
        self.batch_timeout = max(0.0, batch_timeout_ms / 1000.0)

        self._queue: deque[_Request] = deque()
        self._cond = threading.Condition()      # guards _queue and _running
        self._running = False
        self._worker: threading.Thread | None = None
        self.metrics = SchedulerMetrics()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        with self._cond:
            if self._running:
                return
            self._running = True
        self._worker = threading.Thread(target=self._run, name="batch-scheduler",
                                        daemon=True)
        self._worker.start()

    def stop(self, timeout: float | None = 10.0) -> None:
        """Stop accepting work, drain in-flight requests, join the worker."""
        with self._cond:
            self._running = False
            self._cond.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=timeout)
            self._worker = None
        # Fail any requests still queued after shutdown so callers don't hang.
        with self._cond:
            leftover = list(self._queue)
            self._queue.clear()
        for req in leftover:
            if not req.future.done():
                req.future.set_exception(
                    RuntimeError("scheduler stopped before request was served"))

    # ------------------------------------------------------------------ #
    # Submission
    # ------------------------------------------------------------------ #
    def submit(self, prompt: str, **gen_kw) -> Future:
        """Enqueue a prompt; returns a Future resolving to the generated text."""
        fut: Future = Future()
        with self._cond:
            if not self._running:
                fut.set_exception(RuntimeError("scheduler is not running"))
                return fut
            self.metrics.total_requests += 1
            self._queue.append(_Request(prompt, gen_kw, fut))
            self._cond.notify()
        return fut

    def generate(self, prompt: str, *, timeout: float | None = 60.0,
                 **gen_kw) -> str:
        """Blocking convenience: submit and wait for the result."""
        return self.submit(prompt, **gen_kw).result(timeout=timeout)

    # ------------------------------------------------------------------ #
    # Worker loop
    # ------------------------------------------------------------------ #
    def _collect_batch(self) -> list[_Request]:
        """Block (no busy-spin) until work arrives or shutdown, then gather a
        batch: take what's available up to max_batch_size, giving late arrivals
        up to batch_timeout to join."""
        with self._cond:
            # Wait for at least one request (or shutdown).
            while self._running and not self._queue:
                self._cond.wait()
            if not self._queue:
                return []  # shutting down with an empty queue

            batch = [self._queue.popleft()]
            deadline = time.monotonic() + self.batch_timeout
            # Fill the batch, waiting up to the timeout window for more to arrive.
            while len(batch) < self.max_batch_size:
                if self._queue:
                    batch.append(self._queue.popleft())
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._running:
                    break
                self._cond.wait(timeout=remaining)
            return batch

    def _run(self) -> None:
        while True:
            with self._cond:
                if not self._running and not self._queue:
                    break
            batch = self._collect_batch()
            if not batch:
                continue
            self._execute(batch)

    def _execute(self, batch: list[_Request]) -> None:
        # All requests in a batch share generation kwargs (batch_generate takes
        # one set). Group identical kwargs into sub-batches so heterogeneous
        # requests still run correctly.
        groups: dict[tuple, list[_Request]] = {}
        for req in batch:
            key = tuple(sorted(req.gen_kw.items()))
            groups.setdefault(key, []).append(req)

        for key, reqs in groups.items():
            prompts = [r.prompt for r in reqs]
            gen_kw = dict(key)
            try:
                outputs = self.engine.batch_generate(prompts, **gen_kw)
            except Exception as exc:  # one bad batch must not kill the worker
                for r in reqs:
                    if not r.future.done():
                        r.future.set_exception(exc)
                continue
            self.metrics.num_batches += 1
            self.metrics.total_batched += len(reqs)
            self.metrics.batch_sizes.append(len(reqs))
            for r, out in zip(reqs, outputs):
                if not r.future.done():
                    r.future.set_result(out)


# --------------------------------------------------------------------------- #
# Self-test: stand up a nano engine and pound it from several threads.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from ..config import get_config
    from ..data.corpus import build_pretrain_corpus
    from ..data.curation import curate
    from ..data.tokenizer import BPETokenizer
    from ..model.transformer import LyceumLM
    from ..inference.engine import InferenceEngine

    import tempfile
    from pathlib import Path

    cfg = get_config("nano")

    # Tiny corpus -> curate -> train tokenizer -> untrained model -> engine.
    tmp = Path(tempfile.mkdtemp()) / "corpus.txt"
    build_pretrain_corpus(tmp, n_docs=120, seed=0)
    text, _report = curate(tmp.read_text(encoding="utf-8"))
    tok = BPETokenizer(special_tokens=cfg.tokenizer.special_tokens)
    tok.train(text, vocab_size=300)  # keep tiny so the self-test is fast
    model = LyceumLM(cfg.model, tok.vocab_size)
    engine = InferenceEngine(model, tok, cfg)

    sched = ContinuousBatchScheduler(
        engine,
        max_batch_size=cfg.serving.max_batch_size,
        batch_timeout_ms=cfg.serving.batch_timeout_ms,
    )
    sched.start()

    prompts = [f"prompt number {i}" for i in range(5)]
    futures: list[Future] = []
    results: dict[int, str] = {}
    lock = threading.Lock()

    def worker(i: int) -> None:
        fut = sched.submit(prompts[i], max_new_tokens=8, temperature=0.0)
        out = fut.result(timeout=60)
        with lock:
            results[i] = out

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(prompts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    sched.stop(timeout=10)

    print(f"collected {len(results)}/{len(prompts)} results")
    for i in sorted(results):
        print(f"  [{i}] {results[i][:50]!r}")
    print("metrics:", sched.metrics.as_dict())

    assert len(results) == len(prompts), "not all requests were served"
    assert sched.metrics.num_batches >= 1, "no batches ran"
    # Under concurrent load with a 50ms window, batches should coalesce > 1.
    print(f"avg batch size = {sched.metrics.avg_batch_size:.2f} "
          f"(>1 means requests coalesced -> throughput up)")
    print("\nbatching.py self-test OK")
