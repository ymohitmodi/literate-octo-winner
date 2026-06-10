"""Minimal Prometheus-style metrics (no dependency needed).

Three pillars, scaled down: counters + histograms exported in Prometheus text
format at /metrics so the same Grafana/Prometheus stack from the System Design
manual can scrape it. Percentiles (p50/p99) are what users feel, so we keep a
histogram, not just an average.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict


class Metrics:
    def __init__(self):
        self._counters: dict[tuple, float] = defaultdict(float)
        self._hist: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = {}
        self._lock = threading.Lock()

    def inc(self, name: str, value: float = 1.0, **labels):
        with self._lock:
            self._counters[(name, tuple(sorted(labels.items())))] += value

    def observe(self, name: str, value: float):
        with self._lock:
            h = self._hist[name]
            h.append(value)
            if len(h) > 5000:
                del h[:1000]

    def gauge(self, name: str, value: float):
        self._gauges[name] = value

    def percentile(self, name: str, p: float) -> float:
        h = sorted(self._hist.get(name, []))
        if not h:
            return 0.0
        return h[min(len(h) - 1, int(p / 100 * len(h)))]

    def render(self) -> str:
        lines = []
        for (name, labels), v in self._counters.items():
            lbl = ",".join(f'{k}="{val}"' for k, val in labels)
            lines.append(f"{name}{{{lbl}}} {v}" if lbl else f"{name} {v}")
        for name, v in self._gauges.items():
            lines.append(f"{name} {v}")
        for name in self._hist:
            for p in (50, 95, 99):
                lines.append(f'{name}_p{p} {self.percentile(name, p):.4f}')
        return "\n".join(lines) + "\n"


class Timer:
    def __init__(self, metrics: Metrics, name: str):
        self.metrics, self.name = metrics, name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.metrics.observe(self.name, time.time() - self.t0)
