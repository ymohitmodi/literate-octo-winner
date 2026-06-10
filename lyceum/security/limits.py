"""Rate limiting, token/compute budgets, and a circuit breaker.

These cover the availability defenses from both manuals: denial-of-wallet /
resource-exhaustion (cap input/output tokens and request rate) and reliability
(fail fast when a dependency is unhealthy).
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field


class TokenBucket:
    """Burst-friendly per-client rate limiter (the API default)."""

    def __init__(self, rate_per_min: int, burst: int | None = None):
        self.rate = rate_per_min / 60.0
        self.capacity = burst or max(1, rate_per_min)
        self._state: dict[str, tuple[float, float]] = {}

    def allow(self, key: str, cost: float = 1.0) -> tuple[bool, float]:
        now = time.time()
        tokens, last = self._state.get(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.rate)
        if tokens >= cost:
            self._state[key] = (tokens - cost, now)
            return True, tokens - cost
        self._state[key] = (tokens, now)
        retry_after = (cost - tokens) / self.rate
        return False, retry_after


@dataclass
class SpendBudget:
    """Denial-of-wallet defense: hard caps on tokens per request and per client."""
    max_input_tokens: int = 2048
    max_output_tokens: int = 512
    daily_token_cap: int = 100_000
    _spent: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def check_input(self, key: str, n_tokens: int) -> tuple[bool, str]:
        if n_tokens > self.max_input_tokens:
            return False, f"input {n_tokens} > cap {self.max_input_tokens}"
        if self._spent[key] + n_tokens > self.daily_token_cap:
            return False, "daily token budget exhausted"
        return True, "ok"

    def record(self, key: str, n_tokens: int):
        self._spent[key] += n_tokens

    def clamp_output(self, requested: int) -> int:
        return min(requested, self.max_output_tokens)


class CircuitBreaker:
    """CLOSED -> OPEN (fail fast) -> HALF_OPEN (test) -> CLOSED."""

    def __init__(self, fail_threshold: int = 5, reset_after: float = 10.0):
        self.fail_threshold = fail_threshold
        self.reset_after = reset_after
        self.failures = 0
        self.state = "CLOSED"
        self.opened_at = 0.0

    def allow(self) -> bool:
        if self.state == "OPEN":
            if time.time() - self.opened_at > self.reset_after:
                self.state = "HALF_OPEN"
                return True
            return False
        return True

    def record(self, ok: bool):
        if ok:
            self.failures = 0
            self.state = "CLOSED"
        else:
            self.failures += 1
            if self.failures >= self.fail_threshold:
                self.state = "OPEN"
                self.opened_at = time.time()
