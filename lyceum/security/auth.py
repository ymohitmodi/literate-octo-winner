"""Minimal authentication + per-key scopes.

Security manual: each caller has its own identity (never anonymous/shared), and
authorization is least-privilege. Keys are stored only as salted hashes, and
short-lived tokens model the "JIT credential" idea (prefer time-boxed tokens
over standing secrets).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass, field


def _hash(key: str, salt: str) -> str:
    return hashlib.sha256((salt + key).encode()).hexdigest()


@dataclass
class Principal:
    name: str
    scopes: set[str]
    tenant: str = "public"


@dataclass
class AuthService:
    salt: str = field(default_factory=lambda: os.environ.get("LYCEUM_AUTH_SALT", "lyceum-salt"))
    _keys: dict[str, Principal] = field(default_factory=dict)   # hash -> principal
    _tokens: dict[str, tuple[Principal, float]] = field(default_factory=dict)

    def issue_key(self, name: str, scopes: set[str], tenant="public") -> str:
        raw = "mk_" + secrets.token_urlsafe(16)
        self._keys[_hash(raw, self.salt)] = Principal(name, set(scopes), tenant)
        return raw

    def authenticate(self, api_key: str | None) -> Principal | None:
        if not api_key:
            return None
        return self._keys.get(_hash(api_key, self.salt))

    def mint_token(self, principal: Principal, ttl: float = 300.0) -> str:
        tok = secrets.token_urlsafe(24)
        self._tokens[tok] = (principal, time.time() + ttl)
        return tok

    def verify_token(self, token: str) -> Principal | None:
        entry = self._tokens.get(token)
        if not entry:
            return None
        principal, expiry = entry
        if time.time() > expiry:
            del self._tokens[token]   # short-lived: expires, unlike a standing secret
            return None
        return principal

    @staticmethod
    def authorize(principal: Principal, scope: str) -> bool:
        return scope in principal.scopes or "admin" in principal.scopes
