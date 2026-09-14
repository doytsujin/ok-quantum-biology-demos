"""A minimal agentic control plane for hybrid quantum-classical jobs.

Standalone, written from scratch for this demo, following the architecture of an
agentic-control-plane-for-quantum design: a gateway that sits in front of the
quantum backends and provides, for every job,

  - admission control against a tenant policy (which backends are allowed);
  - token-bucket rate limiting per tenant;
  - a cache keyed on circuit identity, so an identical circuit is not re-executed
    (fewer shots / less redundant backend use);
  - a structured event record (QEBR) for observability.

It is deliberately small and provider-neutral. The point is the *shape*: the
descriptor runtime governs the data and the pipeline stage; this gateway governs
access to the quantum resource itself. Together they are a control plane for a
hybrid quantum-classical workflow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class GatewayError(Exception):
    pass


class AdmissionDenied(GatewayError):
    pass


class RateLimitExceeded(GatewayError):
    pass


class DenialReason(Enum):
    BACKEND_NOT_ALLOWED = "backend_not_allowed"
    RATE_LIMITED = "rate_limited"


@dataclass
class TokenBucket:
    capacity: float
    refill_rate: float           # tokens per second
    _tokens: float = field(init=False)
    _last: float = field(init=False)

    def __post_init__(self):
        self._tokens = self.capacity
        self._last = time.monotonic()

    def acquire(self, tokens: float = 1.0) -> bool:
        now = time.monotonic()
        self._tokens = min(self.capacity,
                           self._tokens + (now - self._last) * self.refill_rate)
        self._last = now
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False


@dataclass
class TenantPolicy:
    tenant_id: str
    allowed_backends: List[str] = field(default_factory=list)  # empty => all
    rate_limit_rps: float = 5.0


@dataclass
class CircuitRequest:
    tenant_id: str
    circuit_id: str              # circuit identity (cache key)
    target_backend: str


@dataclass
class QEBREvent:
    """Quantum Event-Based Record — one structured control-plane event."""
    event: str
    tenant_id: str
    circuit_id: str
    backend: str
    timestamp: float
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubmitResult:
    value: Any
    from_cache: bool
    backend: str


@dataclass
class QuantumGateway:
    tenant_policies: Dict[str, TenantPolicy] = field(default_factory=dict)
    _limiters: Dict[str, TokenBucket] = field(default_factory=dict)
    _cache: Dict[str, Any] = field(default_factory=dict)
    events: List[QEBREvent] = field(default_factory=list)

    def register_tenant(self, policy: TenantPolicy) -> None:
        self.tenant_policies[policy.tenant_id] = policy
        self._limiters[policy.tenant_id] = TokenBucket(
            capacity=policy.rate_limit_rps * 2, refill_rate=policy.rate_limit_rps)

    def _emit(self, event: str, req: CircuitRequest, **detail) -> None:
        self.events.append(QEBREvent(event, req.tenant_id, req.circuit_id,
                                     req.target_backend, time.time(), detail))

    def submit(self, req: CircuitRequest, run: Callable[[], Any]) -> SubmitResult:
        policy = self.tenant_policies.get(req.tenant_id)
        # --- admission control: is this backend allowed for this tenant? ---
        if policy and policy.allowed_backends and \
                req.target_backend not in policy.allowed_backends:
            self._emit("admission_denied", req,
                       reason=DenialReason.BACKEND_NOT_ALLOWED.value)
            raise AdmissionDenied(
                f"backend '{req.target_backend}' not allowed for tenant "
                f"'{req.tenant_id}'")
        # --- rate limiting --------------------------------------------------
        limiter = self._limiters.get(req.tenant_id)
        if limiter and not limiter.acquire():
            self._emit("rate_limited", req, reason=DenialReason.RATE_LIMITED.value)
            raise RateLimitExceeded(f"rate limit exceeded for '{req.tenant_id}'")
        self._emit("admitted", req)
        # --- circuit-identity cache -----------------------------------------
        if req.circuit_id in self._cache:
            self._emit("cache_hit", req)
            return SubmitResult(self._cache[req.circuit_id], True, req.target_backend)
        # --- dispatch to backend -------------------------------------------
        value = run()
        self._cache[req.circuit_id] = value
        self._emit("executed", req)
        return SubmitResult(value, False, req.target_backend)

    def count(self, event: str) -> int:
        return sum(1 for e in self.events if e.event == event)
