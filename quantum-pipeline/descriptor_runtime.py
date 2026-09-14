"""A compact, self-contained dataset-descriptor runtime.

This is a small standalone reimplementation of the descriptor-driven pipeline
model — Descriptor, a non-Turing-complete PolicyEngine, a five-state lifecycle
Agent, and a structured TelemetryLogger. It is written from scratch for this
demo so the demo runs with no external repo; it follows the published design of
the descriptor model rather than vendoring anyone's code.

Lifecycle:  Idle -> Reasoning -> Binding -> Executing -> Publishing -> Idle
            with a Fault path when an ENFORCE policy blocks in Reasoning.

The safety property is the barrier between Reasoning and Executing: a policy
violation is detected and logged *before* any irreversible action runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


# --------------------------------------------------------------------------- #
# Descriptor: (id, schema, policies, actions, telemetry, provenance)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Policy:
    name: str
    rule: str
    action: str          # ENFORCE | AUDIT | WARN


@dataclass(frozen=True)
class Action:
    name: str
    requires: List[str]
    outputs: List[str]
    policy_checks: List[str]


@dataclass(frozen=True)
class Descriptor:
    id: str
    data_type: str
    schema: Dict[str, Any]
    provenance: Dict[str, Any]
    policies: List[Policy]
    permissible_actions: List[Action]
    telemetry_schema: Dict[str, Any]

    def action(self, name: str) -> Action | None:
        return next((a for a in self.permissible_actions if a.name == name), None)

    def policies_for(self, action_name: str) -> List[Policy]:
        act = self.action(action_name)
        if act is None:
            return []
        by_name = {p.name: p for p in self.policies}
        return [by_name[n] for n in act.policy_checks if n in by_name]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Descriptor":
        return Descriptor(
            id=d["id"],
            data_type=d.get("dataType", "unknown"),
            schema=d.get("schema", {}),
            provenance=d.get("provenance", {}),
            policies=[Policy(p["name"], p["rule"], p["action"])
                      for p in d.get("policies", [])],
            permissible_actions=[
                Action(a["name"], a.get("requires", []), a.get("outputs", []),
                       a.get("policyChecks", []))
                for a in d.get("permissibleActions", [])],
            telemetry_schema=d.get("telemetrySchema", {}),
        )


@dataclass
class DescriptorRegistry:
    _by_id: Dict[str, Descriptor] = field(default_factory=dict)

    def register(self, descriptor: Descriptor) -> None:
        self._by_id[descriptor.id] = descriptor

    def get(self, descriptor_id: str) -> Descriptor:
        return self._by_id[descriptor_id]


# --------------------------------------------------------------------------- #
# Policy engine: a tiny, non-Turing-complete rule evaluator.
#   rule ::= expr ((AND|OR) expr)*
#   expr ::= ident (>=|<=|==|!=|>|<) value  |  ident IN [v, v, ...]
# --------------------------------------------------------------------------- #
def _coerce(tok: str) -> Any:
    tok = tok.strip()
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        return tok


def _eval_expr(expr: str, state: Dict[str, Any]) -> bool:
    if " IN " in expr:
        lhs, rhs = expr.split(" IN ", 1)
        items = [_coerce(x) for x in rhs.strip().strip("[]").split(",")]
        return state.get(lhs.strip()) in items
    for op in (">=", "<=", "==", "!=", ">", "<"):
        if op in expr:
            lhs, rhs = expr.split(op, 1)
            a, b = state.get(lhs.strip()), _coerce(rhs)
            if a is None:
                return False
            return {
                ">=": a >= b, "<=": a <= b, "==": a == b,
                "!=": a != b, ">": a > b, "<": a < b,
            }[op]
    return False


def evaluate_rule(rule: str, state: Dict[str, Any]) -> bool:
    # OR binds looser than AND
    for or_clause in rule.split(" OR "):
        if all(_eval_expr(e, state) for e in or_clause.split(" AND ")):
            return True
    return False


@dataclass(frozen=True)
class PolicyAuditRecord:
    policy_name: str
    rule: str
    policy_action: str
    result: str          # PASS | FAIL
    blocking: bool
    dataset_id: str
    actor: str
    trace_id: str
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {"record_type": "policy_audit_record", "policy_name": self.policy_name,
                "rule": self.rule, "policy_action": self.policy_action,
                "result": self.result, "blocking": self.blocking,
                "dataset_id": self.dataset_id, "actor": self.actor,
                "trace_id": self.trace_id, "timestamp": self.timestamp}


@dataclass
class PolicyEngine:
    eval_count: int = 0

    def evaluate(self, policies: List[Policy], state: Dict[str, Any],
                 dataset_id: str, actor: str, trace_id: str):
        records, passed = [], True
        for p in policies:
            ok = evaluate_rule(p.rule, state)
            self.eval_count += 1
            blocking = (p.action == "ENFORCE") and not ok
            passed = passed and not blocking
            records.append(PolicyAuditRecord(
                p.name, p.rule, p.action, "PASS" if ok else "FAIL", blocking,
                dataset_id, actor, trace_id, time.time()))
        return passed, records


# --------------------------------------------------------------------------- #
# Telemetry
# --------------------------------------------------------------------------- #
@dataclass
class TelemetryRecord:
    record_type: str
    trace_id: str
    actor: str
    timestamp: float
    payload: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"record_type": self.record_type, "trace_id": self.trace_id,
             "actor": self.actor, "timestamp": self.timestamp,
             "payload": self.payload}
        if self.provenance is not None:
            d["provenance"] = self.provenance
        return d


@dataclass
class TelemetryLogger:
    records: List[TelemetryRecord] = field(default_factory=list)

    def emit(self, record_type: str, trace_id: str, actor: str,
             payload: Dict[str, Any] | None = None,
             provenance: Dict[str, Any] | None = None) -> None:
        self.records.append(TelemetryRecord(
            record_type, trace_id, actor, time.time(), payload or {}, provenance))

    def write_jsonl(self, path) -> int:
        text = "\n".join(json.dumps(r.to_dict()) for r in self.records) + "\n"
        from pathlib import Path
        Path(path).write_text(text)
        return len(text.encode())

    def total(self) -> int:
        return len(self.records)

    def trace_ids(self) -> set:
        return {r.trace_id for r in self.records}


# --------------------------------------------------------------------------- #
# Agent: five-state lifecycle with a reasoning-before-execution barrier.
# --------------------------------------------------------------------------- #
class State(Enum):
    IDLE = "Idle"
    REASONING = "Reasoning"
    BINDING = "Binding"
    EXECUTING = "Executing"
    PUBLISHING = "Publishing"
    FAULT = "Fault"


@dataclass
class Agent:
    name: str
    stage: str
    registry: DescriptorRegistry
    policy_engine: PolicyEngine
    telemetry: TelemetryLogger
    state: State = State.IDLE
    state_transitions: List[str] = field(default_factory=list)

    def _to(self, s: State) -> None:
        self.state_transitions.append(f"{self.state.value}->{s.value}")
        self.state = s

    def run_task(self, descriptor_id: str, action_name: str,
                 dataset_state: Dict[str, Any], sample_id: str,
                 upstream_trace_id: str) -> Dict[str, Any]:
        trace_id, actor = f"{sample_id}:{self.stage}", self.name
        self.telemetry.emit("signal_received", trace_id, actor, {"action": action_name})

        # --- REASONING: discover action, evaluate policies BEFORE acting ---
        self._to(State.REASONING)
        self.telemetry.emit("reasoning_entered", trace_id, actor)
        descriptor = self.registry.get(descriptor_id)
        action = descriptor.action(action_name)
        self.telemetry.emit("action_discovered", trace_id, actor,
                            {"action": action_name, "permitted": action is not None})
        policies = descriptor.policies_for(action_name)
        passed, audits = self.policy_engine.evaluate(
            policies, dataset_state, descriptor_id, actor, trace_id)
        self.telemetry.emit("policy_evaluated", trace_id, actor,
                            {"policies_checked": len(policies), "all_enforced_passed": passed})
        for rec in audits:
            d = rec.to_dict()
            self.telemetry.records.append(TelemetryRecord(
                "policy_audit_record", rec.trace_id, rec.actor, rec.timestamp,
                {k: v for k, v in d.items()
                 if k not in ("record_type", "trace_id", "actor", "timestamp")}))

        if not passed:
            self._to(State.FAULT)
            self.telemetry.emit("policy_refused", trace_id, actor,
                                {"blocked_by": [r.policy_name for r in audits if r.blocking]})
            self._to(State.IDLE)
            return {"success": False, "trace_id": trace_id, "reason": "policy_violation"}

        # --- BINDING / EXECUTING / PUBLISHING ------------------------------
        self._to(State.BINDING)
        self.telemetry.emit("binding_committed", trace_id, actor)
        self._to(State.EXECUTING)
        self.telemetry.emit("executing_entered", trace_id, actor)
        try:
            self._do_work(dataset_state)
        except Exception as exc:                       # e.g. gateway admission denial
            self._to(State.FAULT)
            self.telemetry.emit("execution_fault", trace_id, actor,
                                {"error": type(exc).__name__, "detail": str(exc)})
            self._to(State.IDLE)
            return {"success": False, "trace_id": trace_id, "reason": "execution_fault"}
        self.telemetry.emit("task_executed", trace_id, actor, {"outputs": action.outputs})
        self.telemetry.emit("metadata_captured", trace_id, actor,
                            {"format": descriptor.schema.get("format")})
        self._to(State.PUBLISHING)
        self.telemetry.emit("publishing_entered", trace_id, actor)
        self.telemetry.emit("provenance_linked", trace_id, actor, provenance={
            "source_dataset_id": descriptor.provenance.get("sourceDataset", descriptor_id),
            "upstream_trace_id": upstream_trace_id, "descriptor_id": descriptor.id,
            "actor": actor, "policy_context": [p.name for p in policies],
            "input_refs": action.requires})
        self.telemetry.emit("telemetry_emitted", trace_id, actor,
                            {"log_level": descriptor.telemetry_schema.get("logLevel")})
        self.telemetry.emit("completion_logged", trace_id, actor, {"status": "COMPLETE"})
        self._to(State.IDLE)
        return {"success": True, "trace_id": trace_id, "reason": None}

    def _do_work(self, dataset_state: Dict[str, Any]) -> str:
        return f"{self.stage}_ok"
