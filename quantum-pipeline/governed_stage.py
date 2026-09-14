"""Two governed quantum stages — one that *solves* a circuit, one that *trains*
one — behind a single minimal agentic control plane. This is the pipeline in
miniature: where AI meets differentiable quantum programming, governed alike.

    Sequencing reads
        -> [ De novo assembly via QAOA      ]   quantum OPTIMIZATION (not trained)
        -> [ GC classification via trained VQC ] quantum MACHINE LEARNING (the AI)
        -> ...

Stage 1 is a real genomics step: following QuASeR (Sarkar, Al-Ars, Bertels,
PLoS ONE 2021), reference-free assembly is cast as TSP -> QUBO -> QAOA and solved
on PennyLane; see quaser_assembly.py. Nothing in it is trained.

Stage 2 is genuine machine learning: a variational quantum classifier trained by
gradient descent (gradients flow through the circuit as through a neural net)
that labels the reconstructed contig GC-rich vs AT-rich; see qml_classifier.py.
The same PennyLane stack that *solves* a circuit upstream *trains* one here.

Two governance layers sit around both stages, and together they form a control
plane for a hybrid quantum-classical workflow:

  - the DESCRIPTOR runtime (descriptor_runtime.py) governs each pipeline stage and
    its data: schema, provenance, and resource policies (qubit / iteration /
    training budgets), checked before execution;
  - the GATEWAY (quantum_gateway.py) governs access to the quantum resource:
    admission control against a tenant policy (which backends are allowed),
    rate limiting, a cache keyed on circuit identity, and a QEBR event record.

Runs:
  1. approved backend (default.qubit)   -> assembly admitted, executes, contract PASS
  2. downstream AI stage                -> trained VQC classifies the contig, contract PASS
  3. same assembly circuit again        -> served from the gateway cache
  4. unapproved backend (ibm_hardware)  -> gateway denies admission; stage faults

    python3 governed_stage.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import quaser_assembly as qa
import qml_classifier as qc
from descriptor_runtime import (Agent, Descriptor, DescriptorRegistry,
                                PolicyEngine, TelemetryLogger)
from quantum_gateway import CircuitRequest, QuantumGateway, TenantPolicy

HERE = Path(__file__).resolve().parent
CIRCUIT_ID = f"assembly:{'-'.join(qa.READS)}"


class QuantumAgent(Agent):
    """Agent whose stage work dispatches the QAOA assembly via the gateway."""

    last_result: qa.AssemblyResult | None = None
    last_from_cache: bool = False

    def _do_work(self, dataset_state: Dict[str, Any]) -> str:
        req = CircuitRequest(tenant_id=dataset_state["tenant_id"],
                             circuit_id=CIRCUIT_ID,
                             target_backend=dataset_state["backend"])
        sub = self.gateway.submit(req, lambda: qa.run_assembly(
            backend=dataset_state["backend"],
            steps=int(dataset_state["max_iterations"])))
        r = sub.value
        self.last_result, self.last_from_cache = r, sub.from_cache
        self.telemetry.emit(
            "quantum_result", f"{dataset_state['sample_id']}:{self.stage}", self.name,
            payload={"reconstructed": r.reconstructed, "from_cache": sub.from_cache,
                     "backend": sub.backend},
            provenance={"method": "QuASeR TSP->QUBO->QAOA", "reads": qa.READS,
                        "target": qa.TARGET, "n_qubits": r.n_qubits,
                        "qaoa_layers": r.layers})
        return f"{self.stage}_ok contig={r.reconstructed}"


def _check_contract(r: qa.AssemblyResult) -> list[dict]:
    return [
        {"check": "reconstruction matches target", "passed": r.matches_target,
         "detail": f"{r.reconstructed} vs target {qa.TARGET}"},
        {"check": "matches brute-force optimum length", "passed": r.matches_bruteforce,
         "detail": f"len {len(r.reconstructed)}"},
    ]


def _make_agent(registry, engine, telemetry, gateway) -> QuantumAgent:
    a = QuantumAgent(name="assembly-agent", stage="de_novo_assembly",
                     registry=registry, policy_engine=engine, telemetry=telemetry)
    a.gateway = gateway
    return a


def _run(agent, backend, sample_id):
    state = {"sample_id": sample_id, "tenant_id": "genomics-lab",
             "backend": backend, "n_qubits": 9, "max_iterations": 70}
    return agent.run_task("de-novo-assembly-v1.0", "assemble_reads", state,
                          sample_id, upstream_trace_id=f"{sample_id}:sequencing")


class ClassifierAgent(Agent):
    """The AI stage: trains a variational quantum classifier and labels the
    upstream contig GC-rich vs AT-rich, routed through the same gateway."""

    last_result: dict | None = None
    last_from_cache: bool = False

    def _do_work(self, dataset_state: Dict[str, Any]) -> str:
        contig = dataset_state["contig"]
        req = CircuitRequest(tenant_id=dataset_state["tenant_id"],
                             circuit_id=f"classify-vqc:{contig}",
                             target_backend=dataset_state["backend"])

        def run() -> dict:
            clf = qc.train(backend=dataset_state["backend"],
                           steps=int(dataset_state["max_train_steps"]))
            label, score = clf.classify(contig)
            return {"label": int(label), "score": float(score), "meta": clf.meta}

        sub = self.gateway.submit(req, run)
        r = sub.value
        self.last_result, self.last_from_cache = r, sub.from_cache
        m = r["meta"]
        self.telemetry.emit(
            "classification_result", f"{dataset_state['sample_id']}:{self.stage}", self.name,
            payload={"contig": contig, "label": r["label"], "gc_rich": bool(r["label"]),
                     "score": r["score"], "from_cache": sub.from_cache,
                     "backend": sub.backend, "train_acc": m.train_acc, "test_acc": m.test_acc},
            provenance={"method": "trained VQC (AngleEmbedding + BasicEntanglerLayers)",
                        "task": "GC-rich vs AT-rich", "feature": "nucleotide frequency",
                        "n_qubits": m.n_qubits, "train_steps": m.steps})
        return f"{self.stage}_ok label={'GC' if r['label'] else 'AT'}-rich"


def _check_classifier_contract(contig: str, r: dict) -> list[dict]:
    truth = qc.gc_rich(contig)
    return [
        {"check": "predicted label matches GC truth", "passed": r["label"] == truth,
         "detail": f"pred {'GC' if r['label'] else 'AT'}-rich vs true "
                   f"{'GC' if truth else 'AT'}-rich (score {r['score']:+.3f})"},
        {"check": "trained model meets accuracy floor", "passed": r["meta"].test_acc >= 0.8,
         "detail": f"test_acc {r['meta'].test_acc:.2f}"},
    ]


def _make_classifier(registry, engine, telemetry, gateway) -> ClassifierAgent:
    a = ClassifierAgent(name="classifier-agent", stage="gc_classification",
                        registry=registry, policy_engine=engine, telemetry=telemetry)
    a.gateway = gateway
    return a


def _run_classifier(agent, backend, sample_id, contig, upstream_trace_id):
    state = {"sample_id": sample_id, "tenant_id": "genomics-lab", "backend": backend,
             "n_qubits": qc.N_QUBITS, "max_train_steps": 60, "contig": contig}
    return agent.run_task("gc-classification-v1.0", "classify_contig", state,
                          sample_id, upstream_trace_id=upstream_trace_id)


def main(stage1_only: bool = False) -> int:
    import pennylane as qml
    print("=" * 74)
    if stage1_only:
        print("  Governed quantum stage behind an agentic control plane  [Part 1]")
        print("  QAOA assembly (solve) only — quantum optimization, nothing trained")
    else:
        print("  Two governed quantum stages behind one agentic control plane  [Part 2]")
        print("  QAOA assembly (solve) + trained VQC classifier (train); same governance")
    print(f"  reads={qa.READS}  target={qa.TARGET}  (PennyLane {qml.version()})")
    print("=" * 74)

    registry = DescriptorRegistry()
    for name in ("de-novo-assembly.json", "gc-classification.json"):
        registry.register(Descriptor.from_dict(
            json.loads((HERE / "descriptors" / name).read_text())))
    engine, telemetry = PolicyEngine(), TelemetryLogger()

    gateway = QuantumGateway()
    gateway.register_tenant(TenantPolicy(
        tenant_id="genomics-lab",
        allowed_backends=["default.qubit", "lightning.qubit"], rate_limit_rps=5))

    # --- Run 1: approved backend -> admitted, executes -------------------
    print("\n[1] Approved run  (tenant=genomics-lab, backend=default.qubit)")
    a1 = _make_agent(registry, engine, telemetry, gateway)
    _run(a1, "default.qubit", "READS-001")
    r = a1.last_result
    print(f"    lifecycle : {' -> '.join(s.split('->')[1] for s in a1.state_transitions)}")
    print(f"    gateway   : admitted, executed (from_cache={a1.last_from_cache})")
    print(f"    assembly  : {r.ordering} -> contig '{r.reconstructed}'  "
          f"({r.n_qubits} qubits, p={r.layers})")
    contract = _check_contract(r)
    for c in contract:
        print(f"    contract  : [{'PASS' if c['passed'] else 'FAIL'}] {c['check']} — {c['detail']}")

    # --- Run 2: downstream AI stage -> trained VQC classifies the contig -
    # This is the Part 2 stage: the AI added on top of the Part 1 pipeline.
    clf_contract: list[dict] = []
    if stage1_only:
        print("\n[2] AI stage      (skipped — Part 1 is the pipeline without AI; see Part 2)")
    else:
        print("\n[2] AI stage      (trained VQC labels the contig from [1] — genuine ML)")
        ca = _make_classifier(registry, engine, telemetry, gateway)
        _run_classifier(ca, "default.qubit", "READS-001", r.reconstructed,
                        upstream_trace_id="READS-001:de_novo_assembly")
        cr = ca.last_result
        print(f"    lifecycle : {' -> '.join(s.split('->')[1] for s in ca.state_transitions)}")
        print(f"    training  : trained VQC ({cr['meta'].n_qubits} qubits, {cr['meta'].steps} steps) "
              f"train_acc={cr['meta'].train_acc:.2f} test_acc={cr['meta'].test_acc:.2f}")
        print(f"    inference : contig '{r.reconstructed}' -> "
              f"{'GC' if cr['label'] else 'AT'}-rich  (score {cr['score']:+.3f})")
        clf_contract = _check_classifier_contract(r.reconstructed, cr)
        for c in clf_contract:
            print(f"    contract  : [{'PASS' if c['passed'] else 'FAIL'}] {c['check']} — {c['detail']}")

    # --- Run 3: identical circuit -> gateway cache ----------------------
    print("\n[3] Repeat run   (same assembly circuit) — expect cache hit")
    a2 = _make_agent(registry, engine, telemetry, gateway)
    _run(a2, "default.qubit", "READS-003")
    print(f"    gateway   : from_cache={a2.last_from_cache}  "
          f"(no re-execution — fewer shots / less backend use)")

    # --- Run 4: unapproved backend -> gateway admission denial -----------
    print("\n[4] Unapproved run  (backend=ibm_hardware) — expect admission denial")
    a3 = _make_agent(registry, engine, telemetry, gateway)
    res3 = _run(a3, "ibm_hardware", "READS-002")
    print(f"    lifecycle : {' -> '.join(s.split('->')[1] for s in a3.state_transitions)}")
    print(f"    outcome   : success={res3['success']}  reason={res3['reason']}")
    print(f"    quantum job executed: {a3.last_result is not None}  "
          f"(denied at the gateway before the backend)")

    # --- Observability ---------------------------------------------------
    print("\n[5] Observability")
    print(f"    descriptor telemetry records : {telemetry.total()}")
    print(f"    gateway QEBR events          : {len(gateway.events)} "
          f"(admitted={gateway.count('admitted')}, executed={gateway.count('executed')}, "
          f"cache_hit={gateway.count('cache_hit')}, "
          f"admission_denied={gateway.count('admission_denied')})")

    out = HERE / "output"; out.mkdir(exist_ok=True)
    nbytes = telemetry.write_jsonl(out / "telemetry.jsonl")
    print(f"\n    wrote output/telemetry.jsonl ({nbytes/1024:.1f} KB)")
    all_pass = (all(c["passed"] for c in contract)
                and all(c["passed"] for c in clf_contract)
                and a2.last_from_cache and not res3["success"])
    print(f"    all checks passed: {all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main(stage1_only="--stage1-only" in sys.argv[1:]))
