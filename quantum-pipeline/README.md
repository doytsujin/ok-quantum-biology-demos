# Quantum pipeline: two governed stages on PennyLane

Two things the same PennyLane stack lets you do to a circuit — *solve* one and *train* one — chained into one small pipeline and placed behind two governance layers that together act as a control plane for a hybrid quantum-classical workflow.

```
sequencing reads
    -> [ de novo assembly via QAOA         ]   quantum optimization (nothing trained)
    -> [ GC classification via trained VQC ]   quantum machine learning
```

**Stage 1, assembly (solve).** Follows QuASeR (Sarkar, Al-Ars and Bertels, *PLoS ONE* 16(4):e0249850, 2021): reads → traveling-salesman formulation → QUBO → cost Hamiltonian → QAOA. It runs on a deliberately small instance — three reads tiling `GCGCA`, nine qubits — and the result is checked against the brute-force optimum. Nothing in this stage is trained.

**Stage 2, classification (train).** A variational quantum classifier, trained by gradient descent with gradients flowing through the circuit, labels the reconstructed contig GC-rich or AT-rich. The task is small on purpose; the point is that a trained model is integrated and governed on the same terms as the optimization stage, not that it beats a linear classifier.

**The two governance layers.**

- The **descriptor runtime** (`descriptor_runtime.py`) governs a stage and its data: schema, provenance, and resource policies such as qubit and iteration or training-step budgets, evaluated before anything executes. An agent here is an orchestration controller with a five-state lifecycle, not a language model.
- The **gateway** (`quantum_gateway.py`) governs access to the quantum resource: admission against a tenant's allowed backends, token-bucket rate limiting, a cache keyed on circuit identity so an identical circuit is not re-executed, and a structured event record (`QEBREvent` in the code).

Both are compact modules written from scratch; no other repository is required.

## Files

| File | Role |
|---|---|
| `quaser_assembly.py` | Stage 1: the QuASeR chain on PennyLane (`qml.qaoa`, `@qml.qnode`) with a brute-force check |
| `qml_classifier.py` | Stage 2: a variational quantum classifier (`qml.AngleEmbedding`, `qml.BasicEntanglerLayers`) trained to label GC-rich vs AT-rich |
| `descriptor_runtime.py` | `Descriptor`, a non-Turing-complete `PolicyEngine`, a five-state `Agent`, `TelemetryLogger` |
| `quantum_gateway.py` | Admission control, rate limiting, circuit-identity cache, event record |
| `descriptors/de-novo-assembly.json` | The assembly stage's descriptor (qubit and iteration budgets) |
| `descriptors/gc-classification.json` | The classifier stage's descriptor (qubit and training-step budgets) |
| `governed_stage.py` | Wires both stages together and runs the scenarios below; exits non-zero if any check fails |

## Scenarios

1. **Approved assembly** on `default.qubit`: the descriptor budgets pass, the gateway admits the job, QAOA runs, and the contract confirms the contig `GCGCA`.
2. **Classifier stage**: the trained VQC labels that contig GC-rich under its own descriptor, through the same gateway; the contract checks the label against ground truth and an accuracy floor.
3. **Repeat** of the same assembly circuit: served from the gateway cache, with no re-execution.
4. **Unapproved backend** (`ibm_hardware`, a name only): the gateway refuses admission before the job reaches any backend; the stage faults and the refusal is logged.

## Run

```bash
pip install -r requirements.txt
python3 governed_stage.py                 # both stages
python3 governed_stage.py --stage1-only   # assembly and governance only, no classifier
python3 quaser_assembly.py                # just the QAOA assembly
python3 qml_classifier.py                 # just the classifier
```

`governed_stage.py` writes the descriptor telemetry to `output/telemetry.jsonl`.

## Limits

The cache reuses a result only for an identical circuit identity. It does not decide whether a result computed at one set of parameters may stand in for another; that needs an error bound this code does not compute. Everything runs on a simulator, and every instance is small enough to verify exhaustively, which is also why none of it says anything about quantum advantage.
