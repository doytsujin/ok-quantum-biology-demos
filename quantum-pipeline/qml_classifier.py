"""A trained quantum model: a variational quantum classifier on PennyLane.

This is the *AI* stage of the pipeline — genuinely machine learning, not
optimization. A variational quantum circuit is trained by gradient descent (the
gradients flow through the circuit exactly as they would through a neural
network) to label a DNA region by nucleotide composition: GC-rich vs AT-rich.

It is a small, honest task — GC content is a real and simple genomic property,
and the point is that a *trained quantum model* is integrated into the pipeline
and governed like every other stage, not that it beats a linear classifier.

Downstream of assembly: stage 1 (QAOA) reconstructs a contig; this stage
interprets it. The pipeline does not just rebuild the sequence, it labels it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

N_QUBITS = 4
N_LAYERS = 3
BASES = "ATGC"


def features(seq: str) -> np.ndarray:
    """Nucleotide-frequency feature vector [fA, fT, fG, fC] scaled to angles."""
    n = max(len(seq), 1)
    f = np.array([seq.count(b) for b in BASES], dtype=float) / n
    return f * np.pi                      # angle encoding range


def gc_rich(seq: str) -> int:
    """Label: 1 if GC fraction > 0.5 (GC-rich), else 0 (AT-rich)."""
    n = max(len(seq), 1)
    return int((seq.count("G") + seq.count("C")) / n > 0.5)


def _make_dataset(n: int, length: int, seed: int, margin: float = 0.0625):
    """Balanced GC-rich / AT-rich samples with a margin around 0.5 (drop the
    ambiguous borderline so the label is well defined)."""
    rng = np.random.default_rng(seed)
    lo, hi = 0.5 - margin, 0.5 + margin
    pos, neg = [], []                       # GC-rich, AT-rich
    while len(pos) < n // 2 or len(neg) < n // 2:
        s = "".join(rng.choice(list(BASES), size=length))
        gc = (s.count("G") + s.count("C")) / length
        if gc >= hi and len(pos) < n // 2:
            pos.append(s)
        elif gc <= lo and len(neg) < n // 2:
            neg.append(s)
    seqs = pos + neg
    X = np.array([features(s) for s in seqs])
    y = np.array([gc_rich(s) for s in seqs])
    return X, y, seqs


_dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(_dev)
def _circuit(weights, x):
    qml.AngleEmbedding(x, wires=range(N_QUBITS))
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))      # in [-1, 1]


def _predict_raw(weights, x):
    return _circuit(weights, x)


@dataclass
class ClassifierResult:
    train_acc: float
    test_acc: float
    steps: int
    n_qubits: int
    backend: str


@dataclass
class TrainedClassifier:
    weights: object
    bias: float
    backend: str
    meta: ClassifierResult

    def classify(self, seq: str) -> Tuple[int, float]:
        score = float(_predict_raw(self.weights, features(seq))) + float(self.bias)
        return (1 if score > 0 else 0), score   # label, calibrated score


def train(backend: str = "default.qubit", steps: int = 60,
          n_train: int = 60, n_test: int = 40, length: int = 8,
          seed: int = 3) -> TrainedClassifier:
    Xtr, ytr, _ = _make_dataset(n_train, length, seed)
    Xte, yte, _ = _make_dataset(n_test, length, seed + 1)
    ttr = 2 * ytr - 1                       # targets in {-1, +1}

    shape = qml.BasicEntanglerLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
    rng = np.random.default_rng(seed)
    weights = pnp.array(0.1 * rng.standard_normal(shape), requires_grad=True)
    bias = pnp.array(0.0, requires_grad=True)

    def cost(w, b):
        preds = pnp.stack([_circuit(w, x) for x in Xtr]) + b
        return pnp.mean((preds - ttr) ** 2)

    opt = qml.AdamOptimizer(stepsize=0.15)
    for _ in range(steps):
        weights, bias = opt.step(cost, weights, bias)

    def acc(X, y):
        preds = np.array([1 if float(_circuit(weights, x)) + float(bias) > 0 else 0
                          for x in X])
        return float(np.mean(preds == y))

    meta = ClassifierResult(train_acc=acc(Xtr, ytr), test_acc=acc(Xte, yte),
                            steps=steps, n_qubits=N_QUBITS, backend=backend)
    return TrainedClassifier(weights=weights, bias=bias, backend=backend, meta=meta)


if __name__ == "__main__":
    clf = train()
    print(f"trained VQC ({clf.meta.n_qubits} qubits, {clf.meta.steps} steps)")
    print(f"  train acc: {clf.meta.train_acc:.2f}   test acc: {clf.meta.test_acc:.2f}")
    for s in ["ATGCA", "GGCGC", "ATATA"]:
        label, score = clf.classify(s)
        print(f"  {s}: {'GC-rich' if label else 'AT-rich'}  (score {score:+.3f}, "
              f"true {'GC-rich' if gc_rich(s) else 'AT-rich'})")
