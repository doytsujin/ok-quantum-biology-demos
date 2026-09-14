"""De novo DNA sequence assembly on PennyLane, following QuASeR.

QuASeR (Sarkar, Al-Ars, Bertels, "QuASeR: Quantum Accelerated de novo DNA
Sequence Reconstruction", PLoS ONE 16(4):e0249850, 2021; arXiv:2004.05078)
models reference-free assembly as four steps:

    reads -> TSP (overlap-layout-consensus) -> QUBO -> cost Hamiltonian -> QAOA

This is a faithful, small implementation of that chain on real PennyLane
(`qml.qaoa`). It is a demonstration on a tiny instance, verified
against the brute-force optimum — not a claim of quantum advantage (QuASeR is
explicit about QAOA's gate-fidelity limits at depth).

Pipeline here:
  - reads tile a short target sequence with pairwise suffix/prefix overlaps;
  - assembly = find the read ordering of maximum total overlap (shortest
    superstring) -> a TSP-style assignment QUBO with x[i,p] = read i at slot p;
  - QUBO -> Ising cost Hamiltonian -> QAOA (PennyLane) -> sampled distribution;
  - decode the most probable *feasible* ordering -> reconstructed sequence.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np

# A tiny instance: target GCGCA, three length-3 reads that tile it.
TARGET = "GCGCA"
READS = ["GCG", "CGC", "GCA"]


def overlap(a: str, b: str) -> int:
    """Length of the longest suffix of a that is a prefix of b."""
    for k in range(min(len(a), len(b)), 0, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


def overlap_matrix(reads: List[str]) -> np.ndarray:
    n = len(reads)
    O = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                O[i, j] = overlap(reads[i], reads[j])
    return O


# --------------------------------------------------------------------------- #
# QUBO: x[i,p] = 1 if read i occupies slot p.  qubit index = i*n + p.
#   minimize  A * (constraint penalties)  -  B * (sum of consecutive overlaps)
# --------------------------------------------------------------------------- #
def build_qubo(reads: List[str], A: float = 8.0, B: float = 1.0):
    n = len(reads)
    O = overlap_matrix(reads)
    nq = n * n

    def idx(i, p):
        return i * n + p

    lin = np.zeros(nq)               # diagonal (linear) coefficients
    quad: Dict[Tuple[int, int], float] = {}

    def add_quad(u, v, w):
        if u == v:
            lin[u] += w
        else:
            a, b = (u, v) if u < v else (v, u)
            quad[(a, b)] = quad.get((a, b), 0.0) + w

    # Constraint 1: each slot p holds exactly one read:  (sum_i x[i,p] - 1)^2
    for p in range(n):
        vs = [idx(i, p) for i in range(n)]
        for u in vs:
            add_quad(u, u, -A)                 # -2A x + A x^2 = -A x (x^2=x)
        for u, v in itertools.combinations(vs, 2):
            add_quad(u, v, 2 * A)
    # Constraint 2: each read i in exactly one slot:    (sum_p x[i,p] - 1)^2
    for i in range(n):
        vs = [idx(i, p) for p in range(n)]
        for u in vs:
            add_quad(u, u, -A)
        for u, v in itertools.combinations(vs, 2):
            add_quad(u, v, 2 * A)
    # Objective: reward overlap between reads in consecutive slots.
    for p in range(n - 1):
        for i in range(n):
            for j in range(n):
                if i != j and O[i, j] > 0:
                    add_quad(idx(i, p), idx(j, p + 1), -B * O[i, j])

    return lin, quad, nq


def qubo_to_cost_hamiltonian(lin: np.ndarray, quad: Dict[Tuple[int, int], float]):
    """x_i = (1 - Z_i)/2 -> Ising cost Hamiltonian (offset dropped for QAOA)."""
    nq = len(lin)
    h = np.zeros(nq)
    J: Dict[Tuple[int, int], float] = {}
    for i in range(nq):
        h[i] += -lin[i] / 2.0
    for (i, j), b in quad.items():
        J[(i, j)] = J.get((i, j), 0.0) + b / 4.0
        h[i] += -b / 4.0
        h[j] += -b / 4.0
    coeffs, ops = [], []
    for i in range(nq):
        if abs(h[i]) > 1e-12:
            coeffs.append(h[i]); ops.append(qml.PauliZ(i))
    for (i, j), w in J.items():
        if abs(w) > 1e-12:
            coeffs.append(w); ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return qml.Hamiltonian(coeffs, ops)


# --------------------------------------------------------------------------- #
# QAOA on PennyLane.
# --------------------------------------------------------------------------- #
@dataclass
class AssemblyResult:
    reconstructed: str
    ordering: List[int]
    matches_target: bool
    matches_bruteforce: bool
    feasible_prob: float
    backend: str
    n_qubits: int
    layers: int
    steps: int


def _decode(bitstring: Tuple[int, ...], n: int):
    """Return ordering [read at slot 0, ...] or None if infeasible."""
    slot_read = {}
    for i in range(n):
        for p in range(n):
            if bitstring[i * n + p] == 1:
                if p in slot_read:
                    return None
                slot_read[p] = i
    if len(slot_read) != n or len(set(slot_read.values())) != n:
        return None
    return [slot_read[p] for p in range(n)]


def assemble(reads: List[str], ordering: List[int]) -> str:
    s = reads[ordering[0]]
    for k in ordering[1:]:
        s += reads[k][overlap(s, reads[k]):]
    return s


def brute_force(reads: List[str]) -> Tuple[List[int], str]:
    n = len(reads)
    best, best_len = None, 10 ** 9
    for perm in itertools.permutations(range(n)):
        sup = assemble(reads, list(perm))
        if len(sup) < best_len:
            best, best_len = list(perm), len(sup)
    return best, assemble(reads, best)


def run_assembly(reads: List[str] = READS, backend: str = "default.qubit",
                 layers: int = 3, steps: int = 70) -> AssemblyResult:
    n = len(reads)
    lin, quad, nq = build_qubo(reads)
    cost_h = qubo_to_cost_hamiltonian(lin, quad)
    mixer_h = qml.qaoa.x_mixer(range(nq))

    def qaoa_layer(gamma, beta):
        qml.qaoa.cost_layer(gamma, cost_h)
        qml.qaoa.mixer_layer(beta, mixer_h)

    dev = qml.device(backend, wires=nq)

    @qml.qnode(dev)
    def expval(params):
        for w in range(nq):
            qml.Hadamard(w)
        qml.layer(qaoa_layer, layers, params[0], params[1])
        return qml.expval(cost_h)

    @qml.qnode(dev)
    def probs(params):
        for w in range(nq):
            qml.Hadamard(w)
        qml.layer(qaoa_layer, layers, params[0], params[1])
        return qml.probs(wires=range(nq))

    rng = np.random.default_rng(7)
    params = pnp.array(0.1 * rng.standard_normal((2, layers)), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.1)
    for _ in range(steps):
        params = opt.step(expval, params)

    p = probs(params)
    # decode the most probable *feasible* ordering
    order = np.argsort(p)[::-1]
    best_ordering, feasible_prob = None, 0.0
    for state in order:
        bits = tuple(int(b) for b in format(int(state), f"0{nq}b"))
        dec = _decode(bits, n)
        if dec is not None:
            best_ordering = dec
            feasible_prob = float(p[state])
            break

    bf_order, bf_seq = brute_force(reads)
    recon = assemble(reads, best_ordering) if best_ordering else ""
    return AssemblyResult(
        reconstructed=recon,
        ordering=best_ordering or [],
        matches_target=(recon == TARGET),
        matches_bruteforce=(len(recon) == len(bf_seq) and recon != ""),
        feasible_prob=feasible_prob,
        backend=backend, n_qubits=nq, layers=layers, steps=steps)


if __name__ == "__main__":
    bf_order, bf_seq = brute_force(READS)
    print(f"reads        : {READS}   target: {TARGET}")
    print(f"brute force  : order={bf_order} -> {bf_seq}")
    r = run_assembly()
    print(f"QAOA ({r.n_qubits} qubits, p={r.layers}) on {r.backend}")
    print(f"  ordering   : {r.ordering} -> {r.reconstructed}")
    print(f"  == target  : {r.matches_target}   == brute-force length: {r.matches_bruteforce}")
    print(f"  feasible top-prob: {r.feasible_prob:.3f}")
