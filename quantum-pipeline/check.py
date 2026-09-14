"""
check.py — assert the quantum pipeline's numbers that the README states.

governed_stage.py already fails on its own contracts (contig, label, accuracy
floor, cache hit, refused backend). This adds the figures quoted in prose: the
brute-force ordering, the share of the distribution on the chosen ordering, and
the classifier's accuracies. Everything is seeded, so the values repeat.
Exits non-zero on any mismatch.
"""

from __future__ import annotations

import sys

import quaser_assembly as qa
from qml_classifier import gc_rich, train


def expect(label: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def main() -> int:
    results = []

    bf_order, bf_seq = qa.brute_force(qa.READS)
    r = qa.run_assembly()
    results.append(expect("assembly matches brute force",
                          r.ordering == bf_order == [0, 1, 2] and r.reconstructed == bf_seq == "GCGCA",
                          f"QAOA {r.ordering} -> {r.reconstructed}; brute force {bf_order} -> {bf_seq}"))
    results.append(expect("nine qubits, three layers", r.n_qubits == 9 and r.layers == 3,
                          f"qubits={r.n_qubits}, layers={r.layers}"))
    results.append(expect("most probable valid ordering carries about 8%",
                          0.07 <= r.feasible_prob <= 0.10, f"p={r.feasible_prob:.3f}"))

    clf = train()
    results.append(expect("classifier accuracies",
                          round(clf.meta.train_acc, 2) == 0.83 and round(clf.meta.test_acc, 2) == 0.90,
                          f"train={clf.meta.train_acc:.2f}, test={clf.meta.test_acc:.2f}"))
    label, score = clf.classify("GCGCA")
    results.append(expect("GCGCA labeled GC-rich", label == 1 and gc_rich("GCGCA") == 1,
                          f"score={score:+.3f}"))

    ok = all(results)
    print(f"  all checks passed: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
