# Quantum-biology demos: a governed quantum pipeline and a bioprocess yardstick

Two small, runnable demonstrations of what it takes to put a computational step inside a governed biology workflow. The first runs a published quantum genomics method and a trained quantum classifier on PennyLane, behind a control plane that decides what may run and records what happened. The second is not quantum at all: it is a mechanistic model of a fed-batch CHO bioreactor under a contract of in-process limits, and it is here as the yardstick — the kind of decision record a regulated bioprocess step already has to leave behind.

Neither demo claims quantum advantage. Every instance is small enough to check classically, and each one is checked.

## What is here

| Folder | What it runs | Offline |
|---|---|---|
| [`quantum-pipeline/`](quantum-pipeline/) | QuASeR's *de novo* assembly as QAOA (three reads, nine qubits, verified against brute force), a variational quantum classifier trained downstream of it, a dataset-descriptor runtime and a gateway (admission control, rate limiting, a circuit cache, an event record) | yes |
| [`bioprocess-twin/`](bioprocess-twin/) | A mechanistic fed-batch CHO bioreactor model, a contract of in-process limits, a search controller that picks a feed plan from 256 candidates, and the decision artifact it writes | yes; an optional LLM-driven path needs an API key |

## Run

```bash
make install        # pip install the pinned requirements of both folders
make check          # run both demos and assert their verified outputs (about a minute)
```

Or one at a time: `make quantum`, `make quantum-stage1`, `make assembly`, `make classifier`, `make twin`. Each folder's README describes its pieces.

Tested with Python 3.12, PennyLane 0.45.0, NumPy 2.2.6 and SciPy 1.14.0 on Linux. The simulations are seeded, so the numbers below repeat.

## Verified outputs

`make check` fails unless all of these hold.

**Quantum pipeline**

- QAOA assembly of the reads `GCG`, `CGC`, `GCA` returns the ordering `[0, 1, 2]` and the contig `GCGCA`, which is the brute-force optimum. The most probable *valid* ordering carries about 8% of the output distribution; invalid bitstrings take much of the rest.
- The classifier (four qubits, 60 training steps) reaches 0.83 training and 0.90 test accuracy and labels `GCGCA` GC-rich. It does not beat a one-line GC count, and nothing here says it does.
- A repeated assembly request is served from the gateway's cache; a request for a backend the tenant is not allowed to use is refused before anything executes.

**Bioprocess yardstick**

- With no feed, glucose runs out but the run stays inside the contract, at a final titer of 1607.0 mg/L.
- A naive daily over-feed breaks the contract: lactate reaches 165.6 mM against a 60 mM limit, ammonia 18.01 mM against 8 mM.
- From the day-6 state, the search controller evaluates 256 feed plans and chooses a single 20 mL bolus on day 12: final titer 1642.2 mg/L, lactate 57.5 mM, ammonia 7.27 mM, inside the contract. The decision artifact records the observed state, the prediction, the contract check, the rationale and five alternatives considered.

## What these demos are not

- **Not a quantum-advantage result.** QuASeR is explicit that QAOA's accuracy is limited at depth, and a nine-qubit instance is solved exactly by brute force in the same run.
- **Not hardware.** Everything runs on PennyLane's `default.qubit` simulator. The refused backend in the gateway scenario is a name, not a device.
- **Not a certified cache.** The gateway reuses a result only for an identical circuit identity. Reusing a result at a different parameter point needs an error bound, which this code does not compute.
- **Not a calibrated process model.** The bioreactor parameters are order-of-magnitude realistic, not fit to a clone, and nothing updates the model from a physical bioreactor. The contract's lactate and ammonia limits are in-process limits; the quality attribute they protect (glycosylation) is not simulated.
- **Not a signed record.** The decision artifact is attributed — it names the controller and the model version — but nothing signs it.

## Context

These demos were discussed at BOF-2099, *Is Quantum Ready for Biology?*, a Birds-of-a-Feather session at IEEE Quantum Week 2026 in Toronto. This repository is not an IEEE publication and is not endorsed by IEEE, the IEEE Computer Society or the QCE26 committee.

This is independent research and technical work, unaffiliated with the author's employer. Nothing here represents the employer's views.

## References

- A. Sarkar, Z. Al-Ars, and K. Bertels, "QuASeR: Quantum accelerated de novo DNA sequence reconstruction," *PLoS ONE*, vol. 16, no. 4, e0249850, 2021, doi:10.1371/journal.pone.0249850.
- V. Bergholm et al., "PennyLane: Automatic differentiation of hybrid quantum-classical computations," arXiv:1811.04968.
- J. Monod, "The growth of bacterial cultures," *Annu. Rev. Microbiol.*, vol. 3, pp. 371–394, 1949, doi:10.1146/annurev.mi.03.100149.002103.
- ICH, "Development and manufacture of drug substances Q11," 2012 — definition of a critical quality attribute.
- M. Yang and M. Butler, "Effects of ammonia on CHO cell growth, erythropoietin production, and glycosylation," *Biotechnol. Bioeng.*, vol. 68, no. 4, pp. 370–380, 2000.

## Citation and license

Version 1.0.0 is archived on Zenodo: [doi:10.5281/zenodo.22741061](https://doi.org/10.5281/zenodo.22741061). Citation metadata is in [`CITATION.cff`](CITATION.cff). Code is released under the [MIT License](LICENSE).
