"""
check.py — re-run the bioprocess scenarios and assert the numbers the README states.

Runs offline with the deterministic search controller, whatever the environment
holds: the LLM-driven path is not deterministic and is not what the README quotes.
Exits non-zero on the first mismatch.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.pop("ANTHROPIC_API_KEY", None)

from agent import recommend  # noqa: E402
from bioreactor_twin import FeedBolus, simulate  # noqa: E402
from contracts import evaluate_contract  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def expect(label: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def main() -> int:
    results = []

    nofeed = evaluate_contract(simulate(duration_days=14.0))
    results.append(expect("no feed stays inside the contract",
                          nofeed.feasible and nofeed.kpis["final_titer_mg_L"] == 1607.0,
                          f"feasible={nofeed.feasible}, titer={nofeed.kpis['final_titer_mg_L']}"))

    naive = evaluate_contract(simulate([FeedBolus(day=d, volume_L=0.030) for d in range(3, 11)],
                                       duration_days=14.0))
    results.append(expect("naive daily over-feed breaks the contract",
                          (not naive.feasible) and naive.kpis["max_lactate_mM"] == 165.6
                          and naive.kpis["max_ammonia_mM"] == 18.01,
                          f"lactate={naive.kpis['max_lactate_mM']}, ammonia={naive.kpis['max_ammonia_mM']}"))

    state = simulate(duration_days=6.0, n_points=120).state_at(6.0)
    art = recommend(state, start_day=6.0, duration=14.0)
    feeds = [(f["day"], round(f["volume_L"] * 1000)) for f in art.recommended_feeds]
    results.append(expect("chosen plan", feeds == [(12.0, 20)], f"feeds={feeds}"))
    k = art.predicted_kpis
    results.append(expect("chosen plan inside the contract",
                          art.contract_satisfied and k["final_titer_mg_L"] == 1642.2
                          and k["max_lactate_mM"] == 57.5 and k["max_ammonia_mM"] == 7.27,
                          f"titer={k['final_titer_mg_L']}, lactate={k['max_lactate_mM']}, "
                          f"ammonia={k['max_ammonia_mM']}"))
    results.append(expect("256 plans searched, 5 alternatives recorded",
                          "256 feed strategies" in art.rationale and len(art.alternatives_considered) == 5,
                          f"alternatives={len(art.alternatives_considered)}"))

    with open(os.path.join(HERE, "provenance", "decision_artifact.json")) as fh:
        committed = json.load(fh)
    results.append(expect("committed artifact matches a fresh run",
                          committed["predicted_kpis"] == k
                          and committed["recommended_feeds"] == art.recommended_feeds,
                          "provenance/decision_artifact.json"))

    ok = all(results)
    print(f"  all checks passed: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
