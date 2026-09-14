"""
run_demo.py — end-to-end: physics/chemistry of the process, in agentic form.

1. Run the fed-batch to a mid-run decision point (day 6) with the mechanistic
   twin — this stands in for live sensor data.
2. Show the two failure modes that make a human decision hard:
   no feed (glucose starves) vs. naive over-feed (lactate/ammonia past CQA).
3. Hand the day-6 state to the agent, which uses the twin as a world model to
   recommend a feasible, titer-maximizing feed plan.
4. Write the decision artifact (provenance) to provenance/decision_artifact.json.

Runs offline by default (deterministic controller). Set ANTHROPIC_API_KEY to
let Claude Opus 4.8 drive the loop instead.
"""

from __future__ import annotations

import os

from bioreactor_twin import FeedBolus, simulate, simulate_from
from contracts import evaluate_contract
from agent import recommend

DECISION_DAY = 6.0
DURATION = 14.0


def main() -> None:
    print("=" * 70)
    print("Physics & chemistry of a CHO fed-batch bioreactor — in agentic form")
    print("=" * 70)

    # 1. Baselines: the two ways a human gets it wrong
    batch = evaluate_contract(simulate(duration_days=DURATION))
    naive_feeds = [FeedBolus(day=d, volume_L=0.030) for d in range(3, 11)]
    naive = evaluate_contract(simulate(naive_feeds, duration_days=DURATION))
    print("\nBaseline — no feed (batch):")
    print(f"  titer {batch.kpis['final_titer_mg_L']} mg/L | "
          f"min glucose {batch.kpis['min_glucose_mM']} mM | "
          f"feasible: {batch.feasible} {batch.violations}")
    print("Baseline — naive daily over-feed:")
    print(f"  titer {naive.kpis['final_titer_mg_L']} mg/L | "
          f"max lactate {naive.kpis['max_lactate_mM']} mM | "
          f"max ammonia {naive.kpis['max_ammonia_mM']} mM | "
          f"feasible: {naive.feasible} {naive.violations}")

    # 2. Advance to the decision point and read the state (a 'sensor reading')
    pre = simulate(duration_days=DECISION_DAY, n_points=120)
    state = pre.state_at(DECISION_DAY)
    print(f"\nDecision point — day {DECISION_DAY:g} observed state:")
    for k, val in state.items():
        print(f"  {k:>18}: {val}")

    # 3. Agent recommends over the twin
    driver = "Claude Opus 4.8" if os.getenv("ANTHROPIC_API_KEY") else "mechanistic search (offline)"
    print(f"\nAgent ({driver}) deciding ...")
    artifact = recommend(state, start_day=DECISION_DAY, duration=DURATION)
    print("\n" + artifact.summary_line())
    print("Rationale:", artifact.rationale)

    # 4. Persist the decision artifact (provenance)
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "provenance", "decision_artifact.json")
    artifact.to_json(out)
    print(f"\nDecision artifact written -> {out}")


if __name__ == "__main__":
    main()
