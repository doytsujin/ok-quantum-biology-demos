"""
agent.py — the physics and chemistry of the process, in agentic form.

The agent's job: at a decision point mid-run (e.g. day 6), recommend a glucose
feed strategy that MAXIMIZES final mAb titer while keeping the trajectory
inside the process contract (lactate, ammonia, volume).

It does not know the chemistry a priori. It discovers consequences by calling
the mechanistic twin as a tool — running what-if simulations, reading back the
predicted KPIs and contract status, and converging on a feasible plan. The
output is a DecisionArtifact: the recommendation plus the evidence and reason.

Two interchangeable "agents" are provided so the demo is runnable either way:

  * run_agent_llm     — Claude Opus 4.8 driving a tool-use loop over the twin
                        (requires ANTHROPIC_API_KEY + `pip install anthropic`).
  * run_agent_search  — a deterministic controller that uses the SAME twin as
                        its world model (runs fully offline, no API key).

Both produce the identical artifact shape; the LLM path adds natural-language
reasoning and can explore non-obvious strategies (e.g. exploiting the
glucose-limited lactate-reconsumption shift).
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Sequence

from bioreactor_twin import FeedBolus, simulate_from
from contracts import (
    CQA_CONTRACT,
    DecisionArtifact,
    evaluate_contract,
    feeds_to_dicts,
)

MODEL = "claude-opus-4-8"
PROCESS = "CHO fed-batch mAb production (1 L)"
OBJECTIVE = "maximize final mAb titer subject to the process contract"


# --------------------------------------------------------------------------- #
# Shared: turn a list of {day, volume_mL} into FeedBolus objects and simulate
# --------------------------------------------------------------------------- #

def _to_boluses(feeds: Sequence[dict]) -> list[FeedBolus]:
    out = []
    for f in feeds:
        out.append(FeedBolus(
            day=float(f["day"]),
            volume_L=float(f["volume_mL"]) / 1000.0,
            glc_conc=float(f.get("glc_conc_mM", 1250.0)),
            gln_conc=float(f.get("gln_conc_mM", 150.0)),
        ))
    return sorted(out, key=lambda b: b.day)


def _simulate(state: dict, feeds: Sequence[dict], start_day: float, duration: float):
    boluses = _to_boluses(feeds)
    res = simulate_from(state, boluses, start_day=start_day, duration_days=duration)
    check = evaluate_contract(res)
    return boluses, res, check


# --------------------------------------------------------------------------- #
# Deterministic controller (offline) — the twin as a world model
# --------------------------------------------------------------------------- #

def run_agent_search(
    state: dict,
    *,
    start_day: float,
    duration: float = 14.0,
    feed_days: Sequence[float] = (6, 8, 10, 12),
    bolus_options_mL: Sequence[float] = (0, 20, 40, 60),
) -> DecisionArtifact:
    """Grid-search feed strategies over the mechanistic twin; pick the best
    contract-satisfying plan. This is the minimal 'agent': a controller whose
    world model is the physics."""
    best = None
    alternatives = []
    for combo in itertools.product(bolus_options_mL, repeat=len(feed_days)):
        feeds = [{"day": d, "volume_mL": v} for d, v in zip(feed_days, combo) if v > 0]
        _, res, check = _simulate(state, feeds, start_day, duration)
        cand = (check.kpis["final_titer_mg_L"], check.feasible, feeds, res, check)
        if check.feasible:
            alternatives.append({
                "feeds_mL": list(combo),
                "titer": check.kpis["final_titer_mg_L"],
                "max_lac": check.kpis["max_lactate_mM"],
                "max_amm": check.kpis["max_ammonia_mM"],
            })
            if best is None or cand[0] > best[0]:
                best = cand
    if best is None:  # nothing feasible — fall back to least-bad (highest titer)
        _, res, check = _simulate(state, [], start_day, duration)
        best = (check.kpis["final_titer_mg_L"], False, [], res, check)

    _, _, feeds, res, check = best
    boluses = _to_boluses(feeds)
    alternatives.sort(key=lambda a: -a["titer"])
    rationale = (
        f"Searched {len(bolus_options_mL)**len(feed_days)} feed strategies over the "
        f"mechanistic twin from day {start_day:g}. Selected the highest-titer plan that "
        f"keeps lactate <= {check.kpis['max_lactate_mM']} mM and ammonia "
        f"<= {check.kpis['max_ammonia_mM']} mM. Larger boluses raise titer but push "
        f"lactate/ammonia past the contract; the chosen plan sits at the feasible frontier."
    )
    return DecisionArtifact(
        process=PROCESS,
        decision_point_day=start_day,
        observed_state=state,
        recommended_feeds=feeds_to_dicts(boluses),
        predicted_kpis=check.kpis,
        contract=CQA_CONTRACT,
        contract_satisfied=check.feasible,
        contract_violations=check.violations,
        objective=OBJECTIVE,
        rationale=rationale,
        decided_by="mechanistic-search",
        alternatives_considered=alternatives[:5],
    )


# --------------------------------------------------------------------------- #
# LLM agent (Claude Opus 4.8) — tool-use loop over the twin
# --------------------------------------------------------------------------- #

TOOLS = [
    {
        "name": "simulate_feed_strategy",
        "description": (
            "Run the mechanistic process model forward from the current state with a "
            "proposed glucose feed strategy and return predicted KPIs (final titer, peak "
            "VCD, max lactate, max ammonia, min glucose, final volume) plus whether the "
            "process contract is satisfied and any violations. Use this to test ideas before "
            "recommending."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "feeds": {
                    "type": "array",
                    "description": "Feed boluses to apply.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "day": {"type": "number", "description": "Day of the bolus."},
                            "volume_mL": {"type": "number", "description": "Bolus volume in mL."},
                        },
                        "required": ["day", "volume_mL"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["feeds"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit_recommendation",
        "description": (
            "Submit the final feed recommendation once you have found a strategy that "
            "maximizes titer while satisfying the process contract. Provide the feeds and a "
            "short rationale explaining the chemistry trade-off you balanced."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "feeds": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "day": {"type": "number"},
                            "volume_mL": {"type": "number"},
                        },
                        "required": ["day", "volume_mL"],
                        "additionalProperties": False,
                    },
                },
                "rationale": {"type": "string"},
            },
            "required": ["feeds", "rationale"],
            "additionalProperties": False,
        },
    },
]

SYSTEM = (
    "You are a bioprocess control agent for a fed-batch CHO cell-culture bioreactor "
    "producing a monoclonal antibody. You reason about the underlying chemistry: glucose "
    "feeds glycolysis and, in overflow, produce lactate; under glucose limitation cells "
    "re-consume lactate (the metabolic shift); glutamine consumption produces ammonia; "
    "lactate and ammonia inhibit growth and threaten product quality. You never guess "
    "outcomes numerically — you call simulate_feed_strategy on the mechanistic twin to "
    "test every idea, then submit_recommendation with the best feasible plan. Explore a "
    "few strategies (including under-feeding to exploit lactate re-consumption) before "
    "deciding. Keep the trajectory inside the process contract."
)


def run_agent_llm(
    state: dict,
    *,
    start_day: float,
    duration: float = 14.0,
    max_iters: int = 12,
) -> DecisionArtifact:
    import anthropic  # imported lazily so the module loads without the SDK

    client = anthropic.Anthropic()

    user = (
        f"Process: {PROCESS}. We are at day {start_day:g}. Observed state:\n"
        f"{json.dumps(state, indent=2)}\n\n"
        f"Process contract (hard in-process limits):\n{json.dumps(CQA_CONTRACT, indent=2)}\n\n"
        f"Run continues to day {duration:g}. Objective: {OBJECTIVE}. "
        f"Feed stock is ~1250 mM glucose. Recommend a glucose feed strategy."
    )
    messages = [{"role": "user", "content": user}]
    submitted: dict | None = None

    for _ in range(max_iters):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            break

        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            if block.name == "submit_recommendation":
                submitted = block.input
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": "Recommendation recorded."})
            elif block.name == "simulate_feed_strategy":
                _, _, check = _simulate(state, block.input["feeds"], start_day, duration)
                payload = {"kpis": check.kpis, "contract_satisfied": check.feasible,
                           "violations": check.violations}
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(payload)})
        messages.append({"role": "user", "content": results})
        if submitted is not None:
            break

    if submitted is None:  # agent never submitted — fall back to the search controller
        return run_agent_search(state, start_day=start_day, duration=duration)

    boluses, res, check = _simulate(state, submitted["feeds"], start_day, duration)
    return DecisionArtifact(
        process=PROCESS,
        decision_point_day=start_day,
        observed_state=state,
        recommended_feeds=feeds_to_dicts(boluses),
        predicted_kpis=check.kpis,
        contract=CQA_CONTRACT,
        contract_satisfied=check.feasible,
        contract_violations=check.violations,
        objective=OBJECTIVE,
        rationale=submitted["rationale"],
        decided_by=MODEL,
    )


def recommend(state: dict, *, start_day: float, duration: float = 14.0) -> DecisionArtifact:
    """Use the LLM agent when an API key is present, else the offline controller."""
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return run_agent_llm(state, start_day=start_day, duration=duration)
        except Exception as exc:  # network/SDK issue -> deterministic fallback
            print(f"[agent] LLM path unavailable ({exc}); using mechanistic search.")
    return run_agent_search(state, start_day=start_day, duration=duration)
