"""
contracts.py — the process contract and the decision artifact the agent must
produce.

The contract holds in-process limits (lactate, ammonia, vessel volume). They are
not Critical Quality Attributes themselves: a CQA is a property of the product
(ICH Q11). Ammonia accumulation is linked to changes in antibody glycosylation,
which is one. The identifier CQA_CONTRACT is kept for compatibility.

The mechanistic twin says what *will* happen. The process contract says what is
*allowed* to happen. Together they turn "run a simulation" into "make a
governed decision": every recommendation is checked against explicit limits,
and the reasoning is written down as a provenance record rather than left in
someone's head. This is the same descriptor / contract / provenance pattern
used for agentic datasets — applied to a physical process instead of a table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Sequence

from bioreactor_twin import SimResult, FeedBolus


# --------------------------------------------------------------------------- #
# The contract: in-process limits the agent may not violate
# --------------------------------------------------------------------------- #

CQA_CONTRACT = {
    "max_lactate_mM": 60.0,    # acidification and growth inhibition
    "max_ammonia_mM": 8.0,     # growth inhibition; linked to glycosylation changes
    "max_volume_L": 1.6,       # vessel fill / headspace limit
    # (Residual glucose at harvest is intentionally NOT a hard limit — it is
    #  expected to be low. Mid-run starvation shows up as lost titer, not a
    #  contract breach, which is exactly the trade-off the agent navigates.)
}


@dataclass
class ConstraintCheck:
    feasible: bool
    violations: list[str]
    kpis: dict


def evaluate_contract(res: SimResult) -> ConstraintCheck:
    """Check a simulated trajectory against the process contract."""
    s = res.summary()
    v: list[str] = []
    if s["max_lactate_mM"] > CQA_CONTRACT["max_lactate_mM"]:
        v.append(f"lactate {s['max_lactate_mM']} > {CQA_CONTRACT['max_lactate_mM']} mM")
    if s["max_ammonia_mM"] > CQA_CONTRACT["max_ammonia_mM"]:
        v.append(f"ammonia {s['max_ammonia_mM']} > {CQA_CONTRACT['max_ammonia_mM']} mM")
    if s["final_volume_L"] > CQA_CONTRACT["max_volume_L"]:
        v.append(f"volume {s['final_volume_L']} > {CQA_CONTRACT['max_volume_L']} L")
    return ConstraintCheck(feasible=not v, violations=v, kpis=s)


# --------------------------------------------------------------------------- #
# The decision artifact: an auditable record of WHAT was recommended and WHY
# --------------------------------------------------------------------------- #

@dataclass
class DecisionArtifact:
    process: str
    decision_point_day: float
    observed_state: dict
    recommended_feeds: list[dict]
    predicted_kpis: dict
    contract: dict
    contract_satisfied: bool
    contract_violations: list[str]
    objective: str
    rationale: str
    decided_by: str            # "claude-opus-4-8" | "mechanistic-search" | ...
    twin_version: str = "CHOParams/illustrative-v1"
    alternatives_considered: list[dict] = field(default_factory=list)

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    def summary_line(self) -> str:
        feeds = ", ".join(
            f"{f['volume_L']*1000:.0f} mL @ day {f['day']:g}" for f in self.recommended_feeds
        ) or "no feed"
        ok = "PASS" if self.contract_satisfied else "FAIL"
        return (
            f"[{self.decided_by}] {feeds}  ->  "
            f"titer {self.predicted_kpis['final_titer_mg_L']} mg/L, "
            f"lac {self.predicted_kpis['max_lactate_mM']}, "
            f"amm {self.predicted_kpis['max_ammonia_mM']}  contract:{ok}"
        )


def feeds_to_dicts(feeds: Sequence[FeedBolus]) -> list[dict]:
    return [{"day": f.day, "volume_L": f.volume_L,
             "glc_conc_mM": f.glc_conc, "gln_conc_mM": f.gln_conc} for f in feeds]
