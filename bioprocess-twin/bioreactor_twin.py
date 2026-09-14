"""
bioreactor_twin.py — the physics and chemistry of the process.

A first-principles (mechanistic) process model of a fed-batch CHO cell-culture
bioreactor, the upstream unit operation that produces a monoclonal antibody.
It is a "digital twin" only in the loose sense: nothing updates it from a
physical bioreactor, which the usual definition of a digital twin requires.

This module is the *source of truth*. The agent (agent.py) never invents the
dynamics — it calls this simulator as a tool and reasons over what it returns.
That separation is the whole point: physics and chemistry stay in equations
that can be inspected, versioned, and validated; the agent supplies planning,
constraint handling, and an auditable rationale.

THE CHEMISTRY (reaction network encoded below)
  Carbon / energy:  glucose --glycolysis--> pyruvate --overflow--> LACTATE
                    (Warburg-like overflow when glucose is plentiful)
  Metabolic shift:  under glucose limitation cells RE-CONSUME lactate
                    (pyruvate routed to the TCA cycle) — a real CHO phenomenon
  Nitrogen:         glutamine --glutaminolysis--> glutamate + AMMONIA (NH4+)
  Growth/death:     Monod saturation x non-competitive by-product inhibition;
                    death accelerates with lactate + ammonia load
  Product:          mAb synthesis & secretion, non-growth-associated (qP * Xv)

THE PHYSICS (transport / operating envelope)
  Mass balance on a stirred tank with discrete feed boluses (dilution).
  Gas-liquid O2 mass transfer (kLa) is the classic scale-up limiter and is
  left as a documented hook (see DO note) rather than fully coupled here.

Units (consistent throughout; time in DAYS):
  Xv   viable cell density   [1e9 cells/L == 1e6 cells/mL]
  Glc  glucose               [mM]
  Gln  glutamine             [mM]
  Lac  lactate               [mM]
  Amm  ammonia               [mM]
  Titer  product (mAb)       [mg/L]
  V    working volume        [L]

Parameters are illustrative-but-realistic (order of magnitude), not fit to a
specific clone. Swap in a calibrated set for production use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.integrate import solve_ivp


# --------------------------------------------------------------------------- #
# Kinetic parameters
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CHOParams:
    # Growth / death
    mu_max: float = 0.90        # max specific growth rate              [1/day]
    kd_base: float = 0.020      # baseline specific death rate          [1/day]
    # Monod half-saturation constants
    K_glc: float = 0.50         # glucose                               [mM]
    K_gln: float = 0.30         # glutamine                             [mM]
    # Non-competitive by-product inhibition of growth
    KI_lac: float = 80.0        # lactate inhibition constant           [mM]
    KI_amm: float = 6.0         # ammonia inhibition constant           [mM]
    # Death acceleration from toxic by-products
    kd_lac: float = 0.0008      # [1/day per mM lactate]
    kd_amm: float = 0.010       # [1/day per mM ammonia]
    # Specific consumption / production (per 1e9 cells per day)
    q_glc: float = 0.55         # glucose uptake        [mmol/(1e9 cell.day)]
    q_gln: float = 0.10         # glutamine uptake      [mmol/(1e9 cell.day)]
    Y_lac_glc: float = 1.4      # lactate per glucose consumed       [mol/mol]
    Y_amm_gln: float = 0.85     # ammonia per glutamine consumed     [mol/mol]
    # Product formation (Luedeking-Piret, here non-growth-associated qP)
    q_mab: float = 9.0          # specific productivity   [mg/(1e9 cell.day)]
    # Lactate re-uptake once glucose is scarce (the metabolic shift)
    q_lac_uptake: float = 0.08  # [mmol/(1e9 cell.day)]
    glc_shift_threshold: float = 1.0  # glucose below which the shift engages [mM]


# --------------------------------------------------------------------------- #
# Feed schedule (the lever the agent controls)
# --------------------------------------------------------------------------- #

@dataclass
class FeedBolus:
    """A single bolus addition of concentrated feed at a given time."""
    day: float                 # time of addition                       [day]
    volume_L: float            # bolus volume                           [L]
    glc_conc: float = 1250.0   # glucose conc. of feed stock (~225 g/L) [mM]
    gln_conc: float = 150.0    # glutamine conc. of feed stock          [mM]


# --------------------------------------------------------------------------- #
# Right-hand side of the ODE system
# --------------------------------------------------------------------------- #

def _rhs(t: float, y: np.ndarray, p: CHOParams) -> np.ndarray:
    Xv, Glc, Gln, Lac, Amm, Titer, V = y
    Xv = max(Xv, 0.0)
    Glc = max(Glc, 0.0)
    Gln = max(Gln, 0.0)

    # Specific growth rate: Monod substrate limitation x by-product inhibition.
    # (DO note: a dissolved-oxygen Monod term  Glc/(K_glc+Glc) -> *DO/(K_DO+DO)
    #  plugs in here once OTR = kLa*(C* - C_L) is coupled — the kLa physics.)
    mu = (
        p.mu_max
        * (Glc / (p.K_glc + Glc))
        * (Gln / (p.K_gln + Gln))
        * (p.KI_lac / (p.KI_lac + Lac))
        * (p.KI_amm / (p.KI_amm + Amm))
    )
    # Death accelerates with toxic by-product load
    kd = p.kd_base + p.kd_lac * Lac + p.kd_amm * Amm
    dXv = (mu - kd) * Xv

    # Substrate uptake scales with the growing, nutrient-limited population
    glc_uptake = p.q_glc * Xv * (Glc / (p.K_glc + Glc))
    gln_uptake = p.q_gln * Xv * (Gln / (p.K_gln + Gln))

    # Metabolic shift: glucose-limited cells consume lactate instead of making it
    if Glc < p.glc_shift_threshold and Lac > 0:
        lac_flux = -p.q_lac_uptake * Xv            # net lactate consumption
    else:
        lac_flux = p.Y_lac_glc * glc_uptake        # lactate from glycolytic overflow

    dGlc = -glc_uptake
    dGln = -gln_uptake
    dLac = lac_flux
    dAmm = p.Y_amm_gln * gln_uptake
    dTiter = p.q_mab * Xv                            # mg/L/day
    dV = 0.0                                         # boluses applied discretely

    return np.array([dXv, dGlc, dGln, dLac, dAmm, dTiter, dV])


# --------------------------------------------------------------------------- #
# Trajectory container
# --------------------------------------------------------------------------- #

@dataclass
class SimResult:
    t: np.ndarray
    Xv: np.ndarray
    Glc: np.ndarray
    Gln: np.ndarray
    Lac: np.ndarray
    Amm: np.ndarray
    Titer: np.ndarray
    V: np.ndarray

    def state_at(self, day: float) -> dict:
        """Snapshot of the process at a given day (a 'sensor reading')."""
        i = int(np.argmin(np.abs(self.t - day)))
        return {
            "day": round(float(self.t[i]), 2),
            "VCD_e6_cells_mL": round(float(self.Xv[i]), 2),
            "glucose_mM": round(float(self.Glc[i]), 2),
            "glutamine_mM": round(float(self.Gln[i]), 2),
            "lactate_mM": round(float(self.Lac[i]), 1),
            "ammonia_mM": round(float(self.Amm[i]), 2),
            "titer_mg_L": round(float(self.Titer[i]), 1),
            "volume_L": round(float(self.V[i]), 3),
        }

    def summary(self) -> dict:
        """The KPIs / CQAs an operator or agent actually optimizes over."""
        return {
            "final_titer_mg_L": round(float(self.Titer[-1]), 1),
            "peak_VCD_e6_cells_mL": round(float(self.Xv.max()), 2),
            "max_lactate_mM": round(float(self.Lac.max()), 1),
            "final_lactate_mM": round(float(self.Lac[-1]), 1),
            "max_ammonia_mM": round(float(self.Amm.max()), 2),
            "min_glucose_mM": round(float(self.Glc.min()), 2),
            "final_volume_L": round(float(self.V[-1]), 3),
            "duration_days": round(float(self.t[-1]), 1),
        }


# --------------------------------------------------------------------------- #
# Integrator with discrete feed boluses
# --------------------------------------------------------------------------- #

def default_inoculum() -> dict:
    return {
        "Xv": 0.30,   # 0.3e6 cells/mL inoculation density
        "Glc": 30.0,  # initial glucose
        "Gln": 6.0,   # initial glutamine
        "Lac": 0.0,
        "Amm": 0.0,
        "Titer": 0.0,
        "V": 1.0,     # 1 L working volume
    }


_ORDER = ("Xv", "Glc", "Gln", "Lac", "Amm", "Titer", "V")

# Map the human-readable 'sensor reading' keys (SimResult.state_at) onto the
# integrator's state variables, so an observed state can be fed straight back in.
_SENSOR_TO_STATE = {
    "VCD_e6_cells_mL": "Xv",
    "glucose_mM": "Glc",
    "glutamine_mM": "Gln",
    "lactate_mM": "Lac",
    "ammonia_mM": "Amm",
    "titer_mg_L": "Titer",
    "volume_L": "V",
}


def normalize_state(state: dict) -> dict:
    """Accept either internal (Xv, Glc, ...) or sensor (VCD_e6_cells_mL, ...) keys."""
    out = {}
    for k, val in state.items():
        if k in _SENSOR_TO_STATE:
            out[_SENSOR_TO_STATE[k]] = val
        elif k in _ORDER or k == "_t0":
            out[k] = val
    return out


def simulate(
    feeds: Sequence[FeedBolus] = (),
    *,
    params: CHOParams | None = None,
    duration_days: float = 14.0,
    y0: dict | None = None,
    n_points: int = 200,
) -> SimResult:
    """Integrate the fed-batch from `y0`, applying boluses discretely."""
    p = params or CHOParams()
    init = default_inoculum()
    if y0:
        init.update(y0)
    state = np.array([init[k] for k in _ORDER])

    feed_days = sorted(f.day for f in feeds if init.get("_t0", 0.0) < f.day < duration_days)
    t_start = float(init.get("_t0", 0.0))
    boundaries = [t_start, *feed_days, duration_days]

    ts: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for i in range(len(boundaries) - 1):
        t0, t1 = boundaries[i], boundaries[i + 1]
        if t1 <= t0:
            continue
        seg_pts = max(2, int(n_points * (t1 - t0) / max(duration_days - t_start, 1e-9)))
        sol = solve_ivp(
            _rhs, (t0, t1), state, args=(p,),
            t_eval=np.linspace(t0, t1, seg_pts),
            method="LSODA", rtol=1e-6, atol=1e-8,
        )
        ts.append(sol.t)
        ys.append(sol.y)
        state = sol.y[:, -1].copy()

        for f in feeds:                              # apply boluses landing at t1
            if abs(f.day - t1) < 1e-9:
                Xv, Glc, Gln, Lac, Amm, Titer, V = state
                Vn = V + f.volume_L
                Glc = (Glc * V + f.glc_conc * f.volume_L) / Vn
                Gln = (Gln * V + f.gln_conc * f.volume_L) / Vn
                Xv, Lac, Amm, Titer = (x * V / Vn for x in (Xv, Lac, Amm, Titer))
                state = np.array([Xv, Glc, Gln, Lac, Amm, Titer, Vn])

    return SimResult(
        t=np.concatenate(ts),
        Xv=np.concatenate([y[0] for y in ys]),
        Glc=np.concatenate([y[1] for y in ys]),
        Gln=np.concatenate([y[2] for y in ys]),
        Lac=np.concatenate([y[3] for y in ys]),
        Amm=np.concatenate([y[4] for y in ys]),
        Titer=np.concatenate([y[5] for y in ys]),
        V=np.concatenate([y[6] for y in ys]),
    )


def simulate_from(
    state: dict,
    feeds: Sequence[FeedBolus],
    *,
    start_day: float,
    duration_days: float = 14.0,
    params: CHOParams | None = None,
) -> SimResult:
    """Integrate forward from a mid-run state — the agent's what-if primitive.

    `state` may use sensor keys (VCD_e6_cells_mL, glucose_mM, ...) or internal
    keys (Xv, Glc, ...); both are accepted.
    """
    y0 = normalize_state(state)
    y0["_t0"] = start_day
    return simulate(feeds, params=params, duration_days=duration_days, y0=y0)


if __name__ == "__main__":
    print("== Batch, no feeding ==")
    print(simulate(duration_days=14).summary())
    print("\n== Naive daily 30 mL glucose boluses, days 3-10 ==")
    naive = [FeedBolus(day=d, volume_L=0.030) for d in range(3, 11)]
    print(simulate(naive, duration_days=14).summary())
