# Bioprocess yardstick: a fed-batch CHO bioreactor model and the decision record it leaves

A mechanistic model of a fed-batch CHO cell-culture bioreactor — the upstream step that produces a monoclonal antibody — is the source of truth. A controller reasons over it to choose a feed plan inside a contract of in-process limits, and writes down what it observed, what it chose, why, and what it rejected. Nothing quantum happens here; this is the kind of record a computational step in a regulated bioprocess is expected to leave, used as a yardstick for the quantum pipeline next door.

## Files

| File | Role |
|---|---|
| `bioreactor_twin.py` | The process model: Monod growth with by-product inhibition, glycolytic overflow to lactate, lactate re-consumption under glucose limitation, glutaminolysis to ammonia, antibody formation, and a stirred-tank mass balance with feed boluses |
| `contracts.py` | The contract (lactate, ammonia and vessel-volume limits) and the decision-artifact schema |
| `agent.py` | Two interchangeable controllers over the same model: `run_agent_search`, a deterministic search that runs offline, and `run_agent_llm`, a tool-use loop driven by a language model when an API key is set |
| `run_demo.py` | Baselines, the day-6 decision point, the controller's recommendation, and `provenance/decision_artifact.json` |
| `check.py` | Re-runs the scenarios and asserts the numbers below |

## Run

```bash
pip install -r requirements.txt
python3 run_demo.py        # offline: deterministic search controller
python3 check.py           # asserts the verified outputs
```

To let a language model drive the loop instead, install `anthropic` and set `ANTHROPIC_API_KEY`. The contract, not the controller, decides whether a plan is admissible either way.

## What it shows

- **The contract binds on over-feeding, not on starving.** With no feed, glucose runs out and titer suffers, but the run stays inside the contract (1607.0 mg/L). A naive daily over-feed drives lactate to 165.6 mM and ammonia to 18.01 mM, past their limits of 60 and 8 mM.
- **The controller finds where the limits bind by running what-ifs against the model.** From the day-6 state it evaluates 256 feed plans and chooses one 20 mL bolus on day 12: 1642.2 mg/L, lactate 57.5 mM, ammonia 7.27 mM. That is about 2% more titer than no feed — small, and the point is the record of the choice rather than the size of the gain.
- **The decision artifact** records the process, decision point, observed state, recommended feeds, predicted outcomes, the contract and whether it was satisfied, the objective, the rationale, the controller (`decided_by`), the model version, and five alternatives considered.

## Limits

- Parameters are illustrative and order-of-magnitude realistic, not fit to a clone.
- Nothing updates the model from a physical bioreactor, so it is a mechanistic process model rather than a digital twin in the strict sense.
- Lactate and ammonia limits are in-process limits. A critical quality attribute is a property of the product (ICH Q11); ammonia accumulation is linked to changes in antibody glycosylation, which is one, but glycosylation is not simulated here.
- The decision artifact is attributed, not signed.
- Dissolved oxygen and oxygen transfer are left as a documented hook, not coupled.
