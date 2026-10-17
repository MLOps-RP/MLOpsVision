# Reports and explanations (`reporting.py`, `explain.py`)

Every validation produces two files:

| File | For | Contents |
|---|---|---|
| `result.json` | programs, audit | the complete `ValidationResult` (schema in `schemas.py`) |
| `report.md` | people | the same result as a readable Markdown report |

```python
from mqvg.pipeline import run_validation
from mqvg.reporting import render_markdown, write_report

result = run_validation(bundle, config)
write_report(result, "reports/my_model")      # writes result.json and report.md
text = render_markdown(result)                 # or just get the Markdown
```

`scripts/validate_models.py` does this for the sample models.

## What the report contains

1. **Decision** (PROMOTE / HOLD / ROLLBACK), a one-sentence summary, a reminder
   that it is only a recommendation, and banners when the run is simulated or
   the MQS is not reliable.
2. **Scores**: MQS (shown as "n/a" with the recorded value when the evidence is
   insufficient), the worst-case MQS, overall evidence coverage, and the three
   dimension scores with weight, coverage and provisional flag.
3. **Why this decision**: a table of every rule that fired (kind, effect,
   reason) and a plain-English explanation.
4. **Score breakdown**: for each dimension, every check's status, score, weight
   and the points it contributed. The points add up to the dimension's score.
5. **Evidence gaps**: checks that had no evidence and why. They are never
   counted as passing.
6. **All checks**: table plus the full message of each check, with the evidence
   label (MEASURED / DECLARED / SIMULATED).
7. **Inputs and audit trail**: run id, time, versions, file SHA-256 hashes,
   configuration hash, machine, library versions and timings.
8. **Configuration used** and **Limitations**.

The Scores section also writes the MQS arithmetic with the real numbers (for example `MQS = TVS 100.0 × 0.30 + MBS 76.6 × 0.50 + ORS 100.0 × 0.20 = 88.3`, noting dropped dimensions or the penalty) and a three-layers-at-a-glance table. When `behaviour.hidden_failure_groups` is enabled (profile `configs/hidden_groups.yaml`)
the report also has a **Hidden failure groups** section after the score breakdown: each
group's rule, method, held-out size, observed versus expected error, excess with its
95% interval, adjusted p-value and whether it was confirmed, plus the calibration
safeguard and the features searched. See `docs/hidden_groups.md`.

Nothing is left out: every rule that fired and every check appears in the
report, so the JSON never has to be opened to understand a decision.

## The explanation (`explain_failures`)

Stored in `result.explanations`, one line each, starting with a label:

| Label | Meaning |
|---|---|
| `Decision:` | the one-sentence summary |
| `Why (kind):` | a rule that fired, with its reason |
| `Override:` | a critical rule forced ROLLBACK against the MQS |
| `Watch:` | warnings that did not block a promotion |
| `Weakest checks (MQS points lost):` | the checks that cost the most MQS points, largest first |
| `Evidence gap` / `Not applicable` | checks without evidence, with the reason |
| `Evidence note:` | declared evidence, provisional or unreliable MQS, simulated inputs |
| `Suggested action (id):` | a generic starting point for each failing or warning check, and for evidence problems |

A check's lost points = how far its score fell short of 100, scaled by its
weight within the dimension and the dimension's weight in the MQS. With full
evidence the lost points of all checks plus the MQS equal 100.

### Two different kinds of "explanation"

* **This module explains why the validation gate decided what it did** (which
  rule failed, by how much, what to fix). It is a deterministic trace; no model
  is involved.
* **It does not explain the model's individual predictions.** That is what
  SHAP or LIME do, and they are a planned later feature (Phase B). When added,
  they should supply evidence for a failure (for example which feature drives a
  group gap), not replace this trace.

Suggested actions are generic hints, not guarantees.
