# The complete validation workflow

```
 model file (+ metadata, training-row hashes)      labelled evaluation CSV (separate data)
                    \                                /
                     v                              v
        Layer 1  Training validity (TVS)   leakage, overfit gap, train/eval overlap, metadata
        Layer 2  Model behaviour (MBS)     performance, calibration, robustness, fairness,
                                           hidden failure groups (cross-fitted Tree + K-means)
        Layer 3  Operational readiness (ORS) loadable, schema, latency, determinism, size
                                 |
                                 v
              MQS = TVS*w1 + MBS*w2 + ORS*w3   (missing policy, coverage, lower bound)
                                 |
                                 v
        rules: critical -> ROLLBACK, evidence/blocking -> HOLD, score bands -> PROMOTE/HOLD/ROLLBACK
                                 |
                                 v
              unified report (Markdown + JSON): a RECOMMENDATION only
```

```python
from mqvg.config import load_config
from mqvg.schemas import CandidateBundle
from mqvg.workflow import validate_and_report

out = validate_and_report(CandidateBundle(candidate_id="m", model_path=..., eval_data_path=...,
                                          metadata_path=..., train_row_hashes_path=...),
                          load_config("configs/hidden_groups.yaml"), report_dir="reports/m")
out.decision, out.report_markdown, out.paths
```

Command line: `python scripts/validate_models.py <fixture> --config configs/hidden_groups.yaml`.

## What the unified report shows

Decision and recommendation notice; the three layer scores with the MQS calculation;
every rule that fired; the score breakdown; the **Hidden failure groups** section (rule,
held-out rows, observed vs expected error, excess with 95% CI, adjusted p, confirmed);
evidence gaps; all checks; audit trail; configuration; limitations.

## Policy matrix (planted fixture `hidden_group`; reports in `docs/sample_reports/`)

| Profile | MQS | Decision |
|---|---|---|
| `default` (check off, the blind spot) | 93.2 | PROMOTE |
| `hidden_groups_score_only` | 88.3 | PROMOTE |
| `hidden_groups` | 88.3 | HOLD |
| `hidden_groups_critical` | 88.3 | ROLLBACK (recommendation) |

The control `hidden_group_null` is PROMOTE under all four.

## Guarantees and limits

- UNAVAILABLE evidence (too few rows or errors, miscalibrated model, no probabilities)
  is never a pass: the check carries no score, coverage falls, the MQS lower bound drops;
  `hidden_groups*` with `require_assessment` turn it into a HOLD.
- Nothing is deployed, promoted or rolled back. A ROLLBACK is advice.
- Verified by `tests/test_workflow_e2e.py` for the synthetic logistic-regression
  scenarios only; no claim is made for other model families beyond `docs/supported_models.md`.
- Loading a model file with joblib/pickle executes code: only validate files you trust.
- Subgroups are associations, not causes, and their descriptions can be proxies for
  sensitive characteristics (see `docs/hidden_groups.md`).
