# C3: Automated Model Quality & Validation Gate (MQVG)

Evaluates a trained model **before** it is promoted. It checks three layers,
combines them into a Model Quality Score (MQS) and recommends **PROMOTE**,
**HOLD** or **ROLLBACK**, with an auditable report.

**The decision is a recommendation only.** Nothing is deployed, promoted or
rolled back by this component.

**Scope.** Data collection, data-integrity validation, model training,
deployment and monitoring belong to other components. This one evaluates the
artifacts it is given: a saved model, separate labelled evaluation data and
(optionally) training metadata and training-row hashes.

## The flow

```
model file + evaluation CSV + metadata + training-row hashes
   -> Layer 1 Training validity   (TVS)  metadata, reproducibility, overfit gap, overlap, leakage
   -> Layer 2 Model behaviour     (MBS)  F1/precision/recall/AUC, subgroup gap, fairness,
                                         calibration, robustness, hidden failure groups*
   -> Layer 3 Operational readiness (ORS) loads, schema, determinism, malformed input,
                                         size, latency, memory, dependencies
   -> MQS = 0.30*TVS + 0.50*MBS + 0.20*ORS      (weights are configurable placeholders)
   -> rules: critical rules -> ROLLBACK, missing evidence / must-pass -> HOLD, score bands
   -> report.md + result.json
```
\* off in `configs/default.yaml`, on in the `hidden_groups*` profiles.

A check without evidence is never counted as a pass: it carries no score, lowers
evidence coverage and the worst-case MQS, and can force HOLD.

## Install and run

Python 3.11+.

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

python scripts\validate_models.py good                       # one model
python scripts\validate_models.py                            # the six sample models
python scripts\validate_models.py hidden_group --config configs\hidden_groups.yaml
```

From Python:

```python
from mqvg.config import load_config
from mqvg.schemas import CandidateBundle
from mqvg.workflow import validate_and_report

out = validate_and_report(CandidateBundle(
    candidate_id="m", model_path="model.joblib", eval_data_path="eval.csv",
    metadata_path="metadata.json", train_row_hashes_path="train_row_hashes.txt"),
    load_config("configs/default.yaml"), report_dir="reports/m")
print(out.decision, out.result.scorecard.mqs)
```

**Security:** loading a joblib/pickle file executes code stored in it. Only
validate models from trusted sources.

### HTTP API

```
$env:MQVG_ARTIFACT_ROOT = "C:\path	ortifacts"; $env:MQVG_API_KEY = "change-me"
python -m mqvg.api                                    # http://127.0.0.1:8000/docs
```

A thin wrapper around `validate_and_report`: `POST /v1/validations` returns the recommendation,
`GET /v1/validations/{run_id}` the full result, `.../report` the Markdown report. It is
recommendation-only, confines paths to one artifact root, runs only profiles that scan the model
file, and needs an API key. Read `docs/api.md` (security model, settings, limits) before exposing
it; the contract is `docs/api_openapi.json`.

## Profiles (`configs/`)

| File | Purpose |
|---|---|
| `default.yaml` | Balanced prototype profile; hidden-group check off |
| `strict.yaml` | Tighter thresholds, missing evidence penalised |
| `extended.yaml` | Default plus conflicting labels, training duplicates, model-file safety scan and CPU time |
| `extended_hidden_groups.yaml` | Extended plus hidden failure group detection (the profile to use through the API for subgroup detection) |
| `hidden_groups_score_only.yaml` | Hidden-group check on; affects the score only |
| `hidden_groups.yaml` | Check on; a confirmed group gives HOLD |
| `hidden_groups_critical.yaml` | Check on; a severe group gives a ROLLBACK recommendation |

Every threshold, weight and band is an **initial placeholder**, not a standard
(see the header of `default.yaml` and `docs/assumptions.md`).

## Sample models (`fixtures/`)

`good`, `leaky`, `overfit`, `biased`, `miscalibrated`, `slow`, `corrupt`,
`good_incomplete_metadata`, `good_no_sensitive`, plus `hidden_group` (a planted
failing subgroup) and `hidden_group_null` (a control); `scripts/make_extended_fixtures.py` adds
`label_conflicts`, `duplicate_training`, `unsafe_model` (deliberately unsafe) and `unreviewed_model`. Expected outcomes are in
`fixtures/manifest.yaml` and `fixtures/hidden_groups_manifest.yaml`.

## Tests and studies

```
python -m pytest -q                                   # full suite
python scripts\evaluate_hidden_groups.py --quick      # detection-power study (synthetic)
python scripts\sensitivity_analysis.py                # decision stability; writes docs\sensitivity_analysis.md
python scripts\baseline_comparison.py                 # plain metrics gate vs full gate; writes docs\baseline_comparison.md
python scriptsaseline_comparison.py                # plain metrics gate vs full gate; writes docsaseline_comparison.md
```

The full suite takes several minutes. Latency and memory are wall-clock
measurements, so they vary slightly between runs and machines.

## Layout

| Path | Contents |
|---|---|
| `src/mqvg/` | `config`, `loader`, `model_safety`, `pipeline`, `scoring`, `rules`, `explain`, `reporting`, `workflow`, `hidden_groups`, `checks/`, `api/` |
| `configs/` | validation profiles |
| `fixtures/`, `data/` | sample models and evaluation data |
| `scripts/` | demonstration and study helpers (not part of the gate) |
| `docs/` | `workflow.md`, `scoring.md`, `reporting.md`, `hidden_groups.md`, `supported_models.md`, `assumptions.md`, `sensitivity_analysis.md`, `baseline_comparison.md`, `extended_checks.md`, `api.md`, `api_openapi.json`, `sample_reports/` |

## Current limits

- Tested on scikit-learn style binary classifiers saved with joblib/pickle.
- Thresholds and weights are not calibrated on real-world model collections.
- Explanations are rule-based, not SHAP/LIME.
- Not implemented: a rollback-capability check, conflicting labels in the training data,
  sandboxed model loading, asynchronous jobs, TLS, rate limiting and per-client credentials in the API. The extended profile adds a static model-file
  scan (not a sandbox), conflicting labels in the evaluation data, training duplicates from
  the row hashes, and CPU time (`docs/extended_checks.md`).
- Hidden-group findings are statistical associations, not causes, and the
  features describing a group can be proxies for sensitive characteristics.
