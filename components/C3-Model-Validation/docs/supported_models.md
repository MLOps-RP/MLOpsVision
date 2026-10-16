# Supported models, data and limitations

What the gate can and cannot evaluate. Everything marked **tested** is covered
by an automated test (`tests/test_external_models.py` and the fixture tests);
nothing is claimed beyond that.

## Short answer

The gate evaluates **scikit-learn-style tabular binary classifiers** saved with
joblib or pickle. For anything else it says so and returns `UNAVAILABLE`
instead of guessing. It does not evaluate arbitrary machine-learning models.

## What you provide

| Input | Required? | Format |
|---|---|---|
| Model file | yes | a Python object saved with `joblib` or `pickle` (`.joblib`, `.pkl`, `.pickle`) |
| Evaluation data | yes | comma-separated CSV with a header, including the true target column |
| Training metadata | no, but see below | JSON or YAML (`mqvg.schemas.TrainingMetadata`) |
| Configuration | yes | YAML (`configs/default.yaml`, `configs/strict.yaml`, or your own) |

Without metadata the gate can still measure what it can, but it needs the
target column and positive label (metadata or `CandidateBundle` overrides) and,
for a model without column names, the feature columns. Fairness needs the
sensitive columns declared. Missing items give `UNAVAILABLE` checks, a
`HOLD`, and `mqs_sufficient_evidence = False`; they are never treated as
passing.

## Models

| | Status |
|---|---|
| scikit-learn estimators and Pipelines: logistic regression, decision tree, random forest, gradient boosting, linear SVM | **tested** |
| `.joblib` and plain-pickle `.pkl` files | **tested** |
| A bare estimator fitted on numpy arrays | **tested**, if `feature_columns` is given |
| Models with `predict_proba` or `decision_function` | **tested** (AUC from either; calibration only from `predict_proba`) |
| Text classifiers whose pipeline selects a column of a DataFrame | **tested**; leakage is not assessed for text |
| XGBoost / LightGBM / CatBoost scikit-learn wrappers | **untested** (the libraries are not installed here). They may work if installed in the gate's environment; do not rely on it. |
| ONNX, Keras/TensorFlow, PyTorch, native XGBoost/CatBoost files, PMML | **not supported**: reported as `UNAVAILABLE` and never loaded |
| Models that need a Python module the gate does not have | **not supported**: `UNAVAILABLE` (`missing_dependency`) |
| `predict()` that accepts only numpy arrays, or a vectorizer fitted on a plain list of strings | **not supported**: `input_schema` fails, behaviour checks `UNAVAILABLE` |

## Tasks

| Task | Status |
|---|---|
| Binary classification | **supported** |
| Multiclass classification | not supported (behaviour checks `UNAVAILABLE`, clear message) |
| Regression | not supported (recognised from a continuous numeric target) |
| Multi-label, ranking, sequence labelling, generative / LLM models | not supported |

## Data

* Tabular data, with one column per model feature, read with pandas defaults
  (comma separated, UTF-8). Tab-separated, Parquet, Excel and JSON files are
  not read; a TSV appears as missing columns.
* Feature names must match the model exactly (no fuzzy matching). Missing,
  renamed or wrongly typed columns make `input_schema` fail.
* The model's labels and the evaluation target must use the same values.
  Integer-valued numbers agree (`1`, `1.0`, `"1"`). A mismatch (for example
  the model outputs 0/1 and the target holds text) gives `UNAVAILABLE` with an
  explanation, not a score.
* Fairness needs sensitive attributes as separate columns. Text data has no
  numeric features, so robustness is not applicable, and the leakage probe
  cannot assess free-text or identifier-like columns (reported as not assessed).

## How failures are reported

| Situation | Check status | Decision effect |
|---|---|---|
| File missing, corrupt, or not a model | `model_loadable` **FAIL** | critical rule: ROLLBACK |
| Unsupported format, or the model needs an uninstalled module | `model_loadable` **UNAVAILABLE** | HOLD (required evidence missing); not a statement about quality |
| Labels differ between model and data | behaviour checks **UNAVAILABLE** | HOLD |
| Missing / renamed / mistyped feature columns | `input_schema` **FAIL**, behaviour **UNAVAILABLE** | HOLD |
| Unsupported task | behaviour checks **UNAVAILABLE** | HOLD |
| Too little evidence for a reliable MQS | `mqs_sufficient_evidence = False` with a reason | HOLD; show the MQS as "not reliable" |

## Security

Loading a joblib or pickle file **executes code stored in it**
(`tests/test_external_models.py` demonstrates this with a harmless marker
file). Validate only models from sources you trust. Formats the gate refuses
(`.onnx`, `.pt`, ...) are never opened. Process isolation and an explicit
trust acknowledgement in the CLI/API are planned (Phase B / Step 14).

## Known remaining limitations

* One dataset (Adult) was used to set the placeholder thresholds.
* The memory check is weak evidence and the malformed-input check runs in the
  validating process (see `docs/assumptions.md`).
* There is no command-line tool or API yet; use `mqvg.pipeline.run_validation`
  or `scripts/validate_models.py`.
