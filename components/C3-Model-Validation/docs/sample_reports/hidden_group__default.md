# Model Quality Validation Report: hidden_group

## Decision: PROMOTE

> PROMOTE recommended: MQS 93.2 meets the promote threshold of 75; no critical rule fired, required evidence is present and every must-pass check passed.

*This is a recommendation. Nothing was deployed, promoted or rolled back.*

## Scores (0–100)

**Model Quality Score (MQS): 93.2**  
Worst case if every missing check had failed: 93.2  
Overall evidence coverage: 100%  
Missing-evidence policy: `exclude_and_cap`

| Dimension | Score | Weight | Evidence | Checks with evidence | Provisional |
|---|---|---|---|---|---|
| Training Validity (TVS) | 100.0 | 0.30 | 100% | 5/5 | no |
| Model Behaviour (MBS) | 86.4 | 0.50 | 100% | 8/8 | no |
| Operational Readiness (ORS) | 100.0 | 0.20 | 100% | 8/8 | no |

`MQS = TVS × W1 + MBS × W2 + ORS × W3`

**Calculation:** MQS = TVS 100.0 × 0.30 + MBS 86.4 × 0.50 + ORS 100.0 × 0.20 = 93.2


### The three layers at a glance

| Layer | Score | Checks | Needs attention |
|---|---|---|---|
| Training Validity (TVS) | 100.0 | 5 pass · 0 warn · 0 fail · 0 no evidence | none |
| Model Behaviour (MBS) | 86.4 | 7 pass · 1 warn · 0 fail · 0 no evidence | behaviour.roc_auc |
| Operational Readiness (ORS) | 100.0 | 8 pass · 0 warn · 0 fail · 0 no evidence | none |

Subgroup detection (hidden failure analysis) is switched off in this profile.

## Why this decision

| Rule | Kind | Effect | Reason |
|---|---|---|---|
| MQS meets the promotion threshold | score | PROMOTE | MQS 93.2 is at or above the promotion threshold of 75. |

### Explanation

* Watch: these checks warn but do not block promotion: behaviour.roc_auc.
* Weakest checks (MQS points lost): behaviour.roc_auc (-3.9), behaviour.f1 (-1.1), behaviour.precision (-0.9), behaviour.recall (-0.4), calibration.ece (-0.4).
* Evidence note: these results rely on declared (unverified) producer metadata: training_validity.overfit_gap, training_validity.train_eval_overlap, operational.dependency_match.

## Score breakdown

Each check's points add up to its dimension's score.

### Training Validity (TVS): 100.0

| Check | Status | Score | Weight | Points | Treatment |
|---|---|---|---|---|---|
| training_validity.metadata_completeness | PASS | 100 | 1 | 12.50 | counted |
| training_validity.reproducibility_info | PASS | 100 | 1 | 12.50 | counted |
| training_validity.overfit_gap | PASS | 100 | 2 | 25.00 | counted |
| training_validity.train_eval_overlap | PASS | 100 | 1 | 12.50 | counted |
| leakage.single_feature_predictiveness | PASS | 100 | 3 | 37.50 | counted |

### Model Behaviour (MBS): 86.4

| Check | Status | Score | Weight | Points | Treatment |
|---|---|---|---|---|---|
| behaviour.f1 | PASS | 87 | 2 | 15.11 | counted |
| behaviour.roc_auc | WARN | 55 | 2 | 9.61 | counted |
| behaviour.precision | PASS | 79 | 1 | 6.88 | counted |
| behaviour.recall | PASS | 90 | 1 | 7.82 | counted |
| behaviour.subgroup_f1_gap | PASS | 100 | 1 | 8.70 | counted |
| fairness.selection_rate_ratio | PASS | 100 | 2 | 17.39 | counted |
| fairness.equalized_odds | PASS | 100 | 1 | 8.70 | counted |
| calibration.ece | PASS | 94 | 1.5 | 12.23 | counted |

### Operational Readiness (ORS): 100.0

| Check | Status | Score | Weight | Points | Treatment |
|---|---|---|---|---|---|
| operational.model_loadable | PASS | 100 | 3 | 25.00 | counted |
| operational.input_schema | PASS | 100 | 2 | 16.67 | counted |
| operational.prediction_determinism | PASS | 100 | 1 | 8.33 | counted |
| operational.malformed_input_handling | PASS | 100 | 1 | 8.33 | counted |
| operational.artifact_size | PASS | 100 | 1 | 8.33 | counted |
| operational.latency | PASS | 100 | 2 | 16.67 | counted |
| operational.memory | PASS | 100 | 1 | 8.33 | counted |
| operational.dependency_match | PASS | 100 | 1 | 8.33 | counted |

## Evidence gaps

No gaps: every check had evidence.

## All checks

| Check | Status | Value | Score | Evidence | Required |
|---|---|---|---|---|---|
| training_validity.metadata_completeness | PASS | 1 | 100 | MEASURED |  |
| training_validity.reproducibility_info | PASS | 1 | 100 | MEASURED |  |
| training_validity.overfit_gap | PASS | -0.015 | 100 | DECLARED |  |
| training_validity.train_eval_overlap | PASS | 0 | 100 | DECLARED |  |
| leakage.single_feature_predictiveness | PASS | 0.6798 | 100 | MEASURED | yes |
| behaviour.f1 | PASS | 0.6951 | 87 | MEASURED | yes |
| behaviour.roc_auc | WARN | 0.7486 | 55 | MEASURED | yes |
| behaviour.precision | PASS | 0.6727 | 79 | MEASURED |  |
| behaviour.recall | PASS | 0.7191 | 90 | MEASURED |  |
| behaviour.subgroup_f1_gap | PASS | 0.004715 | 100 | MEASURED |  |
| fairness.selection_rate_ratio | PASS | 0.9996 | 100 | MEASURED | yes |
| fairness.equalized_odds | PASS | 0.00513 | 100 | MEASURED |  |
| calibration.ece | PASS | 0.02498 | 94 | MEASURED |  |
| operational.model_loadable | PASS | yes | 100 | MEASURED | yes |
| operational.input_schema | PASS | yes | 100 | MEASURED | yes |
| operational.prediction_determinism | PASS | yes | 100 | MEASURED |  |
| operational.malformed_input_handling | PASS | yes | 100 | MEASURED |  |
| operational.artifact_size | PASS | 0.003709 | 100 | MEASURED |  |
| operational.latency | PASS | 2.743 | 100 | MEASURED |  |
| operational.memory | PASS | 0.8477 | 100 | MEASURED |  |
| operational.dependency_match | PASS | 1 | 100 | DECLARED |  |

### Check details

* **`training_validity.metadata_completeness`** — PASS: 10 of 10 required training-metadata fields are present.
* **`training_validity.reproducibility_info`** — PASS: 5 of 5 reproducibility fields are present.
* **`training_validity.overfit_gap`** — PASS: Declared training accuracy 0.671 vs measured evaluation accuracy 0.686: gap -0.015. The training value is the producer's claim and was not verified.
* **`training_validity.train_eval_overlap`** — PASS: 0 of 3000 evaluation rows (0.00%) also appear in the training row hashes. Datasets can contain genuine duplicate records (0.00% of evaluation rows duplicate another evaluation row), so a small overlap is not by itself leakage. The hashes are producer-supplied.
* **`leakage.single_feature_predictiveness`** — PASS: The strongest single feature, 'income_k', reaches a cross-validated ROC AUC of 0.680 on its own (6 features, 3000 rows). Values near 1.0 suggest the feature leaks the label.
* **`behaviour.f1`** — PASS: F1 is 0.695 on 3000 evaluation rows (an always-positive baseline scores 0.665).
* **`behaviour.roc_auc`** — WARN: ROC AUC is 0.749 on 3000 evaluation rows (0.5 is chance).
* **`behaviour.precision`** — PASS: Precision is 0.673 on 3000 evaluation rows (an always-positive baseline scores 0.498).
* **`behaviour.recall`** — PASS: Recall is 0.719 on 3000 evaluation rows.
* **`behaviour.subgroup_f1_gap`** — PASS: Largest subgroup F1 gap is 0.005 for 'sex': Female 0.693 vs Male 0.697.
* **`fairness.selection_rate_ratio`** — PASS: Lowest/highest group selection rate is 1.000 for 'sex': Male 53.3% vs Female 53.3%.
* **`fairness.equalized_odds`** — PASS: Equalized-odds difference is 0.005 for 'sex' (true-positive-rate gap 0.002, false-positive-rate gap 0.005).
* **`calibration.ece`** — PASS: Expected calibration error is 0.025 (Brier 0.204, 10 bins, 3000 rows). Average predicted probability 0.519 vs observed positive rate 0.498: over-confident on average (predicts higher than observed).
* **`operational.model_loadable`** — PASS: Model loaded as Pipeline.
* **`operational.input_schema`** — PASS: Model predicted on 1000 evaluation rows using 6 feature columns.
* **`operational.prediction_determinism`** — PASS: Repeated predictions on the same rows were identical.
* **`operational.malformed_input_handling`** — PASS: All 3 malformed-input cases were handled.
* **`operational.artifact_size`** — PASS: Model file is 3.8 KB.
* **`operational.latency`** — PASS: Single-row prediction p95 latency is 2.7 ms (p50 1.9 ms) over 50 calls.
* **`operational.memory`** — PASS: Loading the model and predicting on a small sample grew memory by 1 MB.
* **`operational.dependency_match`** — PASS: 5 of 5 declared library versions match the current environment.

*Evidence: MEASURED = computed by the gate from the model and data; DECLARED = taken from producer metadata, not verified; SIMULATED = injected for a demonstration; UNAVAILABLE = none.*

## Inputs and audit trail

| Item | Value |
|---|---|
| Run id | sample-hidden_group-default |
| Created (UTC) | 2026-01-01T00:00:00+00:00 |
| Gate version | 0.1.0 |
| Schema version | 0.1.0 |
| Candidate | hidden_group |
| Model file | C:\Users\ASUS\OneDrive\Documents\GitHub\MLOpsVision\components\C3-Model-Validation\fixtures\hidden_group\model.joblib |
| Model SHA-256 | d8ba43f61e6789f5c0e931e6255c9f10b6bbcc3303c1e9c5ceeaca4706f6a58b |
| Evaluation data SHA-256 | 2f0944999996f28e9f36a49ebc65a3339d397ae87c88ba63ae7652fc96cc7801 |
| Metadata SHA-256 | 7f41938747f6bd46c7874b7d7550f938414cc34200575f80477a802d81b0bcc2 |
| Configuration profile | default |
| Configuration hash | 2c779cb9b2180fee08d3ab5b75955c265a2349cafa1bd1ff3908bb76840b7c24 |
| Machine | Windows-11-10.0.26200-SP0; 12 CPUs; 15.6 GB RAM |
| Libraries | python 3.12.1, scikit_learn 1.9.1, pandas 3.0.6, numpy 2.5.4, joblib 1.6.0 |
| Stage timings (s) | load 0.08, training_validity 0.16, model_behaviour 0.00, operational_readiness 0.16, score_and_decide 0.00 |

## Configuration used

* Profile `default`: Balanced profile used for the standalone prototype.
* Dimension weights: {'training_validity': 0.3, 'model_behaviour': 0.5, 'operational_readiness': 0.2}
* Promote at MQS ≥ 75.0; roll back below 50.0
* Must-pass checks: training_validity.metadata_completeness, training_validity.overfit_gap, calibration.ece, operational.latency
* Minimum evidence coverage: {'min_dimension': 0.5, 'min_overall': 0.8}
* Critical rules: Model artifact cannot be loaded, Target leakage detected, Severe selection-rate disparity between groups, F1 below the absolute minimum

The thresholds and weights are configuration choices. They need justification for the domain; they are not universal standards.

## Limitations

* This report is a recommendation; no action was taken.
* DECLARED evidence comes from producer metadata and is not verified.
* The gate looks for signs of leakage, overfitting and bias; it cannot prove their absence.
* Only binary classification on tabular data is evaluated (see `docs/supported_models.md`).
* Loading a model file executes code stored in it; validate only models from trusted sources.
