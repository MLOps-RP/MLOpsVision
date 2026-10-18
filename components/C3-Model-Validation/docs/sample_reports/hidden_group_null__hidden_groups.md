# Model Quality Validation Report: hidden_group_null

## Decision: PROMOTE

> PROMOTE recommended: MQS 95.6 meets the promote threshold of 75; no critical rule fired, required evidence is present and every must-pass check passed.

*This is a recommendation. Nothing was deployed, promoted or rolled back.*

## Scores (0–100)

**Model Quality Score (MQS): 95.6**  
Worst case if every missing check had failed: 95.6  
Overall evidence coverage: 100%  
Missing-evidence policy: `exclude_and_cap`

| Dimension | Score | Weight | Evidence | Checks with evidence | Provisional |
|---|---|---|---|---|---|
| Training Validity (TVS) | 100.0 | 0.30 | 100% | 5/5 | no |
| Model Behaviour (MBS) | 91.1 | 0.50 | 100% | 9/9 | no |
| Operational Readiness (ORS) | 100.0 | 0.20 | 100% | 8/8 | no |

`MQS = TVS × W1 + MBS × W2 + ORS × W3`

**Calculation:** MQS = TVS 100.0 × 0.30 + MBS 91.1 × 0.50 + ORS 100.0 × 0.20 = 95.6


### The three layers at a glance

| Layer | Score | Checks | Needs attention |
|---|---|---|---|
| Training Validity (TVS) | 100.0 | 5 pass · 0 warn · 0 fail · 0 no evidence | none |
| Model Behaviour (MBS) | 91.1 | 8 pass · 1 warn · 0 fail · 0 no evidence | behaviour.roc_auc |
| Operational Readiness (ORS) | 100.0 | 8 pass · 0 warn · 0 fail · 0 no evidence | none |

Subgroup detection: **0 confirmed** of 13 candidate groups tested on held-out rows.

## Why this decision

| Rule | Kind | Effect | Reason |
|---|---|---|---|
| MQS meets the promotion threshold | score | PROMOTE | MQS 95.6 is at or above the promotion threshold of 75. |

### Explanation

* Hidden failure group policy (configured): consequence 'hold', assessment required. A confirmed group blocks promotion.
* Watch: these checks warn but do not block promotion: behaviour.roc_auc.
* Weakest checks (MQS points lost): behaviour.roc_auc (-2.8), behaviour.f1 (-0.6), behaviour.precision (-0.6), behaviour.recall (-0.2), calibration.ece (-0.1).
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

### Model Behaviour (MBS): 91.1

| Check | Status | Score | Weight | Points | Treatment |
|---|---|---|---|---|---|
| behaviour.f1 | PASS | 92 | 2 | 14.09 | counted |
| behaviour.roc_auc | WARN | 63 | 2 | 9.77 | counted |
| behaviour.precision | PASS | 84 | 1 | 6.44 | counted |
| behaviour.recall | PASS | 94 | 1 | 7.26 | counted |
| behaviour.subgroup_f1_gap | PASS | 100 | 1 | 7.69 | counted |
| fairness.selection_rate_ratio | PASS | 100 | 2 | 15.38 | counted |
| fairness.equalized_odds | PASS | 100 | 1 | 7.69 | counted |
| calibration.ece | PASS | 97 | 1.5 | 11.25 | counted |
| behaviour.hidden_failure_groups | PASS | 100 | 1.5 | 11.54 | counted |

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

## Hidden failure groups

Groups of rows, defined by feature values, where the model makes more errors than its own probabilities predict. Each group was found on some rows and tested on different, held-out rows.

**Decision consequence (configured): `hold`** — a confirmed group blocks promotion (HOLD). A result that could not be assessed **blocks promotion**. Significance (false discovery rate) 0.05, minimum effect 0.1.

**0 confirmed** of 13 groups tested (methods: tree, kmeans; false discovery rate 0.05; minimum effect 10%). Confirmed groups cover 0.0% of rows.

| Group | Method | Rows | Share | Observed error | Expected error | Excess (95% CI) | Adjusted p | Confirmed |
|---|---|---|---|---|---|---|---|---|
| cluster of similar rows: region = s for 100% of rows (overall 26%); channel = u for 100% of rows (overall 51%); segment = d for 32% of rows (overall 25%) | kmeans | 190 | 6.3% | 33.7% | 29.5% | +4.2% [-2.2%, +11.1%] | 9.9e-01 | no |
| cluster of similar rows: region = s for 100% of rows (overall 24%); segment = a for 100% of rows (overall 26%); channel = u for 61% of rows (overall 51%) | kmeans | 90 | 3.0% | 31.1% | 28.7% | +2.5% [-6.2%, +12.6%] | 9.9e-01 | no |
| cluster of similar rows: region = w for 100% of rows (overall 24%); channel = u for 0% of rows (overall 51%); segment = d for 32% of rows (overall 23%) | kmeans | 188 | 6.3% | 28.7% | 28.8% | -0.1% [-6.1%, +6.7%] | 9.9e-01 | no |
| age <= 20.5 | tree | 87 | 2.9% | 27.6% | 27.6% | +0.0% [-8.3%, +10.2%] | 9.9e-01 | no |
| 20.5 < age <= 23.5 AND income_k > 46.4 | tree | 49 | 1.6% | 24.5% | 24.4% | +0.1% [-9.8%, +13.7%] | 9.9e-01 | no |
| age > 23.5 AND 36.4 < income_k <= 42.4 | tree | 168 | 5.6% | 29.2% | 29.5% | -0.3% [-6.7%, +6.9%] | 9.9e-01 | no |
| cluster of similar rows: region = s for 100% of rows (overall 26%); channel = u for 0% of rows (overall 51%); age averages 39.6 (overall 43.3) | kmeans | 168 | 5.6% | 26.8% | 28.4% | -1.6% [-7.7%, +5.5%] | 9.9e-01 | no |
| 0.67 < tenure_years <= 2.71 AND segment != c | tree | 169 | 5.6% | 27.8% | 29.5% | -1.7% [-7.9%, +5.5%] | 9.9e-01 | no |
| cluster of similar rows: segment = b for 100% of rows (overall 23%); channel = u for 72% of rows (overall 51%); income_k averages 44.2 (overall 50.4) | kmeans | 303 | 10.1% | 25.4% | 26.7% | -1.3% [-5.8%, +3.9%] | 9.9e-01 | no |
| cluster of similar rows: region = w for 100% of rows (overall 24%); channel = u for 100% of rows (overall 51%); income_k averages 46.3 (overall 50.6) | kmeans | 173 | 5.8% | 27.2% | 29.8% | -2.6% [-8.7%, +4.5%] | 9.9e-01 | no |

*Showing 10 of 13 groups tested.*

No group passed both the significance and the effect-size requirement.

Calibration safeguard: ECE 0.016, observed-versus-expected error gap 0.001. Probabilities were recalibrated (Platt map, discovery rows only) on 1 of 2 folds.

Features searched: age, income_k, tenure_years, segment, region, channel.

*Caveats: Expected errors come from the model's own probabilities; the test is only valid if they are calibrated. They are recalibrated per fold on the discovery rows only; grossly wrong probabilities make the check unavailable. Groups are axis-aligned rules or clusters of similar rows, so an oblique or very small region may be missed or found only in part. A group is an association found in the evaluation data, not a causal explanation: it says where the model fails, not why. Features that describe a group may be proxies for sensitive characteristics (for example a relationship status standing in for sex), so the rule can reveal a protected subpopulation even though no sensitive column was named. Detection power depends on the sample size; no group found does not mean none exists.*

## Evidence gaps

No gaps: every check had evidence.

## All checks

| Check | Status | Value | Score | Evidence | Required |
|---|---|---|---|---|---|
| training_validity.metadata_completeness | PASS | 1 | 100 | MEASURED |  |
| training_validity.reproducibility_info | PASS | 1 | 100 | MEASURED |  |
| training_validity.overfit_gap | PASS | -0.0105 | 100 | DECLARED |  |
| training_validity.train_eval_overlap | PASS | 0 | 100 | DECLARED |  |
| leakage.single_feature_predictiveness | PASS | 0.7045 | 100 | MEASURED | yes |
| behaviour.f1 | PASS | 0.7326 | 92 | MEASURED | yes |
| behaviour.roc_auc | WARN | 0.7857 | 63 | MEASURED | yes |
| behaviour.precision | PASS | 0.7116 | 84 | MEASURED |  |
| behaviour.recall | PASS | 0.7548 | 94 | MEASURED |  |
| behaviour.subgroup_f1_gap | PASS | 0.006936 | 100 | MEASURED |  |
| fairness.selection_rate_ratio | PASS | 0.9946 | 100 | MEASURED | yes |
| fairness.equalized_odds | PASS | 0.01159 | 100 | MEASURED |  |
| behaviour.hidden_failure_groups | PASS | 0 | 100 | MEASURED |  |
| calibration.ece | PASS | 0.01604 | 97 | MEASURED |  |
| operational.model_loadable | PASS | yes | 100 | MEASURED | yes |
| operational.input_schema | PASS | yes | 100 | MEASURED | yes |
| operational.prediction_determinism | PASS | yes | 100 | MEASURED |  |
| operational.malformed_input_handling | PASS | yes | 100 | MEASURED |  |
| operational.artifact_size | PASS | 0.003709 | 100 | MEASURED |  |
| operational.latency | PASS | 4.028 | 100 | MEASURED |  |
| operational.memory | PASS | 0 | 100 | MEASURED |  |
| operational.dependency_match | PASS | 1 | 100 | DECLARED |  |

### Check details

* **`training_validity.metadata_completeness`** — PASS: 10 of 10 required training-metadata fields are present.
* **`training_validity.reproducibility_info`** — PASS: 5 of 5 reproducibility fields are present.
* **`training_validity.overfit_gap`** — PASS: Declared training accuracy 0.702 vs measured evaluation accuracy 0.712: gap -0.011. The training value is the producer's claim and was not verified.
* **`training_validity.train_eval_overlap`** — PASS: 0 of 3000 evaluation rows (0.00%) also appear in the training row hashes. Datasets can contain genuine duplicate records (0.00% of evaluation rows duplicate another evaluation row), so a small overlap is not by itself leakage. The hashes are producer-supplied.
* **`leakage.single_feature_predictiveness`** — PASS: The strongest single feature, 'income_k', reaches a cross-validated ROC AUC of 0.705 on its own (6 features, 3000 rows). Values near 1.0 suggest the feature leaks the label.
* **`behaviour.f1`** — PASS: F1 is 0.733 on 3000 evaluation rows (an always-positive baseline scores 0.686).
* **`behaviour.roc_auc`** — WARN: ROC AUC is 0.786 on 3000 evaluation rows (0.5 is chance).
* **`behaviour.precision`** — PASS: Precision is 0.712 on 3000 evaluation rows (an always-positive baseline scores 0.522).
* **`behaviour.recall`** — PASS: Recall is 0.755 on 3000 evaluation rows.
* **`behaviour.subgroup_f1_gap`** — PASS: Largest subgroup F1 gap is 0.007 for 'sex': Male 0.729 vs Female 0.736.
* **`fairness.selection_rate_ratio`** — PASS: Lowest/highest group selection rate is 0.995 for 'sex': Female 55.2% vs Male 55.5%.
* **`fairness.equalized_odds`** — PASS: Equalized-odds difference is 0.012 for 'sex' (true-positive-rate gap 0.007, false-positive-rate gap 0.012).
* **`behaviour.hidden_failure_groups`** — PASS: No feature-defined group had significantly more errors than the model expects (13 groups tested on 3000 rows with tree+kmeans; effect floor 10%, false discovery rate 0.05). This does not prove none exists.
* **`calibration.ece`** — PASS: Expected calibration error is 0.016 (Brier 0.189, 10 bins, 3000 rows). Average predicted probability 0.537 vs observed positive rate 0.522: over-confident on average (predicts higher than observed).
* **`operational.model_loadable`** — PASS: Model loaded as Pipeline.
* **`operational.input_schema`** — PASS: Model predicted on 1000 evaluation rows using 6 feature columns.
* **`operational.prediction_determinism`** — PASS: Repeated predictions on the same rows were identical.
* **`operational.malformed_input_handling`** — PASS: All 3 malformed-input cases were handled.
* **`operational.artifact_size`** — PASS: Model file is 3.8 KB.
* **`operational.latency`** — PASS: Single-row prediction p95 latency is 4.0 ms (p50 2.1 ms) over 50 calls.
* **`operational.memory`** — PASS: Loading the model and predicting on a small sample grew memory by 0 MB.
* **`operational.dependency_match`** — PASS: 5 of 5 declared library versions match the current environment.

*Evidence: MEASURED = computed by the gate from the model and data; DECLARED = taken from producer metadata, not verified; SIMULATED = injected for a demonstration; UNAVAILABLE = none.*

## Inputs and audit trail

| Item | Value |
|---|---|
| Run id | sample-hidden_group_null-hidden_groups |
| Created (UTC) | 2026-01-01T00:00:00+00:00 |
| Gate version | 0.1.0 |
| Schema version | 0.1.0 |
| Candidate | hidden_group_null |
| Model file | C:\Users\ASUS\OneDrive\Documents\GitHub\MLOpsVision\components\C3-Model-Validation\fixtures\hidden_group_null\model.joblib |
| Model SHA-256 | df2b82e5331e542e699b9dd9f2b0ecb0c99d68e0b720d87a79a280859c204353 |
| Evaluation data SHA-256 | 292f7c5200a84e7dce4af9f42a529fb3887ac1fdcbddbbb7f306d2c31ce1d273 |
| Metadata SHA-256 | 24154f3930c93f4f4180a86f5e55cf320eafd33d548155e6d09bbb1e52c4aa4c |
| Configuration profile | hidden_groups |
| Configuration hash | 44de7f652ed422fbef22b58409204613951253118dc93161123b8a01d5f51d97 |
| Machine | Windows-11-10.0.26200-SP0; 12 CPUs; 15.6 GB RAM |
| Libraries | python 3.12.1, scikit_learn 1.9.1, pandas 3.0.6, numpy 2.5.4, joblib 1.6.0 |
| Stage timings (s) | load 0.04, training_validity 0.16, model_behaviour 0.08, operational_readiness 0.18, score_and_decide 0.00 |

## Configuration used

* Profile `hidden_groups`: Default profile with Hidden Failure Group Detection; a confirmed group puts the model on HOLD.
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
