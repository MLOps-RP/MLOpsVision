# Model Quality Validation Report: hidden_group

## Decision: HOLD

> HOLD: the model needs review before promotion. MQS 88.3. 1 issue(s): Hidden failure group found.

*This is a recommendation. Nothing was deployed, promoted or rolled back.*

## Scores (0–100)

**Model Quality Score (MQS): 88.3**  
Worst case if every missing check had failed: 88.3  
Overall evidence coverage: 100%  
Missing-evidence policy: `exclude_and_cap`

| Dimension | Score | Weight | Evidence | Checks with evidence | Provisional |
|---|---|---|---|---|---|
| Training Validity (TVS) | 100.0 | 0.30 | 100% | 5/5 | no |
| Model Behaviour (MBS) | 76.6 | 0.50 | 100% | 9/9 | no |
| Operational Readiness (ORS) | 100.0 | 0.20 | 100% | 8/8 | no |

`MQS = TVS × W1 + MBS × W2 + ORS × W3`

**Calculation:** MQS = TVS 100.0 × 0.30 + MBS 76.6 × 0.50 + ORS 100.0 × 0.20 = 88.3


### The three layers at a glance

| Layer | Score | Checks | Needs attention |
|---|---|---|---|
| Training Validity (TVS) | 100.0 | 5 pass · 0 warn · 0 fail · 0 no evidence | none |
| Model Behaviour (MBS) | 76.6 | 7 pass · 1 warn · 1 fail · 0 no evidence | behaviour.roc_auc, behaviour.hidden_failure_groups |
| Operational Readiness (ORS) | 100.0 | 8 pass · 0 warn · 0 fail · 0 no evidence | none |

Subgroup detection: **6 confirmed** of 15 candidate groups tested on held-out rows.

## Why this decision

| Rule | Kind | Effect | Reason |
|---|---|---|---|
| Hidden failure group found | blocking | HOLD | 6 hidden failure group(s) found. The worst, 'segment = d AND tenure_years > 6.57 AND income_k > 57' (1.1% of rows), has an observed error rate of 76.5% against 27.0% expected (+49.5% points, 95% CI +33.0% to +60.6%, adjusted p 5.2e-10, found by tree and tested on held-out rows). Policy hidden_groups.consequence is 'hold': a confirmed group blocks promotion. |

### Explanation

* Hidden failure group (tree): segment = d AND tenure_years > 6.57 AND income_k > 57 — 34 held-out rows, observed error 76.5% vs 27.0% expected; 6 confirmed group(s) in total.
* Hidden failure group policy (configured): consequence 'hold', assessment required. A confirmed group blocks promotion.
* Weakest checks (MQS points lost): behaviour.hidden_failure_groups (-5.7), behaviour.roc_auc (-3.4), behaviour.f1 (-1.0), behaviour.precision (-0.8), behaviour.recall (-0.4).
* Evidence note: these results rely on declared (unverified) producer metadata: training_validity.overfit_gap, training_validity.train_eval_overlap, operational.dependency_match.
* Suggested action (behaviour.roc_auc): Improve the model's ranking quality (features, model family).
* Suggested action (behaviour.hidden_failure_groups): Investigate the reported group: check its data volume and label quality, add or engineer features that separate it, or retrain with more examples from it.
* These suggestions are starting points, not guarantees.

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

### Model Behaviour (MBS): 76.6

| Check | Status | Score | Weight | Points | Treatment |
|---|---|---|---|---|---|
| behaviour.f1 | PASS | 87 | 2 | 13.37 | counted |
| behaviour.roc_auc | WARN | 55 | 2 | 8.50 | counted |
| behaviour.precision | PASS | 79 | 1 | 6.09 | counted |
| behaviour.recall | PASS | 90 | 1 | 6.91 | counted |
| behaviour.subgroup_f1_gap | PASS | 100 | 1 | 7.69 | counted |
| fairness.selection_rate_ratio | PASS | 100 | 2 | 15.38 | counted |
| fairness.equalized_odds | PASS | 100 | 1 | 7.69 | counted |
| calibration.ece | PASS | 94 | 1.5 | 10.82 | counted |
| behaviour.hidden_failure_groups | FAIL | 1 | 1.5 | 0.12 | counted |

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

**6 confirmed** of 15 groups tested (methods: tree, kmeans; false discovery rate 0.05; minimum effect 10%). Confirmed groups cover 11.9% of rows.

| Group | Method | Rows | Share | Observed error | Expected error | Excess (95% CI) | Adjusted p | Confirmed |
|---|---|---|---|---|---|---|---|---|
| segment = d AND tenure_years > 6.57 AND income_k > 57 | tree | 34 | 1.1% | 76.5% | 27.0% | +49.5% [+33.0%, +60.6%] | 5.2e-10 | yes |
| segment = d AND tenure_years > 6.23 AND income_k > 53.6 | tree | 62 | 2.1% | 74.2% | 33.8% | +40.4% [+28.3%, +49.6%] | 1.1e-10 | yes |
| segment = d AND tenure_years > 6.57 AND income_k <= 57 | tree | 72 | 2.4% | 63.9% | 39.4% | +24.5% [+12.9%, +34.6%] | 6.2e-05 | yes |
| cluster of similar rows: region = w for 100% of rows (overall 24%); segment = d for 100% of rows (overall 25%); tenure_years averages 6.32 (overall 4.98) | kmeans | 82 | 2.7% | 51.2% | 33.0% | +18.2% [+7.6%, +28.7%] | 8.5e-04 | yes |
| cluster of similar rows: segment = d for 100% of rows (overall 23%); region = s for 67% of rows (overall 24%); channel = u for 6% of rows (overall 51%) | kmeans | 92 | 3.1% | 47.8% | 33.1% | +14.7% [+4.8%, +24.8%] | 4.2e-03 | yes |
| cluster of similar rows: segment = d for 100% of rows (overall 25%); region = s for 100% of rows (overall 26%); tenure_years averages 5.6 (overall 4.98) | kmeans | 81 | 2.7% | 42.0% | 31.0% | +10.9% [+0.8%, +21.8%] | 3.8e-02 | yes |
| cluster of similar rows: region = n for 100% of rows (overall 24%); segment = d for 100% of rows (overall 25%); tenure_years averages 5.71 (overall 4.98) | kmeans | 103 | 3.4% | 38.8% | 29.2% | +9.6% [+0.8%, +19.3%] | 3.8e-02 | no |
| segment = d AND tenure_years <= 6.23 AND income_k <= 64 | tree | 183 | 6.1% | 33.3% | 27.2% | +6.1% [-0.3%, +13.2%] | 6.1e-02 | no |
| segment = d AND tenure_years > 6.23 AND income_k <= 53.6 | tree | 63 | 2.1% | 46.0% | 36.4% | +9.6% [-2.1%, +21.8%] | 1.1e-01 | no |
| cluster of similar rows: region = e for 100% of rows (overall 26%); segment = d for 52% of rows (overall 25%); tenure_years averages 5.84 (overall 4.98) | kmeans | 188 | 6.3% | 33.0% | 32.8% | +0.1% [-6.2%, +7.1%] | 7.3e-01 | no |

*Showing 10 of 15 groups tested.*

Calibration safeguard: ECE 0.025, observed-versus-expected error gap 0.008. Probabilities were recalibrated (Platt map, discovery rows only) on 1 of 2 folds.

Features searched: age, income_k, tenure_years, segment, region, channel.

*Caveats: Expected errors come from the model's own probabilities; the test is only valid if they are calibrated. They are recalibrated per fold on the discovery rows only; grossly wrong probabilities make the check unavailable. Groups are axis-aligned rules or clusters of similar rows, so an oblique or very small region may be missed or found only in part. A group is an association found in the evaluation data, not a causal explanation: it says where the model fails, not why. Features that describe a group may be proxies for sensitive characteristics (for example a relationship status standing in for sex), so the rule can reveal a protected subpopulation even though no sensitive column was named. Detection power depends on the sample size; no group found does not mean none exists.*

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
| behaviour.hidden_failure_groups | FAIL | 0.4948 | 1 | MEASURED |  |
| calibration.ece | PASS | 0.02498 | 94 | MEASURED |  |
| operational.model_loadable | PASS | yes | 100 | MEASURED | yes |
| operational.input_schema | PASS | yes | 100 | MEASURED | yes |
| operational.prediction_determinism | PASS | yes | 100 | MEASURED |  |
| operational.malformed_input_handling | PASS | yes | 100 | MEASURED |  |
| operational.artifact_size | PASS | 0.003709 | 100 | MEASURED |  |
| operational.latency | PASS | 4.071 | 100 | MEASURED |  |
| operational.memory | PASS | 0 | 100 | MEASURED |  |
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
* **`behaviour.hidden_failure_groups`** — FAIL: 6 hidden failure group(s) found. The worst, 'segment = d AND tenure_years > 6.57 AND income_k > 57' (1.1% of rows), has an observed error rate of 76.5% against 27.0% expected (+49.5% points, 95% CI +33.0% to +60.6%, adjusted p 5.2e-10, found by tree and tested on held-out rows).
* **`calibration.ece`** — PASS: Expected calibration error is 0.025 (Brier 0.204, 10 bins, 3000 rows). Average predicted probability 0.519 vs observed positive rate 0.498: over-confident on average (predicts higher than observed).
* **`operational.model_loadable`** — PASS: Model loaded as Pipeline.
* **`operational.input_schema`** — PASS: Model predicted on 1000 evaluation rows using 6 feature columns.
* **`operational.prediction_determinism`** — PASS: Repeated predictions on the same rows were identical.
* **`operational.malformed_input_handling`** — PASS: All 3 malformed-input cases were handled.
* **`operational.artifact_size`** — PASS: Model file is 3.8 KB.
* **`operational.latency`** — PASS: Single-row prediction p95 latency is 4.1 ms (p50 2.0 ms) over 50 calls.
* **`operational.memory`** — PASS: Loading the model and predicting on a small sample grew memory by 0 MB.
* **`operational.dependency_match`** — PASS: 5 of 5 declared library versions match the current environment.

*Evidence: MEASURED = computed by the gate from the model and data; DECLARED = taken from producer metadata, not verified; SIMULATED = injected for a demonstration; UNAVAILABLE = none.*

## Inputs and audit trail

| Item | Value |
|---|---|
| Run id | sample-hidden_group-hidden_groups |
| Created (UTC) | 2026-01-01T00:00:00+00:00 |
| Gate version | 0.1.0 |
| Schema version | 0.1.0 |
| Candidate | hidden_group |
| Model file | C:\Users\ASUS\OneDrive\Documents\GitHub\MLOpsVision\components\C3-Model-Validation\fixtures\hidden_group\model.joblib |
| Model SHA-256 | d8ba43f61e6789f5c0e931e6255c9f10b6bbcc3303c1e9c5ceeaca4706f6a58b |
| Evaluation data SHA-256 | 2f0944999996f28e9f36a49ebc65a3339d397ae87c88ba63ae7652fc96cc7801 |
| Metadata SHA-256 | 7f41938747f6bd46c7874b7d7550f938414cc34200575f80477a802d81b0bcc2 |
| Configuration profile | hidden_groups |
| Configuration hash | 44de7f652ed422fbef22b58409204613951253118dc93161123b8a01d5f51d97 |
| Machine | Windows-11-10.0.26200-SP0; 12 CPUs; 15.6 GB RAM |
| Libraries | python 3.12.1, scikit_learn 1.9.1, pandas 3.0.6, numpy 2.5.4, joblib 1.6.0 |
| Stage timings (s) | load 0.04, training_validity 0.13, model_behaviour 0.08, operational_readiness 0.17, score_and_decide 0.00 |

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
