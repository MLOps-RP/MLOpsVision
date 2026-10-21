# Model Quality Validation Report: good

## Decision: PROMOTE

> PROMOTE recommended: MQS 86.9 meets the promote threshold of 75; no critical rule fired, required evidence is present and every must-pass check passed.

*This is a recommendation. Nothing was deployed, promoted or rolled back.*

## Scores (0–100)

**Model Quality Score (MQS): 86.9**  
Worst case if every missing check had failed: 86.9  
Overall evidence coverage: 100%  
Missing-evidence policy: `exclude_and_cap`

| Dimension | Score | Weight | Evidence | Checks with evidence | Provisional |
|---|---|---|---|---|---|
| Training Validity (TVS) | 99.9 | 0.30 | 100% | 7/7 | no |
| Model Behaviour (MBS) | 73.8 | 0.50 | 100% | 8/8 | no |
| Operational Readiness (ORS) | 100.0 | 0.20 | 100% | 10/10 | no |

`MQS = TVS × W1 + MBS × W2 + ORS × W3`

**Calculation:** MQS = TVS 99.9 × 0.30 + MBS 73.8 × 0.50 + ORS 100.0 × 0.20 = 86.9


### The three layers at a glance

| Layer | Score | Checks | Needs attention |
|---|---|---|---|
| Training Validity (TVS) | 99.9 | 7 pass · 0 warn · 0 fail · 0 no evidence | none |
| Model Behaviour (MBS) | 73.8 | 5 pass · 3 warn · 0 fail · 0 no evidence | behaviour.subgroup_f1_gap, fairness.selection_rate_ratio, fairness.equalized_odds |
| Operational Readiness (ORS) | 100.0 | 10 pass · 0 warn · 0 fail · 0 no evidence | none |

Subgroup detection (hidden failure analysis) is switched off in this profile.

## Why this decision

| Rule | Kind | Effect | Reason |
|---|---|---|---|
| MQS meets the promotion threshold | score | PROMOTE | MQS 86.9 is at or above the promotion threshold of 75. |

### Explanation

* Watch: these checks warn but do not block promotion: behaviour.subgroup_f1_gap, fairness.selection_rate_ratio, fairness.equalized_odds.
* Weakest checks (MQS points lost): fairness.selection_rate_ratio (-5.7), fairness.equalized_odds (-1.8), behaviour.subgroup_f1_gap (-1.6), behaviour.f1 (-1.5), behaviour.recall (-1.1).
* Evidence note: these results rely on declared (unverified) producer metadata: training_validity.overfit_gap, training_validity.train_eval_overlap, training_validity.train_duplicate_rows, operational.dependency_match.

## Score breakdown

Each check's points add up to its dimension's score.

### Training Validity (TVS): 99.9

| Check | Status | Score | Weight | Points | Treatment |
|---|---|---|---|---|---|
| training_validity.metadata_completeness | PASS | 100 | 1 | 10.00 | counted |
| training_validity.reproducibility_info | PASS | 100 | 1 | 10.00 | counted |
| training_validity.overfit_gap | PASS | 100 | 2 | 20.00 | counted |
| training_validity.train_eval_overlap | PASS | 99 | 1 | 9.94 | counted |
| training_validity.eval_label_conflicts | PASS | 100 | 1 | 10.00 | counted |
| training_validity.train_duplicate_rows | PASS | 100 | 1 | 9.98 | counted |
| leakage.single_feature_predictiveness | PASS | 100 | 3 | 30.00 | counted |

### Model Behaviour (MBS): 73.8

| Check | Status | Score | Weight | Points | Treatment |
|---|---|---|---|---|---|
| behaviour.f1 | PASS | 83 | 2 | 14.37 | counted |
| behaviour.roc_auc | PASS | 90 | 2 | 15.69 | counted |
| behaviour.precision | PASS | 87 | 1 | 7.59 | counted |
| behaviour.recall | PASS | 75 | 1 | 6.48 | counted |
| behaviour.subgroup_f1_gap | WARN | 64 | 1 | 5.56 | counted |
| fairness.selection_rate_ratio | WARN | 35 | 2 | 6.02 | counted |
| fairness.equalized_odds | WARN | 59 | 1 | 5.09 | counted |
| calibration.ece | PASS | 100 | 1.5 | 13.01 | counted |

### Operational Readiness (ORS): 100.0

| Check | Status | Score | Weight | Points | Treatment |
|---|---|---|---|---|---|
| operational.model_loadable | PASS | 100 | 3 | 20.00 | counted |
| operational.input_schema | PASS | 100 | 2 | 13.33 | counted |
| operational.prediction_determinism | PASS | 100 | 1 | 6.67 | counted |
| operational.malformed_input_handling | PASS | 100 | 1 | 6.67 | counted |
| operational.artifact_size | PASS | 100 | 1 | 6.67 | counted |
| operational.latency | PASS | 100 | 2 | 13.33 | counted |
| operational.memory | PASS | 100 | 1 | 6.67 | counted |
| operational.dependency_match | PASS | 100 | 1 | 6.67 | counted |
| operational.model_file_safety | PASS | 100 | 2 | 13.33 | counted |
| operational.cpu_time | PASS | 100 | 1 | 6.67 | counted |

## Evidence gaps

No gaps: every check had evidence.

## All checks

| Check | Status | Value | Score | Evidence | Required |
|---|---|---|---|---|---|
| training_validity.metadata_completeness | PASS | 1 | 100 | MEASURED |  |
| training_validity.reproducibility_info | PASS | 1 | 100 | MEASURED |  |
| training_validity.overfit_gap | PASS | -0.003749 | 100 | DECLARED |  |
| training_validity.train_eval_overlap | PASS | 0.001229 | 99 | DECLARED |  |
| leakage.single_feature_predictiveness | PASS | 0.7724 | 100 | MEASURED | yes |
| training_validity.eval_label_conflicts | PASS | 0 | 100 | MEASURED |  |
| training_validity.train_duplicate_rows | PASS | 0.0005801 | 100 | DECLARED |  |
| behaviour.f1 | PASS | 0.6611 | 83 | MEASURED | yes |
| behaviour.roc_auc | PASS | 0.9061 | 90 | MEASURED | yes |
| behaviour.precision | PASS | 0.7421 | 87 | MEASURED |  |
| behaviour.recall | PASS | 0.5961 | 75 | MEASURED |  |
| behaviour.subgroup_f1_gap | WARN | 0.1929 | 64 | MEASURED |  |
| fairness.selection_rate_ratio | WARN | 0.3113 | 35 | MEASURED | yes |
| fairness.equalized_odds | WARN | 0.2188 | 59 | MEASURED |  |
| calibration.ece | PASS | 0.01069 | 100 | MEASURED |  |
| operational.model_loadable | PASS | yes | 100 | MEASURED | yes |
| operational.input_schema | PASS | yes | 100 | MEASURED | yes |
| operational.prediction_determinism | PASS | yes | 100 | MEASURED |  |
| operational.malformed_input_handling | PASS | yes | 100 | MEASURED |  |
| operational.artifact_size | PASS | 0.007257 | 100 | MEASURED |  |
| operational.latency | PASS | 4.131 | 100 | MEASURED |  |
| operational.memory | PASS | 0.8906 | 100 | MEASURED |  |
| operational.dependency_match | PASS | 1 | 100 | DECLARED |  |
| operational.model_file_safety | PASS | 0 | 100 | MEASURED | yes |
| operational.cpu_time | PASS | 0.003801 | 100 | MEASURED |  |

### Check details

* **`training_validity.metadata_completeness`** — PASS: 10 of 10 required training-metadata fields are present.
* **`training_validity.reproducibility_info`** — PASS: 5 of 5 reproducibility fields are present.
* **`training_validity.overfit_gap`** — PASS: Declared training accuracy 0.850 vs measured evaluation accuracy 0.854: gap -0.004. The training value is the producer's claim and was not verified.
* **`training_validity.train_eval_overlap`** — PASS: 12 of 9768 evaluation rows (0.12%) also appear in the training row hashes. Datasets can contain genuine duplicate records (0.03% of evaluation rows duplicate another evaluation row), so a small overlap is not by itself leakage. The hashes are producer-supplied.
* **`leakage.single_feature_predictiveness`** — PASS: The strongest single feature, 'relationship', reaches a cross-validated ROC AUC of 0.772 on its own (12 features, 9768 rows). Values near 1.0 suggest the feature leaks the label.
* **`training_validity.eval_label_conflicts`** — PASS: 0 of 9768 evaluation rows (0.00%) share their feature values with a row that has a different label (0 conflicting groups among 3 groups of repeated feature values). Conflicting labels put a ceiling on the accuracy any model can reach and make the measured metrics unreliable.
* **`training_validity.train_duplicate_rows`** — PASS: 17 of 29305 training rows (0.06%) duplicate another training row. Duplicates can make a model look better on rows it has memorised and are often a data-quality problem upstream. The hashes are producer-supplied and cover the feature columns only, so rows with the same features and different labels also count as duplicates.
* **`behaviour.f1`** — PASS: F1 is 0.661 on 9768 evaluation rows (an always-positive baseline scores 0.386).
* **`behaviour.roc_auc`** — PASS: ROC AUC is 0.906 on 9768 evaluation rows (0.5 is chance).
* **`behaviour.precision`** — PASS: Precision is 0.742 on 9768 evaluation rows (an always-positive baseline scores 0.239).
* **`behaviour.recall`** — PASS: Recall is 0.596 on 9768 evaluation rows.
* **`behaviour.subgroup_f1_gap`** — WARN: Largest subgroup F1 gap is 0.193 for 'race': Amer-Indian-Eskimo 0.476 vs White 0.669.
* **`fairness.selection_rate_ratio`** — WARN: Lowest/highest group selection rate is 0.311 for 'sex': Female 7.8% vs Male 24.9%.
* **`fairness.equalized_odds`** — WARN: Equalized-odds difference is 0.219 for 'race' (true-positive-rate gap 0.219, false-positive-rate gap 0.077).
* **`calibration.ece`** — PASS: Expected calibration error is 0.011 (Brier 0.102, 10 bins, 9768 rows). Average predicted probability 0.239 vs observed positive rate 0.239: close to the observed rate on average.
* **`operational.model_loadable`** — PASS: Model loaded as Pipeline.
* **`operational.input_schema`** — PASS: Model predicted on 1000 evaluation rows using 12 feature columns.
* **`operational.prediction_determinism`** — PASS: Repeated predictions on the same rows were identical.
* **`operational.malformed_input_handling`** — PASS: All 3 malformed-input cases were handled.
* **`operational.artifact_size`** — PASS: Model file is 7.4 KB.
* **`operational.latency`** — PASS: Single-row prediction p95 latency is 4.1 ms (p50 3.1 ms) over 50 calls.
* **`operational.memory`** — PASS: Loading the model and predicting on a small sample grew memory by 1 MB.
* **`operational.dependency_match`** — PASS: 5 of 5 declared library versions match the current environment.
* **`operational.model_file_safety`** — PASS: The scan found 13 code references in the model file, all on the allowlist. This is a static scan, not a sandbox: it does not prove the file is safe, and only trusted files should be validated.
* **`operational.cpu_time`** — PASS: Predicting 74000 rows used 0.28 s of process CPU time over 0.31 s of wall time: 0.004 ms of CPU per row (CPU/wall ratio 0.92; above 1 means several cores were busy).

*Evidence: MEASURED = computed by the gate from the model and data; DECLARED = taken from producer metadata, not verified; SIMULATED = injected for a demonstration; UNAVAILABLE = none.*

## Inputs and audit trail

| Item | Value |
|---|---|
| Run id | b5de76cda06f4fafb6f7d80f8d46df6b |
| Created (UTC) | 2026-10-20T06:47:44.929177+00:00 |
| Gate version | 0.1.0 |
| Schema version | 0.1.0 |
| Candidate | good |
| Model file | C:\artifacts\fixtures\good\model.joblib |
| Model SHA-256 | 02cae497c6e22a9afce206ed0d39c4780bc651f96e290d5791fdae3d13471d97 |
| Evaluation data SHA-256 | bba157c9f381e730133239c0f78c5cf8c98f3b16f364eaac58fc54ebc0862722 |
| Metadata SHA-256 | 92efdaa2fb71c693680785b443f398493b19e344847616a3c001a54f9a65b46e |
| Configuration profile | extended |
| Configuration hash | e5cd88eaccee847a641f880954e58a1a4d203197483052cf7df2781130923bc7 |
| Machine | Windows-11-10.0.26200-SP0; 12 CPUs; 15.6 GB RAM |
| Libraries | python 3.12.1, scikit_learn 1.9.1, pandas 3.0.6, numpy 2.5.4, joblib 1.6.0 |
| Stage timings (s) | load 0.12, training_validity 0.40, model_behaviour 0.01, operational_readiness 0.57, score_and_decide 0.00 |

## Configuration used

* Profile `extended`: Default checks plus conflicting labels, training duplicates, model-file safety and CPU time.
* Dimension weights: {'training_validity': 0.3, 'model_behaviour': 0.5, 'operational_readiness': 0.2}
* Promote at MQS ≥ 75.0; roll back below 50.0
* Must-pass checks: training_validity.metadata_completeness, training_validity.overfit_gap, calibration.ece, operational.latency, training_validity.eval_label_conflicts, operational.model_file_safety
* Minimum evidence coverage: {'min_dimension': 0.5, 'min_overall': 0.8}
* Critical rules: Model artifact cannot be loaded, Target leakage detected, Severe selection-rate disparity between groups, F1 below the absolute minimum, Model file references dangerous code

The thresholds and weights are configuration choices. They need justification for the domain; they are not universal standards.

## Limitations

* This report is a recommendation; no action was taken.
* DECLARED evidence comes from producer metadata and is not verified.
* The gate looks for signs of leakage, overfitting and bias; it cannot prove their absence.
* Only binary classification on tabular data is evaluated (see `docs/supported_models.md`).
* Loading a model file executes code stored in it; validate only models from trusted sources.
