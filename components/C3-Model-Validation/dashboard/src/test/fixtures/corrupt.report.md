# Model Quality Validation Report: corrupt

## Decision: ROLLBACK

> ROLLBACK recommended: a critical rule overrides the score. MQS 99.9 (not reliable: no score for model behaviour and operational readiness; overall evidence 31% is below the required 80%; at worst 24.0 if missing checks had failed). Reason(s): Model artifact cannot be loaded.

*This is a recommendation. Nothing was deployed, promoted or rolled back.*

**The MQS is not reliable:** no score for model behaviour and operational readiness; overall evidence 31% is below the required 80%.

## Scores (0–100)

**Model Quality Score (MQS): n/a (insufficient evidence; recorded value 99.9)**  
Worst case if every missing check had failed: 24.0  
Overall evidence coverage: 31%  
Missing-evidence policy: `exclude_and_cap`

| Dimension | Score | Weight | Evidence | Checks with evidence | Provisional |
|---|---|---|---|---|---|
| Training Validity (TVS) | 99.9 | 0.30 | 80% | 6/7 | yes |
| Model Behaviour (MBS) | — | 0.50 | 0% | 0/8 | yes |
| Operational Readiness (ORS) | — | 0.20 | 33% | 3/10 | yes |

`MQS = TVS × W1 + MBS × W2 + ORS × W3` (weights renormalized over the scored dimensions: Training Validity 1.00, Model Behaviour 0.00, Operational Readiness 0.00)

**Calculation:** MQS = TVS 99.9 × 1.00 = 99.9 (Model Behaviour, Operational Readiness: no score, left out)

* Training Validity (TVS): Provisional: 80% of the applicable check weight had evidence (exclude_and_cap).
* Model Behaviour (MBS): Evidence coverage 0% is below the minimum 50%, so this dimension has no score.
* Operational Readiness (ORS): Evidence coverage 33% is below the minimum 50%, so this dimension has no score.

### The three layers at a glance

| Layer | Score | Checks | Needs attention |
|---|---|---|---|
| Training Validity (TVS) | 99.9 | 6 pass · 0 warn · 0 fail · 1 no evidence | training_validity.overfit_gap |
| Model Behaviour (MBS) | — | 0 pass · 0 warn · 0 fail · 8 no evidence | behaviour.f1, behaviour.roc_auc, behaviour.precision, behaviour.recall, behaviour.subgroup_f1_gap, fairness.selection_rate_ratio (+2 more) |
| Operational Readiness (ORS) | — | 2 pass · 0 warn · 1 fail · 7 no evidence | operational.model_loadable, operational.input_schema, operational.prediction_determinism, operational.malformed_input_handling, operational.latency, operational.memory (+2 more) |

Subgroup detection (hidden failure analysis) is switched off in this profile.

## Why this decision

**A critical rule overrode the score.**

| Rule | Kind | Effect | Reason |
|---|---|---|---|
| Model artifact cannot be loaded | critical | ROLLBACK | Model artifact cannot be loaded: check 'operational.model_loadable' is FAIL (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: )). |
| Required evidence is missing | evidence | HOLD | Required checks have no evidence: behaviour.f1, behaviour.roc_auc, fairness.selection_rate_ratio, operational.input_schema, operational.model_file_safety. |
| Evidence coverage is too low | evidence | HOLD | Overall evidence coverage 31% is below the required 80%. |
| Must-pass check 'training_validity.overfit_gap' did not pass | blocking | HOLD | Check 'training_validity.overfit_gap' is UNAVAILABLE; it must PASS before promotion (Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))). |
| Must-pass check 'calibration.ece' did not pass | blocking | HOLD | Check 'calibration.ece' is UNAVAILABLE; it must PASS before promotion (Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))). |
| Must-pass check 'operational.latency' did not pass | blocking | HOLD | Check 'operational.latency' is UNAVAILABLE; it must PASS before promotion (Cannot test: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))). |
| Must-pass check 'operational.model_file_safety' did not pass | blocking | HOLD | Check 'operational.model_file_safety' is UNAVAILABLE; it must PASS before promotion (The model file could not be scanned: not a readable pickle stream: EOFError: . It is not counted as safe.). |

### Explanation

* Override: critical rule(s) force ROLLBACK (Model artifact cannot be loaded) although the MQS of 99.9 is above the promote threshold of 75. A good score cannot outweigh a critical failure.
* Weakest checks (MQS points lost): training_validity.train_eval_overlap (-0.1).
* Evidence gap (not counted as passing): training_validity.overfit_gap is UNAVAILABLE. Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* Evidence gap (not counted as passing): behaviour.f1 is UNAVAILABLE. Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* Evidence gap (not counted as passing): behaviour.roc_auc is UNAVAILABLE. Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* Evidence gap (not counted as passing): behaviour.precision is UNAVAILABLE. Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* Evidence gap (not counted as passing): behaviour.recall is UNAVAILABLE. Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* Evidence gap (not counted as passing): behaviour.subgroup_f1_gap is UNAVAILABLE. Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* Evidence gap (not counted as passing): fairness.selection_rate_ratio is UNAVAILABLE. Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* Evidence gap (not counted as passing): fairness.equalized_odds is UNAVAILABLE. Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* Evidence gap (not counted as passing): calibration.ece is UNAVAILABLE. Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* Evidence gap (not counted as passing): operational.input_schema is UNAVAILABLE. Cannot test: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* Evidence gap (not counted as passing): operational.prediction_determinism is UNAVAILABLE. Cannot test: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* Evidence gap (not counted as passing): operational.malformed_input_handling is UNAVAILABLE. Cannot test: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* Evidence gap (not counted as passing): operational.latency is UNAVAILABLE. Cannot test: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* Evidence gap (not counted as passing): operational.memory is UNAVAILABLE. Model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: )).
* Evidence gap (not counted as passing): operational.model_file_safety is UNAVAILABLE. The model file could not be scanned: not a readable pickle stream: EOFError: . It is not counted as safe.
* Evidence gap (not counted as passing): operational.cpu_time is UNAVAILABLE. Cannot test: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* Evidence note: these results rely on declared (unverified) producer metadata: training_validity.train_eval_overlap, training_validity.train_duplicate_rows, operational.dependency_match.
* Evidence note: the MQS is not reliable: no score for model behaviour and operational readiness; overall evidence 31% is below the required 80%.
* Suggested action (operational.model_loadable): Check the model path, and that it was saved with joblib or pickle in a compatible environment.
* Suggested action (required_evidence_missing): Supply the missing inputs (metadata, target and positive label, sensitive columns) so the required checks can run.
* Suggested action (overall_coverage_below_minimum): Supply more of the inputs the checks need; coverage rises as checks gain evidence.
* These suggestions are starting points, not guarantees.

## Score breakdown

Each check's points add up to its dimension's score.

### Training Validity (TVS): 99.9

| Check | Status | Score | Weight | Points | Treatment |
|---|---|---|---|---|---|
| training_validity.metadata_completeness | PASS | 100 | 1 | 12.50 | counted |
| training_validity.reproducibility_info | PASS | 100 | 1 | 12.50 | counted |
| training_validity.overfit_gap | UNAVAILABLE | — | 2 | 0.00 | excluded unavailable |
| training_validity.train_eval_overlap | PASS | 99 | 1 | 12.42 | counted |
| training_validity.eval_label_conflicts | PASS | 100 | 1 | 12.50 | counted |
| training_validity.train_duplicate_rows | PASS | 100 | 1 | 12.48 | counted |
| leakage.single_feature_predictiveness | PASS | 100 | 3 | 37.50 | counted |

### Model Behaviour (MBS): no score

| Check | Status | Score | Weight | Points | Treatment |
|---|---|---|---|---|---|
| behaviour.f1 | UNAVAILABLE | — | 2 | 0.00 | excluded unavailable |
| behaviour.roc_auc | UNAVAILABLE | — | 2 | 0.00 | excluded unavailable |
| behaviour.precision | UNAVAILABLE | — | 1 | 0.00 | excluded unavailable |
| behaviour.recall | UNAVAILABLE | — | 1 | 0.00 | excluded unavailable |
| behaviour.subgroup_f1_gap | UNAVAILABLE | — | 1 | 0.00 | excluded unavailable |
| fairness.selection_rate_ratio | UNAVAILABLE | — | 2 | 0.00 | excluded unavailable |
| fairness.equalized_odds | UNAVAILABLE | — | 1 | 0.00 | excluded unavailable |
| calibration.ece | UNAVAILABLE | — | 1.5 | 0.00 | excluded unavailable |

### Operational Readiness (ORS): no score

| Check | Status | Score | Weight | Points | Treatment |
|---|---|---|---|---|---|
| operational.model_loadable | FAIL | 0 | 3 | 0.00 | counted |
| operational.input_schema | UNAVAILABLE | — | 2 | 0.00 | excluded unavailable |
| operational.prediction_determinism | UNAVAILABLE | — | 1 | 0.00 | excluded unavailable |
| operational.malformed_input_handling | UNAVAILABLE | — | 1 | 0.00 | excluded unavailable |
| operational.artifact_size | PASS | 100 | 1 | 20.00 | counted |
| operational.latency | UNAVAILABLE | — | 2 | 0.00 | excluded unavailable |
| operational.memory | UNAVAILABLE | — | 1 | 0.00 | excluded unavailable |
| operational.dependency_match | PASS | 100 | 1 | 20.00 | counted |
| operational.model_file_safety | UNAVAILABLE | — | 2 | 0.00 | excluded unavailable |
| operational.cpu_time | UNAVAILABLE | — | 1 | 0.00 | excluded unavailable |

## Evidence gaps

Checks without evidence are never counted as passing.

| Check | Status | Reason |
|---|---|---|
| training_validity.overfit_gap | UNAVAILABLE | Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: )) |
| behaviour.f1 | UNAVAILABLE | Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: )) |
| behaviour.roc_auc | UNAVAILABLE | Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: )) |
| behaviour.precision | UNAVAILABLE | Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: )) |
| behaviour.recall | UNAVAILABLE | Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: )) |
| behaviour.subgroup_f1_gap | UNAVAILABLE | Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: )) |
| fairness.selection_rate_ratio | UNAVAILABLE | Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: )) |
| fairness.equalized_odds | UNAVAILABLE | Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: )) |
| calibration.ece | UNAVAILABLE | Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: )) |
| operational.input_schema | UNAVAILABLE | Cannot test: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: )) |
| operational.prediction_determinism | UNAVAILABLE | Cannot test: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: )) |
| operational.malformed_input_handling | UNAVAILABLE | Cannot test: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: )) |
| operational.latency | UNAVAILABLE | Cannot test: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: )) |
| operational.memory | UNAVAILABLE | Model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: )). |
| operational.model_file_safety | UNAVAILABLE | The model file could not be scanned: not a readable pickle stream: EOFError: . It is not counted as safe. |
| operational.cpu_time | UNAVAILABLE | Cannot test: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: )) |

## All checks

| Check | Status | Value | Score | Evidence | Required |
|---|---|---|---|---|---|
| training_validity.metadata_completeness | PASS | 1 | 100 | MEASURED |  |
| training_validity.reproducibility_info | PASS | 1 | 100 | MEASURED |  |
| training_validity.overfit_gap | UNAVAILABLE | — | — | UNAVAILABLE |  |
| training_validity.train_eval_overlap | PASS | 0.001229 | 99 | DECLARED |  |
| leakage.single_feature_predictiveness | PASS | 0.7724 | 100 | MEASURED | yes |
| training_validity.eval_label_conflicts | PASS | 0 | 100 | MEASURED |  |
| training_validity.train_duplicate_rows | PASS | 0.0005801 | 100 | DECLARED |  |
| behaviour.f1 | UNAVAILABLE | — | — | UNAVAILABLE | yes |
| behaviour.roc_auc | UNAVAILABLE | — | — | UNAVAILABLE | yes |
| behaviour.precision | UNAVAILABLE | — | — | UNAVAILABLE |  |
| behaviour.recall | UNAVAILABLE | — | — | UNAVAILABLE |  |
| behaviour.subgroup_f1_gap | UNAVAILABLE | — | — | UNAVAILABLE |  |
| fairness.selection_rate_ratio | UNAVAILABLE | — | — | UNAVAILABLE | yes |
| fairness.equalized_odds | UNAVAILABLE | — | — | UNAVAILABLE |  |
| calibration.ece | UNAVAILABLE | — | — | UNAVAILABLE |  |
| operational.model_loadable | FAIL | no | 0 | MEASURED | yes |
| operational.input_schema | UNAVAILABLE | — | — | UNAVAILABLE | yes |
| operational.prediction_determinism | UNAVAILABLE | — | — | UNAVAILABLE |  |
| operational.malformed_input_handling | UNAVAILABLE | — | — | UNAVAILABLE |  |
| operational.artifact_size | PASS | 0.002903 | 100 | MEASURED |  |
| operational.latency | UNAVAILABLE | — | — | UNAVAILABLE |  |
| operational.memory | UNAVAILABLE | — | — | UNAVAILABLE |  |
| operational.dependency_match | PASS | 1 | 100 | DECLARED |  |
| operational.model_file_safety | UNAVAILABLE | — | — | UNAVAILABLE | yes |
| operational.cpu_time | UNAVAILABLE | — | — | UNAVAILABLE |  |

### Check details

* **`training_validity.metadata_completeness`** — PASS: 10 of 10 required training-metadata fields are present.
* **`training_validity.reproducibility_info`** — PASS: 5 of 5 reproducibility fields are present.
* **`training_validity.overfit_gap`** — UNAVAILABLE: Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* **`training_validity.train_eval_overlap`** — PASS: 12 of 9768 evaluation rows (0.12%) also appear in the training row hashes. Datasets can contain genuine duplicate records (0.03% of evaluation rows duplicate another evaluation row), so a small overlap is not by itself leakage. The hashes are producer-supplied.
* **`leakage.single_feature_predictiveness`** — PASS: The strongest single feature, 'relationship', reaches a cross-validated ROC AUC of 0.772 on its own (12 features, 9768 rows). Values near 1.0 suggest the feature leaks the label.
* **`training_validity.eval_label_conflicts`** — PASS: 0 of 9768 evaluation rows (0.00%) share their feature values with a row that has a different label (0 conflicting groups among 3 groups of repeated feature values). Conflicting labels put a ceiling on the accuracy any model can reach and make the measured metrics unreliable.
* **`training_validity.train_duplicate_rows`** — PASS: 17 of 29305 training rows (0.06%) duplicate another training row. Duplicates can make a model look better on rows it has memorised and are often a data-quality problem upstream. The hashes are producer-supplied and cover the feature columns only, so rows with the same features and different labels also count as duplicates.
* **`behaviour.f1`** — UNAVAILABLE: Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* **`behaviour.roc_auc`** — UNAVAILABLE: Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* **`behaviour.precision`** — UNAVAILABLE: Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* **`behaviour.recall`** — UNAVAILABLE: Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* **`behaviour.subgroup_f1_gap`** — UNAVAILABLE: Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* **`fairness.selection_rate_ratio`** — UNAVAILABLE: Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* **`fairness.equalized_odds`** — UNAVAILABLE: Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* **`calibration.ece`** — UNAVAILABLE: Cannot evaluate: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* **`operational.model_loadable`** — FAIL: cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: )
* **`operational.input_schema`** — UNAVAILABLE: Cannot test: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* **`operational.prediction_determinism`** — UNAVAILABLE: Cannot test: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* **`operational.malformed_input_handling`** — UNAVAILABLE: Cannot test: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* **`operational.artifact_size`** — PASS: Model file is 3.0 KB.
* **`operational.latency`** — UNAVAILABLE: Cannot test: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))
* **`operational.memory`** — UNAVAILABLE: Model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: )).
* **`operational.dependency_match`** — PASS: 5 of 5 declared library versions match the current environment.
* **`operational.model_file_safety`** — UNAVAILABLE: The model file could not be scanned: not a readable pickle stream: EOFError: . It is not counted as safe.
* **`operational.cpu_time`** — UNAVAILABLE: Cannot test: model could not be loaded (cannot load model: it could not be parsed safely and was not loaded (not a readable pickle stream: EOFError: ))

*Evidence: MEASURED = computed by the gate from the model and data; DECLARED = taken from producer metadata, not verified; SIMULATED = injected for a demonstration; UNAVAILABLE = none.*

## Inputs and audit trail

| Item | Value |
|---|---|
| Run id | 2a9c7673357743cfb94614f0ff3aacb7 |
| Created (UTC) | 2026-10-20T06:47:50.979202+00:00 |
| Gate version | 0.1.0 |
| Schema version | 0.1.0 |
| Candidate | corrupt |
| Model file | C:\artifacts\fixtures\corrupt\model.joblib |
| Model SHA-256 | 656dcafa3448215c7ed3e23d792e5a57428b26a5a37a2fe6fbeb51246b29e8b1 |
| Evaluation data SHA-256 | bba157c9f381e730133239c0f78c5cf8c98f3b16f364eaac58fc54ebc0862722 |
| Metadata SHA-256 | d216e659b5d601aeaf4900c572c052cc503332b2e954d4f054b7369dc711b4be |
| Configuration profile | extended |
| Configuration hash | e5cd88eaccee847a641f880954e58a1a4d203197483052cf7df2781130923bc7 |
| Machine | Windows-11-10.0.26200-SP0; 12 CPUs; 15.6 GB RAM |
| Libraries | python 3.12.1, scikit_learn 1.9.1, pandas 3.0.6, numpy 2.5.4, joblib 1.6.0 |
| Stage timings (s) | load 0.02, training_validity 0.36, model_behaviour 0.00, operational_readiness 0.00, score_and_decide 0.00 |

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
