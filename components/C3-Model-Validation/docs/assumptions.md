# Assumptions, decisions and open research justifications

Record each threshold or design choice here with its reason. Items marked
**OPEN** still need literature, experiment or supervisor input.

## Fixtures

- Dataset: UCI Adult via Fairlearn, split 60/20/20 stratified, seed 42.
  Not representative of every domain.
- Sensitive attributes: `sex`, `race`. The good model excludes them from its
  features but is still not fair (parity difference 0.171 by sex, selection-rate
  ratio 0.31). **OPEN:** fairness thresholds decide whether this is WARN or PASS.
- Adult has naturally duplicated records: 12 of 9,768 eval rows (0.12%) exactly
  match a training row. The overlap check must compare against this baseline
  and must not fail on any non-zero overlap. **OPEN:** choose the tolerance.
- Fairness metric choice: the parity *difference* is capped by the majority
  group's selection rate (0.171 for good vs 0.249 for the clearly biased
  fixture), while the selection-rate *ratio* separates them clearly
  (0.31 vs 0.006). Use a ratio or equalized-odds metric as well.
- Latency depends on hardware (about 7 ms vs 78 ms single-row on the
  development machine). Always record a machine profile with latency results.
- `manifest.yaml` expected decisions are provisional until thresholds exist.
  Do not tune thresholds only to make them pass.

## Schemas

- No per-check severity; criticality comes from configured rules.
- Declared metrics in metadata are never to be used as measured results.
- Model loading executes pickle code, so only trusted local paths are
  accepted in this prototype. **Limitation** to state in the thesis.

## Open questions (see the implementation plan)

- Meaning of ROLLBACK (reject candidate vs revert deployed model).
- Whether to compare against an incumbent model (champion/challenger).
- Default missing-evidence policy (exclude-and-cap vs penalize).
- How the MQS itself will be validated (injected defects, expert agreement).

## Operational readiness checks (Step 5)

- **Memory check is weak evidence.** It measures resident-memory growth in the
  validating process. It includes library imports on the first load and reads
  about 0 MB when those libraries are already imported (seen for the `leaky`
  fixture, 68 MB for `good`). **OPEN / Phase B:** measure in a fresh subprocess.
  That would also isolate model loading (a security gain) and let the gate
  survive a model that hard-crashes.
- **Malformed-input check is in-process.** It can tell that a model rejects bad
  input but cannot detect a hard crash (such as a segmentation fault), which
  would end the whole run.
- **Latency is single-row, one call at a time, on the validating machine.**
  It says nothing about batch throughput or a production server. Always read
  it next to the recorded machine profile.
- **Dependency match is DECLARED evidence.** It compares the versions the
  producer declared against the current environment; the declaration itself is
  unverified. Python patch versions are treated as compatible.
- **Loading executes code.** `joblib.load` runs pickle code, so the loader
  accepts only trusted local paths. State this as a limitation in the thesis.

## Performance metrics (Step 6)

- Binary classification only. Multiclass and regression targets, or a declared
  non-binary task type, make the behaviour checks UNAVAILABLE (not wrong
  numbers). **OPEN:** extend in Phase B.
- All metrics are computed by the gate from the model's own predictions.
  Metrics recorded in metadata are never used; for the good fixture the two
  agree to 1e-9, which cross-checks both code paths.
- Positive label: taken from the bundle or metadata. If undeclared it is
  inferred as `model.classes_[1]` and the result says so. This is a
  scikit-learn convention, not a fact; declare it where it matters.
- Performance alone cannot reveal leakage or overfitting: the `leaky` fixture
  scores F1 0.954 / AUC 0.994 and passes every behaviour check, and `overfit`
  passes F1. This is the argument for the separate training-validity dimension
  and for critical rules (use it in the thesis).
- Reported baselines (always-positive F1/precision, AUC 0.5) are context only
  and are not part of the score. **OPEN:** decide whether a lift-over-baseline
  check should be scored.
- Precision is undefined when nothing is predicted positive; it is reported as
  0.0 with `precision_undefined` set.
- Behaviour checks that are not built yet (fairness, calibration, subgroup,
  robustness) report ERROR with "not implemented", never PASS, until their step.

## Fairness and subgroup checks (Step 7)

- Metrics are computed in the gate (so small groups and undefined rates can be
  handled explicitly) and cross-checked against Fairlearn in the tests
  (`demographic_parity_ratio/difference`, `equalized_odds_difference`,
  `MetricFrame` F1 difference): they agree on random 3-group data and the
  fixtures.
- Groups smaller than `evaluation.min_group_size` (default 30) are excluded
  and listed. **OPEN:** justify the minimum; a statistical-power argument
  would be better than a round number. Confidence intervals are not computed
  yet, so a gap from groups of ~30 is noisy. **Phase B.**
- Only the declared sensitive columns are examined, one at a time.
  Intersectional groups (for example sex x race) and non-sensitive slices
  are left to hidden-failure discovery (Phase B).
- Declared columns missing from the evaluation data are skipped and named in
  the message; if none can be evaluated the check is UNAVAILABLE. A partial
  result (one column evaluated, another skipped) is still scored, with the
  skipped column recorded in the details. **OPEN:** should a skipped declared
  column cap the score or block PROMOTE?
- If no group receives a positive prediction, the selection-rate ratio is
  vacuous; it is reported as 1.0 and flagged. A model that predicts nothing is
  "equal" across groups, so rely on F1/recall to catch it.
- `behaviour.subgroup_f1_gap` currently uses sensitive-attribute groups only.
- Results on the fixtures: `good` has selection-rate ratio 0.311 (sex) and
  equalized-odds difference 0.219 (race): WARN under the default thresholds
  and FAIL under strict; `biased` has ratio 0.006 and fails everywhere.

## Calibration and robustness (Step 8)

- ECE uses equal-width bins (`evaluation.ece_bins`, default 10). It depends on
  the bin count and is biased upward on small samples, so the per-bin counts,
  max calibration error and Brier score are reported with it. The gate's ECE
  agrees with a scikit-learn `calibration_curve` reference and with the
  fixture verifier (good 0.011, miscalibrated 0.240). **OPEN:** justify the
  ECE thresholds and consider adaptive (equal-mass) bins.
- Calibration is UNAVAILABLE for models that only give decision-function
  scores: those are not probabilities.
- Overfitting is visible through calibration: the `overfit` tree passes F1
  (0.617) but has ECE 0.185 and fails. Use this when arguing that no single
  metric is enough.
- Robustness is a simple sensitivity probe: Gaussian noise on numeric features
  only, scaled to each feature's spread (0.1x by default), 5 seeded repeats.
  It ignores valid ranges and feature correlations, does not perturb
  categorical features, and is not an adversarial or distribution-shift test.
  It stays **disabled in the default profile**. The good fixture loses only
  about 0.9% relative F1 at 0.1x noise, so the default level may be too mild
  to discriminate. **OPEN / Phase B:** more perturbation types and levels,
  and calibrate the thresholds against known-fragile models.
- A model with no numeric features gets NOT_APPLICABLE for robustness; a clean
  F1 of zero makes the relative drop undefined (UNAVAILABLE).

## Training validity checks (Step 9)

What a pretrained model can and cannot show (state this plainly in the thesis):

| Check | Evidence | Limit |
|---|---|---|
| metadata completeness, reproducibility info | MEASURED presence | values are not verified |
| overfit gap | DECLARED train metric vs MEASURED eval metric | the producer's training metric is a claim; no training data access |
| train/eval overlap | DECLARED (producer-supplied row hashes) | proves nothing if the hashes are wrong or partial |
| single-feature leakage probe | MEASURED | a heuristic: finds a feature that predicts the label alone, not diffuse leakage; a legitimately strong feature also scores high |

- The gate cannot prove a split was correct or that leakage is absent; it can
  only find signs. Split correctness (stratification, temporal ordering) is
  not assessed beyond the overlap hashes. **OPEN:** a `split_documented`
  check, or train-data access for a true overlap/overfit measurement.
- **Missing metadata silently removes protection.** With `good_incomplete_metadata`
  the overfit gap and overlap are UNAVAILABLE, so an overfit model with no
  declared training metrics would not be flagged by training validity (it
  would still show up in calibration, as the `overfit` fixture does). The
  scoring step must make this visible: low coverage, provisional MQS, and
  `must_pass_checks` such as overfit_gap blocking PROMOTE.
- Leakage probe: cross-validated depth-4 decision tree on each feature alone,
  ROC AUC, max over features. On the Adult fixture an honest model's best
  feature ('relationship') reaches 0.772; the leaky fixture's derived flag
  reaches 0.968. The default thresholds (pass_at 0.90, fail_at 0.95) leave a
  wide gap between those; **OPEN:** justify them on more datasets, since a
  dataset with one genuinely dominant feature will score high.
- Overlap is judged against a baseline: datasets contain genuine duplicates
  (Adult: 0.12% of eval rows). The result reports the evaluation set's own
  duplicate rate for context. **OPEN:** whether to compare overlap to the
  expected duplicate rate rather than to a fixed fraction.
- Relative `train_row_hashes_file` paths resolve against the working
  directory, then the metadata file's folder. Hashing follows the convention in
  `mqvg/hashing.py`, which a test pins to `scripts/fixture_utils.py`.
- Overfit gap uses accuracy when declared, otherwise F1; the unit is a
  difference in that metric. It can be negative (eval better than train).

## Scoring (Step 10)

Scores on the sample fixtures (default profile, exclude_and_cap; placeholder
thresholds, so read the *pattern*, not the numbers):

| Fixture | TVS | MBS | ORS | MQS | lower bound | coverage |
|---|---|---|---|---|---|---|
| good | 99.9 | 73.8 | 100 | 86.9 | 86.9 | 1.00 |
| leaky | 68.4 | 86.5 | 100 | 83.8 | 83.8 | 1.00 |
| biased | 97.7 | 52.1 | 100 | 75.4 | 75.4 | 1.00 |
| overfit | 81.7 | 54.8 | 100 | 71.9 | 71.9 | 1.00 |
| miscalibrated | 99.9 | 60.7 | 100 | 80.3 | 80.3 | 1.00 |
| corrupt | 99.9 | n/a | n/a | 99.9 | 22.5 | 0.31 |
| good_no_sensitive | 100 | 87.6 | 100 | 93.8 | 74.8 | 0.79 |

Findings (use these in the thesis):

1. **A high MQS can hide a critical failure.** `leaky` (83.8) and `biased`
   (75.4, above the 75 promote line) score well; only critical rules
   (leakage, fairness) stop them. This is the case for critical overrides.
2. **Excluding missing evidence rewards omission.** `good_no_sensitive` scores
   93.8, higher than `good` (86.9), because the unfavourable fairness checks
   dropped out of the average. The lower bound (74.8) and blocking PROMOTE on
   missing required evidence are the safeguards. The `penalize` policy avoids
   the incentive but is harsher. **OPEN:** which default to use, and whether
   to cap the MQS when required evidence is missing.
3. **A dimension can be dropped despite a known FAIL.** `corrupt` loses its
   operational score (coverage 0.42 < 0.50) even though `model_loadable`
   failed, leaving MQS 99.9 from training validity alone. The critical rule
   `model_not_loadable` and the lower bound (22.5) handle it, but the number
   alone misleads. **OPEN:** should an explicit FAIL on a required check keep a
   dimension scored regardless of coverage?
4. **Thresholds sit close together.** `biased` (75.4) and `good` (86.9) are
   only 11 points apart and `biased` clears `promote_at`. Run a sensitivity
   analysis over weights and thresholds before relying on any cutoff.
5. **Dimension weights are a modelling choice** (0.30 / 0.50 / 0.20 here). They
   need justification and a sensitivity analysis.

## Rules engine (Step 11)

Decisions on the sample fixtures (default profile; thresholds were not tuned
to fit the manifest, which was written first):

| Fixture | MQS | Decision | Why |
|---|---|---|---|
| good | 86.9 | PROMOTE | meets the threshold, all must-pass checks pass |
| leaky | 83.8 | ROLLBACK | critical `target_leakage`; MQS alone would PROMOTE |
| biased | 75.4 | ROLLBACK | critical `severe_selection_rate_disparity`; MQS alone would PROMOTE |
| corrupt | 99.9 (at worst 22.5) | ROLLBACK | critical `model_not_loadable` |
| overfit | 71.9 | HOLD | must-pass overfit gap and calibration fail |
| miscalibrated | 80.3 | HOLD | must-pass calibration fails |
| good_incomplete_metadata | 79.7 | HOLD | must-pass metadata completeness and overfit gap |
| good_no_sensitive | 93.8 (at worst 74.8) | HOLD | required fairness evidence missing; coverage 79% < 80% |
| slow | 85.9 | PROMOTE or HOLD | HOLD only if single-row latency exceeds 50 ms on the validating machine |

Findings and open decisions:

1. **Hardware-dependent outcome.** `slow` was 35 ms p95 on an idle machine (PASS)
   and 70-110 ms when busy (WARN). The manifest originally said HOLD; it now
   says HOLD or PROMOTE, and a test checks both the conditional rule and a
   deterministic HOLD under a tightened limit. Absolute latency limits do not
   transfer between machines. **OPEN / Phase B:** express the limit relative to a
   reference model measured on the same machine.
2. **The strict profile is unattainable for this baseline.** It ROLLBACKs even
   `good`, because its critical `fail_at` for the selection-rate ratio (0.50)
   is above what the baseline reaches (0.31). Strict thresholds need to be
   set from what is achievable on the dataset, or the profile is meaningless.
3. **`penalize` turns missing evidence into ROLLBACK** (strict: `good_no_sensitive`
   gets MQS 45.6 < 60). Under `exclude_and_cap` the same case is a HOLD.
   Missing evidence is "not yet known" in the first and "assumed bad" in the
   second. **OPEN:** which is right for the deployment decision.
4. Decisions use the policy MQS, never the lower bound, so that missing
   evidence leads to HOLD (not ROLLBACK) under `exclude_and_cap`.
5. `overridden_by_critical` is only true when the score alone would not have
   produced a ROLLBACK; a critical rule that agrees with a low score is
   reported but is not an "override".
6. A must-pass check that is UNAVAILABLE or ERROR blocks PROMOTE (it is not
   PASS); NOT_APPLICABLE does not. A required check with no evidence blocks too.
   **OPEN:** the lists of required and must-pass checks are design choices.
7. A numeric critical rule cannot match a check that has no measured value.

## Pipeline (Step 12)

- `run_validation(bundle, config)` is the single entry point: load -> three
  check groups -> scorecard -> rules -> `ValidationResult`. It is what the CLI
  and the API will wrap (Step 14); neither may contain validation logic.
- **Contract:** a bad candidate never raises; it produces findings and a
  decision (even a candidate where nothing exists gives a ROLLBACK result).
  Only an unusable configuration raises `ConfigError`. A crashing check becomes
  an ERROR result; a crashing check group makes all of that dimension's
  enabled checks ERROR. ERROR is never a pass and blocks a required or
  must-pass check from allowing PROMOTE.
- **Reproducibility:** identical inputs and config give an identical result
  except `run_id`, `created_at_utc`, `environment` (machine profile, timings)
  and the machine-dependent latency/memory checks. A test compares two runs
  exactly with those two checks switched off.
- A full run on the sample fixtures takes about 0.7 s (the leakage probe is
  the largest part).
- `ValidationResult.explanations` is still empty: the failure explanation and
  the readable report are Step 13.
- Security: the model file is unpickled in the validating process. Run the
  gate only on models from trusted sources; process isolation is Phase B.

## External models (Step A)

Found by running 14 foreign-model scenarios through the pipeline and fixed:

1. **Label mismatch gave wrong numbers.** A model outputting 0/1 against a text target scored F1 0.000 / AUC 0.094 and was rolled back. Labels are now normalized (1, 1.0, "1" agree) and validated against `model.classes_` (or against the predictions if there is no `classes_`); a mismatch is `UNAVAILABLE` with an explanation. The unsafe decision-function fallback with an unverified sign was removed.
2. **MQS 100.0 on 26% evidence.** `ScoreCard.mqs_sufficient_evidence` and `insufficient_evidence_reason` were added (the numeric `mqs` is unchanged). The rules engine HOLDs when a dimension has no score although overall coverage looks fine (`mqs_insufficient_evidence`), and the summary says the MQS is not reliable. Reports and APIs should show such an MQS as "n/a (insufficient evidence)" next to the worst-case value.
3. **Unreadable was reported as bad.** A model the gate cannot read (unsupported format, or a module it does not have) is now `UNAVAILABLE`, not `FAIL`, so it leads to HOLD and not to a critical ROLLBACK. Unsupported formats are never opened. A missing, corrupt or not-a-model file is still `FAIL`. **OPEN:** whether a missing file should also be `UNAVAILABLE` (it is likely a path mistake).
4. **Leakage probe false reassurance on text.** It scored free text ~0.5 and passed. Free-text or identifier-like columns are now listed as not assessed; if every feature is such a column the check is `UNAVAILABLE`.
5. Clearer messages for a continuous (regression-like) target and for models that expect a different input form.

Not done in Step A (still open): an explicit trust-this-model flag (planned for the CLI/API step), subprocess isolation, a numpy input option, TSV/Parquet input, multiclass and regression support.

## Reporting and explanation (Step 13)

- The report is a pure rendering of the `ValidationResult`: it makes no judgement of its own. Every rule that fired and every check appears in it, tested by comparing the JSON with the Markdown.
- `explain.py` traces the GATE decision; it does not explain the model predictions (SHAP/LIME, Phase B). Suggested actions are generic hints. **OPEN:** review the hint wording with your supervisor before quoting it.
- Bug found by a test: a must-pass check that was UNAVAILABLE (for example latency) produced the check-specific advice "reduce model size". Advice is now limited to checks that actually FAILED or WARNED; missing evidence gets the "supply the inputs" advice.
- An MQS without sufficient evidence is shown as "n/a (insufficient evidence; recorded value ...)" with the worst case beside it.
- **OPEN:** an HTML report with charts, and a comparison report between two models, are Phase B.
