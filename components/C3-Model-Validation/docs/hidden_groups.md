# Hidden Failure Group Detection

Check id: `behaviour.hidden_failure_groups` · module: [`hidden_groups.py`](../src/mqvg/hidden_groups.py)
· check wrapper: [`checks/hidden_failure.py`](../src/mqvg/checks/hidden_failure.py)
· results: [`hidden_groups_results.md`](hidden_groups_results.md)

## Why

Overall metrics, fairness checks and subgroup checks only examine groups someone
named in advance (the sensitive columns). A model can pass all of them and still
fail badly on a group nobody named. The fixture `hidden_group` demonstrates it:
error is 61% inside a region covering about 9% of the rows and 28% elsewhere, no
sensitive column describes the region, and **every existing check passes, so the
default profile PROMOTES the model**. This check searches the model's own feature
space for such groups.

## Turning it on

The check is **off in `configs/default.yaml`** so existing scores and decisions do
not change (adding a ninth behaviour check would shift every model's MBS and MQS).
Enable it with one of three profiles that extend the default and differ **only** in
what a confirmed group is allowed to do to the decision (`hidden_groups.consequence`):

| Profile | `consequence` | `require_assessment` | Confirmed group | Not assessed |
|---|---|---|---|---|
| `default` | `score_only` | false | check is off: no effect | no effect |
| `hidden_groups_score_only` | `score_only` | false | lowers the MQS only | lowers coverage, never a pass |
| `hidden_groups` | `hold` | true | HOLD (rule `hidden_failure_groups_hold`) | HOLD (rule `hidden_failure_groups_not_assessed`) |
| `hidden_groups_critical` | `critical` | true | WARN: HOLD; FAIL (excess >= `fail_at`): ROLLBACK recommendation | HOLD |

```
python scripts/validate_models.py hidden_group hidden_group_null --config configs/hidden_groups.yaml
```

A detected subgroup never forces ROLLBACK unless the policy is explicitly `critical`.
Whatever the policy, the explanation and report state which one was applied, and a
recommendation is never a deployment action. Other settings:

| Setting | Default in the profiles | Effect |
|---|---|---|
| `checks.behaviour.hidden_failure_groups.weight` | 1.5 | contribution to the Model Behaviour score |
| `checks.…thresholds` | pass 0 / fail 0.25 / floor 0.50 | the value is the largest confirmed excess error rate, so any confirmed group is at best a WARN; `fail_at` is also the "severe" line for `critical` |
| `hidden_groups.alpha`, `min_effect` | 0.05, 0.10 | significance (false discovery rate) and minimum excess error |
| `hidden_groups.consequence`, `require_assessment` | see table | decision consequence |
| `hidden_groups.*` | see below | the algorithm's settings |

The policy replaces editing `must_pass_checks` or `critical_rules`; those lists are
unchanged in every profile. A policy set while the check is disabled is rejected as a
configuration error.

## Associations, not causes; proxy features

A reported group is a statistical **association**: rows with these feature values were
mispredicted more often than the model's probabilities predict, on rows not used to find
the group. It is not a causal explanation of why the model fails. Features that
describe a group can be **proxies for sensitive characteristics** (for example an
address or an occupation); the group description may therefore reveal or approximate a
protected attribute even when no sensitive column was supplied. Review the features
searched (listed in the report) and handle group descriptions as sensitive.

## Complete workflow

See `docs/workflow.md` for the end-to-end flow and `docs/sample_reports/` for one report
per policy.

## Algorithm

Input: the evaluated rows (model features only), true labels, the model's predictions
and its positive-class probabilities.

1. **Expected error per row.** From the model's own probability *p*: `1 − p` if it
   predicted the positive class, `p` otherwise. If the probabilities are calibrated,
   the number of errors in any group is a sum of independent Bernoulli variables with
   known means, which gives an honest null hypothesis. A group is interesting only if
   it has **more** errors than that.
2. **Cross-fitted discovery.** Rows are split into `folds` (default 2) random folds.
   Candidate groups are discovered on all folds but one and **tested on the held-out
   fold**; the roles rotate. A group is never evaluated on the rows used to find it.
   Discovery sees the discovery-fold features and residuals (error minus expected
   error) only; the test fold's labels are never used (a test mutates the test fold's
   labels and checks that what is discovered does not change).
3. **Two discovery methods.**
   * *Decision tree* (depth 3) on the residual: groups are readable rules, for example
     `segment = d AND tenure_years > 6.57 AND income_k > 57`.
   * *K-means* (8 clusters) on the standardized features, weighted by each discovery
     row's positive excess error so cluster centres settle where excess errors
     concentrate. A cluster has no rule; it is described by its most distinctive
     features, for example `cluster of similar rows: segment = d for 100% of rows (overall 25%); …`.
   Only groups whose discovery-fold excess error is positive are tested.
4. **Test and correct.** Each held-out group gets a one-sided normal-approximation test of
   observed against expected errors (continuity-corrected). **Benjamini–Hochberg**
   controls the false discovery rate (`alpha`, default 0.05) over **every group tested**,
   across folds and methods.
5. **Effect size.** A group is *confirmed* only if it is significant **and** its
   observed error rate exceeds the expected rate by at least `min_effect` (default 10
   percentage points). The check's value is the largest excess among confirmed groups
   (0 if none).

### Reported for each group

Rule or cluster profile · detection method · held-out rows and share of all rows ·
observed errors, observed rate, expected rate · effect size with a 95% interval
(Wilson interval of the observed rate minus the expected rate) · p-value and
BH-adjusted p-value · confirmed or not. The first `report_groups` (default 10) are kept
in the check details, confirmed ones first; the report has a "Hidden failure groups"
section and the explanation names the worst group.

### Calibration safeguards

The null hypothesis needs probabilities that are right in general.

* **Recalibration (default on).** For each fold, a Platt map (a logistic fit on the
  log-odds) is fitted on the **discovery rows only** and used for the expected errors
  of that fold, but **only if the discovery rows' ECE exceeds
  `recalibration_trigger_ece`** (0.03). It corrects systematic over- or
  under-confidence without using held-out labels.
  The trigger is approximate: ECE is biased upward on a half-size fold, so a
  well-calibrated model occasionally crosses it on one fold. That is harmless
  (a Platt map of calibrated probabilities is nearly the identity) and the report
  states which folds were recalibrated.
* **Hard limit.** If ECE or the overall observed-versus-expected error gap exceeds
  `max_ece` / `max_overall_gap` (0.15), the check is **UNAVAILABLE**. These limits are
  deliberately loose: a large hidden group raises both by itself (ECE up to about 0.11
  was measured on a model that was fine elsewhere), whereas the grossly miscalibrated
  models measured 0.19 to 0.34. A tight limit would disqualify the very problem
  being sought.
* If confirmed groups cover more than half of the rows the check is UNAVAILABLE
  (the model is wrong in general, not in a hidden group).

### Other UNAVAILABLE cases (never a pass)

No probability estimates (including models that only have a decision function), fewer
than `min_rows` (1000) evaluation rows, fewer than `min_errors` (50) errors, no usable
feature, or an unavailable prediction stage (no target, label mismatch, and so on).

## Features, privacy and fairness

* **Default: every feature the model uses**, including non-sensitive ones, and never
  the target. Constant columns and free-text or identifier-like columns are left out
  and listed in the details. Categorical columns keep their `max_categories` most
  frequent levels (the number folded away is reported).
* The model's own feature set is used. Sensitive columns that exist in the evaluation
  data but that the model did not use are **not** searched unless you list them in
  `hidden_groups.extra_columns`; `hidden_groups.exclude_features` withholds model
  features.
* **A group description can reveal a demographic subpopulation even without a sensitive
  column.** On the Adult `biased` fixture the check reported `relationship = Wife`
  (observed error 45% against 7% expected) without being told the sex attribute. That
  is useful for finding proxies, but the same rule describes a protected group, so
  treat reports as sensitive and share them accordingly. Only aggregate information
  (a rule and counts) is stored, never individual rows.
* Finding a group is not a fairness verdict: the fairness checks remain the place
  for group-level disparity metrics, and this check does not replace them.
* Withholding features avoids searching on them but does not stop correlated
  features from describing the same group.

## Evaluation

Reproduce with `python scripts/evaluate_hidden_groups.py` (about 80 s with 8 jobs);
the full tables are in [hidden_groups_results.md](hidden_groups_results.md) and the raw
per-run data in `reports/hidden_groups_eval/results.csv`.

**Design.** Synthetic tabular data (3 numeric, 3 categorical features) with a planted
region where labels flip with probability 0.8 (0.4 for the weaker case). *Misspecification*
plants it in training and evaluation data, so a linear model cannot represent it;
*shift* plants it in the evaluation data only, so even gradient boosting fails there.
Ten planted scenarios and three controls with no group (logistic, gradient boosting, and
a logistic model made over-confident), evaluation sizes 1000, 2000, 5000 and 10000, 20
repeats each: 1040 runs, giving 5200 method-level results (the five method variants).

**Development versus final study.** Design choices (the tree against k-means, the
residual weighting of k-means, the form of the calibration safeguard) were made while
looking at other seeds and the five original patterns. The final study uses seeds that
were never used in development, and two patterns (`three_way_categorical`,
`young_low_income`; marked † in the tables) that were created afterwards. The
miscalibration handling was developed on null data and then evaluated on fresh seeds
only through the over-confident control.

**Findings** (800 planted and 240 control detections for the combined method):

* **Power depends on the group's size and on the sample.** Three of the four
  axis-aligned groups of 6 to 10% of the rows were found in 16 to 20 of 20 runs from
  n = 2000 (two_feature 16/20, single_threshold 18/20, categorical_pair 19/20); the
  fourth, young_low_income (7%), needed more data (8/20 at 2000, 19/20 at 5000). A 2%
  group was found in at most 9/20 even with 10,000 rows; a 3% three-way categorical
  group in 16/20 at 10,000. At n = 1000 most groups are missed.
* **Localization is partial.** Flagged groups often cover only part of the planted
  region: 612 of 800 planted runs had a confirmed group, but 483 covered at least half
  of the planted rows. The oblique (non-axis-aligned) region was detected 18 to 20 of 20
  times from n = 5000 but with only 62 to 64% recall of its rows.
* **Weaker failures need more data.** With 40% flips the group was found in 13 to 14 of
  20 runs from n = 5000, covering about two thirds of its rows.
* **Shifted subgroups with a flexible model** (gradient boosting) were found in 20/20
  (two_feature) and 13/20 to 18/20 (held-out young_low_income) from n = 5000.
* **K-means is the weaker partner.** Alone it detected 77 of 800, the tree 467, the
  combination 483. The combination helped mainly for categorical conjunctions at small
  samples (categorical_pair at n = 1000: tree 4, k-means 8, both 9 of 20;
  three_way_categorical at n = 2000: tree 1, k-means 7, both 7). Without the residual
  weighting k-means detected 50.
* **False alarms.** On the three controls the combined method confirmed a group in
  10 of 240 runs (4.2%, 95% interval 2.3 to 7.5%), the tree in 6 and k-means in 12,
  consistent with the nominal 5% but not guaranteed: the gradient-boosting control
  gave 6 of 80 (7.5%) for the combined method, with up to 3 of 20 in a single cell
  (4 of 20 for the tree alone at n = 2000).
* **Recalibration.** Without it, false alarms were 65/80 on the over-confident control
  and 33/80 on the gradient-boosting control; with it, 1/80 and 6/80. It costs power
  where it is triggered by a real hidden group (shift scenarios: 58 against 73 and 37
  against 49 detections of 80).
* **Cost.** About 0.1 s per detection at n = 1000 to 0.22 s at n = 10,000.

**On real data** (the six Adult models, no ground truth; one run each): nothing was
reported for `good`, `leaky` and `slow`; `miscalibrated` and `overfit` were UNAVAILABLE
(ECE 0.24 and 0.185); `biased` produced six confirmed groups led by `relationship = Wife`.
This is an observation on one set of fixtures, not a validation.

## Limitations

* **Synthetic evaluation only.** One data family, one kind of planted failure (label
  flips in a region), tabular binary classification. The results show behaviour on
  these patterns; they do not show that the method works in general, and no claim of
  general effectiveness is made.
* **Needs probabilities and a reasonably calibrated model.** The recalibration is a
  two-parameter map; it cannot fix distortions it cannot express, and it can be
  weakened by the very group being sought (visible in the shift scenarios).
* **Limited shapes.** Axis-aligned rules and clusters of similar rows. Oblique or very
  small regions are found late or only in part.
* **Minimum detectable group.** Groups with fewer than 30 held-out rows, or less than 2%
  of the held-out fold, are not tested. Power grows with the sample; "no group found" is
  not evidence that none exists.
* **Approximate statistics.** The test is a normal approximation; BH assumes the tests
  are independent or positively dependent, which is plausible here (the groups of one
  tree are disjoint, folds and methods overlap) but unproven. The false-alarm rates
  above are estimates from 20 repeats per cell.
* **Descriptive, not causal.** A confirmed group says where the model fails more than it
  expects, not why.
* **Placeholders.** `alpha`, `min_effect`, the group-size minimums, the tree depth, the
  number of clusters and the check's thresholds and weight all need justification for a
  domain.

## Commands

```
python scripts/make_hidden_group_fixtures.py        # create and verify the two fixtures
python scripts/validate_models.py hidden_group hidden_group_null --config configs/hidden_groups.yaml
python scripts/evaluate_hidden_groups.py            # the evaluation study (--quick for a smoke run)
python -m pytest tests/test_hidden_groups.py tests/test_hidden_failure_check.py tests/test_hidden_groups_study.py
```
