# Scoring (`src/mqvg/scoring.py`)

```
check value --normalize--> score 0-100 --weighted average--> dimension score --weights--> MQS
```

## 1. Check score and status (`normalize.py`)

For a non-binary check with thresholds `floor`, `fail_at`, `pass_at`, `target`:

* score = 0 at `floor`, 100 at `target`, linear in between, clipped to 0-100
* PASS if the value is at least as good as `pass_at`; FAIL if worse than
  `fail_at`; WARN in between
* binary checks score 100 (PASS) or 0 (FAIL)

## 2. Dimension score (TVS, MBS, ORS)

Each dimension is the weighted average of its enabled checks, using the
per-check weights in the config. Every check's share is reported as `points`,
and the points add up to the dimension score.

| Check outcome | `exclude_and_cap` (default) | `penalize` (strict) |
|---|---|---|
| PASS / WARN / FAIL | counted | counted |
| UNAVAILABLE / ERROR / no result | left out of the average; lowers coverage | scores 0 |
| NOT_APPLICABLE | ignored entirely (not in the average, not in coverage) | ignored entirely |

* **Coverage** = weight of checks with evidence / weight of applicable checks.
* Coverage below `coverage.min_dimension` -> the dimension gets **no score**.
* Coverage below 1 -> the dimension is **provisional**.

## 3. Model Quality Score

`MQS = TVS*W1 + MBS*W2 + ORS*W3`, weights from the config (sum to 1).

* `exclude_and_cap`: dimensions without a score are dropped and the remaining
  weights are renormalized (`effective_weights`); the MQS is provisional. No
  scored dimension -> no MQS.
* `penalize`: a dimension without a score contributes 0.
* `overall_coverage` = sum of weight x coverage over the dimensions.

## 4. The lower bound

Excluding missing evidence is optimistic, so every scorecard carries
`mqs_lower_bound`: the MQS if every missing check and unscored dimension had
scored 0. It equals the MQS under `penalize`, and is never above it. Show the
two together ("99.9, at worst 22.5"). Do **not** reject on the lower bound:
missing evidence should lead to HOLD, not ROLLBACK.

## 5. What scoring does not do

It never changes a number to enforce a decision. Critical failures, missing
required evidence, low overall coverage and `must_pass_checks` are applied by
the rules engine (Step 11), which can stop a high MQS from becoming PROMOTE.
