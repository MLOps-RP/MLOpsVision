"""Turn a measured value into a 0-100 score and a PASS/WARN/FAIL status.

Pure functions, no I/O. Semantics are those documented in config.py:

* score is 0 at `floor` and 100 at `target`, linear in between, clipped;
* status is PASS if the value is at least as good as `pass_at`, FAIL if it is
  worse than `fail_at`, and WARN in between.

The same linear formula serves both directions: for a lower-is-better metric
floor > target, and the ratio is still 0 at floor and 1 at target.
"""
from __future__ import annotations

import math

from mqvg.config import Thresholds
from mqvg.schemas import CheckStatus

BINARY_PASS_SCORE = 100.0
BINARY_FAIL_SCORE = 0.0


def normalize(value: float, thresholds: Thresholds,
              direction: str) -> tuple[float, CheckStatus]:
    if direction not in ("higher_is_better", "lower_is_better"):
        raise ValueError(f"cannot normalize a '{direction}' metric")
    if not math.isfinite(value):
        raise ValueError(f"value must be finite, got {value}")

    span = thresholds.target - thresholds.floor
    score = (value - thresholds.floor) / span * 100.0
    score = min(100.0, max(0.0, score))

    if direction == "higher_is_better":
        passed = value >= thresholds.pass_at
        failed = value < thresholds.fail_at
    else:
        passed = value <= thresholds.pass_at
        failed = value > thresholds.fail_at
    status = (CheckStatus.PASS if passed
              else CheckStatus.FAIL if failed else CheckStatus.WARN)
    return score, status


def binary_outcome(passed: bool) -> tuple[float, CheckStatus]:
    if passed:
        return BINARY_PASS_SCORE, CheckStatus.PASS
    return BINARY_FAIL_SCORE, CheckStatus.FAIL
