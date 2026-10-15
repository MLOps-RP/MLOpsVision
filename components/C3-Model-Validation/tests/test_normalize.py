import math

import pytest

from mqvg.config import Thresholds
from mqvg.normalize import binary_outcome, normalize
from mqvg.schemas import CheckStatus

HIGHER = Thresholds(target=0.9, pass_at=0.7, fail_at=0.5, floor=0.1)
LOWER = Thresholds(target=0.0, pass_at=0.05, fail_at=0.10, floor=0.30)


@pytest.mark.parametrize("value, score, status", [
    (0.9, 100, CheckStatus.PASS),
    (1.2, 100, CheckStatus.PASS),    # better than target is clipped
    (0.7, 75, CheckStatus.PASS),     # exactly at pass_at passes
    (0.6, 62.5, CheckStatus.WARN),
    (0.5, 50, CheckStatus.WARN),     # exactly at fail_at is still WARN
    (0.4999, 49.9875, CheckStatus.FAIL),
    (0.1, 0, CheckStatus.FAIL),
    (-3.0, 0, CheckStatus.FAIL),     # worse than floor is clipped
])
def test_higher_is_better(value, score, status):
    got_score, got_status = normalize(value, HIGHER, "higher_is_better")
    assert got_score == pytest.approx(score)
    assert got_status == status


@pytest.mark.parametrize("value, score, status", [
    (0.0, 100, CheckStatus.PASS),
    (-0.5, 100, CheckStatus.PASS),
    (0.05, pytest.approx(83.3333, abs=1e-3), CheckStatus.PASS),
    (0.08, pytest.approx(73.3333, abs=1e-3), CheckStatus.WARN),
    (0.10, pytest.approx(66.6667, abs=1e-3), CheckStatus.WARN),
    (0.1001, pytest.approx(66.63, abs=1e-2), CheckStatus.FAIL),
    (0.30, 0, CheckStatus.FAIL),
    (5.0, 0, CheckStatus.FAIL),
])
def test_lower_is_better(value, score, status):
    got_score, got_status = normalize(value, LOWER, "lower_is_better")
    assert got_score == score
    assert got_status == status


def test_score_never_leaves_0_100():
    for value in (-1e9, -1, 0, 0.5, 1, 1e9):
        for thresholds, direction in ((HIGHER, "higher_is_better"),
                                      (LOWER, "lower_is_better")):
            score, _ = normalize(value, thresholds, direction)
            assert 0 <= score <= 100


def test_better_values_never_lower_the_score():
    values = [i / 100 for i in range(-20, 140)]
    higher = [normalize(v, HIGHER, "higher_is_better")[0] for v in values]
    lower = [normalize(v, LOWER, "lower_is_better")[0] for v in values]
    assert higher == sorted(higher)
    assert lower == sorted(lower, reverse=True)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        normalize(bad, HIGHER, "higher_is_better")


def test_binary_direction_cannot_be_normalized():
    with pytest.raises(ValueError, match="binary"):
        normalize(1.0, HIGHER, "binary")


def test_binary_outcome():
    assert binary_outcome(True) == (100.0, CheckStatus.PASS)
    assert binary_outcome(False) == (0.0, CheckStatus.FAIL)
