import pytest

from src.common.config import load_config
from src.c4.severity import severity

CFG = load_config("severity")


def signals(ratio=0.0, pred_ratio=0.0, fired=(), fired_share=0.0, perf_drop=0.0):
    return {
        "features": {f: {"ratio": ratio} for f in fired},
        "fired_features": list(fired),
        "prediction": {"ratio": pred_ratio},
        "fired_share": fired_share,
        "performance": {"perf_drop": perf_drop},
    }


def test_config_weights_sum_to_one_and_levels_increase():
    assert sum(CFG["weights"].values()) == pytest.approx(1.0)
    lv = CFG["levels"]
    assert lv["medium"] < lv["high"] < lv["critical"]


def test_no_drift_is_level_none():
    out = severity.compute(signals(), "NO_DRIFT", 0, True, CFG)
    assert out["level"] == "None" and out["score"] == 0.0


def test_small_signal_is_low():
    out = severity.compute(signals(1.2, 0.5, ["age"], 0.1), "COVARIATE_BENIGN", 1, True, CFG)
    assert out["level"] == "Low"


def test_medium_example_matches_hand_calculation():
    # magnitude .5, persistence .4, scope .2, performance .25
    # score = .25*.5 + .20*.4 + .15*.2 + .40*.25 = 0.335
    out = severity.compute(signals(2.0, 0.5, ["age"], 0.2, 0.05), "CONCEPT_DRIFT", 2, True, CFG)
    assert out["components"] == {"magnitude": 0.5, "persistence": 0.4, "scope": 0.2, "performance": 0.25}
    assert out["score"] == pytest.approx(0.335)
    assert out["level"] == "Medium"


def test_all_components_maxed_is_critical():
    out = severity.compute(signals(5.0, 5.0, ["age"], 1.0, 0.5), "CONCEPT_DRIFT", 10, True, CFG)
    assert out["score"] == pytest.approx(1.0)
    assert out["level"] == "Critical"


def test_sudden_performance_collapse_is_critical_immediately():
    out = severity.compute(signals(perf_drop=0.30), "CONCEPT_DRIFT", 1, True, CFG)
    assert out["level"] == "Critical"


def test_performance_ignored_when_labels_not_valid():
    out = severity.compute(signals(perf_drop=0.30), "SUSPECTED", 1, False, CFG)
    assert out["perf_drop_used"] == 0.0
    assert out["level"] != "Critical"
