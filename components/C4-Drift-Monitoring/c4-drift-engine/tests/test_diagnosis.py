import numpy as np
import pandas as pd

from src.c4.diagnosis import diagnose as dg

NO_PERF = {"available": False}


def perf(**kw):
    base = {"available": True, "earliest_labelled_window": 16, "label_share_fired": False,
            "per_class_stable": False, "label_share_delta": 0.0, "tpr_delta": 0.0, "tnr_delta": 0.0,
            "perf_fired": False, "perf_drop": 0.0, "f1": 0.7}
    base.update(kw)
    return base


def signals(fired=(), fired_share=0.0, pred_fired=False, performance=None, dq=None):
    return {
        "features": {f: {"test": "psi", "score": 0.4, "threshold": 0.1, "ratio": 4.0} for f in fired},
        "fired_features": list(fired),
        "fired_share": fired_share,
        "prediction": {"fired": pred_fired, "ratio": 0.0},
        "performance": performance or NO_PERF,
        "data_quality": dq or {"fired": False, "issues": []},
    }


def test_no_signals_is_no_drift():
    assert dg.diagnose(signals(), None)["diagnosis"] == "NO_DRIFT"


def test_data_quality_wins_over_everything():
    s = signals(["age"], 0.3, performance=perf(perf_fired=True, perf_drop=0.3),
                dq={"fired": True, "issues": ["hours: null rate +40%"]})
    assert dg.diagnose(s, 15)["diagnosis"] == "DATA_QUALITY"


def test_label_shift_when_class_mix_moves_but_recall_is_stable():
    s = signals(performance=perf(label_share_fired=True, per_class_stable=True, label_share_delta=0.25))
    assert dg.diagnose(s, 15)["diagnosis"] == "LABEL_SHIFT"


def test_concept_drift_when_performance_drops_but_inputs_do_not_move():
    s = signals(performance=perf(perf_fired=True, perf_drop=0.15))
    out = dg.diagnose(s, 15)
    assert out["diagnosis"] == "CONCEPT_DRIFT" and out["confidence"] == "confirmed"


def test_covariate_degrading_when_inputs_drift_and_performance_drops():
    s = signals(["age", "hours"], 0.3, performance=perf(perf_fired=True, perf_drop=0.12))
    assert dg.diagnose(s, 15)["diagnosis"] == "COVARIATE_DEGRADING"


def test_covariate_benign_when_inputs_drift_but_performance_is_stable():
    s = signals(["race"], 0.3, performance=perf())
    assert dg.diagnose(s, 15)["diagnosis"] == "COVARIATE_BENIGN"


def test_suspected_when_inputs_drift_and_no_labels_yet():
    out = dg.diagnose(signals(["age"], 0.3), 15)
    assert out["diagnosis"] == "SUSPECTED" and out["confidence"] == "provisional"


def test_pre_drift_labels_do_not_count_as_evidence():
    s = signals(["age"], 0.3, performance=perf(earliest_labelled_window=10))
    assert dg.diagnose(s, 15)["diagnosis"] == "SUSPECTED"


def test_performance_is_valid():
    assert dg.performance_is_valid(perf(earliest_labelled_window=16), 15)
    assert not dg.performance_is_valid(perf(earliest_labelled_window=14), 15)
    assert not dg.performance_is_valid(NO_PERF, 15)
    assert dg.performance_is_valid(perf(earliest_labelled_window=0), None)


def test_rank_affected_features_weights_ratio_by_importance():
    s = signals(["age", "race"])
    s["features"]["race"]["ratio"] = 6.0
    ranked = dg.rank_affected_features(s, {"age": 0.5, "race": 0.01})
    assert [r["feature"] for r in ranked] == ["age", "race"]


def test_affected_segments_finds_the_group_that_got_worse():
    n = 100
    X = pd.DataFrame({"sex": ["Male"] * n + ["Female"] * n})
    y = np.zeros(2 * n, dtype=int)
    pred = np.concatenate([np.zeros(n), np.r_[np.ones(50), np.zeros(50)]]).astype(int)
    base = {"segment_error": {"sex": {"Male": 0.1, "Female": 0.1}}}
    seg = dg.affected_segments(X, y, pred, base)
    assert seg[0]["segment"] == "sex = Female"
    assert seg[0]["increase"] == 0.4
