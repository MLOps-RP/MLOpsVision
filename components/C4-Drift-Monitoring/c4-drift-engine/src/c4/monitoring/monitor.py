"""
MONITORING (processing step 1)

Combines all signals for one window and compares each with its calibrated threshold.
Output: one 'signals' dictionary per window, used by diagnosis and severity.
"""
import numpy as np
import pandas as pd

from src.c4.monitoring import feature_tests, prediction_tests, data_quality


def compute_signals(window: int, window_X: pd.DataFrame, conf: np.ndarray, baseline: dict,
                    thresholds: dict, reference: pd.DataFrame, perf: dict, cfg: dict) -> dict:
    feats = feature_tests.compute(window_X, baseline, reference)
    preds = prediction_tests.compute(conf, baseline, cfg["confidence_bins"])
    dq = data_quality.check(window_X, baseline, cfg["data_quality"])

    # ---- compare every feature score with its own calibrated threshold ----
    fired = []
    for f, r in feats.items():
        thr = thresholds["features"][f]
        r["threshold"] = thr
        r["ratio"] = r["score"] / thr if thr > 0 else 0.0
        r["fired"] = r["score"] > thr
        if r["fired"]:
            fired.append(f)

    pred_ratio = max(preds["prediction_share_delta"] / thresholds["prediction_share"],
                     preds["confidence_mean_delta"] / thresholds["confidence_mean"],
                     preds["confidence_jsd"] / thresholds["confidence_jsd"])
    preds["ratio"] = pred_ratio
    preds["fired"] = pred_ratio > 1

    # ---- performance (only when labels have arrived) ----
    if perf.get("available"):
        thr = thresholds["perf_drop"]
        perf["perf_fired"] = bool(perf["perf_drop"] > thr or
                                  (perf["adwin_alarm"] and perf["perf_drop"] > 0.5 * thr))
        perf["label_share_fired"] = abs(perf["label_share_delta"]) > thresholds["label_share"]
        perf["per_class_stable"] = (abs(perf["tpr_delta"]) <= thresholds["tpr_delta"] and
                                    abs(perf["tnr_delta"]) <= thresholds["tnr_delta"])

    return {
        "window": window,
        "features": feats,
        "fired_features": fired,
        "fired_share": len(fired) / len(feats),
        "prediction": preds,
        "performance": perf,
        "data_quality": dq,
    }
