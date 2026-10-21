"""Model-output drift: is the model predicting a different mix, or with different confidence?"""
import numpy as np

from src.c4.monitoring.stats import confidence_hist, jsd_arrays


def compute(conf: np.ndarray, baseline: dict, n_bins: int = 10) -> dict:
    conf = np.asarray(conf, dtype=float)
    pred = (conf >= 0.5).astype(int)
    b = baseline["prediction"]
    return {
        "prediction_share": float(pred.mean()),
        "prediction_share_delta": float(abs(pred.mean() - b["positive_share"])),
        "confidence_mean": float(conf.mean()),
        "confidence_mean_delta": float(abs(conf.mean() - b["confidence_mean"])),
        "confidence_jsd": jsd_arrays(b["confidence_hist"], confidence_hist(conf, n_bins)),
    }
