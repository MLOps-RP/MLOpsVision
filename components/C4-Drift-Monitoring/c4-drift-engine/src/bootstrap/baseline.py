"""
Build the VERSIONED BASELINE for one model: the 'normal' picture of the data
and of the model's behaviour. C4 compares every production window against it.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.common.config import LABEL, load_config, feature_lists, all_features
from src.c4.monitoring.stats import (decile_edges, bin_shares, category_shares,
                                     confidence_hist, classification_summary)


def feature_importance(pipe, X: pd.DataFrame, y: pd.Series) -> dict:
    """Permutation importance on the ORIGINAL features (works for any model)."""
    r = permutation_importance(pipe, X, y, n_repeats=3, random_state=42, scoring="f1", n_jobs=1)
    imp = np.clip(r.importances_mean, 0, None)
    total = imp.sum() or 1.0
    return {f: float(v / total) for f, v in zip(X.columns, imp)}


def build_baseline(pipe, val: pd.DataFrame, model_version: str, model_type: str,
                   preprocessing_version: str) -> dict:
    cfg = load_config("monitoring")
    num, cat = feature_lists()
    X, y = val[all_features()], val[LABEL]

    conf = pipe.predict_proba(X)[:, 1]
    pred = (conf >= 0.5).astype(int)

    numeric = {}
    for f in num:
        edges = decile_edges(X[f], cfg["psi_bins"])
        numeric[f] = {
            "bin_edges": edges,
            "bin_shares": bin_shares(X[f], edges).tolist(),
            "min": float(X[f].min()), "max": float(X[f].max()),
            "mean": float(X[f].mean()),
            "null_rate": float(X[f].isna().mean()),
        }

    categorical = {}
    for f in cat:
        categorical[f] = {
            "shares": category_shares(X[f]),
            "null_rate": float(X[f].isna().mean()),
        }

    # error rate per group (e.g. marital-status = Married-civ-spouse), used to find
    # WHICH segments get worse when concept drift happens
    err = pd.Series((pred != y.to_numpy()).astype(float), index=X.index)
    segment_error = {}
    for f in cat:
        g = err.groupby(X[f].astype(str))
        stats = g.agg(["mean", "count"])
        segment_error[f] = {str(k): float(r["mean"]) for k, r in stats.iterrows() if r["count"] >= 40}

    return {
        "model_name": "income-classifier",
        "model_version": str(model_version),
        "model_type": model_type,
        "preprocessing_version": preprocessing_version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline_data": "validation split",
        "n_rows": int(len(val)),
        "numeric_features": num,
        "categorical_features": cat,
        "numeric": numeric,
        "categorical": categorical,
        "prediction": {
            "positive_share": float(pred.mean()),
            "confidence_mean": float(conf.mean()),
            "confidence_hist": confidence_hist(conf, cfg["confidence_bins"]).tolist(),
        },
        "label": {"positive_share": float(y.mean())},
        "performance": classification_summary(y, pred),
        "segment_error": segment_error,
        "feature_importance": feature_importance(pipe, X, y),
    }
