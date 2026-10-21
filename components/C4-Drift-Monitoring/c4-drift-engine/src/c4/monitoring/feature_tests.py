"""
Feature drift tests for one production window vs the baseline.

numeric features     -> PSI (main score) + KS test (statistic and p-value, secondary evidence)
categorical features -> Jensen-Shannon distance between category shares
"""
import pandas as pd
from scipy.stats import ks_2samp

from src.c4.monitoring.stats import bin_shares, psi, category_shares, jsd_from_dicts


def numeric_drift(values: pd.Series, base: dict, reference: pd.Series) -> dict:
    cur = pd.to_numeric(values, errors="coerce").dropna()
    shares = bin_shares(cur, base["bin_edges"])
    score = psi(base["bin_shares"], shares)
    if len(cur) > 1:
        ks = ks_2samp(reference.dropna(), cur)
        ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
    else:
        ks_stat, ks_p = 1.0, 0.0
    return {"kind": "numeric", "test": "PSI", "score": score,
            "ks_stat": ks_stat, "ks_p": ks_p, "mean": float(cur.mean()) if len(cur) else None}


def categorical_drift(values: pd.Series, base: dict) -> dict:
    score = jsd_from_dicts(base["shares"], category_shares(values))
    return {"kind": "categorical", "test": "JSD", "score": score}


def compute(window_X: pd.DataFrame, baseline: dict, reference: pd.DataFrame) -> dict:
    """Return {feature: {...score...}} for every monitored feature."""
    out = {}
    for f in baseline["numeric_features"]:
        out[f] = numeric_drift(window_X[f], baseline["numeric"][f], reference[f])
    for f in baseline["categorical_features"]:
        out[f] = categorical_drift(window_X[f], baseline["categorical"][f])
    return out
