"""Stability S_j: how often feature j lands in the top-k over K bootstrap resamples (MI + SHAP; RFE skipped for cost)."""
import numpy as np
import pandas as pd

from c1.selection.aggregate import aggregate
from c1.selection.rankers import mi_rank, shap_rank


def selection_frequency(X: pd.DataFrame, y, k: int, K: int = 10, seed: int = 0, max_rows: int = 1500) -> pd.Series:
    y, rng = np.asarray(y), np.random.default_rng(seed)
    cnt = pd.Series(0.0, index=X.columns)
    for b in range(K):
        idx = rng.choice(len(X), min(len(X), max_rows), replace=True)
        Xb, yb = X.iloc[idx], y[idx]
        P = aggregate(pd.DataFrame({"mi": mi_rank(Xb, yb, seed + b), "shap": shap_rank(Xb, yb, seed + b)}))
        cnt[P.nlargest(k).index] += 1
    return cnt / K


def nogueira(masks: np.ndarray) -> float:
    """Nogueira et al. 2018 stability of M selected sets over d features (masks: M x d bool). 1 = identical sets."""
    M, d = masks.shape
    kbar = masks.sum(1).mean()
    if M < 2 or kbar in (0, d):
        return 1.0
    p = masks.mean(0)
    return float(1 - (M / (M - 1) * p * (1 - p)).mean() / ((kbar / d) * (1 - kbar / d)))
