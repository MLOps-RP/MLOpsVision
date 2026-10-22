"""Controlled stress setting for Gap 1 (NOT natural data): subsample minority classes to a target share
and inject MAR missingness. Labelled 'stress' everywhere it is reported."""
import numpy as np
import pandas as pd

from experiments.inject_missingness import inject


def subsample_minority(X: pd.DataFrame, y: np.ndarray, share: float, seed: int, floor: int = 40):
    """All non-majority classes together make `share` of the rows (never below `floor` rows each, never above what exists)."""
    rng = np.random.default_rng(seed)
    cls, cnt = np.unique(y, return_counts=True)
    maj = cls[cnt.argmax()]
    per = max(floor, int(round(cnt.max() * share / (1 - share) / max(len(cls) - 1, 1))))
    keep = [np.flatnonzero(y == maj)]
    for c, n in zip(cls, cnt):
        if c != maj:
            idx = np.flatnonzero(y == c)
            keep.append(rng.choice(idx, min(per, n), replace=False))
    keep = np.sort(np.concatenate(keep))
    return X.iloc[keep].reset_index(drop=True), y[keep]


def stress(X, y, seed, share=0.03, rate=0.2, mech="MAR"):
    X, y = subsample_minority(X, y, share, seed)
    num = X.select_dtypes("number")
    Xm, _ = inject(num, rate, mech, seed)
    X = X.copy()
    X[num.columns] = Xm
    return X, y
