"""Dataset meta-features (FR1). Cheap, label-aware only where the selector needs it."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from c1.schemas import DatasetProfile

_MAX_MISS_COLS = 50  # ponytail: first 50 columns-with-missing for MAR/target tests; sample if wider data matters


def _cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    t = pd.crosstab(a, b).values
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(t, correction=False)[0]
    return float(np.sqrt(chi2 / (t.sum() * (min(t.shape) - 1))))


def profile(X: pd.DataFrame, y) -> DatasetProfile:
    y = np.asarray(y)
    num = X.select_dtypes("number")
    miss = X.isna()
    cols = miss.columns[miss.any()][:_MAX_MISS_COLS]

    mar, tgt = 0.0, 0.0
    for c in cols:
        ind = miss[c].astype(float)
        if ind.nunique() < 2:
            continue
        with np.errstate(all="ignore"):
            r = num.drop(columns=[c], errors="ignore").corrwith(ind).abs().max()
        mar = max(mar, 0.0 if pd.isna(r) else float(r))
        tgt = max(tgt, _cramers_v(ind.values, y))

    counts = np.unique(y, return_counts=True)[1]
    p = counts / counts.sum()
    ent = float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
    n, d = X.shape
    return DatasetProfile(
        n_rows=n, n_features=d, n_numeric=num.shape[1], n_categorical=d - num.shape[1],
        frac_categorical=(d - num.shape[1]) / d, dim_ratio=n / d,
        mean_abs_skew=float(np.nan_to_num(num.skew().abs().mean())) if num.shape[1] else 0.0,
        missing_ratio=float(miss.values.mean()),
        rows_with_missing_ratio=float(miss.any(axis=1).mean()),
        n_cols_with_missing=int(miss.any().sum()),
        max_col_missing=float(miss.mean().max()),
        mar_signal=mar, miss_target_assoc=tgt,
        n_classes=len(counts), imbalance_ratio=float(counts.max() / counts.min()),
        min_class_count=int(counts.min()), minority_share=float(p.min()), class_entropy=ent,
    )
