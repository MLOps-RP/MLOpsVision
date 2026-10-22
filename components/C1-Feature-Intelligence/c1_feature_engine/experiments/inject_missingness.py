"""Synthetic MCAR / MAR / MNAR injection on numeric columns, with ground-truth mask (R3)."""
import numpy as np
import pandas as pd


def inject(X: pd.DataFrame, rate: float, mechanism: str, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (X_missing, mask) where mask is True on removed cells. Numeric columns only.
    MCAR: uniform. MAR: P(miss) rises with rank of a random *other* column. MNAR: rises with the column's own value.
    P = min(1, 2*rate*rank_pct) keeps mean rate ~= `rate` (rank_pct has mean 0.5)."""
    rng = np.random.default_rng(seed)
    num = X.select_dtypes("number").columns.tolist()
    mask = pd.DataFrame(False, index=X.index, columns=X.columns)
    for c in num:
        if mechanism == "MCAR":
            p = np.full(len(X), rate)
        else:
            driver = c if mechanism == "MNAR" else rng.choice([k for k in num if k != c] or [c])
            p = np.minimum(1.0, 2 * rate * X[driver].rank(pct=True).fillna(0.5).to_numpy())
        mask[c] = rng.random(len(X)) < p
    Xm = X.copy()
    Xm[mask] = np.nan
    return Xm, mask
