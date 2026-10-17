"""Redundancy R_j (mRMR-style, greedy): max |Spearman| between feature j and any feature ranked above it by P."""
import numpy as np
import pandas as pd

from c1.selection.rankers import prep


def redundancy(X: pd.DataFrame, order: list[str], max_rows: int = 2000, seed: int = 0) -> pd.Series:
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
    Z, _ = prep(X[order])
    C = np.abs(np.nan_to_num(pd.DataFrame(Z).rank().corr().to_numpy()))
    np.fill_diagonal(C, 0)
    r = [0.0] + [float(C[j, :j].max()) for j in range(1, len(order))]
    return pd.Series(r, index=order)


def mean_pairwise(X: pd.DataFrame, cols: list[str], max_rows: int = 2000, seed: int = 0) -> float:
    """Set-level redundancy for reporting: mean over features of max |Spearman| with another member."""
    if len(cols) < 2:
        return 0.0
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
    Z, _ = prep(X[cols])
    C = np.abs(np.nan_to_num(pd.DataFrame(Z).rank().corr().to_numpy()))
    np.fill_diagonal(C, 0)
    return float(C.max(axis=1).mean())
