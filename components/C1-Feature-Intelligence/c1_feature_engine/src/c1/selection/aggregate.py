"""Rank aggregation -> predictive power P_j in [0, 1] (1 = best)."""
import pandas as pd


def aggregate(scores: pd.DataFrame, method: str = "rank_mean") -> pd.Series:
    """rank_mean: mean normalised rank (ties averaged). borda: Borda points (ties broken by order), rescaled to [0,1]."""
    n = len(scores)
    if n == 1:
        return pd.Series(1.0, index=scores.index)
    r = scores.rank(ascending=False, method="average" if method == "rank_mean" else "first")  # 1 = best
    if method not in ("rank_mean", "borda"):
        raise ValueError(method)
    return ((n - r) / (n - 1)).mean(axis=1)    # borda points (n-r) summed / (m*(n-1)) == mean normalised rank
