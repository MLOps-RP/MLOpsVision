"""Transformation library (FR4). Pure functions; any fitted statistic (edges, maps) lives in the generator."""
import numpy as np
import pandas as pd

# name -> (fn, applicable(train_series)). Domain guards keep log/sqrt from producing NaN on negatives.
# z-score is deliberately absent: it is a linear rescale, so it is a no-op after StandardScaler / for trees.
UNARY = {
    "log1p": (np.log1p, lambda s: s.min() >= 0),
    "sqrt": (np.sqrt, lambda s: s.min() >= 0),
    "square": (np.square, lambda s: True),
}

BINARY = {
    "ratio": lambda a, b: a / b.where(b != 0),
    "product": lambda a, b: a * b,
    "diff": lambda a, b: a - b,
    "sum": lambda a, b: a + b,
}


def quantile_edges(s: pd.Series, q: int = 5) -> np.ndarray:
    return np.unique(s.quantile(np.linspace(0, 1, q + 1)[1:-1]).to_numpy())


def apply_bins(s: pd.Series, edges: np.ndarray) -> pd.Series:
    out = pd.Series(np.searchsorted(edges, s.to_numpy(), side="right").astype(float), index=s.index)
    return out.where(s.notna())


def smoothed_target_map(col: pd.Series, pos: np.ndarray, prior: float, m: float = 10.0) -> pd.Series:
    """Per-category smoothed positive rate: (sum + m*prior) / (n + m)."""
    g = pd.DataFrame({"c": col.to_numpy(), "t": pos}).groupby("c")["t"].agg(["sum", "count"])
    return (g["sum"] + m * prior) / (g["count"] + m)
