"""
Small statistics helpers used by both the baseline builder and the monitor.
Keeping them in one file guarantees the baseline and the live windows
are measured in exactly the same way.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

EPS = 1e-4  # avoids log(0) and division by zero


# ---------------- numeric features ----------------
def decile_edges(values: pd.Series, n_bins: int = 10) -> list[float]:
    """Interior bin edges taken from the baseline's quantiles (duplicates removed)."""
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    qs = np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1])
    return sorted(set(float(q) for q in qs))


def bin_shares(values: pd.Series, edges: list[float]) -> np.ndarray:
    """Share of non-missing values falling in each bin defined by `edges`."""
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    n_bins = len(edges) + 1
    if len(v) == 0:
        return np.full(n_bins, 1.0 / n_bins)
    idx = np.searchsorted(np.asarray(edges), v, side="right")
    counts = np.bincount(idx, minlength=n_bins)
    return counts / counts.sum()


def psi(base_shares, cur_shares) -> float:
    """Population Stability Index: sum((cur - base) * ln(cur / base))."""
    b = np.asarray(base_shares, dtype=float) + EPS
    c = np.asarray(cur_shares, dtype=float) + EPS
    b, c = b / b.sum(), c / c.sum()
    return float(np.sum((c - b) * np.log(c / b)))


# ---------------- categorical features ----------------
def category_shares(values: pd.Series) -> dict:
    """Share of each category among non-missing values."""
    v = values.dropna().astype(str)
    if len(v) == 0:
        return {}
    return (v.value_counts(normalize=True)).to_dict()


def jsd_from_dicts(base: dict, cur: dict) -> float:
    """Jensen-Shannon distance (0 = identical, 1 = completely different)."""
    keys = sorted(set(base) | set(cur))
    p = np.array([base.get(k, 0.0) for k in keys]) + EPS
    q = np.array([cur.get(k, 0.0) for k in keys]) + EPS
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2))


def jsd_arrays(p, q) -> float:
    p = np.asarray(p, dtype=float) + EPS
    q = np.asarray(q, dtype=float) + EPS
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2))


# ---------------- model outputs ----------------
def confidence_hist(conf, n_bins: int = 10) -> np.ndarray:
    counts, _ = np.histogram(np.asarray(conf, dtype=float), bins=n_bins, range=(0.0, 1.0))
    total = counts.sum()
    return counts / total if total else np.full(n_bins, 1.0 / n_bins)


def classification_summary(y_true, y_pred) -> dict:
    """Accuracy, F1 and per-class recall (TPR for class 1, TNR for class 0)."""
    from sklearn.metrics import accuracy_score, f1_score
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    pos, neg = y_true == 1, y_true == 0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tpr": float((y_pred[pos] == 1).mean()) if pos.any() else float("nan"),
        "tnr": float((y_pred[neg] == 0).mean()) if neg.any() else float("nan"),
        "positive_share": float(y_true.mean()),
        "n": int(len(y_true)),
    }
