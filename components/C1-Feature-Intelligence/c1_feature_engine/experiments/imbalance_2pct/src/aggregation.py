"""
aggregation.py  -  turn 1350 raw fold rows into the summary tables in results/.

KEY IDEA - the unit of analysis is the DATASET
----------------------------------------------
Folds, seeds and models inside one dataset are strongly correlated (same rows, same thinning). Treating the
1350 rows as independent would make every confidence interval far too narrow. So:

    raw row  --mean over seeds, folds, models-->  one number per (method, dataset)   (5 numbers per method)
    uncertainty = t-interval across those 5 datasets   (df = 4)

That is why CIs are wide, and why only PAIRED differences between methods (see statistics.py) are informative.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .config import MET, METHODS


def tci(v, level=.95):
    """Two-sided Student-t confidence interval of the mean of v. Returns (lo, hi); (nan, nan) if n < 2."""
    v = np.asarray(v, float)
    n = len(v)
    if n < 2:
        return np.nan, np.nan
    h = stats.t.ppf((1 + level) / 2, n - 1) * v.std(ddof=1) / np.sqrt(n)    # half-width = t * s / sqrt(n)
    return v.mean() - h, v.mean() + h


def agg(d, by):
    """Group raw rows by `by` (e.g. ["dataset","method"]) and summarise every metric in MET.

    mean / sd are over ALL raw rows of the group.  The CI is a t-interval over the per-SEED means,
    so seeds (not individual correlated folds) are the sampling unit inside a group.
    Used for results/per_dataset_results.csv, per_model_results.csv, per_dataset_model_results.csv.
    """
    rows = []
    for k, g in d.groupby(by):
        k = k if isinstance(k, tuple) else (k,)
        r = dict(zip(by, k))
        r["n_cells"] = len(g)
        seedm = g.groupby("seed")[MET].mean()           # collapse folds/models first -> one value per seed
        for m in MET:
            lo, hi = tci(seedm[m])
            r[f"{m}_mean"], r[f"{m}_sd"], r[f"{m}_ci_lo"], r[f"{m}_ci_hi"] = g[m].mean(), g[m].std(), lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


def overall(d):
    """Headline table (results/aggregate_overall.csv): one row per method.

    Step 1: mean over seeds x models x folds  ->  `dm`, one value per (method, dataset)  [5 per method]
    Step 2: mean and t-CI over those 5 dataset values.
    Returns (overall table in METHODS order, dm) - `dm` is reused by the figures (the black dots).
    """
    dm = d.groupby(["method", "dataset"])[MET].mean().reset_index()
    rows = []
    for m, g in dm.groupby("method"):
        r = {"method": m, "n_datasets": len(g)}
        for k in MET:
            lo, hi = tci(g[k])
            r[f"{k}_mean"], r[f"{k}_sd"], r[f"{k}_ci_lo"], r[f"{k}_ci_hi"] = g[k].mean(), g[k].std(), lo, hi
        rows.append(r)
    return pd.DataFrame(rows).set_index("method").loc[METHODS].reset_index(), dm


def gains_over_b0(dm):
    """results/comparisons.csv: how much each method beats the default B0, per dataset then averaged.

    gain(method, dataset) = metric(method, dataset) - metric(B0, dataset)      -> a PAIRED difference
    reported: mean gain over the 5 datasets, its 95% t-CI, and how many datasets improved on macro-F1.
    """
    b0 = dm[dm.method == "B0"].set_index("dataset")
    comp = []
    for m in METHODS:
        row = {"method": m}
        for k in ["f1_macro", "bal_acc", "min_f1", "pr_auc_min"]:
            g = dm[dm.method == m].set_index("dataset")[k] - b0[k]
            lo, hi = tci(g)
            row[f"gain_{k}_mean"], row[f"gain_{k}_ci_lo"], row[f"gain_{k}_ci_hi"] = g.mean(), lo, hi
        row["datasets_improved_f1"] = int(((dm[dm.method == m].set_index("dataset").f1_macro - b0.f1_macro) > 0).sum())
        comp.append(row)
    return pd.DataFrame(comp)
