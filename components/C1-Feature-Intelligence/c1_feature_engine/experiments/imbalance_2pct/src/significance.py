"""
statistics.py  -  significance testing (results/statistical_tests.csv).

Question answered: "is Rules better than comparator C, and is that more than noise?"

Design of the tests
-------------------
* PAIRED: for every (dataset, seed, fold, model) cell Rules and C were run on the identical split, so we take
  the cell-wise difference  metric(rules) - metric(C).
* Those differences are averaged WITHIN each dataset first -> 5 numbers (one per dataset). The dataset is the
  inferential unit (n = 5), NOT the hundreds of correlated cells.
* On those 5 numbers we report:
    mean_diff, 95% t-CI, Cohen's d_z (effect size), how many datasets favour Rules / the comparator,
    exact two-sided Wilcoxon signed-rank p, and a paired one-sample t-test p.
* LIMIT OF THE DESIGN: with n = 5 the smallest possible exact Wilcoxon p is 2/2^5 = 0.0625 > 0.05, so NO comparison
  can ever be "significant" here. The consistent sign across datasets and the effect size are the informative signals.
* Multiple comparisons: Holm correction over the 4 PRE-SPECIFIED primary tests (macro-F1 vs B0, B_thr, Random, Brute).
  B_prac and all non-macro-F1 metrics are labelled "exploratory" and are NOT corrected.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .aggregation import tci


def dataset_diffs(d, a, b, metric):
    """Per-dataset mean of the paired difference (method a - method b) on `metric`."""
    x = d.pivot_table(index=["dataset", "seed", "fold", "model"], columns="method", values=metric)
    return (x[a] - x[b]).groupby("dataset").mean()


def tests(d):
    rows = []
    # (metric, is_primary): only macro-F1 is the pre-specified primary metric.
    for metric, primary in [("f1_macro", True), ("bal_acc", False), ("min_f1", False), ("pr_auc_min", False)]:
        for c in ["B0", "B_thr", "random", "brute", "B_prac"]:
            diff = dataset_diffs(d, "rules", c, metric)           # 5 values: rules - c, one per dataset
            lo, hi = tci(diff)
            # Exact Wilcoxon signed-rank (no normal approximation - n is tiny). If all diffs are 0, p = 1.
            w = stats.wilcoxon(diff, method="exact").pvalue if (diff != 0).any() else 1.0
            t = stats.ttest_1samp(diff, 0).pvalue if diff.std() > 0 else np.nan
            rows.append(dict(
                metric=metric, comparison=f"rules - {c}", n_datasets=len(diff),
                mean_diff=diff.mean(), ci95_lo=lo, ci95_hi=hi,
                cohens_dz=diff.mean() / diff.std(ddof=1) if diff.std() > 0 else np.nan,   # paired effect size
                datasets_rules_better=int((diff > 0).sum()), datasets_rules_worse=int((diff < 0).sum()),
                wilcoxon_exact_p=w, ttest_p=t,
                # B_prac is excluded from the primary family even for macro-F1 (it was not one of the 4 pre-specified).
                family="primary" if primary and c != "B_prac" else "exploratory"))
    t = pd.DataFrame(rows)

    # Holm step-down correction over the primary family (4 tests):
    #   sort p ascending, multiply the i-th smallest by (m - i), cap at 1, enforce monotonicity with a running max.
    fam = (t.family == "primary")
    p = t.loc[fam, "wilcoxon_exact_p"]
    o = p.sort_values()
    m = len(o)
    adj = np.maximum.accumulate([min(1, (m - i) * v) for i, v in enumerate(o)])
    t.loc[o.index, "p_holm"] = adj
    return t
