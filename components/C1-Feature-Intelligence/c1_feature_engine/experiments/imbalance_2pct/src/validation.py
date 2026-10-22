"""
validation.py  -  sanity checks run on results/raw_fold_results.csv BEFORE any number is reported.

If any assertion fails, analysis stops. Nothing downstream (tables, tests, figures) is trusted until these pass.
"""
from .config import METHODS


def validate(d, man):
    """d = raw fold results, man = dataset_manifest.csv. Returns the number of selector fallbacks (reported, not hidden)."""
    # 1. Completeness: datasets*seeds (= len(man)) x 5 folds x 3 models x 6 methods rows, no more, no less.
    exp = len(man) * 5 * 3 * len(METHODS)
    assert len(d) == exp, f"expected {exp} rows, got {len(d)}"

    # 2. No cell was counted twice.
    key = ["dataset", "seed", "fold", "model", "method"]
    assert not d.duplicated(key).any(), "duplicate result rows"

    # 3. Every (dataset, seed, fold, model) has all 6 methods -> comparisons are properly paired.
    assert set(d.method) == set(METHODS) and d.groupby(key[:4]).method.nunique().eq(6).all(), "missing method"

    # 4. All score columns are valid probabilities-like values in [0, 1] (also catches NaN, since NaN fails .between).
    for m in ["f1_macro", "bal_acc", "roc_auc", "pr_auc_min", "min_precision", "min_recall", "min_f1", "accuracy"]:
        assert d[m].between(-1e-9, 1 + 1e-9).all(), f"{m} out of [0,1] or NaN"

    # 5. The positive class used for the minority metrics really is the minority class recorded at data-prep time,
    #    and the share seen during CV matches the manifest. Guards against label mix-ups between datasets.
    mm = man.set_index(["dataset", "seed"])
    for (n, s), g in d.groupby(["dataset", "seed"]):
        assert (g.pos_label == mm.loc[(n, s), "minority_label"]).all(), f"positive label != minority for {n},{s}"
        assert abs(g.minority_share.mean() - mm.loc[(n, s), "prepared_minority_share"]) < .01, f"minority share mismatch {n},{s}"

    # 6. Every test fold contributed confusion counts.
    assert (d[["tn", "fp", "fn", "tp"]].sum(axis=1) > 0).all()

    # 7. Fallbacks (selected pipeline failed to fit -> anchor used) are COUNTED and written into the report.
    return int(d.opts.str.contains("fallback").sum())
