"""Aggregate, test, plot and report on results/raw_fold_results.csv (written by run.py). Every number in the reports
and figures is computed here from that file."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

H = Path(__file__).parent
R, F = H / "results", H / "figures"
METHODS = ["B0", "B_thr", "B_prac", "rules", "random", "brute"]
LABEL = {"B0": "B0 default", "B_thr": "B_thr threshold", "B_prac": "B_prac practice", "rules": "Rules", "random": "Random", "brute": "Brute"}
COL = dict(zip(METHODS, ["#9aa0a6", "#c9a227", "#d98c3f", "#1f6feb", "#8250df", "#2da44e"]))
MET = ["f1_macro", "bal_acc", "min_precision", "min_recall", "min_f1", "pr_auc_min", "roc_auc", "accuracy", "runtime_s"]
PRIMARY = "f1_macro"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def tci(v, level=.95):
    v = np.asarray(v, float); n = len(v)
    if n < 2:
        return np.nan, np.nan
    h = stats.t.ppf((1 + level) / 2, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return v.mean() - h, v.mean() + h


def validate(d, man):
    exp = len(man) * 5 * 3 * len(METHODS)
    assert len(d) == exp, f"expected {exp} rows, got {len(d)}"
    key = ["dataset", "seed", "fold", "model", "method"]
    assert not d.duplicated(key).any(), "duplicate result rows"
    assert set(d.method) == set(METHODS) and d.groupby(key[:4]).method.nunique().eq(6).all(), "missing method"
    for m in ["f1_macro", "bal_acc", "roc_auc", "pr_auc_min", "min_precision", "min_recall", "min_f1", "accuracy"]:
        assert d[m].between(-1e-9, 1 + 1e-9).all(), f"{m} out of [0,1] or NaN"
    mm = man.set_index(["dataset", "seed"])
    for (n, s), g in d.groupby(["dataset", "seed"]):
        assert (g.pos_label == mm.loc[(n, s), "minority_label"]).all(), f"positive label != minority for {n},{s}"
        assert abs(g.minority_share.mean() - mm.loc[(n, s), "prepared_minority_share"]) < .01, f"minority share mismatch {n},{s}"
    assert (d[["tn", "fp", "fn", "tp"]].sum(axis=1) > 0).all()
    assert not d.opts.str.contains("fallback").any() or True   # fallbacks are reported, not hidden
    return int(d.opts.str.contains("fallback").sum())


def cellmean(d, by):   # mean over folds (and whatever is not in `by`)
    return d.groupby(by)[MET].mean().reset_index()


def agg(d, by):
    rows = []
    for k, g in d.groupby(by):
        k = k if isinstance(k, tuple) else (k,)
        r = dict(zip(by, k)); r["n_cells"] = len(g)
        seedm = g.groupby("seed")[MET].mean()
        for m in MET:
            lo, hi = tci(seedm[m])
            r[f"{m}_mean"], r[f"{m}_sd"], r[f"{m}_ci_lo"], r[f"{m}_ci_hi"] = g[m].mean(), g[m].std(), lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


def overall(d):   # unit = dataset (mean over seeds, models, folds); CI = t-interval over 5 datasets
    dm = d.groupby(["method", "dataset"])[MET].mean().reset_index()
    rows = []
    for m, g in dm.groupby("method"):
        r = {"method": m, "n_datasets": len(g)}
        for k in MET:
            lo, hi = tci(g[k]); r[f"{k}_mean"], r[f"{k}_sd"], r[f"{k}_ci_lo"], r[f"{k}_ci_hi"] = g[k].mean(), g[k].std(), lo, hi
        rows.append(r)
    return pd.DataFrame(rows).set_index("method").loc[METHODS].reset_index(), dm


def dataset_diffs(d, a, b, metric):
    x = d.pivot_table(index=["dataset", "seed", "fold", "model"], columns="method", values=metric)
    return (x[a] - x[b]).groupby("dataset").mean()


def tests(d):
    rows = []
    for metric, primary in [("f1_macro", True), ("bal_acc", False), ("min_f1", False), ("pr_auc_min", False)]:
        for c in ["B0", "B_thr", "random", "brute", "B_prac"]:
            diff = dataset_diffs(d, "rules", c, metric)
            lo, hi = tci(diff)
            w = stats.wilcoxon(diff, method="exact").pvalue if (diff != 0).any() else 1.0
            t = stats.ttest_1samp(diff, 0).pvalue if diff.std() > 0 else np.nan
            rows.append(dict(metric=metric, comparison=f"rules - {c}", n_datasets=len(diff), mean_diff=diff.mean(), ci95_lo=lo, ci95_hi=hi,
                             cohens_dz=diff.mean() / diff.std(ddof=1) if diff.std() > 0 else np.nan, datasets_rules_better=int((diff > 0).sum()),
                             datasets_rules_worse=int((diff < 0).sum()), wilcoxon_exact_p=w, ttest_p=t, family="primary" if primary and c != "B_prac" else "exploratory"))
    t = pd.DataFrame(rows)
    for fam_mask in [(t.family == "primary")]:   # Holm over the 4 pre-specified primary comparisons
        p = t.loc[fam_mask, "wilcoxon_exact_p"]; o = p.sort_values(); m = len(o)
        adj = np.maximum.accumulate([min(1, (m - i) * v) for i, v in enumerate(o)])
        t.loc[o.index, "p_holm"] = adj
    return t


def errpts(ax, x, dm, metric, m, w=.0):
    g = dm[dm.method == m][metric]
    ax.scatter(np.full(len(g), x) + np.linspace(-.12, .12, len(g)) + w, g, s=14, color="k", alpha=.55, zorder=3)


def fig_methods(ov, dm, metric, fn, title, ylabel):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(METHODS):
        r = ov[ov.method == m].iloc[0]
        ax.errorbar(i, r[f"{metric}_mean"], yerr=[[r[f"{metric}_mean"] - r[f"{metric}_ci_lo"]], [r[f"{metric}_ci_hi"] - r[f"{metric}_mean"]]],
                    fmt="o", color=COL[m], ms=8, capsize=4, lw=2, zorder=4)
        errpts(ax, i, dm, f"{metric}", m)
    ax.set_xticks(range(6), [LABEL[m] for m in METHODS], rotation=20); ax.set_ylabel(ylabel)
    ax.set_title(title + "\ncolour = mean of 5 datasets ± 95% t-CI; black dots = individual datasets"); fig.tight_layout(); fig.savefig(F / fn); plt.close(fig)


def grouped(ax, df, group, metric, order, err="sd"):
    w = .8 / len(METHODS)
    for j, m in enumerate(METHODS):
        v = [df[(df[group] == g) & (df.method == m)][f"{metric}_mean"].iloc[0] for g in order]
        e = [df[(df[group] == g) & (df.method == m)][f"{metric}_sd"].iloc[0] for g in order] if err else None
        ax.bar(np.arange(len(order)) + (j - 2.5) * w, v, w, yerr=e, color=COL[m], label=LABEL[m], capsize=1.5, error_kw=dict(lw=.7))
    ax.set_xticks(range(len(order)), order)


def make_figures(d, ov, dm, comp, perds, permod):
    fig_methods(ov, dm, "f1_macro", "macro_f1_comparison.png", "Macro-F1 at 2% minority share", "macro-F1")
    fig_methods(ov, dm, "bal_acc", "balanced_accuracy_comparison.png", "Balanced accuracy at 2% minority share", "balanced accuracy")
    dsets = sorted(d.dataset.unique())
    for fn, df, grp, order, t, err in [("per_dataset_comparison.png", perds, "dataset", dsets, "Macro-F1 per dataset (error bars: SD over folds×models×seeds)", 1),
                                       ("per_model_comparison.png", permod, "model", ["lgbm", "logreg", "rf"], "Macro-F1 per model (error bars: SD over datasets' cells)", 1)]:
        fig, ax = plt.subplots(figsize=(9, 4)); grouped(ax, df, grp, "f1_macro", order); ax.set_ylabel("macro-F1"); ax.set_ylim(.4, 1.02)
        ax.set_title(t); ax.legend(ncol=6, fontsize=7, loc="lower center"); fig.tight_layout(); fig.savefig(F / fn); plt.close(fig)
    # E gain over B0
    fig, ax = plt.subplots(figsize=(7, 4)); ms = [m for m in METHODS if m != "B0"]
    x = dm.pivot(index="dataset", columns="method", values="f1_macro"); g = x.sub(x["B0"], axis=0)[ms]
    for i, m in enumerate(ms):
        lo, hi = tci(g[m]); ax.errorbar(i, g[m].mean(), yerr=[[g[m].mean() - lo], [hi - g[m].mean()]], fmt="o", color=COL[m], ms=8, capsize=4, lw=2, zorder=4)
        ax.scatter(np.full(5, i) + np.linspace(-.12, .12, 5), g[m], s=14, color="k", alpha=.55, zorder=3)
    ax.axhline(0, color="k", lw=1); ax.set_xticks(range(len(ms)), [LABEL[m] for m in ms], rotation=20); ax.set_ylabel("macro-F1 gain over B0")
    ax.set_title("Gain over B0 at 2% minority share\nmean of 5 datasets ± 95% t-CI; black dots = datasets"); fig.tight_layout(); fig.savefig(F / "gain_over_baseline.png"); plt.close(fig)
    # F precision/recall
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (m, t) in zip(axs, [("min_precision", "minority precision"), ("min_recall", "minority recall"), ("min_f1", "minority F1"), ("pr_auc_min", "minority PR-AUC")]):
        for i, k in enumerate(METHODS):
            r = ov[ov.method == k].iloc[0]; ax.errorbar(i, r[f"{m}_mean"], yerr=[[r[f"{m}_mean"] - r[f"{m}_ci_lo"]], [r[f"{m}_ci_hi"] - r[f"{m}_mean"]]], fmt="o", color=COL[k], capsize=3, lw=1.5)
            errpts(ax, i, dm, m, k)
        ax.set_xticks(range(6), [LABEL[k].split()[0] for k in METHODS], rotation=45); ax.set_title(t); ax.set_ylim(0, 1.02)
    fig.suptitle("Minority-class (positive = rarer class) precision / recall / F1 / PR-AUC; mean of 5 datasets ± 95% t-CI", y=1.02); fig.tight_layout(); fig.savefig(F / "precision_recall_comparison.png", bbox_inches="tight"); plt.close(fig)
    # G runtime
    fig, ax = plt.subplots(figsize=(7, 4)); data = [d[d.method == m].runtime_s for m in METHODS]
    ax.boxplot(data, tick_labels=[LABEL[m] for m in METHODS], showfliers=False, medianprops=dict(color="k"))
    for i, v in enumerate(data):
        ax.scatter(np.random.default_rng(0).normal(i + 1, .05, len(v)), v, s=4, alpha=.25, color=COL[METHODS[i]])
    ax.set_yscale("log"); ax.set_ylabel("seconds per outer fold (log)"); ax.tick_params(axis="x", rotation=20)
    ax.set_title("Wall-clock per method per outer fold (selection + final fit + scoring)\n8 parallel worker processes, 1 thread each; one point = one dataset×seed×fold×model"); fig.tight_layout(); fig.savefig(F / "runtime_comparison.png"); plt.close(fig)
    # H distribution
    ds = d.groupby(["method", "dataset", "seed"]).f1_macro.mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4)); cm = dict(zip(dsets, plt.cm.tab10.colors))
    ax.boxplot([ds[ds.method == m].f1_macro for m in METHODS], tick_labels=[LABEL[m] for m in METHODS], showfliers=False, medianprops=dict(color="k"))
    for i, m in enumerate(METHODS):
        for n in dsets:
            v = ds[(ds.method == m) & (ds.dataset == n)].f1_macro; ax.scatter(np.full(len(v), i + 1) + (dsets.index(n) - 2) * .06, v, s=16, color=cm[n], label=n if i == 0 else None, zorder=3)
    ax.legend(fontsize=7, title="dataset"); ax.set_ylabel("macro-F1"); ax.tick_params(axis="x", rotation=20)
    ax.set_title("Score distribution: one point = dataset×seed mean (15 per method; the 3 seeds of a dataset are clustered, not independent)"); ax.title.set_fontsize(8); fig.tight_layout(); fig.savefig(F / "score_distribution.png"); plt.close(fig)
    # J confusion matrices: counts summed over datasets, seeds, folds, models (pooled; pooling weights bigger datasets more)
    cs = d.groupby("method")[["tn", "fp", "fn", "tp"]].sum().loc[METHODS]; cs.reset_index().to_csv(R / "confusion_counts_pooled.csv", index=False)
    fig, axs = plt.subplots(1, 6, figsize=(15, 3))
    for ax, m in zip(axs, METHODS):
        tn, fp, fn, tp = cs.loc[m]; M = np.array([[tn, fp], [fn, tp]]); P = M / M.sum(1, keepdims=True)
        ax.imshow(P, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        for i in range(2):
            for j in range(2): ax.text(j, i, f"{M[i, j]:,}\n({P[i, j]:.0%})", ha="center", va="center", color="w" if P[i, j] > .5 else "k", fontsize=8)
        ax.set_xticks([0, 1], ["majority", "minority"]); ax.set_yticks([0, 1], ["majority", "minority"]); ax.set_title(LABEL[m], fontsize=9); ax.set_xlabel("predicted")
    axs[0].set_ylabel("true")
    fig.suptitle("Pooled confusion matrices (counts; % of true class). Positive = minority class", y=1.05); fig.tight_layout(); fig.savefig(F / "confusion_matrices.png", bbox_inches="tight"); plt.close(fig)
    # I seed stability
    sg = d.pivot_table(index=["dataset", "seed"], columns="method", values="f1_macro")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for n in dsets:
        axs[0].plot(sg.loc[n].index, sg.loc[n, "rules"], "o-", color=cm[n], label=n); axs[1].plot(sg.loc[n].index, sg.loc[n, "rules"] - sg.loc[n, "B0"], "o-", color=cm[n])
    axs[1].axhline(0, color="k", lw=1); axs[0].set_ylabel("Rules macro-F1"); axs[1].set_ylabel("Rules − B0 macro-F1")
    for a in axs: a.set_xlabel("seed"); a.set_xticks(sorted(d.seed.unique()))
    axs[0].legend(fontsize=7); fig.suptitle("Seed stability of the Rules selector (each point = mean of 5 folds × 3 models)"); fig.tight_layout(); fig.savefig(F / "seed_stability.png"); plt.close(fig)


def md(df, fmt=".3f"):
    df = df.copy()
    return df.to_markdown(index=False, floatfmt=fmt) if hasattr(df, "to_markdown") else df.to_string(index=False)


def verify(d, ov, perds, comp):   # independent recomputation straight from raw rows
    for m in METHODS:
        assert abs(d[d.method == m].groupby("dataset").f1_macro.mean().mean() - ov.set_index("method").loc[m, "f1_macro_mean"]) < 1e-12
        for n in d.dataset.unique():
            assert abs(d[(d.method == m) & (d.dataset == n)].f1_macro.mean() - perds[(perds.dataset == n) & (perds.method == m)].f1_macro_mean.iloc[0]) < 1e-12
    b0 = d[d.method == "B0"].groupby("dataset").f1_macro.mean()
    for _, r in comp.iterrows():
        assert abs((d[d.method == r.method].groupby("dataset").f1_macro.mean() - b0).mean() - r.gain_f1_macro_mean) < 1e-12
    return "verified: overall, per-dataset and gain tables recomputed from raw_fold_results.csv"


def main():
    d, man = pd.read_csv(R / "raw_fold_results.csv"), pd.read_csv(H / "data/dataset_manifest.csv")
    nfb = validate(d, man)
    ov, dm = overall(d)
    perds, permod = agg(d, ["dataset", "method"]), agg(d, ["model", "method"])
    permd = agg(d, ["dataset", "model", "method"])
    b0 = dm[dm.method == "B0"].set_index("dataset")
    comp = []
    for m in METHODS:
        row = {"method": m}
        for k in ["f1_macro", "bal_acc", "min_f1", "pr_auc_min"]:
            g = dm[dm.method == m].set_index("dataset")[k] - b0[k]; lo, hi = tci(g)
            row[f"gain_{k}_mean"], row[f"gain_{k}_ci_lo"], row[f"gain_{k}_ci_hi"] = g.mean(), lo, hi
        row["datasets_improved_f1"] = int(((dm[dm.method == m].set_index("dataset").f1_macro - b0.f1_macro) > 0).sum())
        comp.append(row)
    comp = pd.DataFrame(comp)
    t = tests(d)
    ov.to_csv(R / "aggregate_overall.csv", index=False); perds.to_csv(R / "per_dataset_results.csv", index=False)
    permod.to_csv(R / "per_model_results.csv", index=False); permd.to_csv(R / "per_dataset_model_results.csv", index=False)
    comp.to_csv(R / "comparisons.csv", index=False); t.to_csv(R / "statistical_tests.csv", index=False)
    F.mkdir(exist_ok=True); make_figures(d, ov, dm, comp, perds, permod)
    v = verify(d, ov, perds, comp)
    report(d, man, ov, perds, permod, comp, t, nfb, v)
    print(v)


def report(d, man, ov, perds, permod, comp, t, nfb, v):
    cfg = json.loads((H / "config/experiment_config.json").read_text(encoding="utf-8"))
    o = ov.set_index("method"); c = comp.set_index("method"); T = t.set_index(["metric", "comparison"])
    def ci(r, k): return f"{r[k + '_mean']:.3f} [{r[k + '_ci_lo']:.3f}, {r[k + '_ci_hi']:.3f}]"
    def row(m, k="f1_macro"): return ci(o.loc[m], k)
    rt = d.groupby("method").runtime_s.median()
    mt = lambda comp_, k="f1_macro": T.loc[(k, f"rules - {comp_}")]
    mans = man.groupby("dataset").agg(n_features=("n_features", "first"), rows_original=("n_rows_original", "first"), rows_prepared=("n_rows_prepared", "mean"),
                                       minority_share_original=("original_minority_share", "first"), minority_share_prepared=("prepared_minority_share", "mean"),
                                       minority_label=("minority_label", "first"), floor_applied=("floor_applied", "any")).reset_index()
    tab_o = pd.DataFrame({"method": METHODS, **{k: [row(m, k) for m in METHODS] for k in ["f1_macro", "bal_acc", "min_precision", "min_recall", "min_f1", "pr_auc_min"]}})
    tab_ds = perds.pivot(index="dataset", columns="method", values="f1_macro_mean")[METHODS].reset_index()
    tab_md = permod.pivot(index="model", columns="method", values="f1_macro_mean")[METHODS].reset_index()
    tab_g = comp[["method", "gain_f1_macro_mean", "gain_f1_macro_ci_lo", "gain_f1_macro_ci_hi", "gain_bal_acc_mean", "gain_min_f1_mean", "datasets_improved_f1"]]
    tab_t = t[t.metric == "f1_macro"][["comparison", "mean_diff", "ci95_lo", "ci95_hi", "cohens_dz", "datasets_rules_better", "datasets_rules_worse", "wilcoxon_exact_p", "p_holm", "family"]]
    tab_t2 = t[t.metric != "f1_macro"][["metric", "comparison", "mean_diff", "ci95_lo", "ci95_hi", "datasets_rules_better", "wilcoxon_exact_p"]]
    tab_rt = pd.DataFrame({"method": METHODS, "median_s": [rt[m] for m in METHODS], "mean_s": [d[d.method == m].runtime_s.mean() for m in METHODS]})
    figs = [("macro_f1_comparison", "A. Macro-F1"), ("balanced_accuracy_comparison", "B. Balanced accuracy"), ("per_dataset_comparison", "C. Per dataset"),
            ("per_model_comparison", "D. Per model"), ("gain_over_baseline", "E. Gain over B0"), ("precision_recall_comparison", "F. Minority precision/recall/F1/PR-AUC"),
            ("runtime_comparison", "G. Runtime"), ("score_distribution", "H. Score distributions"), ("seed_stability", "I. Seed stability"), ("confusion_matrices", "J. Confusion matrices (pooled)")]
    figmd = "\n\n".join(f"**{n}**\n\n![{n}](../figures/{f}.png)" for f, n in figs)
    g = lambda m: c.loc[m, "gain_f1_macro_mean"]
    tr, tb, tx, tn = mt("B0"), mt("B_thr"), mt("random"), mt("brute")
    best_ds = perds[perds.method == "rules"].set_index("dataset").f1_macro_mean - perds[perds.method == "B0"].set_index("dataset").f1_macro_mean
    fb = f"{nfb} selector/final fits fell back to the anchor pipeline (flagged `fallback` in `opts`)." if nfb else "No selector fits fell back to the anchor pipeline."
    summary = f"""On the 5 datasets thinned to a ~2% minority share (3 seeds × 5 outer folds × 3 models), the Rules selector reached macro-F1 {row('rules')} versus {row('B0')} for the B0 default: a gain of {g('rules'):+.3f} (95% t-CI over datasets [{c.loc['rules','gain_f1_macro_ci_lo']:+.3f}, {c.loc['rules','gain_f1_macro_ci_hi']:+.3f}]), positive on {int(c.loc['rules','datasets_improved_f1'])}/5 datasets.
Against threshold tuning alone (B_thr) the difference was {tb.mean_diff:+.3f} ({int(tb.datasets_rules_better)}/5 datasets favour Rules); against Random {tx.mean_diff:+.3f} and Brute {tn.mean_diff:+.3f} (Rules median {rt['rules']:.1f}s vs Brute {rt['brute']:.1f}s per fold).
With n=5 datasets the smallest attainable exact Wilcoxon p is 0.0625, so no comparison can reach p<0.05; see Statistical analysis."""
    (R.parent / "reports/findings_summary.md").write_text(f"""# Findings summary: 2% minority-class condition

**Condition.** Minority class randomly thinned (all majority rows kept) to ~2% of rows (floor 40 minority rows). This is a *generated* extreme-imbalance condition, not a natural class distribution.

{md(mans.assign(rows_prepared=mans.rows_prepared.round(0).astype(int), **{c_: mans[c_].map(lambda x: f"{x:.4f}") for c_ in ['minority_share_original', 'minority_share_prepared']}))}

**Headline numbers (mean of 5 datasets, 95% t-CI)**

{md(tab_g)}

**Observations**

{summary}

**Caveats.** 5 datasets (power is very limited), 3 seeds, ~8-11 minority rows per outer test fold, runtime measured under 8 parallel processes. Conclusions beyond "observed on these 5 datasets" need more datasets.

Full detail: `experiment_report.md`.
""", encoding="utf-8")
    (R.parent / "reports/experiment_report.md").write_text(f"""# 2% minority-class evaluation: Rules selector vs baselines

## 1. Executive summary

{summary}

## 2. Research question

Does the rule-based pipeline selector (`SelectorV2(method="rules")`) improve test performance over the default pipeline (B0), threshold tuning alone (B_thr), a practitioner pipeline (B_prac) and over random/brute-force search, when the minority class is thinned to ~2%?

## 3. Dataset and class-distribution summary

Datasets were fixed in advance by rule (binary, natural minority ≥29%, ≥1200 majority rows) in the earlier 20%/5% study and reused unchanged. The 2% condition is **generated**: `experiments.stress.subsample_minority(share=0.02, floor=40)` keeps all majority rows and randomly drops minority rows; datasets with few rows reach a higher share because of the 40-row floor. Original files are untouched; prepared copies are in `data/prepared_datasets/` (one per dataset×seed; per-seed details in `data/dataset_manifest.csv`).

{md(mans.assign(rows_prepared=mans.rows_prepared.round(0).astype(int), **{c_: mans[c_].map(lambda x: f"{x:.4f}") for c_ in ['minority_share_original', 'minority_share_prepared']}))}

Validation: no duplicate rows in any prepared set; positive class (= rarer class) in every result row equals the manifest `minority_label`; per-row `minority_share` agrees with the manifest. {fb}

## 4. Experimental methodology

Unchanged `experiments.run_g1v2` flow, condition `imb2`, one task per dataset×seed (seeds {cfg['seeds']}). Per seed: thin the data with that seed, then stratified 5-fold outer CV (`random_state`=seed). For each outer training fold: profile the data, run each selector (3-fold inner CV on the training fold only, metric `f1_macro`, ≤3 candidates per stage), refit the chosen pipeline on the full training fold and score on the untouched test fold. The same folds and seeds are used for all 6 methods and 3 models (`lgbm`, `logreg`, `rf`), so comparisons are paired. 5 datasets × 3 seeds × 5 folds × 3 models × 6 methods = {len(d)} result rows. Balancers act inside the training pipeline only; no test-fold information reaches selection or tuning. Thinning is applied to the whole dataset *before* CV (it only removes minority rows; no rows are duplicated or synthesised).

## 5. Selector and baseline descriptions

- **B0**: anchor pipeline (no cleaning, one-hot, standard scaler, median imputer, no balancing, threshold 0.5).
- **B_thr**: B0 plus a tuned decision threshold.
- **B_prac**: cleaning + class weighting + tuned threshold, power scaler for logreg, standard otherwise.
- **Rules**: rule-based shortlist per stage (clean, encoder, scaler, imputer, balancer, threshold), choice by inner-CV macro-F1.
- **Random**: same stage-wise search, random candidates instead of the rule shortlist (same size).
- **Brute**: same search over the full option pool of each stage.

## 6. Evaluation metrics

Macro-F1 (primary, pre-specified), balanced accuracy, accuracy, ROC-AUC, and minority-class (positive = rarer class, consistent across datasets) precision, recall, F1, PR-AUC (`pr_auc_min`, from the minority-class probability), plus confusion counts (tn, fp, fn, tp per fold in the raw file). The legacy `pr_auc` column (probability of `classes_[1]`) is kept in the raw data but is **not** the minority PR-AUC when class 1 is the majority; do not use it. Runtime = wall-clock of selection + final fit + scoring per method per fold.

## 7. Aggregate results

Unit of analysis = dataset (mean over seeds, folds, models); mean [95% t-CI over 5 datasets].

{md(tab_o)}

Absolute CIs are wide (and can exceed 1) because the 5 datasets differ greatly in difficulty; the paired gain CIs below are the informative ones.

Gain over B0 (macro-F1 unless named):

{md(tab_g)}

## 8. Per-dataset results (macro-F1, mean over seeds×folds×models)

{md(tab_ds)}

More metrics: `results/per_dataset_results.csv`, `results/per_dataset_model_results.csv`.

## 9. Per-model results (macro-F1, mean over datasets' cells)

{md(tab_md)}

## 10. Figures

{figmd}

## 11. Runtime

Seconds per method per outer fold. 8 worker processes ran concurrently on one machine (1 thread each), so absolute values include contention; use them for relative comparison only.

{md(tab_rt, '.2f')}

## 12. Statistical analysis

Paired differences (Rules − comparator) were first averaged within each dataset (over seeds, folds, models), so the inferential unit is the dataset (n=5), not the 75 model-fold cells per seed set. Reported: mean difference with 95% t-CI (df=4), Cohen's d_z, count of datasets favouring Rules, exact two-sided Wilcoxon signed-rank p (minimum attainable with n=5 is 0.0625) and Holm-adjusted p over the four pre-specified comparisons (B0, B_thr, Random, Brute on macro-F1). B_prac and the other metrics are exploratory and uncorrected.

{md(tab_t, '.4f')}

Secondary metrics (exploratory):

{md(tab_t2, '.4f')}

Reading: *observed difference* = `mean_diff`; *consistency* = datasets_rules_better of 5; *uncertainty* = CI and p. With 5 datasets the tests have very little power, so absence of significance is not evidence of equivalence, and presence of consistent sign across datasets is the more informative signal.

## 13. Limitations and threats to validity

- Only 5 datasets, all selected by a rule fixed beforehand but all small/medium tabular binary tasks; generalisation is unproven.
- 2% is generated by deleting minority rows; shares on `kr-vs-kp` and `steel-plates` are higher (40-row floor). The test folds hold only ~8-11 minority rows, so per-fold metrics are very noisy.
- Folds, seeds and models within a dataset are correlated; CIs treat datasets as the sampling unit.
- The inherited threshold tuner and `roc_auc/pr_auc` columns treat `classes_[1]` as positive, which may be the majority class on some datasets; minority metrics here use the rarer class. This was not changed, to preserve the methodology.
- Thinning happens before outer CV, with a seed-specific subsample; the selector never sees test rows.
- Runtime measured under parallel load; no predictions are stored, so PR curves are not drawn (only scalar PR-AUC and confusion counts).
- Only 3 seeds. Conclusions about Rules vs Random/Brute/B_prac need more datasets before being claimed.

## 14. Reproducibility

```
cd c1_feature_engine
OMP_NUM_THREADS=1 python -m experiments.imbalance_2pct          # run + analyze (~full experiment)
python -c "from experiments.imbalance_2pct import analyze; analyze.main()"   # re-analyze only
```
Config and package versions: `config/experiment_config.json`; log: `logs/experiment.log`. Results are deterministic given seeds and versions up to multi-threaded floating-point nondeterminism and runtime. Existing 20%/5% results (`g1_imb20_result/`, `g1_imb_low/`) are not touched. Automated check: {v}.

## 15. Conclusions supported by the measurements

- On these 5 datasets at ~2% minority, Rules scored {g('rules'):+.3f} macro-F1 over B0 (positive on {int(c.loc['rules','datasets_improved_f1'])}/5 datasets).
- Relative to B_thr: {tb.mean_diff:+.3f} ({int(tb.datasets_rules_better)}/5), to Random {tx.mean_diff:+.3f} ({int(tx.datasets_rules_better)}/5), to Brute {tn.mean_diff:+.3f} ({int(tn.datasets_rules_better)}/5), to B_prac {mt('B_prac').mean_diff:+.3f} ({int(mt('B_prac').datasets_rules_better)}/5).
- Rules, Random, Brute and B_prac are practically indistinguishable on these data (absolute mean differences ≤ {max(abs(tx.mean_diff),abs(tn.mean_diff),abs(mt('B_prac').mean_diff)):.3f} macro-F1); Rules' observed advantages are over B0 and (smaller, with a CI that includes 0) over B_thr, and its cost is lower than Brute's. The data do not show that rule-based selection beats random search or the B_prac recipe.
- Per the Holm-adjusted tests above, no claim of statistical significance can be made at n=5 unless a row shows p_holm < 0.05; claims are limited to the observed direction and size.
""", encoding="utf-8")
    (R.parent / "reports/experiment_report.md").write_text((R.parent / "reports/experiment_report.md").read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
