"""
plotting.py  -  HOW EVERY FIGURE IN figures/ IS DRAWN.

All figures read ONLY from tables computed by aggregation.py / the raw csv; no number is typed in by hand.
Common visual language:
    * colour  = method (fixed in config.COL)
    * coloured marker + whisker = mean over the 5 datasets with its 95% t-CI
    * small black dots          = the 5 individual dataset values behind that mean (so spread is visible)

Figure -> file map
    A  macro_f1_comparison.png          primary metric, all methods
    B  balanced_accuracy_comparison.png secondary metric
    C  per_dataset_comparison.png       macro-F1 broken down by dataset
    D  per_model_comparison.png         macro-F1 broken down by model
    E  gain_over_baseline.png           paired gain over B0
    F  precision_recall_comparison.png  minority precision / recall / F1 / PR-AUC
    G  runtime_comparison.png           cost per outer fold (log scale)
    H  score_distribution.png           dataset x seed spread
    I  seed_stability.png               is Rules stable across seeds?
    J  confusion_matrices.png           pooled confusion counts per method
"""
import matplotlib
matplotlib.use("Agg")                                  # headless backend: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np

from .aggregation import tci
from .config import COL, FIGURES, LABEL, METHODS, RESULTS

# One global style so all figures look alike (light grid, no top/right frame, 200 dpi PNGs).
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


# --------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------
def errpts(ax, x, dm, metric, m, w=.0):
    """Draw the per-dataset dots of method `m` at x (slightly jittered side-to-side so they do not overlap)."""
    g = dm[dm.method == m][metric]
    ax.scatter(np.full(len(g), x) + np.linspace(-.12, .12, len(g)) + w, g, s=14, color="k", alpha=.55, zorder=3)


def _mean_ci(ax, x, r, metric, color, **kw):
    """Coloured mean marker at x with asymmetric whiskers from the stored CI columns (`<metric>_ci_lo/hi`)."""
    mu = r[f"{metric}_mean"]
    ax.errorbar(x, mu, yerr=[[mu - r[f"{metric}_ci_lo"]], [r[f"{metric}_ci_hi"] - mu]], fmt="o", color=color, **kw)


def fig_methods(ov, dm, metric, fn, title, ylabel):
    """Figures A and B: one mean +- CI marker per method plus the 5 dataset dots."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(METHODS):
        r = ov[ov.method == m].iloc[0]
        _mean_ci(ax, i, r, metric, COL[m], ms=8, capsize=4, lw=2, zorder=4)
        errpts(ax, i, dm, metric, m)
    ax.set_xticks(range(6), [LABEL[m] for m in METHODS], rotation=20)
    ax.set_ylabel(ylabel)
    ax.set_title(title + "\ncolour = mean of 5 datasets ± 95% t-CI; black dots = individual datasets")
    fig.tight_layout(); fig.savefig(FIGURES / fn); plt.close(fig)


def grouped(ax, df, group, metric, order, err="sd"):
    """Grouped bars: x = each value of `group` (dataset or model), one bar per method, error bar = SD."""
    w = .8 / len(METHODS)                               # bar width so 6 bars fill 80% of a group slot
    for j, m in enumerate(METHODS):
        sel = lambda g: df[(df[group] == g) & (df.method == m)]
        v = [sel(g)[f"{metric}_mean"].iloc[0] for g in order]
        e = [sel(g)[f"{metric}_sd"].iloc[0] for g in order] if err else None
        ax.bar(np.arange(len(order)) + (j - 2.5) * w, v, w, yerr=e, color=COL[m], label=LABEL[m],
               capsize=1.5, error_kw=dict(lw=.7))
    ax.set_xticks(range(len(order)), order)


# --------------------------------------------------------------------------------------------------
# C, D : breakdown by dataset / by model
# --------------------------------------------------------------------------------------------------
def fig_breakdowns(d, perds, permod):
    dsets = sorted(d.dataset.unique())
    for fn, df, grp, order, t in [
        ("per_dataset_comparison.png", perds, "dataset", dsets,
         "Macro-F1 per dataset (error bars: SD over folds×models×seeds)"),
        ("per_model_comparison.png", permod, "model", ["lgbm", "logreg", "rf"],
         "Macro-F1 per model (error bars: SD over datasets' cells)"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4))
        grouped(ax, df, grp, "f1_macro", order)
        ax.set_ylabel("macro-F1"); ax.set_ylim(.4, 1.02)
        ax.set_title(t); ax.legend(ncol=6, fontsize=7, loc="lower center")
        fig.tight_layout(); fig.savefig(FIGURES / fn); plt.close(fig)


# --------------------------------------------------------------------------------------------------
# E : paired gain over the default pipeline
# --------------------------------------------------------------------------------------------------
def fig_gain(dm):
    fig, ax = plt.subplots(figsize=(7, 4))
    ms = [m for m in METHODS if m != "B0"]
    x = dm.pivot(index="dataset", columns="method", values="f1_macro")   # rows = datasets, cols = methods
    g = x.sub(x["B0"], axis=0)[ms]                                       # method - B0, per dataset (paired)
    for i, m in enumerate(ms):
        lo, hi = tci(g[m])
        ax.errorbar(i, g[m].mean(), yerr=[[g[m].mean() - lo], [hi - g[m].mean()]], fmt="o", color=COL[m],
                    ms=8, capsize=4, lw=2, zorder=4)
        ax.scatter(np.full(5, i) + np.linspace(-.12, .12, 5), g[m], s=14, color="k", alpha=.55, zorder=3)
    ax.axhline(0, color="k", lw=1)                                       # above the line = better than B0
    ax.set_xticks(range(len(ms)), [LABEL[m] for m in ms], rotation=20)
    ax.set_ylabel("macro-F1 gain over B0")
    ax.set_title("Gain over B0 at 2% minority share\nmean of 5 datasets ± 95% t-CI; black dots = datasets")
    fig.tight_layout(); fig.savefig(FIGURES / "gain_over_baseline.png"); plt.close(fig)


# --------------------------------------------------------------------------------------------------
# F : minority-class metrics (the ones that matter at 2%)
# --------------------------------------------------------------------------------------------------
def fig_minority(ov, dm):
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    panels = [("min_precision", "minority precision"), ("min_recall", "minority recall"),
              ("min_f1", "minority F1"), ("pr_auc_min", "minority PR-AUC")]
    for ax, (m, t) in zip(axs, panels):
        for i, k in enumerate(METHODS):
            r = ov[ov.method == k].iloc[0]
            _mean_ci(ax, i, r, m, COL[k], capsize=3, lw=1.5)
            errpts(ax, i, dm, m, k)
        ax.set_xticks(range(6), [LABEL[k].split()[0] for k in METHODS], rotation=45)   # short names: "B0", "Rules", ...
        ax.set_title(t); ax.set_ylim(0, 1.02)
    fig.suptitle("Minority-class (positive = rarer class) precision / recall / F1 / PR-AUC; mean of 5 datasets ± 95% t-CI", y=1.02)
    fig.tight_layout(); fig.savefig(FIGURES / "precision_recall_comparison.png", bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------------------------------
# G : runtime
# --------------------------------------------------------------------------------------------------
def fig_runtime(d):
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [d[d.method == m].runtime_s for m in METHODS]
    ax.boxplot(data, tick_labels=[LABEL[m] for m in METHODS], showfliers=False, medianprops=dict(color="k"))
    for i, v in enumerate(data):                        # every raw cell as a faint jittered point (fixed rng -> reproducible plot)
        ax.scatter(np.random.default_rng(0).normal(i + 1, .05, len(v)), v, s=4, alpha=.25, color=COL[METHODS[i]])
    ax.set_yscale("log")                                # fixed baselines take ~0.1s, Brute takes minutes -> log axis
    ax.set_ylabel("seconds per outer fold (log)"); ax.tick_params(axis="x", rotation=20)
    ax.set_title("Wall-clock per method per outer fold (selection + final fit + scoring)\n"
                 "8 parallel worker processes, 1 thread each; one point = one dataset×seed×fold×model")
    fig.tight_layout(); fig.savefig(FIGURES / "runtime_comparison.png"); plt.close(fig)


# --------------------------------------------------------------------------------------------------
# H : score distribution (dataset x seed means)
# --------------------------------------------------------------------------------------------------
def fig_distribution(d):
    dsets = sorted(d.dataset.unique())
    cm = dict(zip(dsets, plt.cm.tab10.colors))          # one colour per dataset
    ds = d.groupby(["method", "dataset", "seed"]).f1_macro.mean().reset_index()   # 15 points per method
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([ds[ds.method == m].f1_macro for m in METHODS], tick_labels=[LABEL[m] for m in METHODS],
               showfliers=False, medianprops=dict(color="k"))
    for i, m in enumerate(METHODS):
        for n in dsets:
            v = ds[(ds.method == m) & (ds.dataset == n)].f1_macro
            ax.scatter(np.full(len(v), i + 1) + (dsets.index(n) - 2) * .06, v, s=16, color=cm[n],
                       label=n if i == 0 else None, zorder=3)          # label once so the legend lists each dataset once
    ax.legend(fontsize=7, title="dataset"); ax.set_ylabel("macro-F1"); ax.tick_params(axis="x", rotation=20)
    ax.set_title("Score distribution: one point = dataset×seed mean (15 per method; the 3 seeds of a dataset are clustered, not independent)")
    ax.title.set_fontsize(8)
    fig.tight_layout(); fig.savefig(FIGURES / "score_distribution.png"); plt.close(fig)
    return cm


# --------------------------------------------------------------------------------------------------
# J : pooled confusion matrices
# --------------------------------------------------------------------------------------------------
def fig_confusion(d):
    """Sum tn/fp/fn/tp over every dataset, seed, fold and model for each method, then draw a 2x2 heat-map.

    Cell colour/percent = share of the TRUE class (row-normalised), so the minority row directly shows recall.
    Caveat: pooling by summing weights bigger datasets more; this is a descriptive picture, not an inferential one.
    """
    cs = d.groupby("method")[["tn", "fp", "fn", "tp"]].sum().loc[METHODS]
    cs.reset_index().to_csv(RESULTS / "confusion_counts_pooled.csv", index=False)     # same numbers saved as a table
    fig, axs = plt.subplots(1, 6, figsize=(15, 3))
    for ax, m in zip(axs, METHODS):
        tn, fp, fn, tp = cs.loc[m]
        M = np.array([[tn, fp], [fn, tp]])               # rows = true (majority, minority); cols = predicted
        P = M / M.sum(1, keepdims=True)                  # row-normalise
        ax.imshow(P, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]:,}\n({P[i, j]:.0%})", ha="center", va="center",
                        color="w" if P[i, j] > .5 else "k", fontsize=8)              # white text on dark cells
        ax.set_xticks([0, 1], ["majority", "minority"]); ax.set_yticks([0, 1], ["majority", "minority"])
        ax.set_title(LABEL[m], fontsize=9); ax.set_xlabel("predicted")
    axs[0].set_ylabel("true")
    fig.suptitle("Pooled confusion matrices (counts; % of true class). Positive = minority class", y=1.05)
    fig.tight_layout(); fig.savefig(FIGURES / "confusion_matrices.png", bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------------------------------
# I : seed stability of the Rules selector
# --------------------------------------------------------------------------------------------------
def fig_seed_stability(d, cm):
    dsets = sorted(d.dataset.unique())
    sg = d.pivot_table(index=["dataset", "seed"], columns="method", values="f1_macro")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for n in dsets:
        axs[0].plot(sg.loc[n].index, sg.loc[n, "rules"], "o-", color=cm[n], label=n)                       # level per seed
        axs[1].plot(sg.loc[n].index, sg.loc[n, "rules"] - sg.loc[n, "B0"], "o-", color=cm[n])              # gain over B0 per seed
    axs[1].axhline(0, color="k", lw=1)
    axs[0].set_ylabel("Rules macro-F1"); axs[1].set_ylabel("Rules − B0 macro-F1")
    for a in axs:
        a.set_xlabel("seed"); a.set_xticks(sorted(d.seed.unique()))
    axs[0].legend(fontsize=7)
    fig.suptitle("Seed stability of the Rules selector (each point = mean of 5 folds × 3 models)")
    fig.tight_layout(); fig.savefig(FIGURES / "seed_stability.png"); plt.close(fig)


# --------------------------------------------------------------------------------------------------
# entry point: draw all ten
# --------------------------------------------------------------------------------------------------
def make_figures(d, ov, dm, perds, permod):
    FIGURES.mkdir(exist_ok=True)
    fig_methods(ov, dm, "f1_macro", "macro_f1_comparison.png", "Macro-F1 at 2% minority share", "macro-F1")             # A
    fig_methods(ov, dm, "bal_acc", "balanced_accuracy_comparison.png", "Balanced accuracy at 2% minority share", "balanced accuracy")  # B
    fig_breakdowns(d, perds, permod)                    # C, D
    fig_gain(dm)                                        # E
    fig_minority(ov, dm)                                # F
    fig_runtime(d)                                      # G
    cm = fig_distribution(d)                            # H  (also returns the dataset colour map)
    fig_confusion(d)                                    # J
    fig_seed_stability(d, cm)                           # I
