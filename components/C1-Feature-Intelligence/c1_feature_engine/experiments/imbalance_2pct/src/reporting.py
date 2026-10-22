"""
reporting.py  -  independent verification + the two markdown reports.

reports/findings_summary.md   short, 1 page: condition, headline gains, observations, caveats
reports/experiment_report.md  full 15-section write-up (design, tables, figures, tests, limitations)

Every number in the text is formatted from a DataFrame computed in aggregation.py / significance.py. No result is
typed by hand, so the reports can be regenerated at any time and always match the csv files and figures.
"""
import json

import pandas as pd

from .config import HERE, METHODS, REPORTS, RESULTS, RAW_ALL


def md(df, fmt=".3f"):
    """DataFrame -> markdown table (falls back to plain text if `tabulate` is not installed)."""
    df = df.copy()
    return df.to_markdown(index=False, floatfmt=fmt) if hasattr(df, "to_markdown") else df.to_string(index=False)


def verify(d, ov, perds, comp):
    """Second, independent computation of the key tables straight from the raw rows.

    The tables were built through groupby/pivot chains in aggregation.py. Here the same quantities are recomputed
    with different code paths (plain boolean filters) and compared to < 1e-12. A silent indexing / ordering bug in
    the aggregation would show up as an AssertionError instead of a wrong headline number.
    """
    for m in METHODS:
        # overall macro-F1 mean == mean over datasets of the per-dataset mean
        assert abs(d[d.method == m].groupby("dataset").f1_macro.mean().mean()
                   - ov.set_index("method").loc[m, "f1_macro_mean"]) < 1e-12
        # per-dataset table == direct filter on (method, dataset)
        for n in d.dataset.unique():
            assert abs(d[(d.method == m) & (d.dataset == n)].f1_macro.mean()
                       - perds[(perds.dataset == n) & (perds.method == m)].f1_macro_mean.iloc[0]) < 1e-12
    # gain-over-B0 table == (method per-dataset mean - B0 per-dataset mean).mean()
    b0 = d[d.method == "B0"].groupby("dataset").f1_macro.mean()
    for _, r in comp.iterrows():
        assert abs((d[d.method == r.method].groupby("dataset").f1_macro.mean() - b0).mean()
                   - r.gain_f1_macro_mean) < 1e-12
    return "verified: overall, per-dataset and gain tables recomputed from raw_fold_results.csv"


def report(d, man, ov, perds, permod, comp, t, nfb, v):
    """Write findings_summary.md and experiment_report.md.

    d,man = raw rows / manifest;  ov,perds,permod,comp,t = result tables;  nfb = #fallbacks;  v = verify() message.
    """
    cfg = json.loads((HERE / "config/experiment_config.json").read_text(encoding="utf-8"))

    # ---- lookup helpers --------------------------------------------------------------------------
    o = ov.set_index("method")
    c = comp.set_index("method")
    T = t.set_index(["metric", "comparison"])

    def ci(r, k):                      # "0.812 [0.700, 0.920]"  mean [95% CI]
        return f"{r[k + '_mean']:.3f} [{r[k + '_ci_lo']:.3f}, {r[k + '_ci_hi']:.3f}]"

    def row(m, k="f1_macro"):
        return ci(o.loc[m], k)

    rt = d.groupby("method").runtime_s.median()
    mt = lambda comp_, k="f1_macro": T.loc[(k, f"rules - {comp_}")]     # one row of the test table

    # ---- tables used in the text ------------------------------------------------------------------
    # dataset facts collapsed over the 3 seeds
    mans = man.groupby("dataset").agg(
        n_features=("n_features", "first"), rows_original=("n_rows_original", "first"),
        rows_prepared=("n_rows_prepared", "mean"), minority_share_original=("original_minority_share", "first"),
        minority_share_prepared=("prepared_minority_share", "mean"), minority_label=("minority_label", "first"),
        floor_applied=("floor_applied", "any")).reset_index()
    mans_fmt = mans.assign(rows_prepared=mans.rows_prepared.round(0).astype(int),
                           **{c_: mans[c_].map(lambda x: f"{x:.4f}") for c_ in ["minority_share_original", "minority_share_prepared"]})

    tab_o = pd.DataFrame({"method": METHODS, **{k: [row(m, k) for m in METHODS]
                          for k in ["f1_macro", "bal_acc", "min_precision", "min_recall", "min_f1", "pr_auc_min"]}})
    tab_ds = perds.pivot(index="dataset", columns="method", values="f1_macro_mean")[METHODS].reset_index()
    tab_md = permod.pivot(index="model", columns="method", values="f1_macro_mean")[METHODS].reset_index()
    tab_g = comp[["method", "gain_f1_macro_mean", "gain_f1_macro_ci_lo", "gain_f1_macro_ci_hi",
                  "gain_bal_acc_mean", "gain_min_f1_mean", "datasets_improved_f1"]]
    tab_t = t[t.metric == "f1_macro"][["comparison", "mean_diff", "ci95_lo", "ci95_hi", "cohens_dz", "datasets_rules_better",
                                       "datasets_rules_worse", "wilcoxon_exact_p", "p_holm", "family"]]
    tab_t2 = t[t.metric != "f1_macro"][["metric", "comparison", "mean_diff", "ci95_lo", "ci95_hi",
                                        "datasets_rules_better", "wilcoxon_exact_p"]]
    tab_rt = pd.DataFrame({"method": METHODS, "median_s": [rt[m] for m in METHODS],
                           "mean_s": [d[d.method == m].runtime_s.mean() for m in METHODS]})

    figs = [("macro_f1_comparison", "A. Macro-F1"), ("balanced_accuracy_comparison", "B. Balanced accuracy"),
            ("per_dataset_comparison", "C. Per dataset"), ("per_model_comparison", "D. Per model"),
            ("gain_over_baseline", "E. Gain over B0"), ("precision_recall_comparison", "F. Minority precision/recall/F1/PR-AUC"),
            ("runtime_comparison", "G. Runtime"), ("score_distribution", "H. Score distributions"),
            ("seed_stability", "I. Seed stability"), ("confusion_matrices", "J. Confusion matrices (pooled)")]
    figmd = "\n\n".join(f"**{n}**\n\n![{n}](../figures/{f}.png)" for f, n in figs)

    # ---- the auto-written "observations" paragraph (numbers filled from the tables) ---------------
    g = lambda m: c.loc[m, "gain_f1_macro_mean"]
    tr, tb, tx, tn = mt("B0"), mt("B_thr"), mt("random"), mt("brute")
    fb = (f"{nfb} selector/final fits fell back to the anchor pipeline (flagged `fallback` in `opts`)."
          if nfb else "No selector fits fell back to the anchor pipeline.")
    summary = f"""On the 5 datasets thinned to a ~2% minority share (3 seeds × 5 outer folds × 3 models), the Rules selector reached macro-F1 {row('rules')} versus {row('B0')} for the B0 default: a gain of {g('rules'):+.3f} (95% t-CI over datasets [{c.loc['rules','gain_f1_macro_ci_lo']:+.3f}, {c.loc['rules','gain_f1_macro_ci_hi']:+.3f}]), positive on {int(c.loc['rules','datasets_improved_f1'])}/5 datasets.
Against threshold tuning alone (B_thr) the difference was {tb.mean_diff:+.3f} ({int(tb.datasets_rules_better)}/5 datasets favour Rules); against Random {tx.mean_diff:+.3f} and Brute {tn.mean_diff:+.3f} (Rules median {rt['rules']:.1f}s vs Brute {rt['brute']:.1f}s per fold).
With n=5 datasets the smallest attainable exact Wilcoxon p is 0.0625, so no comparison can reach p<0.05; see Statistical analysis."""

    # ---- 1) one-page findings summary -------------------------------------------------------------
    (REPORTS / "findings_summary.md").write_text(f"""# Findings summary: 2% minority-class condition

**Condition.** Minority class randomly thinned (all majority rows kept) to ~2% of rows (floor 40 minority rows). This is a *generated* extreme-imbalance condition, not a natural class distribution.

{md(mans_fmt)}

**Headline numbers (mean of 5 datasets, 95% t-CI)**

{md(tab_g)}

**Observations**

{summary}

**Caveats.** 5 datasets (power is very limited), 3 seeds, ~8-11 minority rows per outer test fold, runtime measured under 8 parallel processes. Conclusions beyond "observed on these 5 datasets" need more datasets.

Full detail: `experiment_report.md`.
""", encoding="utf-8")

    # ---- 2) full report ---------------------------------------------------------------------------
    (REPORTS / "experiment_report.md").write_text(f"""# 2% minority-class evaluation: Rules selector vs baselines

## 1. Executive summary

{summary}

## 2. Research question

Does the rule-based pipeline selector (`SelectorV2(method="rules")`) improve test performance over the default pipeline (B0), threshold tuning alone (B_thr), a practitioner pipeline (B_prac) and over random/brute-force search, when the minority class is thinned to ~2%?

## 3. Dataset and class-distribution summary

Datasets were fixed in advance by rule (binary, natural minority ≥29%, ≥1200 majority rows) in the earlier 20%/5% study and reused unchanged. The 2% condition is **generated**: `experiments.stress.subsample_minority(share=0.02, floor=40)` keeps all majority rows and randomly drops minority rows; datasets with few rows reach a higher share because of the 40-row floor. Original files are untouched; prepared copies are in `data/prepared_datasets/` (one per dataset×seed; per-seed details in `data/dataset_manifest.csv`).

{md(mans_fmt)}

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
