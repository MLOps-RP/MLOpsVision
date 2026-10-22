"""
evaluation.py  -  HOW THE TESTS ARE RUN.

Input : one already-prepared (2% minority) dataset CSV per dataset x seed.
Output: one raw-results CSV per dataset x seed, one row per (outer fold, model, method).

The design in one picture
-------------------------
    for each dataset (5) and seed (3)                      -> a "task" (15 tasks, run in parallel)
      outer stratified 5-fold CV (random_state = seed)
        for each outer fold:
          profile the TRAIN fold                           (class balance, missingness, types, ...)
          for each model in {lgbm, logreg, rf}:
            for each method in {B0, B_thr, B_prac, rules, random, brute}:
              1. choose a preprocessing pipeline
                   - B0 / B_thr / B_prac : fixed recipe, no search
                   - rules / random / brute : stage-wise search using INNER 3-fold CV on the TRAIN fold only
              2. refit the chosen pipeline on the full TRAIN fold
              3. score on the untouched TEST fold  -> one raw row

Why this is a fair, paired comparison
-------------------------------------
* All 6 methods and all 3 models see exactly the same train/test split (same seed, same fold).
* Selection and tuning only ever touch the train fold; no test-fold information leaks into them.
* Thinning of the minority class happened BEFORE this code (on the whole dataset), so every fold of a
  seed is a split of the same 2% dataset. Thinning only deletes minority rows: nothing is duplicated or
  synthesised, so the CV folds contain no copies of each other's rows.

Expected size of the raw table: 5 datasets x 3 seeds x 5 folds x 3 models x 6 methods = 1350 rows.
"""
import time
import warnings

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

# Project packages (these implement the selector itself; they are the thing being TESTED, not rewritten here).
from c1.preprocessing.v2 import ANCHOR, SelectorV2, fit_final   # ANCHOR = default pipeline options; SelectorV2 = the search
from c1.profiling.profiler import profile                        # dataset profile fed to the rule engine

from .config import DATA_DIR, METHODS, OUTER_FOLDS, INNER_FOLDS, SELECTION_METRIC, RAW_DIR, RAW_ALL, MODELS
from .metrics import ext_metrics


# --------------------------------------------------------------------------------------------------
# Reading the already-prepared data (no manipulation here, just load X / y back)
# --------------------------------------------------------------------------------------------------
def load_prepared(name, seed):
    """Read data/prepared_datasets/<name>__imb2__seed<seed>.csv  ->  (X DataFrame, y ndarray)."""
    d = pd.read_csv(DATA_DIR / f"{name}__imb2__seed{seed}.csv")
    return d.drop(columns="target"), d["target"].to_numpy()


# --------------------------------------------------------------------------------------------------
# The three non-searching baselines
# --------------------------------------------------------------------------------------------------
def fixed(method, model):
    """Pipeline options of a baseline that does NOT search. Stages: clean, encoder, scaler, imputer, balancer, threshold."""
    if method == "B0":                                  # default: ANCHOR as is (threshold 0.5, no balancing)
        return dict(ANCHOR)
    if method == "B_thr":                               # default + decision threshold tuned for the metric
        return {**ANCHOR, "threshold": "tuned"}
    # B_prac = "common best practice": clean the data, class-weight the loss, tune the threshold.
    # logreg gets a power scaler (helps linear models), tree models keep the standard scaler.
    return {**ANCHOR, "clean": "clean", "balancer": "class_weight", "threshold": "tuned",
            "scaler": "power" if model == "logreg" else "standard"}


# --------------------------------------------------------------------------------------------------
# One task = one dataset x one seed
# --------------------------------------------------------------------------------------------------
def run_task(name, seed, out_dir):
    """Run the full 5-fold x 3-model x 6-method evaluation for one prepared dataset and write its raw csv."""
    warnings.filterwarnings("ignore")                   # sklearn/lightgbm convergence chatter would flood parallel logs

    cfg = yaml.safe_load(open("configs/default.yaml"))  # inner-fold count, max candidates per stage, ...
    rules = yaml.safe_load(open("configs/rules_all.yaml"))   # the rule base the "rules" selector fires from

    X, y = load_prepared(name, seed)
    rows = []

    # Outer CV: stratified so that EVERY test fold still contains minority rows (~8-11 of them at 2%).
    # random_state = seed makes the split reproducible and identical for every method/model.
    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(outer.split(X, y)):
        Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]

        prof = profile(Xtr, ytr)                        # profile uses TRAIN rows only (no leakage)

        for model in MODELS:
            for method in METHODS:
                t0, scores = time.time(), ""            # runtime = selection + final fit + scoring
                try:
                    if method in ("rules", "random", "brute"):
                        # Searching methods: stage-by-stage selection with INNER CV on the train fold.
                        #   rules  -> candidates proposed by the rule base (<= max_candidates per stage)
                        #   random -> same number of candidates, picked at random (controls for "just any search")
                        #   brute  -> the entire option pool of each stage (upper bound on search effort/cost)
                        sel = SelectorV2(rules, model, INNER_FOLDS, SELECTION_METRIC,
                                         cfg["max_candidates"], seed, method)
                        d = sel.select(Xtr, ytr, prof)
                        opts, scores = d["opts"], str(d["scores"])   # scores = per-stage inner-CV score of every candidate
                    else:
                        opts = fixed(method, model)

                    # Refit the chosen pipeline on the WHOLE train fold, then score on the test fold.
                    m = ext_metrics(fit_final(opts, model, Xtr, ytr, seed), Xte, yte)

                except Exception as e:
                    # If the chosen combination cannot be fitted on the full train fold, fall back to the
                    # anchor pipeline and FLAG it. Fallbacks are counted and reported, never hidden.
                    opts = dict(ANCHOR)
                    m = ext_metrics(fit_final(ANCHOR, model, Xtr, ytr, seed), Xte, yte)
                    opts["note"] = "fallback:" + type(e).__name__

                rows.append({
                    "dataset": name, "split": "test", "cond": "imb2", "seed": seed, "fold": fold,
                    "model": model, "method": method,
                    **m,                                                  # all metrics of this test fold
                    "opts": str(opts),                                    # which pipeline was chosen (audit trail)
                    "stage_scores": scores,                               # inner-CV scores behind that choice
                    "runtime_s": time.time() - t0,
                    "n_rows": len(Xtr),
                    "minority_share": float(min((ytr == c).mean() for c in set(ytr))),   # checked against the manifest later
                })

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f"{name}__imb2.csv", index=False)
    return name, seed


# --------------------------------------------------------------------------------------------------
# All tasks, in parallel, then merge
# --------------------------------------------------------------------------------------------------
def run_all(datasets, seeds, n_jobs):
    """15 independent tasks (5 datasets x 3 seeds) on `n_jobs` worker processes, then concatenate the raw csvs."""
    from joblib import Parallel, delayed

    Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_task)(n, s, RAW_DIR / f"seed{s}") for n in datasets for s in seeds)

    # results/raw_fold_results.csv is the ONE file every later table, test and figure is computed from.
    files = sorted(RAW_DIR.glob("seed*/*.csv"))
    pd.concat([pd.read_csv(f) for f in files]).to_csv(RAW_ALL, index=False)
    return len(files)
