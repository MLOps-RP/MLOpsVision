"""
pipeline.py  -  the two entry points that tie the modules together, in the order things actually happen.

    STEP 1  run_experiment()   evaluation.py   prepared csvs -> results/raw/*/*.csv -> results/raw_fold_results.csv
    STEP 2  analyze()          everything else raw_fold_results.csv -> results/*.csv, figures/*.png, reports/*.md

Step 2 only reads results/raw_fold_results.csv, so it can be re-run any number of times (e.g. after tweaking a
plot) without repeating the ~8 minute experiment.

(Reference copy - not meant to be executed as part of this documentation task.)
"""
import json
import platform
import sys
import time

import pandas as pd

from .aggregation import agg, gains_over_b0, overall
from .config import (DATASETS, HERE, INNER_FOLDS, MANIFEST, METHODS, MODELS, N_JOBS, OUTER_FOLDS, RAW_ALL,
                     REPORTS, RESULTS, SEEDS, SELECTION_METRIC)
from .evaluation import run_all
from .plotting import make_figures
from .reporting import report, verify
from .significance import tests
from .validation import validate


def run_experiment():
    """STEP 1 - run every test (see evaluation.py for what one test is)."""
    t0 = time.time()

    # Record exactly what was run (design + library versions) so the result folder is self-describing.
    (HERE / "config").mkdir(exist_ok=True)
    (HERE / "config/experiment_config.json").write_text(json.dumps({
        "datasets": DATASETS, "share": 0.02, "floor": 40, "seeds": SEEDS, "models": MODELS, "jobs": N_JOBS,
        "folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS, "metric": SELECTION_METRIC, "methods": METHODS,
        "condition": "imb2 = experiments.stress.subsample_minority(share=0.02, floor=40)",
        "positive_class": "minority class (fewest rows)",
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {m: __import__(m).__version__ for m in ["numpy", "pandas", "sklearn", "scipy", "lightgbm", "imblearn"]},
    }, indent=2), encoding="utf-8")

    # 15 tasks (5 datasets x 3 seeds) in parallel, then merge into results/raw_fold_results.csv
    n_files = run_all(DATASETS, SEEDS, N_JOBS)
    print(f"run done: {n_files} CSVs in {time.time() - t0:.0f}s", file=sys.stderr)


def analyze():
    """STEP 2 - from raw fold rows to validated tables, tests, figures and reports."""
    d = pd.read_csv(RAW_ALL)                                  # 1350 rows: the only input of the analysis
    man = pd.read_csv(MANIFEST)                               # facts about the prepared data (minority label, share, ...)

    # 2.1 refuse to continue if the raw table is incomplete / inconsistent
    nfb = validate(d, man)                                    # returns the number of fallback fits

    # 2.2 summary tables (unit of analysis = dataset)
    ov, dm = overall(d)                                       # headline: method x metric, mean + CI over 5 datasets
    perds = agg(d, ["dataset", "method"])                     # per dataset
    permod = agg(d, ["model", "method"])                      # per model
    permd = agg(d, ["dataset", "model", "method"])            # per dataset x model (most detailed)
    comp = gains_over_b0(dm)                                  # paired gain of every method over B0

    # 2.3 paired significance tests: Rules vs B0 / B_thr / Random / Brute (primary) and B_prac (exploratory)
    t = tests(d)

    # 2.4 save every table
    RESULTS.mkdir(exist_ok=True)
    ov.to_csv(RESULTS / "aggregate_overall.csv", index=False)
    perds.to_csv(RESULTS / "per_dataset_results.csv", index=False)
    permod.to_csv(RESULTS / "per_model_results.csv", index=False)
    permd.to_csv(RESULTS / "per_dataset_model_results.csv", index=False)
    comp.to_csv(RESULTS / "comparisons.csv", index=False)
    t.to_csv(RESULTS / "statistical_tests.csv", index=False)

    # 2.5 draw the ten figures (also writes results/confusion_counts_pooled.csv)
    make_figures(d, ov, dm, perds, permod)

    # 2.6 recompute key numbers a second way; any mismatch aborts before a report is written
    v = verify(d, ov, perds, comp)

    # 2.7 write the markdown reports from the verified tables
    REPORTS.mkdir(exist_ok=True)
    report(d, man, ov, perds, permod, comp, t, nfb, v)
    print(v)


def main():
    """Full run: experiment, then analysis (what `python -m experiments.imbalance_2pct` does)."""
    run_experiment()
    analyze()


if __name__ == "__main__":
    main()
