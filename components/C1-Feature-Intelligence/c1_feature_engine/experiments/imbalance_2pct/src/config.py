"""
config.py  -  every constant the 2% minority-class study uses, in one place.

NOTE ON SCOPE
-------------
This `src/` folder documents HOW THE TESTS ARE RUN and HOW THE RESULTS ARE ANALYSED / PLOTTED.
It deliberately does NOT contain the data-manipulation code (loading the raw OpenML datasets, the
stratified 10 000-row cap, `subsample_minority(share=0.02, floor=40)`, writing the prepared CSVs).
That step already happened; its output is what we start from:

    data/prepared_datasets/<dataset>__imb2__seed<seed>.csv     (features + a `target` column)
    data/dataset_manifest.csv                                   (one row per dataset x seed)

Nothing in this folder needs to be executed to be understood; it is a commented reference copy of
the pipeline that produced everything in results/, figures/ and reports/.
"""
from pathlib import Path

# --------------------------------------------------------------------------------------------------
# Paths (all relative to experiments/imbalance_2pct/)
# --------------------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parents[1]          # .../experiments/imbalance_2pct
DATA_DIR = HERE / "data" / "prepared_datasets"      # INPUT : already-thinned datasets (2% minority)
MANIFEST = HERE / "data" / "dataset_manifest.csv"   # INPUT : per dataset x seed facts (minority label, share, ...)
RESULTS = HERE / "results"                          # OUTPUT: csv tables
RAW_DIR = RESULTS / "raw"                           # OUTPUT: one csv per dataset x seed  (results/raw/seed42/...)
RAW_ALL = RESULTS / "raw_fold_results.csv"          # OUTPUT: all raw csvs concatenated (the single source of truth)
FIGURES = HERE / "figures"                          # OUTPUT: png plots
REPORTS = HERE / "reports"                          # OUTPUT: markdown reports

# --------------------------------------------------------------------------------------------------
# Experimental design
# --------------------------------------------------------------------------------------------------
DATASETS = ["spambase", "phoneme", "mushroom", "kr-vs-kp", "steel-plates"]   # 5 binary tabular datasets
SEEDS = [42, 43, 44]                    # 3 independent thinning/CV seeds
MODELS = ["lgbm", "logreg", "rf"]       # 3 downstream classifiers
OUTER_FOLDS = 5                         # stratified outer CV  -> the TEST folds the numbers come from
INNER_FOLDS = 3                         # inner CV used only by the selectors, on the TRAIN fold only
SELECTION_METRIC = "f1_macro"           # metric the selectors optimise in the inner CV
N_JOBS = 8                              # parallel worker processes (1 thread each -> OMP_NUM_THREADS=1)

# The six compared methods (the "arms" of the experiment). Order here = order on every plot/table.
METHODS = ["B0", "B_thr", "B_prac", "rules", "random", "brute"]
LABEL = {
    "B0": "B0 default",             # anchor pipeline: no cleaning, one-hot, standard scaler, median imputer, no balancing, thr 0.5
    "B_thr": "B_thr threshold",     # B0 + tuned decision threshold
    "B_prac": "B_prac practice",    # cleaning + class weighting + tuned threshold (a "common best practice" recipe)
    "rules": "Rules",               # OUR method: rule-based shortlist per stage, choice by inner-CV macro-F1
    "random": "Random",             # same stage-wise search, but random candidates (same shortlist size)
    "brute": "Brute",               # same stage-wise search over the FULL option pool of each stage
}

# Fixed colour per method so a method looks identical in every figure.
COL = dict(zip(METHODS, ["#9aa0a6", "#c9a227", "#d98c3f", "#1f6feb", "#8250df", "#2da44e"]))

# Metrics carried through aggregation. `min_*` / `pr_auc_min` are MINORITY-class metrics
# (positive class = the rarer class, consistently across datasets).
MET = ["f1_macro", "bal_acc", "min_precision", "min_recall", "min_f1", "pr_auc_min", "roc_auc", "accuracy", "runtime_s"]
PRIMARY = "f1_macro"                    # the single pre-specified primary metric
