"""Thin adapter between the Streamlit demo and the EXISTING C1 Gap-1 engine.

Nothing of the engine is re-implemented here. Reused as-is:
    c1.profiling.profiler.profile              dataset meta-features
    c1.preprocessing.v2.SelectorV2             Gap 1: stage-wise preprocessing search (inner CV, train data only)
    c1.preprocessing.v2.build_v2               Gap 1: pipeline construction
    c1.pipeline.FeatureIntelligenceEngine      Gap 2: generation -> MI/SHAP/RFE -> stability/redundancy -> FQS -> GA subset

What this file adds: CSV cleaning like experiments/datasets.load(), a progress hook (subclass of SelectorV2 that only
reports what the engine is doing), and an export of the CSV-level transformations of the chosen pipeline.
"""
import inspect
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline as SkPipeline

warnings.filterwarnings("ignore")      # same as the experiments: candidates that cannot run (e.g. ADASYN) simply lose the comparison

ROOT = Path(__file__).resolve().parents[1]                 # .../c1_feature_engine
for _p in (ROOT, ROOT / "src"):                            # make `c1` and `experiments` importable
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from c1.preprocessing.v2 import STAGES, SelectorV2, build_v2   # noqa: E402
from c1.profiling.profiler import profile                                          # noqa: E402
from c1.pipeline import FeatureIntelligenceEngine                                  # noqa: E402
from c1.selection.evolutionary import ga_select                                    # noqa: E402
from c1.selection.fqs import DEFAULT_WEIGHTS                                       # noqa: E402

# The model the engine itself uses unless told otherwise (SelectorV2.__init__ default) - the demo does not ask the user.
DEFAULT_MODEL = inspect.signature(SelectorV2.__init__).parameters["model"].default

CFG = yaml.safe_load(open(ROOT / "configs/default.yaml"))        # inner_folds=3, metric=f1_macro, max_candidates=3
RULES = yaml.safe_load(open(ROOT / "configs/rules_all.yaml"))    # the same rule base the imbalance_2pct run used

# Where does each stage act? (this is what the demo UI uses to separate CSV transformations from model-level operations)
STAGE_KIND = {
    "clean": "CSV transformation",
    "encoder": "CSV transformation",
    "scaler": "CSV transformation",
    "imputer": "CSV transformation",
    "balancer": "see note",          # depends on the chosen value, resolved in balancer_kind()
    "threshold": "Model-level (prediction time)",
}


def balancer_kind(value: str) -> str:
    if value == "none":
        return "None (no balancing)"
    if value == "class_weight":
        return "Model-level (class weights in the loss)"
    return "Training-time resampling (creates/removes rows; NOT applied to the exported CSV)"


# --------------------------------------------------------------------------------------------------
# 1. CSV preparation + dataset analysis
# --------------------------------------------------------------------------------------------------
def prepare_xy(df: pd.DataFrame, target: str):
    """Same cleaning as experiments/datasets.load(): non-numeric columns -> str with NaN; target -> integer codes
    (pd.factorize, first-seen class = 0). Rows with a missing target are dropped.
    If the file has both a `target` and a `class_name` column, `class_name` (a text copy of the target) is excluded from
    the features. Returns (X, y_codes, class_names, excluded_column_or_None)."""
    df = df.loc[df[target].notna()].reset_index(drop=True)
    excluded = "class_name" if ("class_name" in df.columns and "target" in df.columns and target != "class_name") else None
    X = df.drop(columns=[target] + ([excluded] if excluded else [])).copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(object).map(lambda v: np.nan if pd.isna(v) else str(v))
    codes, uniques = pd.factorize(df[target].astype(str))
    return X, codes, [str(u) for u in uniques], excluded


def analyze(df: pd.DataFrame, target: str) -> dict:
    vc = df[target].astype(str).value_counts(dropna=False)
    dist = pd.DataFrame({"class": vc.index, "count": vc.values, "percent": (vc.values / vc.sum() * 100).round(2)})
    miss = df.isna().sum()
    return {
        "shape": df.shape,
        "missing_total": int(miss.sum()),
        "missing_by_col": miss[miss > 0].rename("missing").to_frame().assign(percent=lambda m: (m.missing / len(df) * 100).round(2)),
        "duplicates": int(df.duplicated().sum()),
        "class_dist": dist,
        "minority_pct": float(dist["percent"].min()),
        "imbalance_ratio": float(dist["count"].max() / dist["count"].min()) if len(dist) > 1 else float("nan"),
        "n_missing_target": int(df[target].isna().sum()),
    }


def check_usable(y: np.ndarray) -> str | None:
    """Return an error message if the engine cannot run on this target, else None."""
    cnt = np.bincount(y)
    if len(cnt) < 2:
        return "The target has only one class."
    if cnt.min() < 6:
        return f"The rarest class has only {cnt.min()} rows; the engine's 3-fold inner CV needs at least 6 per class."
    return None


# --------------------------------------------------------------------------------------------------
# 2. Run the real C1 selector (with a progress hook)
# --------------------------------------------------------------------------------------------------
class _ProgressSelector(SelectorV2):
    """SelectorV2 unchanged, plus callbacks that REPORT what it evaluates. It changes no decision."""
    def __init__(self, *a, on_event=None, **k):
        super().__init__(*a, **k)
        self.on_event, self.proposed, self._stage = on_event or (lambda *x: None), {}, None

    def _cands(self, stage, p):
        c = super()._cands(stage, p)
        self._stage, self.proposed[stage] = stage, list(c)
        self.on_event("stage", stage, list(c))
        return c

    def _score(self, opts, X, y):
        s = super()._score(opts, X, y)
        self.on_event("score", self._stage, opts[self._stage], s)
        return s

    def _threshold_scores(self, opts, X, y):
        s = super()._threshold_scores(opts, X, y)
        self.on_event("threshold", "threshold", s, None)
        return s


def run_c1(X: pd.DataFrame, y: np.ndarray, model: str = DEFAULT_MODEL, seed: int = 42, on_event=None) -> dict:
    """Profile the data and run the real stage-wise selector (mode='rules'). Returns the engine's actual decision."""
    t0 = time.time()
    prof = profile(X, y)
    on_event and on_event("profile", None, prof.model_dump(), None)
    sel = _ProgressSelector(RULES, model, CFG["inner_folds"], CFG["metric"], CFG["max_candidates"], seed, "rules",
                            on_event=on_event)
    d = sel.select(X, y, prof)                                  # {"opts", "scores", "runtime_s"}
    return {"opts": d["opts"], "scores": d["scores"], "proposed": sel.proposed, "profile": prof.model_dump(),
            "model": model, "seed": seed, "runtime_s": time.time() - t0}


def stage_table(res: dict) -> pd.DataFrame:
    """One row per stage: what C1 chose and what it actually did at that stage."""
    rows = []
    for st in STAGES:
        val = res["opts"][st]
        if st in res["scores"]:
            status = "searched: " + ", ".join(f"{k}={v:.3f}" for k, v in res["scores"][st].items())
        elif st in res["proposed"]:
            status = "not searched (rules proposed only the default)"
        else:
            status = "skipped (not applicable to this data)"
        kind = balancer_kind(val) if st == "balancer" else STAGE_KIND[st]
        rows.append({"stage": st, "selected": val, "applied as": kind, "evidence (inner-CV macro-F1)": status})
    return pd.DataFrame(rows)


# user-facing labels for the engine's stages / methods (display only; the values themselves come from the engine)
_LABEL = {"clean": ("Data Cleaning", "Remove unusable columns and group very rare categories"),
          "imputer": ("Missing Value Handling", "Fill in missing values"),
          "encoder": ("Categorical Encoding", "Convert text categories into numbers"),
          "scaler": ("Feature Scaling", "Put numeric columns on a comparable scale"),
          "balancer": ("Class Imbalance Handling", "Compensate for classes with far fewer examples"),
          "threshold": ("Prediction Threshold", "Decide how confident a prediction must be to choose the smaller class")}
_METHOD = {"none": "None", "clean": "Clean", "onehot": "One-hot encoding", "ordinal": "Ordinal encoding", "target": "Target encoding",
           "standard": "Standardization", "power": "Power transform", "median": "Median", "mean": "Mean",
           "median_ind": "Median + missing indicators", "knn": "KNN imputation", "mice": "Iterative (MICE) imputation",
           "class_weight": "Class weights", "smote": "SMOTE", "smote_nc": "SMOTE-NC", "adasyn": "ADASYN", "tomek": "Tomek links",
           "smote_tomek": "SMOTE + Tomek links", "0.5": "Standard (0.5)", "tuned": "Tuned for balanced accuracy"}


def preprocessing_table(res: dict) -> pd.DataFrame:
    """User-friendly view of the engine's decision: stage, selected method, purpose, where it acts."""
    rows = []
    for st in STAGES:
        v, (name, purpose) = res["opts"][st], _LABEL[st]
        method = _METHOD.get(v, v)
        if st not in res["scores"] and st not in res["proposed"]:
            method, kind = f"{method} (not needed for this data)", "Not applied"
        elif st in ("clean", "imputer", "encoder", "scaler"):
            kind = "Not applied" if v == "none" else "Data transformation (included in the exported file)"
        elif st == "balancer":
            kind = {"none": "Not applied", "class_weight": "Training-time operation (model setting, not in the file)"}.get(
                v, "Training-time operation (resampling during model training, not in the file)")
        else:
            kind = "Prediction-time operation (not in the file)" if v == "tuned" else "Not applied"
        rows.append({"Preprocessing Stage": name, "Selected Method": method, "Purpose": purpose, "Application Type": kind})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# 3. Export the CSV-level transformation of the chosen pipeline
# --------------------------------------------------------------------------------------------------
def transform_dataset(X: pd.DataFrame, y: np.ndarray, opts: dict, model: str, seed: int = 42) -> pd.DataFrame:
    """Apply ONLY the row-preserving steps of the pipeline built by the engine (build_v2): 'clean' (optional) and 'pre'
    (impute -> encode -> scale). The sampler and the model are not applied. Fitted on the whole uploaded file, so use
    this export for inspection / downstream work, not as a leakage-free evaluation set."""
    pipe = build_v2(opts, model, X, seed)
    steps = [(n, s) for n, s in pipe.steps if n in ("clean", "pre")]
    tr = SkPipeline(steps)
    Z = tr.fit_transform(X, y)
    names = [n.split("__", 1)[-1] for n in tr.named_steps["pre"].get_feature_names_out()]
    return pd.DataFrame(np.asarray(Z, dtype=float), columns=names)


def gap1_export(X: pd.DataFrame, y: np.ndarray, opts: dict, model: str, seed: int = 42):
    """Same as transform_dataset, plus `origin`: {output column -> original uploaded column it came from}.
    One-hot columns are mapped exactly (via the fitted encoder's categories); other blocks by column name."""
    pipe = build_v2(opts, model, X, seed)
    tr = SkPipeline([(n, s) for n, s in pipe.steps if n in ("clean", "pre")])
    Z = tr.fit_transform(X, y)
    pre = tr.named_steps["pre"]
    full = list(pre.get_feature_names_out())
    origin = {}
    for block, bpipe, cols in pre.transformers_:
        if block == "remainder":
            continue
        names = [n for n in full if n.startswith(block + "__")]
        enc = bpipe.named_steps.get("enc") if hasattr(bpipe, "named_steps") else None
        if block == "cat" and enc is not None and hasattr(enc, "categories_") and opts.get("encoder") == "onehot":
            owners = [c for c, cats in zip(cols, enc.categories_) for _ in cats]
        else:
            owners = []
            for n in names:
                short = n.split("__", 1)[1]
                hit = [c for c in cols if short == c or short == "missingindicator_" + c or short.startswith(c + "_")]
                owners.append(max(hit, key=len) if hit else "?")
        origin.update({n.split("__", 1)[1]: o for n, o in zip(names, owners)})
    Zdf = pd.DataFrame(np.asarray(Z, dtype=float), columns=[n.split("__", 1)[-1] for n in full])
    return Zdf, origin


# --------------------------------------------------------------------------------------------------
# 4. Gap 2: the engine's own feature generation + ranking + FQS + GA subset search
# --------------------------------------------------------------------------------------------------
def run_gap2(Z: pd.DataFrame, y: np.ndarray, seed: int = 42) -> dict:
    """Run the existing FeatureIntelligenceEngine on the Gap-1 output. use_selector=False because Gap 1 has already
    chosen the preprocessing (the engine's own 2-stage v1 selector would be a second, redundant Gap-1 run).
    Everything else is the engine's default: generation on, FQS default weights, GA subset search."""
    t0 = time.time()
    eng = FeatureIntelligenceEngine(CFG, seed=seed, use_selector=False).fit(Z, y)
    return {"eng": eng, "runtime_s": time.time() - t0}


def gap2_strategy(eng) -> str:
    """The strategy as configured in the engine object that actually ran (not typed by hand)."""
    w = {**DEFAULT_WEIGHTS, **(eng.weights or {})}
    ga = {k: v.default for k, v in inspect.signature(ga_select).parameters.items() if v.default is not inspect._empty}
    ga.update(eng.ga)
    return (f"FQS = {w['P']}·P + {w['S']}·S − {w['M']}·M − {w['R']}·R (P: MI+SHAP+RFE rank, S: {eng.boot}-bootstrap stability, "
            f"M: missing ratio, R: redundancy) → top {eng.max_genes} candidates → genetic algorithm "
            f"(population {ga['pop']}, ≤{ga['gens']} generations, seeded from FQS, seed {eng.seed})")


# Readable names for the transformations in c1.generation (op key in the engine's lineage -> (technique, family, how it is built))
_GEN_OPS = {
    "log1p": ("Log transform", "Mathematical (single column)", "log(1 + x), only when the column has no negative values"),
    "sqrt": ("Square root", "Mathematical (single column)", "√x, only when the column has no negative values"),
    "square": ("Square", "Mathematical (single column)", "x²"),
    "quantile_bin": ("Quantile binning", "Binning (single column)", "values grouped into quantile-based bins"),
    "ratio": ("Ratio", "Interaction (two columns)", "a ÷ b"),
    "product": ("Product", "Interaction (two columns)", "a × b"),
    "diff": ("Difference", "Interaction (two columns)", "a − b"),
    "sum": ("Sum", "Interaction (two columns)", "a + b"),
    "freq_enc": ("Frequency encoding", "Categorical encoding", "how common each category is"),
    "target_enc": ("Target encoding", "Categorical encoding", "smoothed target rate per category (out-of-fold on training rows)"),
    "grp_mean": ("Group mean", "Group statistic", "mean of a numeric column within each category"),
    "grp_std": ("Group std. deviation", "Group statistic", "spread of a numeric column within each category"),
}


def feature_engineering_report(eng, Z: pd.DataFrame) -> dict:
    """Everything the 'Automated Feature Engineering' section shows, read from the engine object that actually ran:
    eng.gen_ (the FeatureGenerator), its fitted settings, eng.lineage_ and eng.G_ (the candidate pool).
    `Z` is the matrix that was passed to eng.fit (= the generator's input). The only recomputation is the generator's own
    `_build` on `Z` with its already-fitted statistics, to learn how many candidates it created before its de-duplication."""
    gen = eng.gen_
    n_in = Z.shape[1]
    if gen is None:
        return {"enabled": False, "n_input": n_in, "n_pool": eng.G_.shape[1], "generated": pd.DataFrame(), "techniques": pd.DataFrame()}
    kept = list(gen.keep_)
    built = list(gen._build(Z)[0].columns)
    rows = [(n, _GEN_OPS.get(eng.lineage_[n]["op"], (eng.lineage_[n]["op"],))[0], ", ".join(eng.lineage_[n]["parents"]))
            for n in kept]
    gen_df = pd.DataFrame(rows, columns=["Generated Feature Name", "Transformation Type", "Source Columns"])
    ops_kept = pd.Series([eng.lineage_[n]["op"] for n in kept], dtype=object).value_counts()
    tech = []
    for op, (name, fam, how) in _GEN_OPS.items():
        if op in ops_kept.index:
            tech.append({"Technique": name, "Family": fam, "How it is built": how, "Features created": int(ops_kept[op])})
    return {"enabled": True, "n_input": n_in, "n_pool": eng.G_.shape[1], "n_built": len(built), "kept": kept,
            "removed": [n for n in built if n not in set(kept)], "generated": gen_df,
            "techniques": pd.DataFrame(tech), "top_cols": list(gen.top_), "n_cat_seen": len(gen.cat_),
            "settings": {"top_n": gen.top_n, "cap": gen.cap, "corr_thr": gen.thr}}


def selected_table(eng, origin: dict) -> pd.DataFrame:
    rows = []
    for f in eng.feature_scores_:
        if f.selected:
            src = sorted({origin.get(p, p) for p in (f.parents or [f.name])})
            rows.append({"feature": f.name, "type": ("Original column" if src == [f.name] else "Encoded column") if f.op == "raw" else f"Generated feature ({f.op})",
                         "from uploaded column(s)": ", ".join(src), "P": f.P, "S": f.S, "M": f.M, "R": f.R, "FQS": f.fqs})
    return pd.DataFrame(rows)


def with_target(Z: pd.DataFrame, y: np.ndarray, classes: list[str], target: str) -> pd.DataFrame:
    return Z.assign(**{target: np.asarray(classes, dtype=object)[y]})
