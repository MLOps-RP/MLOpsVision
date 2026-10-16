"""Training validity checks: was this model trained and documented soundly?

A pretrained model rarely carries its whole history, so these checks are
explicit about their evidence:

* metadata completeness / reproducibility info: MEASURED facts about what the
  metadata contains (the contents themselves are not verified);
* overfit gap and train/eval overlap: DECLARED evidence, because they rely on
  producer-supplied training metrics and row hashes the gate cannot verify;
* leakage probe: MEASURED from the evaluation data and labels alone.

Anything that cannot be assessed is UNAVAILABLE, never a pass. The gate cannot
prove a split was correct or that no leakage exists; it can only find signs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from mqvg.catalogue import CATALOGUE
from mqvg.checks.base import (
    CheckContext, CheckFunction, guarded, scored_result, unavailable_result,
)
from mqvg.checks.predictions import (
    f1_from_labels, get_predictions, resolve_labels,
)
from mqvg.config import GateConfig
from mqvg.hashing import read_csv_as_text, row_hashes
from mqvg.loader import LoadedCandidate
from mqvg.schemas import CheckResult, Dimension, EvidenceSource

TRAINING_VALIDITY_CHECKS: dict[str, CheckFunction] = {}

# A string column is "free text or identifier-like" (and not assessable by the
# one-feature probe) if its values average at least this many words, or if more
# than this share of its rows are distinct.
FREE_TEXT_MIN_WORDS = 4
FREE_TEXT_MAX_DISTINCT_SHARE = 0.5
FREE_TEXT_SAMPLE = 2000
LEAKAGE_TREE_DEPTH = 4
LEAKAGE_MIN_LEAF = 20
TOP_FEATURES_REPORTED = 5


def _register(check_id: str) -> Callable[[CheckFunction], CheckFunction]:
    def decorator(func: CheckFunction) -> CheckFunction:
        TRAINING_VALIDITY_CHECKS[check_id] = guarded(check_id)(func)
        return TRAINING_VALIDITY_CHECKS[check_id]
    return decorator


# --------------------------------------------------------------------------
# Metadata presence
# --------------------------------------------------------------------------
def _presence(ctx: CheckContext, check_id: str, fields: list[str],
              what: str) -> CheckResult:
    c = ctx.candidate
    if c.metadata is None:
        why = (c.metadata_error if c.metadata_provided
               else "no training metadata was provided")
        return scored_result(
            ctx, check_id, 0.0, f"No {what} is available: {why}.",
            details={"missing": list(fields), "present": [], "reason": why})
    missing = c.metadata.missing_fields(fields)
    present = [f for f in fields if f not in missing]
    fraction = len(present) / len(fields)
    note = f" Missing: {missing}." if missing else ""
    return scored_result(
        ctx, check_id, fraction,
        f"{len(present)} of {len(fields)} {what} fields are present.{note}",
        details={"missing": missing, "present": present,
                 "note": "Presence is checked; the values are not verified."})


@_register("training_validity.metadata_completeness")
def metadata_completeness(ctx: CheckContext) -> CheckResult:
    return _presence(ctx, "training_validity.metadata_completeness",
                     ctx.config.metadata.required_fields,
                     "required training-metadata")


@_register("training_validity.reproducibility_info")
def reproducibility_info(ctx: CheckContext) -> CheckResult:
    return _presence(ctx, "training_validity.reproducibility_info",
                     ctx.config.metadata.reproducibility_fields,
                     "reproducibility")


# --------------------------------------------------------------------------
# Overfit gap
# --------------------------------------------------------------------------
@_register("training_validity.overfit_gap")
def overfit_gap(ctx: CheckContext) -> CheckResult:
    check_id = "training_validity.overfit_gap"
    declared = None if ctx.candidate.metadata is None \
        else ctx.candidate.metadata.training_metrics
    if not declared:
        return unavailable_result(
            ctx, check_id,
            "No training metrics are declared in the metadata, so the "
            "train/eval gap cannot be assessed (the gate has no access to the "
            "training data).")
    predictions, reason = get_predictions(ctx.candidate)
    if predictions is None:
        return unavailable_result(ctx, check_id, reason)

    measured = {"accuracy": float(np.mean(predictions.y_true == predictions.y_pred))}
    f1 = f1_from_labels(predictions.y_true, predictions.y_pred)
    if f1 is not None:
        measured["f1"] = f1
    metric = next((m for m in ("accuracy", "f1")
                   if m in declared and m in measured), None)
    if metric is None:
        return unavailable_result(
            ctx, check_id,
            f"The declared training metrics {sorted(declared)} include neither "
            "'accuracy' nor 'f1', or the matching evaluation metric is undefined.")
    train_value = declared[metric]
    if not 0.0 <= train_value <= 1.0:
        return unavailable_result(
            ctx, check_id,
            f"The declared training {metric} ({train_value}) is not within [0, 1].")
    gap = train_value - measured[metric]
    return scored_result(
        ctx, check_id, gap,
        f"Declared training {metric} {train_value:.3f} vs measured evaluation "
        f"{metric} {measured[metric]:.3f}: gap {gap:+.3f}. The training value "
        "is the producer's claim and was not verified.",
        evidence=EvidenceSource.DECLARED,
        details={"metric": metric, "declared_train_value": train_value,
                 "measured_eval_value": measured[metric],
                 "measured_eval_metrics": measured,
                 "declared_training_metrics": declared})


# --------------------------------------------------------------------------
# Train / eval overlap
# --------------------------------------------------------------------------
def _hash_file(candidate: LoadedCandidate) -> tuple[Path | None, str | None]:
    explicit = candidate.bundle.train_row_hashes_path
    if explicit is not None:
        if explicit.is_file():
            return explicit, None
        return None, f"training row-hash file not found: {explicit}"
    declared = (None if candidate.metadata is None
                else candidate.metadata.train_row_hashes_file)
    if not declared:
        return None, ("No training row hashes were provided; set "
                      "train_row_hashes_path or declare train_row_hashes_file "
                      "in the metadata.")
    tries = [Path(declared)]
    if candidate.bundle.metadata_path is not None:
        tries.append(candidate.bundle.metadata_path.parent / declared)
    for path in tries:
        if path.is_file():
            return path, None
    return None, ("the declared training row-hash file was not found "
                  f"(tried {[str(t) for t in tries]})")


def _eval_text(candidate: LoadedCandidate) -> pd.DataFrame:
    if "eval_text" not in candidate.cache:
        candidate.cache["eval_text"] = read_csv_as_text(
            candidate.bundle.eval_data_path)
    return candidate.cache["eval_text"]


@_register("training_validity.train_eval_overlap")
def train_eval_overlap(ctx: CheckContext) -> CheckResult:
    check_id = "training_validity.train_eval_overlap"
    c = ctx.candidate
    if c.eval_df is None:
        return unavailable_result(
            ctx, check_id, f"Evaluation data unavailable ({c.eval_error}).")
    path, problem = _hash_file(c)
    if path is None:
        return unavailable_result(ctx, check_id, problem)
    columns = None if c.metadata is None else c.metadata.row_hash_columns
    if not columns:
        return unavailable_result(
            ctx, check_id,
            "The metadata does not say which columns the training row hashes "
            "cover (row_hash_columns).")
    missing = [col for col in columns if col not in c.eval_df.columns]
    if missing:
        return unavailable_result(
            ctx, check_id,
            f"The evaluation data lacks columns used for the row hashes: {missing}.")

    train_hashes = {line.strip() for line in
                    path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not train_hashes:
        return unavailable_result(ctx, check_id,
                                  f"The row-hash file {path} is empty.")
    eval_hashes = row_hashes(_eval_text(c), columns)
    overlap = sum(h in train_hashes for h in eval_hashes)
    fraction = overlap / len(eval_hashes)
    internal_duplicates = 1 - len(set(eval_hashes)) / len(eval_hashes)
    return scored_result(
        ctx, check_id, fraction,
        f"{overlap} of {len(eval_hashes)} evaluation rows ({fraction:.2%}) also "
        "appear in the training row hashes. Datasets can contain genuine "
        f"duplicate records ({internal_duplicates:.2%} of evaluation rows "
        "duplicate another evaluation row), so a small overlap is not by "
        "itself leakage. The hashes are producer-supplied.",
        evidence=EvidenceSource.DECLARED,
        details={"overlap_rows": overlap, "eval_rows": len(eval_hashes),
                 "train_hashes": len(train_hashes),
                 "eval_internal_duplicate_fraction": internal_duplicates,
                 "hash_file": str(path), "hashed_columns": columns})


# --------------------------------------------------------------------------
# Leakage probe
# --------------------------------------------------------------------------
def is_free_text_like(values: pd.Series) -> bool:
    """True for string columns the one-feature probe cannot judge.

    The probe treats each distinct string as a category, which tells nothing
    for free text or identifiers: every value is (nearly) unique, so the probe
    would report ~0.5 and look reassuring although it examined nothing.
    """
    present = values.dropna()
    if present.empty:
        return False
    sample = present.iloc[:FREE_TEXT_SAMPLE]
    if not sample.map(lambda v: isinstance(v, str)).all():
        return False
    words = sample.map(lambda v: len(v.split())).mean()
    distinct_share = present.nunique() / len(present)
    return bool(words >= FREE_TEXT_MIN_WORDS
                or (len(present) >= 50 and distinct_share > FREE_TEXT_MAX_DISTINCT_SHARE))


def single_feature_auc(values: pd.Series, y: np.ndarray, folds: int,
                       seed: int) -> float:
    """Cross-validated ROC AUC of a shallow tree that sees ONE feature."""
    if values.nunique(dropna=True) < 2:
        return 0.5                       # a constant carries no information
    tree = DecisionTreeClassifier(max_depth=LEAKAGE_TREE_DEPTH,
                                  min_samples_leaf=LEAKAGE_MIN_LEAF,
                                  random_state=seed)
    if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        X = values.to_numpy(dtype=float).reshape(-1, 1)
        model = Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("tree", tree)])
    else:
        X = values.astype(object).where(values.notna(), np.nan).to_numpy().reshape(-1, 1)
        model = Pipeline([
            ("impute", SimpleImputer(strategy="constant",
                                     fill_value="__missing__")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
            ("tree", tree)])
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probability = cross_val_predict(model, X, y.astype(int), cv=cv,
                                    method="predict_proba")[:, 1]
    return float(roc_auc_score(y, probability))


@_register("leakage.single_feature_predictiveness")
def single_feature_predictiveness(ctx: CheckContext) -> CheckResult:
    """A feature that predicts the label almost perfectly on its own is a
    classic sign of target leakage (it may be derived from the label).

    This is a heuristic: a legitimately strong feature also scores high, and
    leakage spread over several features is not caught.
    """
    check_id = "leakage.single_feature_predictiveness"
    c = ctx.candidate
    info, problem = resolve_labels(c)
    if info is None:
        return unavailable_result(ctx, check_id, problem)
    features = c.feature_columns
    if features is None:
        return unavailable_result(
            ctx, check_id,
            "The model's feature columns are unknown (no loaded model with "
            "feature_names_in_ and none listed in the metadata).")
    missing = [f for f in features if f not in c.eval_df.columns]
    if missing:
        return unavailable_result(
            ctx, check_id, f"The evaluation data lacks feature columns {missing}.")

    evaluation = ctx.config.evaluation
    frame = c.eval_df.loc[info.row_mask, features].reset_index(drop=True)
    y = info.y_true
    if min(int(y.sum()), int((~y).sum())) < evaluation.leakage_cv_folds:
        return unavailable_result(
            ctx, check_id,
            f"Fewer than {evaluation.leakage_cv_folds} examples of one class, so "
            "the cross-validated probe cannot run.")
    sampled = len(frame) > evaluation.leakage_max_rows
    if sampled:
        rng = np.random.default_rng(evaluation.random_seed)
        index = np.sort(rng.choice(len(frame), evaluation.leakage_max_rows,
                                   replace=False))
        frame, y = frame.iloc[index].reset_index(drop=True), y[index]
        if min(int(y.sum()), int((~y).sum())) < evaluation.leakage_cv_folds:
            return unavailable_result(
                ctx, check_id, "The sampled rows contain too few examples of one class.")

    aucs, failed = {}, {}
    text_like = [f for f in features if is_free_text_like(frame[f])]
    if len(text_like) == len(features):
        return unavailable_result(
            ctx, check_id,
            f"Every feature column {text_like} is free text or identifier-like, "
            "which this probe cannot assess (it would report ~0.5 without "
            "examining anything). Leakage in text features is not checked.",
            details={"not_assessed_columns": text_like})
    for feature in features:
        if feature in text_like:
            continue
        try:
            aucs[feature] = single_feature_auc(
                frame[feature], y, evaluation.leakage_cv_folds,
                evaluation.random_seed)
        except Exception as exc:  # noqa: BLE001 - one odd column must not stop the rest
            failed[feature] = f"{type(exc).__name__}: {exc}"
    if not aucs:
        return unavailable_result(
            ctx, check_id, "No feature could be examined.",
            details={"failed_features": failed})

    ranked = sorted(aucs.items(), key=lambda item: item[1], reverse=True)
    top, top_auc = ranked[0]
    return scored_result(
        ctx, check_id, top_auc,
        f"The strongest single feature, '{top}', reaches a cross-validated ROC "
        f"AUC of {top_auc:.3f} on its own ({len(aucs)} features, {len(frame)} "
        "rows). Values near 1.0 suggest the feature leaks the label."
        + (f" Not assessed (free text or identifier-like): {text_like}."
           if text_like else ""),
        details={"top_feature": top,
                 "top_features": dict(ranked[:TOP_FEATURES_REPORTED]),
                 "all_features": dict(ranked), "failed_features": failed,
                 "not_assessed_columns": text_like,
                 "rows_used": len(frame), "sampled": sampled,
                 "cv_folds": evaluation.leakage_cv_folds,
                 "tree_depth": LEAKAGE_TREE_DEPTH,
                 "positive_label": info.positive_label,
                 "caveat": "A legitimately strong feature also scores high; "
                           "leakage spread over several features is not detected."})


# --------------------------------------------------------------------------
def run_training_validity_checks(candidate: LoadedCandidate,
                                 config: GateConfig) -> list[CheckResult]:
    """Run every enabled training-validity check, in catalogue order."""
    ctx = CheckContext(candidate=candidate, config=config)
    return [TRAINING_VALIDITY_CHECKS[spec.id](ctx)
            for spec in CATALOGUE
            if spec.dimension == Dimension.TRAINING_VALIDITY
            and config.checks[spec.id].enabled]
