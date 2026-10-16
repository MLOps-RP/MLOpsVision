"""Evaluation predictions shared by the model-behaviour checks.

The model is run on the evaluation data once per validation and the result
is cached on the candidate, so F1, fairness, calibration and the rest all
see identical predictions. When predictions cannot be produced honestly, the
reason is returned so each check can report itself UNAVAILABLE.

Scope: binary classification. Labels are compared as normalized text (see
`normalize_label`) so that 1, 1.0, "1" and True from different sources agree.

A model and its evaluation data must use the same label values. If they do
not (for example the model outputs 0/1 but the target holds ">50K"), no metric
is computed: any number would blame the model for a data mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mqvg.loader import LoadedCandidate
from mqvg.schemas import TaskType

# More distinct numeric target values than this looks like regression.
CONTINUOUS_TARGET_MIN_DISTINCT = 20


@dataclass(frozen=True)
class EvalPredictions:
    y_true: np.ndarray        # bool: the true label is the positive class
    y_pred: np.ndarray        # bool: the model predicted the positive class
    y_score: np.ndarray | None  # P(positive) or a monotone score, if available
    score_source: str | None  # "predict_proba" or "decision_function"
    row_mask: np.ndarray      # bool over candidate.eval_df: rows used here
    target_column: str
    positive_label: str
    label_source: str         # where the positive label came from
    dropped_rows: int         # rows ignored because their label was missing

    @property
    def n_rows(self) -> int:
        return int(len(self.y_true))


Result = tuple[EvalPredictions | None, str | None]


def normalize_label(value) -> str:
    """Canonical text for a class label.

    Integer-valued numbers collapse (1, 1.0 and "1.0" all become "1") so a
    CSV column and a model that disagree only about int versus float still
    match. Everything else is compared as its text.
    """
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return text
    if not np.isfinite(number):
        return text
    return str(int(number)) if number.is_integer() else text


def _normalized(values) -> np.ndarray:
    """normalize_label over an array, computing each distinct value once."""
    series = pd.Series(np.asarray(values).ravel())
    mapping = {v: normalize_label(v) for v in series.unique()}
    return series.map(mapping).to_numpy()


def f1_from_labels(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """F1 of the positive class; None when it is undefined (no positives at
    all, true or predicted)."""
    tp = int(np.sum(y_true & y_pred))
    denominator = 2 * tp + int(np.sum(~y_true & y_pred)) + int(np.sum(y_true & ~y_pred))
    return 2 * tp / denominator if denominator else None


def prediction_blocker(candidate: LoadedCandidate) -> str | None:
    """Why the model cannot be run on the evaluation data, or None if it can."""
    if candidate.model is None:
        return f"model could not be loaded ({candidate.model_error})"
    if candidate.eval_df is None:
        return f"evaluation data unavailable ({candidate.eval_error})"
    features = candidate.feature_columns
    if features is None:
        return ("cannot determine the model's feature columns (the model has "
                "no feature_names_in_ and the metadata lists none)")
    missing = [c for c in features if c not in candidate.eval_df.columns]
    if missing:
        return f"evaluation data lacks model feature columns {missing}"
    return None


def model_classes(model) -> list[str] | None:
    """The model's class labels, normalized, or None if it has none."""
    classes = getattr(model, "classes_", None)
    if classes is None or np.ndim(classes) != 1:
        return None
    return [normalize_label(c) for c in classes]


def _mismatch(observed: list[str], model_side: list[str], what: str) -> str:
    return (
        f"Label mismatch: the evaluation target contains {observed} but "
        f"{what} {model_side}. The model and the evaluation data appear to use "
        "different label encodings (for example 0/1 versus text), so no "
        "metric can be computed. Re-encode the evaluation target, or supply a "
        "model trained on the same labels.")


def _resolve_positive_label(candidate: LoadedCandidate,
                            observed: list[str]) -> tuple[str | None, str, str | None]:
    """Return (label, source, problem)."""
    declared = candidate.positive_label
    if declared is not None:
        label = normalize_label(declared)
        if label not in observed:
            return None, "declared", (
                f"declared positive label {label!r} does not appear in the "
                f"target column (values: {observed}); either the label is "
                "misspelled or the evaluation data has no positive examples")
        return label, "declared in bundle or metadata", None
    classes = model_classes(candidate.model)
    if classes is not None and len(classes) == 2:
        if classes[1] in observed:
            return classes[1], "inferred: second class of model.classes_", None
        if not set(observed) <= set(classes):
            return None, "undeclared", _mismatch(
                observed, classes, "the model's classes are")
    return None, "undeclared", (
        "positive label is not declared and could not be inferred; set "
        "positive_label in the metadata or bundle")


def _scores(model, X, positive_label: str) -> tuple[np.ndarray | None, str | None]:
    classes = model_classes(model) or []
    try:
        if hasattr(model, "predict_proba") and positive_label in classes:
            proba = np.asarray(model.predict_proba(X))
            return proba[:, classes.index(positive_label)], "predict_proba"
        if (hasattr(model, "decision_function") and len(classes) == 2
                and positive_label in classes):
            raw = np.asarray(model.decision_function(X), dtype=float)
            if raw.ndim == 1:
                return (raw if classes[1] == positive_label else -raw,
                        "decision_function")
    except Exception:  # noqa: BLE001 - scores are optional
        pass
    return None, None


@dataclass(frozen=True)
class LabelInfo:
    """The evaluation labels, resolved without needing the model to run."""
    y_true: np.ndarray        # bool over the kept rows
    row_mask: np.ndarray      # bool over candidate.eval_df: rows with a label
    target_column: str
    positive_label: str
    label_source: str
    dropped_rows: int
    observed: list[str]       # distinct (normalized) target values


def resolve_labels(candidate: LoadedCandidate
                   ) -> tuple[LabelInfo | None, str | None]:
    """Target column, positive label and true labels, or the reason they
    cannot be resolved. Needs the evaluation data but not the model."""
    if candidate.eval_df is None:
        return None, f"evaluation data unavailable ({candidate.eval_error})"
    task = candidate.metadata.task_type if candidate.metadata else None
    if task is not None and task != TaskType.BINARY_CLASSIFICATION:
        return None, (f"Task type '{task.value}' is not supported yet; only "
                      "binary classification is evaluated.")

    target = candidate.target_column
    if target is None:
        return None, ("The target column is not declared; set target_column in "
                      "the metadata or bundle.")
    frame = candidate.eval_df
    if target not in frame.columns:
        return None, f"Target column '{target}' is not in the evaluation data."

    keep = frame[target].notna().to_numpy()
    if not keep.any():
        return None, f"Every value in target column '{target}' is missing."
    labels = _normalized(frame.loc[keep, target])
    observed = sorted(set(labels))
    if len(observed) > 2:
        numeric = (pd.api.types.is_numeric_dtype(frame[target])
                   and not pd.api.types.is_bool_dtype(frame[target]))
        if numeric and len(observed) > CONTINUOUS_TARGET_MIN_DISTINCT:
            return None, (
                f"The target has {len(observed)} distinct numeric values, so "
                "this looks like a regression task, which is not supported "
                "yet (only binary classification is evaluated).")
        return None, (f"The target has {len(observed)} classes; multiclass "
                      "evaluation is not supported yet.")

    positive, source, problem = _resolve_positive_label(candidate, observed)
    if positive is None:
        return None, problem
    return LabelInfo(
        y_true=(labels == positive), row_mask=keep,
        target_column=target, positive_label=positive, label_source=source,
        dropped_rows=int((~keep).sum()), observed=observed), None


def _label_space_problem(model, info: LabelInfo, predicted: np.ndarray
                         ) -> str | None:
    """Why the model's labels cannot be compared with the evaluation labels."""
    classes = model_classes(model)
    if classes is not None:
        if not set(info.observed) <= set(classes) or info.positive_label not in classes:
            return _mismatch(info.observed, classes, "the model's classes are")
        return None
    # No classes_ attribute: judge by what the model actually predicted.
    seen = sorted(set(predicted))
    if len(info.observed) == 2:
        if not set(seen) <= set(info.observed):
            return _mismatch(info.observed, seen, "the model predicts")
    elif not set(seen) & set(info.observed):
        return _mismatch(info.observed, seen, "the model predicts")
    return None


def _compute(candidate: LoadedCandidate) -> Result:
    blocker = prediction_blocker(candidate)
    if blocker:
        return None, "Cannot evaluate: " + blocker
    info, problem = resolve_labels(candidate)
    if info is None:
        return None, problem

    X = candidate.eval_df.loc[info.row_mask, candidate.feature_columns]
    try:
        raw = np.asarray(candidate.model.predict(X))
    except Exception as exc:  # noqa: BLE001
        return None, f"The model failed to predict: {type(exc).__name__}: {exc}"
    if len(raw) != len(X):
        return None, (f"The model returned {len(raw)} predictions for "
                      f"{len(X)} rows. The model may expect a different input "
                      "form (for example a single text column or a plain "
                      "array) than the DataFrame of feature columns the gate "
                      "supplies.")
    predicted = _normalized(raw)

    mismatch = _label_space_problem(candidate.model, info, predicted)
    if mismatch:
        return None, mismatch

    score, score_source = _scores(candidate.model, X, info.positive_label)
    return EvalPredictions(
        y_true=info.y_true, y_pred=(predicted == info.positive_label),
        y_score=score, score_source=score_source,
        row_mask=info.row_mask, target_column=info.target_column,
        positive_label=info.positive_label, label_source=info.label_source,
        dropped_rows=info.dropped_rows,
    ), None


def get_predictions(candidate: LoadedCandidate) -> Result:
    """Predictions on the evaluation data, computed once per candidate."""
    if "predictions" not in candidate.cache:
        candidate.cache["predictions"] = _compute(candidate)
    return candidate.cache["predictions"]
