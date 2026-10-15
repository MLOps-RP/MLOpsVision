"""Evaluation predictions shared by the model-behaviour checks.

The model is run on the evaluation data once per validation and the result
is cached on the candidate, so F1, fairness, calibration and the rest all
see identical predictions. When predictions cannot be produced honestly, the
reason is returned so each check can report itself UNAVAILABLE.

Scope: binary classification. Labels are compared as strings so that 1,
"1" and True from different sources agree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mqvg.loader import LoadedCandidate
from mqvg.schemas import TaskType


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


def _resolve_positive_label(candidate: LoadedCandidate,
                            observed: list[str]) -> tuple[str | None, str, str | None]:
    """Return (label, source, problem)."""
    declared = candidate.positive_label
    if declared is not None:
        label = str(declared)
        if label not in observed:
            return None, "declared", (
                f"declared positive label {label!r} does not appear in the "
                f"target column (values: {observed}); either the label is "
                "misspelled or the evaluation data has no positive examples")
        return label, "declared in bundle or metadata", None
    classes = getattr(candidate.model, "classes_", None)
    if classes is not None and len(classes) == 2:
        label = str(classes[1])
        if label in observed:
            return label, "inferred: second class of model.classes_", None
    return None, "undeclared", (
        "positive label is not declared and could not be inferred; set "
        "positive_label in the metadata or bundle")


def _scores(model, X, positive_label: str) -> tuple[np.ndarray | None, str | None]:
    classes = [str(c) for c in getattr(model, "classes_", [])]
    try:
        if hasattr(model, "predict_proba") and positive_label in classes:
            proba = np.asarray(model.predict_proba(X))
            return proba[:, classes.index(positive_label)], "predict_proba"
        if hasattr(model, "decision_function") and len(classes) == 2:
            raw = np.asarray(model.decision_function(X), dtype=float)
            if raw.ndim == 1:
                return (raw if classes[1] == positive_label else -raw,
                        "decision_function")
    except Exception:  # noqa: BLE001 - scores are optional
        pass
    return None, None


def _compute(candidate: LoadedCandidate) -> Result:
    blocker = prediction_blocker(candidate)
    if blocker:
        return None, "Cannot evaluate: " + blocker

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
    labels = frame.loc[keep, target].astype(str)
    observed = sorted(labels.unique())
    if len(observed) > 2:
        return None, (f"The target has {len(observed)} classes; multiclass "
                      "evaluation is not supported yet.")

    positive, source, problem = _resolve_positive_label(candidate, observed)
    if positive is None:
        return None, problem

    X = frame.loc[keep, candidate.feature_columns]
    try:
        predicted = np.asarray(candidate.model.predict(X)).astype(str)
    except Exception as exc:  # noqa: BLE001
        return None, f"The model failed to predict: {type(exc).__name__}: {exc}"
    if len(predicted) != len(X):
        return None, (f"The model returned {len(predicted)} predictions for "
                      f"{len(X)} rows.")

    score, score_source = _scores(candidate.model, X, positive)
    return EvalPredictions(
        y_true=(labels.to_numpy() == positive),
        y_pred=(predicted == positive),
        y_score=score, score_source=score_source,
        row_mask=keep, target_column=target, positive_label=positive,
        label_source=source, dropped_rows=int((~keep).sum()),
    ), None


def get_predictions(candidate: LoadedCandidate) -> Result:
    """Predictions on the evaluation data, computed once per candidate."""
    if "predictions" not in candidate.cache:
        candidate.cache["predictions"] = _compute(candidate)
    return candidate.cache["predictions"]
