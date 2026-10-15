"""Model behaviour checks: how well does the model perform on evaluation data?

All metrics here are MEASURED by the gate from the model's own predictions on
the supplied evaluation data. Metrics a producer wrote into the metadata are
never used. Binary classification only.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score

from mqvg.catalogue import CATALOGUE
from mqvg.checks.base import (
    CheckContext, CheckFunction, guarded, scored_result, unavailable_result,
)
from mqvg.checks.calibration import CALIBRATION_CHECKS
from mqvg.checks.fairness import FAIRNESS_CHECKS
from mqvg.checks.predictions import EvalPredictions, get_predictions
from mqvg.checks.robustness import ROBUSTNESS_CHECKS
from mqvg.config import GateConfig
from mqvg.loader import LoadedCandidate
from mqvg.schemas import CheckResult, Dimension

BEHAVIOUR_CHECKS: dict[str, CheckFunction] = {}


def _register(check_id: str) -> Callable[[CheckFunction], CheckFunction]:
    def decorator(func: CheckFunction) -> CheckFunction:
        BEHAVIOUR_CHECKS[check_id] = guarded(check_id)(func)
        return BEHAVIOUR_CHECKS[check_id]
    return decorator


def confusion(p: EvalPredictions) -> dict[str, int]:
    return {
        "tp": int(np.sum(p.y_true & p.y_pred)),
        "fp": int(np.sum(~p.y_true & p.y_pred)),
        "fn": int(np.sum(p.y_true & ~p.y_pred)),
        "tn": int(np.sum(~p.y_true & ~p.y_pred)),
    }


def classification_metrics(p: EvalPredictions) -> dict:
    """Precision, recall and F1 with the context needed to interpret them.

    Precision is undefined when nothing is predicted positive; it is reported
    as 0.0 with `precision_undefined` set so the report can say so.
    """
    c = confusion(p)
    predicted_pos, actual_pos = c["tp"] + c["fp"], c["tp"] + c["fn"]
    precision = c["tp"] / predicted_pos if predicted_pos else 0.0
    recall = c["tp"] / actual_pos if actual_pos else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    prevalence = actual_pos / p.n_rows
    return {
        "confusion": c,
        "precision": precision, "recall": recall, "f1": f1,
        "precision_undefined": predicted_pos == 0,
        "n_rows": p.n_rows,
        "dropped_rows_missing_label": p.dropped_rows,
        "positive_label": p.positive_label,
        "positive_label_source": p.label_source,
        "prevalence": prevalence,
        # What a trivial classifier would score, for context in reports.
        "baseline_always_positive": {
            "precision": prevalence, "recall": 1.0,
            "f1": 2 * prevalence / (1 + prevalence),
            "roc_auc": 0.5,
        },
    }


def _ready(ctx: CheckContext, check_id: str
           ) -> tuple[EvalPredictions | None, CheckResult | None]:
    """Predictions with both classes present, or an UNAVAILABLE result."""
    predictions, reason = get_predictions(ctx.candidate)
    if predictions is None:
        return None, unavailable_result(ctx, check_id, reason)
    if predictions.y_true.all() or not predictions.y_true.any():
        which = "positive" if predictions.y_true.all() else "negative"
        return None, unavailable_result(
            ctx, check_id,
            f"All {predictions.n_rows} evaluation rows are {which} examples, "
            "so this metric is undefined.",
            details={"positive_label": predictions.positive_label})
    return predictions, None


def _metric_check(ctx: CheckContext, check_id: str, key: str, label: str
                  ) -> CheckResult:
    predictions, unavailable = _ready(ctx, check_id)
    if unavailable:
        return unavailable
    metrics = classification_metrics(predictions)
    value = metrics[key]
    extra = (" No row was predicted positive, so precision is undefined "
             "and counted as 0." if key == "precision"
             and metrics["precision_undefined"] else "")
    # An always-positive classifier has recall 1 by construction, so a
    # baseline is only quoted for the other metrics.
    context = ""
    if key != "recall":
        baseline = metrics["baseline_always_positive"][key]
        context = f" (an always-positive baseline scores {baseline:.3f})"
    return scored_result(
        ctx, check_id, value,
        f"{label} is {value:.3f} on {metrics['n_rows']} evaluation rows"
        f"{context}.{extra}",
        details=metrics)


@_register("behaviour.f1")
def f1(ctx: CheckContext) -> CheckResult:
    return _metric_check(ctx, "behaviour.f1", "f1", "F1")


@_register("behaviour.precision")
def precision(ctx: CheckContext) -> CheckResult:
    return _metric_check(ctx, "behaviour.precision", "precision", "Precision")


@_register("behaviour.recall")
def recall(ctx: CheckContext) -> CheckResult:
    return _metric_check(ctx, "behaviour.recall", "recall", "Recall")


@_register("behaviour.roc_auc")
def roc_auc(ctx: CheckContext) -> CheckResult:
    check_id = "behaviour.roc_auc"
    predictions, unavailable = _ready(ctx, check_id)
    if unavailable:
        return unavailable
    if predictions.y_score is None:
        return unavailable_result(
            ctx, check_id,
            "The model provides no probability or decision scores, so ROC AUC "
            "cannot be computed.")
    value = float(roc_auc_score(predictions.y_true, predictions.y_score))
    return scored_result(
        ctx, check_id, value,
        f"ROC AUC is {value:.3f} on {predictions.n_rows} evaluation rows "
        "(0.5 is chance).",
        details={"n_rows": predictions.n_rows,
                 "score_source": predictions.score_source,
                 "positive_label": predictions.positive_label,
                 "positive_label_source": predictions.label_source})


BEHAVIOUR_CHECKS.update(FAIRNESS_CHECKS)
BEHAVIOUR_CHECKS.update(CALIBRATION_CHECKS)
BEHAVIOUR_CHECKS.update(ROBUSTNESS_CHECKS)


def run_behaviour_checks(candidate: LoadedCandidate,
                         config: GateConfig) -> list[CheckResult]:
    """Run every enabled behaviour check, in catalogue order."""
    ctx = CheckContext(candidate=candidate, config=config)
    return [BEHAVIOUR_CHECKS[spec.id](ctx)
            for spec in CATALOGUE
            if spec.dimension == Dimension.MODEL_BEHAVIOUR
            and config.checks[spec.id].enabled]
