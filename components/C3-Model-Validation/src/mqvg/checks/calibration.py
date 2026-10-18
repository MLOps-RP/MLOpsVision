"""Calibration check: do the model's probabilities match observed frequencies?

A calibrated model that says "0.8" is right about 80% of the time. Calibration
matters when probabilities drive decisions (thresholds, costs, ranking).

Expected calibration error (ECE) splits [0, 1] into equal-width bins and
averages |mean predicted probability - observed positive rate| over the bins,
weighted by bin size. Caveats worth stating in any report: ECE depends on the
number of bins and is biased upward on small samples, so it is reported with
the per-bin counts.
"""
from __future__ import annotations

import numpy as np

from mqvg.checks.base import (
    CheckContext, CheckFunction, guarded, scored_result, unavailable_result,
)
from mqvg.checks.predictions import get_predictions
from mqvg.schemas import CheckResult

CALIBRATION_CHECKS: dict[str, CheckFunction] = {}

# Average predicted vs observed rate must differ by more than this before the
# message calls the model over- or under-confident.
DIRECTION_NOTE_THRESHOLD = 0.01


def reliability_summary(y_true: np.ndarray, scores: np.ndarray,
                        bins: int) -> dict:
    """ECE, maximum calibration error, Brier score and the per-bin table."""
    n = len(scores)
    # equal-width bins; a score of exactly 1.0 belongs to the last bin
    index = np.minimum((scores * bins).astype(int), bins - 1)
    observed = y_true.astype(float)
    table, ece, mce = [], 0.0, 0.0
    for b in range(bins):
        in_bin = index == b
        count = int(in_bin.sum())
        row = {"lower": b / bins, "upper": (b + 1) / bins, "n": count,
               "mean_score": None, "observed_rate": None, "gap": None}
        if count:
            mean_score = float(scores[in_bin].mean())
            rate = float(observed[in_bin].mean())
            gap = abs(mean_score - rate)
            row.update(mean_score=mean_score, observed_rate=rate, gap=gap)
            ece += count / n * gap
            mce = max(mce, gap)
        table.append(row)
    return {
        "ece": ece, "max_calibration_error": mce,
        "brier_score": float(np.mean((scores - observed) ** 2)),
        "mean_score": float(scores.mean()),
        "observed_positive_rate": float(observed.mean()),
        "bins": bins, "n_rows": n, "reliability": table,
    }


@guarded("calibration.ece")
def calibration_ece(ctx: CheckContext) -> CheckResult:
    check_id = "calibration.ece"
    predictions, reason = get_predictions(ctx.candidate)
    if predictions is None:
        return unavailable_result(ctx, check_id, reason)
    if predictions.y_score is None:
        return unavailable_result(
            ctx, check_id,
            "The model provides no probability scores, so calibration cannot "
            "be assessed.")
    if predictions.score_source != "predict_proba":
        return unavailable_result(
            ctx, check_id,
            f"The model's scores come from {predictions.score_source}, which "
            "are not probabilities, so calibration is undefined.")
    scores = np.asarray(predictions.y_score, dtype=float)
    if not np.all(np.isfinite(scores)) or scores.min() < 0 or scores.max() > 1:
        return unavailable_result(
            ctx, check_id,
            "The model's probabilities are not all finite values in [0, 1].")

    summary = reliability_summary(predictions.y_true, scores,
                                  ctx.config.evaluation.ece_bins)
    shift = summary["mean_score"] - summary["observed_positive_rate"]
    if abs(shift) <= DIRECTION_NOTE_THRESHOLD:
        direction = "close to the observed rate on average"
    elif shift > 0:
        direction = "over-confident on average (predicts higher than observed)"
    else:
        direction = "under-confident on average (predicts lower than observed)"
    return scored_result(
        ctx, check_id, summary["ece"],
        f"Expected calibration error is {summary['ece']:.3f} (Brier "
        f"{summary['brier_score']:.3f}, {summary['bins']} bins, "
        f"{summary['n_rows']} rows). Average predicted probability "
        f"{summary['mean_score']:.3f} vs observed positive rate "
        f"{summary['observed_positive_rate']:.3f}: {direction}.",
        details={**summary, "positive_label": predictions.positive_label})


CALIBRATION_CHECKS["calibration.ece"] = calibration_ece
