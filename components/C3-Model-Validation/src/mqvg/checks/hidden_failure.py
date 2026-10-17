"""behaviour.hidden_failure_groups: groups where the model fails more than it expects.

Overall metrics, fairness and subgroup checks only look at groups someone
named in advance (the sensitive columns). This check searches the model's own
feature space for groups with unusually many errors (see `mqvg.hidden_groups`
for the method). The scalar it reports is the largest excess error rate of a
confirmed group, 0 when none is found (lower is better).

It is UNAVAILABLE, never a pass, when it cannot be run honestly: no
probability scores, too few rows or errors, or probabilities too poorly
calibrated for the test to be valid.

Feature use and privacy: by default the search uses every feature the model
uses (including any sensitive feature the model was given) and never the
target column. Groups are reported as rules over feature values, so a rule can
describe a demographic subpopulation even if no sensitive column was named; it
is aggregate information (a rule and counts), not individual rows. Use
`hidden_groups.exclude_features` to withhold features and
`hidden_groups.extra_columns` to include columns the model did not use.
"""
from __future__ import annotations

from typing import Callable

from mqvg.checks.base import (
    CheckContext, CheckFunction, guarded, scored_result, unavailable_result,
)
from mqvg.checks.predictions import get_predictions
from mqvg.hidden_groups import Detection, detect
from mqvg.schemas import CheckResult

HIDDEN_FAILURE_CHECKS: dict[str, CheckFunction] = {}
CHECK_ID = "behaviour.hidden_failure_groups"

CAVEATS = [
    "Expected errors come from the model's own probabilities; the test is "
    "only valid if they are calibrated. They are recalibrated per fold on the "
    "discovery rows only; grossly wrong probabilities make the check unavailable.",
    "Groups are axis-aligned rules or clusters of similar rows, so an oblique "
    "or very small region may be missed or found only in part.",
    "A group is an association found in the evaluation data, not a causal "
    "explanation: it says where the model fails, not why.",
    "Features that describe a group may be proxies for sensitive characteristics "
    "(for example a relationship status standing in for sex), so the rule can "
    "reveal a protected subpopulation even though no sensitive column was named.",
    "Detection power depends on the sample size; no group found does not mean "
    "none exists.",
]


def _details(detection: Detection, ctx: CheckContext, extra_missing: list[str]) -> dict:
    cfg = ctx.config.hidden_groups
    ordered = sorted(detection.tests,
                     key=lambda g: (not g.confirmed, -g.effect if g.confirmed else g.p_value))
    covered = detection.covered_rows() if detection.available else None
    return {
        "methods": list(cfg.methods), "folds": cfg.folds, "alpha": cfg.alpha,
        "min_effect": cfg.min_effect, "kmeans_weighting": cfg.kmeans_weighting,
        "seed": detection.seed, "n_rows": detection.n_rows,
        "n_errors": detection.n_errors, "ece": detection.ece,
        "overall_gap": detection.overall_gap,
        "recalibration_enabled": cfg.recalibrate,
        "recalibrated_folds": detection.recalibrated_folds,
        "features_used": detection.features_used,
        "features_excluded": detection.features_excluded,
        "pooled_categories": detection.pooled_categories,
        "extra_columns_missing": extra_missing,
        "groups_tested": len(detection.tests),
        "groups_confirmed": sum(g.confirmed for g in detection.tests),
        "share_of_rows_in_confirmed_groups":
            None if covered is None else float(covered.mean()),
        "groups": [g.as_dict() for g in ordered[:cfg.report_groups]],
        "caveats": CAVEATS,
    }


@guarded(CHECK_ID)
def hidden_failure_groups(ctx: CheckContext) -> CheckResult:
    cfg = ctx.config.hidden_groups
    predictions, reason = get_predictions(ctx.candidate)
    if predictions is None:
        return unavailable_result(ctx, CHECK_ID, reason)
    if predictions.y_score is None or predictions.score_source != "predict_proba":
        return unavailable_result(
            ctx, CHECK_ID,
            "Hidden failure groups need the model's probability estimates "
            "(predict_proba) to derive expected errors; "
            + ("decision-function scores are not probabilities."
               if predictions.y_score is not None else "none are available."))

    frame = ctx.candidate.eval_df.loc[predictions.row_mask]
    target = predictions.target_column
    features = [c for c in ctx.candidate.feature_columns
                if c not in cfg.exclude_features and c != target]
    extra = [c for c in cfg.extra_columns
             if c in frame.columns and c not in features and c != target]
    extra_missing = [c for c in cfg.extra_columns if c not in frame.columns]
    if not features and not extra:
        return unavailable_result(ctx, CHECK_ID, "No feature is left to search after exclusions.")

    detection = detect(
        frame[features + extra].reset_index(drop=True), predictions.y_true,
        predictions.y_pred, predictions.y_score, cfg,
        seed=ctx.config.evaluation.random_seed, ece_bins=ctx.config.evaluation.ece_bins)
    details = _details(detection, ctx, extra_missing)
    if not detection.available:
        return unavailable_result(
            ctx, CHECK_ID, f"Hidden failure groups not assessed: {detection.reason}.",
            details=details)

    confirmed = detection.confirmed
    methods = "+".join(cfg.methods)
    if not confirmed:
        message = (
            f"No feature-defined group had significantly more errors than the model "
            f"expects ({len(detection.tests)} groups tested on {detection.n_rows} rows "
            f"with {methods}; effect floor {cfg.min_effect:.0%}, false discovery rate "
            f"{cfg.alpha:g}). This does not prove none exists.")
    else:
        worst = confirmed[0]
        message = (
            f"{len(confirmed)} hidden failure group(s) found. The worst, "
            f"'{worst.rule}' ({worst.share:.1%} of rows), has an observed error rate of "
            f"{worst.observed_rate:.1%} against {worst.expected_rate:.1%} expected "
            f"({worst.effect:+.1%} points, 95% CI {worst.effect_ci_low:+.1%} to "
            f"{worst.effect_ci_high:+.1%}, adjusted p {worst.q_value:.1e}, found by "
            f"{worst.method} and tested on held-out rows).")
    return scored_result(ctx, CHECK_ID, detection.max_effect, message, details=details)


HIDDEN_FAILURE_CHECKS[CHECK_ID] = hidden_failure_groups
