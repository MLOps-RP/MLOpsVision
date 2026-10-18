"""Robustness check: how much does performance drop under small input noise?

A simple, reproducible probe: add Gaussian noise to every numeric feature
(standard deviation = `robustness_noise_scale` x that feature's own standard
deviation in the evaluation data), re-predict, and compare F1 with the clean
F1. Repeated with different but seeded noise.

Limitations (state them wherever the result is quoted):
* numeric features only; categorical features are not perturbed;
* noise can produce implausible values (for example a negative age) because it
  ignores each feature's valid range and its correlations with other features;
* it is a sensitivity probe, not an adversarial or distribution-shift test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mqvg.checks.base import (
    CheckContext, CheckFunction, guarded, scored_result, unavailable_result,
)
from mqvg.checks.predictions import f1_from_labels, get_predictions
from mqvg.schemas import CheckResult, CheckStatus

ROBUSTNESS_CHECKS: dict[str, CheckFunction] = {}


@guarded("robustness.noise_degradation")
def noise_degradation(ctx: CheckContext) -> CheckResult:
    check_id = "robustness.noise_degradation"
    predictions, reason = get_predictions(ctx.candidate)
    if predictions is None:
        return unavailable_result(ctx, check_id, reason)

    clean_f1 = f1_from_labels(predictions.y_true, predictions.y_pred)
    if not clean_f1:     # None (undefined) or 0 (a relative drop is meaningless)
        return unavailable_result(
            ctx, check_id,
            "Clean F1 is zero or undefined, so a relative drop under noise "
            "cannot be computed.")

    evaluation = ctx.config.evaluation
    X = ctx.candidate.eval_df.loc[predictions.row_mask,
                                  ctx.candidate.feature_columns]
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
               and not pd.api.types.is_bool_dtype(X[c])]
    spreads = {c: float(X[c].std(ddof=0)) for c in numeric}
    perturbed = [c for c in numeric if spreads[c] > 0]
    if not perturbed:
        return unavailable_result(
            ctx, check_id,
            "The model has no numeric feature with any variation to perturb.",
            status=CheckStatus.NOT_APPLICABLE)

    noisy_f1s = []
    for repeat in range(evaluation.robustness_repeats):
        rng = np.random.default_rng(evaluation.random_seed + repeat)
        noisy = X.copy()
        for column in perturbed:
            sigma = evaluation.robustness_noise_scale * spreads[column]
            noisy[column] = X[column].astype(float) + rng.normal(0, sigma, len(X))
        predicted = np.asarray(ctx.candidate.model.predict(noisy)).astype(str)
        f1 = f1_from_labels(predictions.y_true,
                            predicted == predictions.positive_label)
        noisy_f1s.append(0.0 if f1 is None else f1)

    mean_noisy = float(np.mean(noisy_f1s))
    drop = (clean_f1 - mean_noisy) / clean_f1
    return scored_result(
        ctx, check_id, drop,
        f"F1 falls from {clean_f1:.3f} to {mean_noisy:.3f} on average "
        f"({drop:.1%} relative drop) when Gaussian noise of "
        f"{evaluation.robustness_noise_scale:g} x each feature's spread is added "
        f"to {len(perturbed)} numeric feature(s), over "
        f"{evaluation.robustness_repeats} seeded repeats.",
        details={"clean_f1": clean_f1, "noisy_f1_per_repeat": noisy_f1s,
                 "mean_noisy_f1": mean_noisy,
                 "noise_scale": evaluation.robustness_noise_scale,
                 "repeats": evaluation.robustness_repeats,
                 "seed": evaluation.random_seed,
                 "perturbed_columns": perturbed,
                 "feature_std": {c: spreads[c] for c in perturbed},
                 "not_perturbed_columns": [c for c in X.columns
                                           if c not in perturbed]})


ROBUSTNESS_CHECKS["robustness.noise_degradation"] = noise_degradation
