"""Aggregate check results into dimension scores and the Model Quality Score.

    MQS = TVS * W1 + MBS * W2 + ORS * W3          (weights sum to 1)

Each dimension score is the weighted average of its checks' 0-100 scores.
How missing evidence is treated is decided by the configured policy and is
never silent:

* UNAVAILABLE / ERROR checks are never counted as passes. Under
  `exclude_and_cap` they leave the average and lower the dimension's
  coverage; under `penalize` they score 0.
* NOT_APPLICABLE checks never applied, so they leave both the average and the
  coverage denominator and cost nothing.
* A dimension whose coverage is below `coverage.min_dimension` gets no score.
  Under `exclude_and_cap` that dimension drops out of the MQS (the remaining
  weights are renormalized and the MQS is marked provisional); under
  `penalize` it contributes 0.
* Excluding missing evidence is optimistic (a model can score higher by
  omitting an unfavourable check), so every scorecard also reports
  `mqs_lower_bound`, the MQS if every missing check had scored 0. Use it to
  show a range; do not reject on it, because missing evidence should lead to
  HOLD, not ROLLBACK.
* "Cap" refers to the decision, not the number: this module never inflates or
  trims an MQS, but a provisional score or missing required evidence stops the
  rules engine from deciding PROMOTE.

Coverage is the share of an applicable check weight that has evidence.
"""
from __future__ import annotations

from typing import Sequence

from mqvg.catalogue import CHECKS_BY_ID
from mqvg.config import GateConfig
from mqvg.schemas import (
    CheckContribution, CheckResult, CheckStatus, CheckTreatment, Dimension,
    DimensionScore, MissingPolicy, ScoreCard,
)


def _index_results(results: Sequence[CheckResult],
                   config: GateConfig) -> dict[str, CheckResult]:
    enabled = {i for i, c in config.checks.items() if c.enabled}
    by_id: dict[str, CheckResult] = {}
    for result in results:
        if result.id in by_id:
            raise ValueError(f"duplicate result for check '{result.id}'")
        if result.id not in enabled:
            raise ValueError(
                f"result for '{result.id}', which is unknown or disabled in "
                f"profile '{config.profile}'")
        by_id[result.id] = result
    return by_id


def _dimension_score(dimension: Dimension, results: dict[str, CheckResult],
                     config: GateConfig,
                     policy: MissingPolicy) -> DimensionScore:
    members = [i for i, c in config.checks.items()
               if c.enabled and CHECKS_BY_ID[i].dimension == dimension]

    rows = []   # (check_id, weight, status, score, treatment)
    for check_id in members:
        weight = config.checks[check_id].weight
        result = results.get(check_id)
        status = None if result is None else result.status
        if status == CheckStatus.NOT_APPLICABLE:
            rows.append((check_id, weight, status, None, CheckTreatment.NOT_APPLICABLE))
        elif result is not None and result.has_evidence:
            rows.append((check_id, weight, status, result.score, CheckTreatment.COUNTED))
        else:
            missing = (CheckTreatment.PENALIZED_AS_ZERO
                       if policy == MissingPolicy.PENALIZE
                       else CheckTreatment.EXCLUDED_UNAVAILABLE)
            rows.append((check_id, weight, status, None, missing))

    applicable = [r for r in rows if r[4] != CheckTreatment.NOT_APPLICABLE]
    applicable_weight = sum(r[1] for r in applicable)
    available = [r for r in applicable if r[4] == CheckTreatment.COUNTED]
    available_weight = sum(r[1] for r in available)

    coverage = available_weight / applicable_weight if applicable_weight else 0.0
    denominator = (applicable_weight if policy == MissingPolicy.PENALIZE
                   else available_weight)

    def points(row) -> float:
        _, weight, _, score, treatment = row
        if treatment != CheckTreatment.COUNTED or not denominator:
            return 0.0
        return weight * score / denominator

    contributions = [
        CheckContribution(check_id=r[0], weight=r[1], status=r[2], score=r[3],
                          treatment=r[4], points=points(r))
        for r in rows]

    note = None
    score: float | None = None
    if not applicable:
        note = "No applicable check in this dimension, so it has no score."
    elif coverage < config.coverage.min_dimension or not available:
        note = (f"Evidence coverage {coverage:.0%} is below the minimum "
                f"{config.coverage.min_dimension:.0%}, so this dimension has "
                "no score.")
    else:
        score = min(100.0, max(0.0, sum(c.points for c in contributions)))
        if coverage < 1:
            note = (f"Provisional: {coverage:.0%} of the applicable check "
                    f"weight had evidence ({policy.value}).")

    return DimensionScore(
        dimension=dimension, score=score, coverage=coverage,
        provisional=coverage < 1 or score is None,
        available_checks=len(available), total_checks=len(applicable),
        note=note, contributions=contributions)


def compute_scorecard(results: Sequence[CheckResult],
                      config: GateConfig) -> ScoreCard:
    """Dimension scores, coverage and the MQS for one validation run.

    `results` should hold one result per enabled check; an enabled check with
    no result is treated as unavailable. A result for an unknown or disabled
    check, or two results for one check, is a programming error.
    """
    by_id = _index_results(results, config)
    policy = config.missing_policy
    dimensions = {d: _dimension_score(d, by_id, config, policy) for d in Dimension}
    weights = config.weights.as_dict()

    scored = {d: s for d, s in dimensions.items() if s.score is not None}
    effective: dict[Dimension, float] | None = None
    mqs: float | None

    if policy == MissingPolicy.PENALIZE:
        mqs = sum(weights[d] * (s.score or 0.0) for d, s in dimensions.items())
    else:
        total = sum(weights[d] for d in scored)
        if not scored or total <= 0:
            mqs = None
        else:
            effective = {d: (weights[d] / total if d in scored else 0.0)
                         for d in Dimension}
            mqs = sum(effective[d] * s.score for d, s in scored.items())

    # Worst case: every missing check scores 0, and a dimension without a
    # score contributes 0. Equals the MQS under the penalize policy.
    if policy == MissingPolicy.PENALIZE:
        lower_bound = mqs
    else:
        penalized = {d: _dimension_score(d, by_id, config, MissingPolicy.PENALIZE)
                     for d in Dimension}
        lower_bound = sum(weights[d] * (s.score or 0.0)
                          for d, s in penalized.items())

    clip = lambda v: None if v is None else min(100.0, max(0.0, v))  # noqa: E731
    overall_coverage = min(1.0, max(0.0, sum(
        weights[d] * s.coverage for d, s in dimensions.items())))

    # Is there enough evidence to quote the MQS as a quality score?
    problems = []
    if mqs is None:
        problems.append("no MQS could be computed")
    unscored = [d.value.replace("_", " ") for d, s in dimensions.items()
                if s.score is None]
    if unscored and mqs is not None:
        problems.append("no score for " + " and ".join(unscored))
    if overall_coverage < config.coverage.min_overall:
        problems.append(f"overall evidence {overall_coverage:.0%} is below the "
                        f"required {config.coverage.min_overall:.0%}")

    return ScoreCard(
        dimensions=dimensions, weights=weights, effective_weights=effective,
        mqs=clip(mqs), mqs_lower_bound=clip(lower_bound),
        overall_coverage=overall_coverage,
        provisional=any(s.provisional for s in dimensions.values()),
        missing_policy=policy,
        mqs_sufficient_evidence=not problems,
        insufficient_evidence_reason="; ".join(problems) or None)
