"""Plain-English explanation of a validation result.

This explains why the GATE reached its decision: which rules fired, which
checks cost the most MQS points, which evidence is missing or only declared,
and what to do about each failure. It is a deterministic trace of the result.

It is NOT an explanation of the model's individual predictions. Prediction
explanations (SHAP, LIME) are a separate, later feature; where they are used
they should support, not replace, this trace.

The suggested actions are generic starting points, not guarantees.
"""
from __future__ import annotations

from typing import Sequence

from mqvg.config import GateConfig
from mqvg.schemas import (
    CheckResult, CheckStatus, CheckTreatment, Decision, DecisionRecord,
    EvidenceSource, RuleKind, ScoreCard,
)

NO_EVIDENCE = (CheckStatus.UNAVAILABLE, CheckStatus.ERROR)
MAX_COSTLY_CHECKS = 5
MIN_COST_REPORTED = 0.05      # MQS points
MAX_ACTIONS = 8
MESSAGE_LIMIT = 170

CHECK_HINTS = {
    "leakage.single_feature_predictiveness":
        "Inspect the named feature. If it is derived from the target or only "
        "known after the outcome, remove it and retrain.",
    "training_validity.overfit_gap":
        "Reduce model complexity or add regularisation, and tune on a "
        "separate validation set.",
    "training_validity.train_eval_overlap":
        "Remove evaluation rows that also appear in the training data, or "
        "re-split the data.",
    "training_validity.metadata_completeness":
        "Complete the training metadata; the check lists the missing fields.",
    "training_validity.reproducibility_info":
        "Record the random seed, library versions, data hash and code commit.",
    "behaviour.f1": "Improve the model or features; compare against a simple baseline.",
    "behaviour.roc_auc": "Improve the model's ranking quality (features, model family).",
    "behaviour.precision": "Review the decision threshold and the false-positive cost.",
    "behaviour.recall": "Review the decision threshold and the false-negative cost.",
    "behaviour.subgroup_f1_gap":
        "Examine the weakest group: check its data volume and quality, and "
        "whether features behave differently for it.",
    "fairness.selection_rate_ratio":
        "Investigate the group gap: review training-data balance and proxy "
        "features, and consider mitigation (for example reweighting or "
        "group-aware thresholds).",
    "fairness.equalized_odds":
        "Compare error rates between groups and consider mitigation; this "
        "can trade off against accuracy.",
    "calibration.ece":
        "Recalibrate the probabilities (for example Platt scaling or isotonic "
        "regression) on held-out data.",
    "robustness.noise_degradation":
        "Test with noisier training data or regularisation; check feature "
        "sensitivity.",
    "behaviour.hidden_failure_groups":
        "Investigate the reported group: check its data volume and label "
        "quality, add or engineer features that separate it, or retrain with "
        "more examples from it.",
    "operational.model_loadable":
        "Check the model path, and that it was saved with joblib or pickle in "
        "a compatible environment.",
    "operational.input_schema":
        "Make the evaluation data have exactly the columns and types the "
        "model was trained on.",
    "operational.prediction_determinism":
        "Fix random seeds or remove nondeterminism in prediction.",
    "operational.malformed_input_handling":
        "Add input validation so malformed rows are rejected, not predicted.",
    "operational.artifact_size": "Reduce model size (fewer or shallower trees, pruning).",
    "operational.latency":
        "Reduce model size or complexity, or validate on the target hardware.",
    "operational.memory": "Reduce model size, or validate on the target hardware.",
    "operational.dependency_match":
        "Run with the same library versions the model was trained with.",
}

RULE_HINTS = {
    "required_evidence_missing":
        "Supply the missing inputs (metadata, target and positive label, "
        "sensitive columns) so the required checks can run.",
    "overall_coverage_below_minimum":
        "Supply more of the inputs the checks need; coverage rises as checks "
        "gain evidence.",
    "mqs_insufficient_evidence":
        "Supply the inputs for the dimension without a score.",
    "mqs_unavailable": "Supply the evaluation data, labels and metadata.",
    "mqs_below_reject_threshold": "Improve the weakest checks listed above.",
    "mqs_below_promote_threshold": "Improve the weakest checks listed above.",
}


def _short(message: str) -> str:
    text = " ".join(message.split())
    return text if len(text) <= MESSAGE_LIMIT else text[:MESSAGE_LIMIT - 1] + "…"


def mqs_point_losses(scorecard: ScoreCard) -> list[tuple[str, float]]:
    """MQS points each check cost, largest first.

    A check's loss is how far its score fell short of 100, scaled by its
    weight within the dimension and the dimension's weight in the MQS.
    """
    weights = scorecard.effective_weights or scorecard.weights
    losses = []
    for dimension, dim in scorecard.dimensions.items():
        if dim.score is None:
            continue
        counted = [c for c in dim.contributions if c.treatment in (
            CheckTreatment.COUNTED, CheckTreatment.PENALIZED_AS_ZERO)]
        denominator = sum(c.weight for c in counted)
        if not denominator:
            continue
        for c in counted:
            possible = c.weight * 100.0 / denominator
            losses.append((c.check_id,
                           (possible - c.points) * weights[dimension]))
    return sorted(losses, key=lambda item: (-item[1], item[0]))


def explain_failures(checks: Sequence[CheckResult], scorecard: ScoreCard,
                     decision: DecisionRecord, config: GateConfig,
                     simulated_inputs: Sequence[str] = ()) -> list[str]:
    """Explanation lines, each starting with a label such as 'Why:'."""
    lines = [f"Decision: {decision.summary}"]

    for rule in decision.triggered_rules:
        if rule.effect != Decision.PROMOTE:
            lines.append(f"Why ({rule.kind.value}): {rule.reason}")

    if decision.overridden_by_critical and decision.mqs is not None:
        names = ", ".join(r.name for r in decision.triggered_rules
                          if r.kind == RuleKind.CRITICAL)
        lines.append(
            f"Override: critical rule(s) force ROLLBACK ({names}) although the "
            f"MQS of {decision.mqs:.1f} is "
            f"{'above' if decision.mqs >= config.decision.promote_at else 'below'}"
            f" the promote threshold of {config.decision.promote_at:g}. A good "
            "score cannot outweigh a critical failure.")

    for check in checks:
        if check.id == "behaviour.hidden_failure_groups" and check.has_evidence:
            groups = [g for g in check.details.get("groups", []) if g.get("confirmed")]
            if groups:
                worst = groups[0]
                lines.append(
                    f"Hidden failure group ({worst['method']}): {worst['rule']} — "
                    f"{worst['n_rows']} held-out rows, observed error "
                    f"{worst['observed_error_rate']:.1%} vs {worst['expected_error_rate']:.1%} "
                    f"expected; {len(groups)} confirmed group(s) in total.")

    hidden = next((c for c in checks if c.id == "behaviour.hidden_failure_groups"), None)
    if hidden is not None:
        policy = config.hidden_groups
        lines.append(
            f"Hidden failure group policy (configured): consequence '{policy.consequence}', "
            f"assessment {'required' if policy.require_assessment else 'not required'}. "
            + {"score_only": "The result affects the score only.",
               "hold": "A confirmed group blocks promotion.",
               "critical": "A severe confirmed group is a critical failure; a lesser one "
                           "blocks promotion."}[policy.consequence])

    watch = [c.id for c in checks if c.status == CheckStatus.WARN]
    if watch and decision.decision == Decision.PROMOTE:
        lines.append("Watch: these checks warn but do not block promotion: "
                     + ", ".join(watch) + ".")

    costly = [(i, l) for i, l in mqs_point_losses(scorecard)
              if l >= MIN_COST_REPORTED][:MAX_COSTLY_CHECKS]
    if costly:
        lines.append("Weakest checks (MQS points lost): " + ", ".join(
            f"{check_id} (-{loss:.1f})" for check_id, loss in costly) + ".")

    for check in checks:
        if check.status in NO_EVIDENCE:
            lines.append(f"Evidence gap (not counted as passing): {check.id} is "
                         f"{check.status.value}. {_short(check.message)}")
        elif check.status == CheckStatus.NOT_APPLICABLE:
            lines.append(f"Not applicable: {check.id}. {_short(check.message)}")

    declared = [c.id for c in checks if c.evidence_source == EvidenceSource.DECLARED]
    if declared:
        lines.append("Evidence note: these results rely on declared (unverified) "
                     "producer metadata: " + ", ".join(declared) + ".")
    if not scorecard.mqs_sufficient_evidence:
        lines.append("Evidence note: the MQS is not reliable: "
                     f"{scorecard.insufficient_evidence_reason}.")
    elif scorecard.provisional:
        lines.append("Evidence note: the MQS is provisional; some checks lacked "
                     "evidence and were left out of the average.")
    if simulated_inputs:
        lines.append("Evidence note: this run uses SIMULATED inputs ("
                     + ", ".join(simulated_inputs) + "); results are not real "
                     "measurements.")

    actions, seen = [], set()
    actionable = (CheckStatus.FAIL, CheckStatus.WARN)
    for check in checks:
        if check.status in actionable:
            actions.append((check.id, CHECK_HINTS.get(check.id)))
    for rule in decision.triggered_rules:
        if rule.rule_id in RULE_HINTS:
            actions.append((rule.rule_id, RULE_HINTS[rule.rule_id]))
        # a must-pass rule on a check that was not measured needs inputs, not
        # a model change, so its check-specific advice would mislead
    suggestions = []
    for key, hint in actions:
        if hint and hint not in seen and decision.decision != Decision.PROMOTE:
            seen.add(hint)
            suggestions.append(f"Suggested action ({key}): {hint}")
    actions = suggestions
    lines += actions[:MAX_ACTIONS]
    if actions:
        lines.append("These suggestions are starting points, not guarantees.")
    return lines
