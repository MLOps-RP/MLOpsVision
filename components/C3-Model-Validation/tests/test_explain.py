import numpy as np
import pytest

from mqvg.catalogue import CATALOGUE
from mqvg.explain import (
    CHECK_HINTS, MAX_ACTIONS, RULE_HINTS, explain_failures, mqs_point_losses,
)
from mqvg.rules import evaluate_rules
from mqvg.schemas import CheckStatus, Dimension, EvidenceSource
from mqvg.scoring import compute_scorecard

from builders import ERROR, NA, UNAVAILABLE, enabled_ids, make_results

TV, MB, OR = (Dimension.TRAINING_VALIDITY, Dimension.MODEL_BEHAVIOUR,
              Dimension.OPERATIONAL_READINESS)
LEAKAGE = "leakage.single_feature_predictiveness"
MB_SCORES = {"behaviour.f1": 80, "behaviour.roc_auc": 90, "behaviour.precision": 70,
             "behaviour.recall": 60, "behaviour.subgroup_f1_gap": 50,
             "fairness.selection_rate_ratio": 40, "fairness.equalized_odds": 30,
             "calibration.ece": 100}


def explain(config, results=None, simulated=(), **kwargs):
    results = results if results is not None else make_results(config, **kwargs)
    card = compute_scorecard(results, config)
    decision = evaluate_rules(results, card, config)
    return explain_failures(results, card, decision, config, simulated), card, decision


def starting(lines, label):
    return [line for line in lines if line.startswith(label)]


# ---------------------------------------------------------- the headline
def test_first_line_is_the_decision_summary(default_config):
    lines, _, decision = explain(default_config)
    assert lines[0] == f"Decision: {decision.summary}"


def test_a_clean_promotion_has_no_failure_lines(default_config):
    lines, _, decision = explain(default_config)
    assert decision.decision.value == "PROMOTE"
    assert not starting(lines, "Why")
    assert not starting(lines, "Suggested action")
    assert not starting(lines, "Weakest checks")          # nothing lost any points


def test_warnings_that_do_not_block_are_listed_for_a_promotion(default_config):
    lines, _, decision = explain(
        default_config, scores={"fairness.selection_rate_ratio": ("WARN", 60)})
    assert decision.decision.value == "PROMOTE"
    assert starting(lines, "Watch")[0].endswith("fairness.selection_rate_ratio.")


# --------------------------------------------------------- why and override
def test_every_non_promote_rule_gets_a_why_line(default_config):
    lines, _, decision = explain(default_config, scores={
        "calibration.ece": ("WARN", 60), "operational.latency": ("FAIL", 10)})
    reasons = [r.reason for r in decision.triggered_rules if r.effect.value != "PROMOTE"]
    assert len(starting(lines, "Why (")) == len(reasons) >= 2
    for reason in reasons:
        assert f"Why (blocking): {reason}" in lines


def test_a_critical_override_is_explained_with_the_score_and_threshold(default_config):
    lines, card, decision = explain(
        default_config, scores={LEAKAGE: ("FAIL", 10)})
    assert decision.overridden_by_critical
    override = starting(lines, "Override")[0]
    assert "Target leakage detected" in override
    assert f"MQS of {card.mqs:.1f} is above the promote threshold of 75" in override


def test_rollback_driven_by_the_score_has_no_override_line(default_config):
    lines, _, decision = explain(default_config, base=30)
    assert decision.decision.value == "ROLLBACK" and not decision.overridden_by_critical
    assert not starting(lines, "Override")


# ---------------------------------------------------------- MQS point costs
def test_point_losses_match_hand_calculation(default_config):
    results = make_results(default_config, scores=MB_SCORES, per_dimension={TV: 100, OR: 100})
    card = compute_scorecard(results, default_config)
    losses = dict(mqs_point_losses(card))
    # a check's loss = (weight*100 - weight*score) / sum_of_weights * dimension weight
    assert losses["fairness.selection_rate_ratio"] == pytest.approx(2 * 60 / 11.5 * 0.5)
    assert losses["fairness.equalized_odds"] == pytest.approx(1 * 70 / 11.5 * 0.5)
    assert losses["behaviour.f1"] == pytest.approx(2 * 20 / 11.5 * 0.5)
    assert losses["calibration.ece"] == pytest.approx(0.0)
    assert losses["operational.latency"] == pytest.approx(0.0)
    ranked = [check for check, _ in mqs_point_losses(card)]
    assert ranked[:3] == ["fairness.selection_rate_ratio", "fairness.equalized_odds",
                          "behaviour.subgroup_f1_gap"]


def test_the_weakest_checks_line_lists_at_most_five_in_order(default_config):
    lines, _, _ = explain(default_config, scores=MB_SCORES)
    weakest = starting(lines, "Weakest checks")[0]
    listed = weakest.split(": ", 1)[1].rstrip(".").split("), ")
    assert len(listed) == 5
    assert listed[0].startswith("fairness.selection_rate_ratio (-")


def test_losses_account_for_every_lost_point_when_nothing_is_missing(default_config):
    rng = np.random.default_rng(0)
    for _ in range(20):
        scores = {i: float(rng.uniform(0, 100)) for i in enabled_ids(default_config)}
        results = make_results(default_config, scores=scores)
        card = compute_scorecard(results, default_config)
        assert sum(l for _, l in mqs_point_losses(card)) + card.mqs == pytest.approx(100)


def test_unscored_dimensions_do_not_appear_in_the_losses(default_config):
    card = compute_scorecard(make_results(
        default_config, per_dimension={MB: UNAVAILABLE}), default_config)
    assert not any(check in dict(mqs_point_losses(card))
                   for check in ("behaviour.f1", "fairness.equalized_odds"))


# ------------------------------------------------------------ evidence
def test_evidence_gaps_are_listed_as_not_counted_as_passing(default_config):
    lines, _, _ = explain(default_config, scores={
        "fairness.selection_rate_ratio": UNAVAILABLE, "operational.memory": ERROR})
    gaps = starting(lines, "Evidence gap")
    assert len(gaps) == 2
    assert "not counted as passing" in gaps[0]
    assert any("fairness.selection_rate_ratio is UNAVAILABLE" in g for g in gaps)
    assert any("operational.memory is ERROR" in g for g in gaps)


def test_not_applicable_checks_are_listed_separately(default_config):
    lines, _, _ = explain(default_config, scores={"calibration.ece": NA})
    assert starting(lines, "Not applicable: calibration.ece")
    assert not starting(lines, "Evidence gap")


def test_declared_evidence_is_flagged(default_config):
    results = make_results(default_config)
    results = [r.model_copy(update={"evidence_source": EvidenceSource.DECLARED})
               if r.id in ("training_validity.overfit_gap", "operational.dependency_match")
               else r for r in results]
    lines, _, _ = explain(default_config, results=results)
    note = starting(lines, "Evidence note: these results rely on declared")[0]
    assert "training_validity.overfit_gap" in note and "operational.dependency_match" in note


def test_an_unreliable_mqs_is_flagged_with_its_reason(default_config):
    lines, card, _ = explain(default_config, per_dimension={MB: UNAVAILABLE})
    assert not card.mqs_sufficient_evidence
    note = starting(lines, "Evidence note: the MQS is not reliable")[0]
    assert card.insufficient_evidence_reason in note


def test_a_provisional_mqs_is_flagged(default_config):
    lines, card, _ = explain(default_config, scores={"behaviour.precision": UNAVAILABLE})
    assert card.mqs_sufficient_evidence and card.provisional
    assert starting(lines, "Evidence note: the MQS is provisional")


def test_simulated_inputs_are_declared(default_config):
    lines, _, _ = explain(default_config, simulated=["metadata", "eval data"])
    assert "SIMULATED inputs (metadata, eval data)" in starting(
        lines, "Evidence note: this run uses SIMULATED")[0]
    assert not starting(explain(default_config)[0], "Evidence note: this run uses SIMULATED")


# ------------------------------------------------------------ suggestions
def test_failing_checks_get_a_suggested_action(default_config):
    lines, _, _ = explain(default_config, scores={
        "calibration.ece": ("FAIL", 10), LEAKAGE: ("FAIL", 10)})
    actions = starting(lines, "Suggested action")
    assert any("calibration.ece" in a and "Recalibrate" in a for a in actions)
    assert any(LEAKAGE in a and "remove it and retrain" in a for a in actions)
    assert lines[-1] == "These suggestions are starting points, not guarantees."


def test_unavailable_checks_do_not_get_misleading_check_advice(default_config):
    lines, _, _ = explain(default_config, scores={"operational.latency": UNAVAILABLE})
    actions = starting(lines, "Suggested action")
    assert not any("operational.latency" in a for a in actions)
    assert not any("Reduce model size" in a for a in actions)


def test_missing_required_evidence_suggests_supplying_inputs(default_config):
    lines, _, _ = explain(default_config,
                          scores={"fairness.selection_rate_ratio": UNAVAILABLE})
    assert any("(required_evidence_missing): Supply the missing inputs" in a
               for a in starting(lines, "Suggested action"))


def test_suggestions_are_capped(default_config):
    every_check_fails = {i: ("FAIL", 5) for i in enabled_ids(default_config)}
    lines, _, _ = explain(default_config, scores=every_check_fails)
    assert len(starting(lines, "Suggested action")) <= MAX_ACTIONS


def test_no_suggestions_for_a_promotion_even_with_warnings(default_config):
    lines, _, decision = explain(
        default_config, scores={"behaviour.subgroup_f1_gap": ("WARN", 60)})
    assert decision.decision.value == "PROMOTE"
    assert not starting(lines, "Suggested action")


# --------------------------------------------------------------- hygiene
def test_every_catalogue_check_has_a_hint():
    assert {s.id for s in CATALOGUE} == set(CHECK_HINTS)


def test_hints_do_not_reference_unknown_rules():
    assert set(RULE_HINTS) <= {
        "required_evidence_missing", "overall_coverage_below_minimum",
        "mqs_insufficient_evidence", "mqs_unavailable",
        "mqs_below_reject_threshold", "mqs_below_promote_threshold"}


def test_explanations_are_deterministic(default_config):
    kwargs = dict(scores={LEAKAGE: ("FAIL", 10), "calibration.ece": ("WARN", 60)})
    assert explain(default_config, **kwargs)[0] == explain(default_config, **kwargs)[0]


def test_the_pipeline_attaches_the_explanation(tiny_bundle, default_config):
    from mqvg.pipeline import run_validation
    result = run_validation(tiny_bundle, default_config)
    assert result.explanations
    assert result.explanations[0] == f"Decision: {result.decision.summary}"
