from pathlib import Path

import pytest
import yaml

from mqvg.catalogue import CHECKS_BY_ID
from mqvg.checks.behaviour import run_behaviour_checks
from mqvg.checks.operational import run_operational_checks
from mqvg.checks.training_validity import run_training_validity_checks
from mqvg.config import load_config
from mqvg.loader import load_candidate
from mqvg.rules import evaluate_rules
from mqvg.schemas import (
    ActionMode, CandidateBundle, CheckStatus, Decision, DecisionRecord,
    Dimension, RuleKind,
)
from mqvg.scoring import compute_scorecard

from builders import ERROR, NA, UNAVAILABLE, make_results
from conftest import CONFIGS, ROOT

TV, MB, OR = (Dimension.TRAINING_VALIDITY, Dimension.MODEL_BEHAVIOUR,
              Dimension.OPERATIONAL_READINESS)


def decide(config, **kwargs) -> DecisionRecord:
    results = make_results(config, **kwargs)
    return evaluate_rules(results, compute_scorecard(results, config), config)


def rule_ids(decision):
    return [r.rule_id for r in decision.triggered_rules]


def by_id(decision):
    return {r.rule_id: r for r in decision.triggered_rules}


# ------------------------------------------------------------------ PROMOTE
def test_clean_high_scoring_model_is_promoted(default_config):
    d = decide(default_config)
    assert d.decision == Decision.PROMOTE
    assert d.action_mode == ActionMode.RECOMMENDATION
    assert d.mqs == pytest.approx(100) and not d.overridden_by_critical
    assert rule_ids(d) == ["mqs_meets_promote_threshold"]
    assert "PROMOTE recommended" in d.summary


def test_mqs_exactly_at_the_promote_threshold_promotes(default_config):
    d = decide(default_config, base=75)              # every dimension 75 -> MQS 75
    assert d.mqs == pytest.approx(75) and d.decision == Decision.PROMOTE


def test_provisional_score_can_still_promote_but_says_so(default_config):
    # one non-required check unavailable: coverage stays high, score provisional
    d = decide(default_config, scores={"behaviour.precision": UNAVAILABLE})
    assert d.decision == Decision.PROMOTE
    assert "provisional" in d.summary


# --------------------------------------------------------------------- HOLD
def test_mqs_between_the_thresholds_holds(default_config):
    d = decide(default_config, base=60)
    assert d.decision == Decision.HOLD
    assert rule_ids(d) == ["mqs_below_promote_threshold"]
    assert "between the rejection threshold (50) and the promotion threshold (75)" \
        in d.triggered_rules[0].reason


def test_mqs_exactly_at_the_reject_threshold_holds(default_config):
    d = decide(default_config, base=50)
    assert d.mqs == pytest.approx(50) and d.decision == Decision.HOLD


def test_required_missing_evidence_holds_despite_a_perfect_score(default_config):
    d = decide(default_config,
               scores={"fairness.selection_rate_ratio": UNAVAILABLE})
    assert d.mqs == pytest.approx(100)
    assert d.decision == Decision.HOLD
    rule = by_id(d)["required_evidence_missing"]
    assert rule.kind == RuleKind.EVIDENCE
    assert rule.observed == ["fairness.selection_rate_ratio"]
    assert rule.check_ids == ["fairness.selection_rate_ratio"]


@pytest.mark.parametrize("missing", [UNAVAILABLE, ERROR])
def test_error_counts_as_missing_required_evidence(default_config, missing):
    d = decide(default_config, scores={"behaviour.f1": missing})
    assert "required_evidence_missing" in rule_ids(d)


def test_required_check_that_is_not_applicable_does_not_block(default_config):
    d = decide(default_config, scores={"fairness.selection_rate_ratio": NA})
    assert d.decision == Decision.PROMOTE


def test_missing_optional_evidence_does_not_block(default_config):
    d = decide(default_config, scores={"behaviour.precision": UNAVAILABLE,
                                       "behaviour.recall": UNAVAILABLE})
    assert d.decision == Decision.PROMOTE


def test_low_overall_coverage_holds(make_config):
    config = make_config(lambda d: d["coverage"].update(min_overall=0.99))
    d = decide(config, scores={"behaviour.precision": UNAVAILABLE})
    assert d.decision == Decision.HOLD
    assert "overall_coverage_below_minimum" in rule_ids(d)
    rule = by_id(d)["overall_coverage_below_minimum"]
    assert rule.threshold == 0.99 and rule.observed < 0.99


def test_no_mqs_holds(default_config):
    d = decide(default_config, base=UNAVAILABLE)
    assert d.mqs is None and d.decision == Decision.HOLD
    assert "mqs_unavailable" in rule_ids(d)
    assert "no MQS could be computed" in d.summary


# ---------------------------------------------------------- must-pass checks
MUST_PASS = "calibration.ece"


@pytest.mark.parametrize("value, shown", [
    (("WARN", 60), "WARN"), (("FAIL", 20), "FAIL"),
    (UNAVAILABLE, "UNAVAILABLE"), (ERROR, "ERROR")])
def test_a_must_pass_check_that_is_not_pass_blocks_promotion(default_config,
                                                             value, shown):
    d = decide(default_config, scores={MUST_PASS: value})
    assert d.decision == Decision.HOLD
    rule = by_id(d)[f"must_pass:{MUST_PASS}"]
    assert rule.kind == RuleKind.BLOCKING and rule.observed == shown
    assert rule.threshold == "PASS" and f"is {shown}" in rule.reason


def test_must_pass_that_passes_or_does_not_apply_does_not_block(default_config):
    assert decide(default_config).decision == Decision.PROMOTE
    assert decide(default_config, scores={MUST_PASS: NA}).decision == Decision.PROMOTE


def test_each_failing_must_pass_check_gets_its_own_rule(default_config):
    d = decide(default_config, scores={"calibration.ece": ("WARN", 60),
                                       "operational.latency": ("FAIL", 10)})
    assert {"must_pass:calibration.ece", "must_pass:operational.latency"} \
        <= set(rule_ids(d))


# ----------------------------------------------------------------- ROLLBACK
def test_mqs_below_the_reject_threshold_rolls_back(default_config):
    d = decide(default_config, base=40)
    assert d.decision == Decision.ROLLBACK
    assert rule_ids(d) == ["mqs_below_reject_threshold"]
    assert not d.overridden_by_critical                 # the score did it
    assert "below the rejection threshold" in d.summary


LEAKAGE = "leakage.single_feature_predictiveness"


def test_a_critical_failure_overrides_a_high_mqs(default_config):
    """The headline behaviour: a critical rule beats a good score."""
    d = decide(default_config, per_dimension={TV: 95, MB: 90, OR: 95},
               scores={LEAKAGE: ("FAIL", 10)})
    assert d.mqs > default_config.decision.promote_at      # the score alone would PROMOTE
    assert d.decision == Decision.ROLLBACK
    assert d.overridden_by_critical
    rule = by_id(d)["target_leakage"]
    assert rule.kind == RuleKind.CRITICAL and rule.effect == Decision.ROLLBACK
    assert rule.check_ids == [LEAKAGE] and rule.observed == "FAIL"
    assert "overrides the score" in d.summary


def test_critical_rule_that_agrees_with_the_score_is_not_an_override(default_config):
    d = decide(default_config, base=30, scores={LEAKAGE: ("FAIL", 0)})
    assert d.decision == Decision.ROLLBACK
    assert {"target_leakage", "mqs_below_reject_threshold"} <= set(rule_ids(d))
    assert not d.overridden_by_critical


def test_every_matching_critical_rule_is_recorded(default_config):
    d = decide(default_config, scores={
        LEAKAGE: ("FAIL", 10), "fairness.selection_rate_ratio": ("FAIL", 0),
        "behaviour.f1": ("FAIL", 5)})
    assert {"target_leakage", "severe_selection_rate_disparity",
            "performance_below_minimum"} <= set(rule_ids(d))


def test_rollback_beats_hold_but_hold_rules_are_still_reported(default_config):
    d = decide(default_config, scores={
        LEAKAGE: ("FAIL", 10), "calibration.ece": ("WARN", 60)})
    assert d.decision == Decision.ROLLBACK
    assert "must_pass:calibration.ece" in rule_ids(d)    # reported, not decisive


def test_critical_status_match_covers_error_but_not_unavailable(default_config):
    loadable = "operational.model_loadable"
    assert "model_not_loadable" in rule_ids(
        decide(default_config, scores={loadable: ("FAIL", 0)}))
    assert "model_not_loadable" in rule_ids(
        decide(default_config, scores={loadable: ERROR}))
    assert "model_not_loadable" not in rule_ids(
        decide(default_config, scores={loadable: UNAVAILABLE}))


def test_a_check_with_no_result_is_treated_as_unavailable(make_config):
    def tweak(d):
        d["critical_rules"] = [{"id": "gone", "name": "Evidence gone",
                                "condition": {"check": "behaviour.recall",
                                              "status_in": ["UNAVAILABLE"]}}]
    config = make_config(tweak)
    results = [r for r in make_results(config) if r.id != "behaviour.recall"]
    d = evaluate_rules(results, compute_scorecard(results, config), config)
    assert "gone" in rule_ids(d) and d.decision == Decision.ROLLBACK


# --------------------------------------------------- numeric critical rules
@pytest.fixture
def value_rules(make_config):
    def tweak(d):
        d["critical_rules"] = [
            {"id": "f1_too_low", "name": "F1 too low",
             "condition": {"check": "behaviour.f1", "value_below": 0.3}},
            {"id": "latency_too_high", "name": "Latency too high",
             "condition": {"check": "operational.latency", "value_above": 500}}]
    return make_config(tweak)


def test_numeric_critical_rules_compare_the_measured_value(value_rules):
    d = decide(value_rules, scores={"behaviour.f1": {"raw": 0.2},
                                    "operational.latency": {"raw": 400}})
    assert "f1_too_low" in rule_ids(d) and "latency_too_high" not in rule_ids(d)
    rule = by_id(d)["f1_too_low"]
    assert rule.observed == 0.2 and rule.threshold == {"value_below": 0.3}
    assert d.decision == Decision.ROLLBACK


def test_numeric_rule_boundary_is_strict(value_rules):
    d = decide(value_rules, scores={"behaviour.f1": {"raw": 0.3}})
    assert "f1_too_low" not in rule_ids(d)


def test_numeric_rule_cannot_match_without_a_measured_value(value_rules):
    d = decide(value_rules, scores={"behaviour.f1": UNAVAILABLE})
    assert "f1_too_low" not in rule_ids(d)               # no value, no match
    d = decide(value_rules, scores={"behaviour.f1": {"raw": None}})
    assert "f1_too_low" not in rule_ids(d)
    d = decide(value_rules, scores={"behaviour.f1": {"raw": True}})
    assert "f1_too_low" not in rule_ids(d)               # booleans are not values


# ------------------------------------------------ the score vs the lower bound
def test_decision_uses_the_policy_mqs_not_the_lower_bound(default_config):
    results = make_results(default_config, per_dimension={MB: UNAVAILABLE})
    card = compute_scorecard(results, default_config)
    assert card.mqs > 75 > card.mqs_lower_bound          # optimistic vs worst case
    d = evaluate_rules(results, card, default_config)
    assert d.decision == Decision.HOLD                   # missing evidence, not reject
    assert "mqs_below_reject_threshold" not in rule_ids(d)
    assert "at worst" in d.summary


def test_penalize_policy_lets_missing_evidence_drive_a_rollback(make_config):
    config = make_config(lambda d: d.update(missing_policy="penalize"))
    d = decide(config, per_dimension={MB: UNAVAILABLE, OR: UNAVAILABLE})
    assert d.decision == Decision.ROLLBACK
    assert "mqs_below_reject_threshold" in rule_ids(d)


# -------------------------------------------------------------- profiles
def test_strict_profile_adds_critical_rules():
    strict = load_config(CONFIGS / "strict.yaml")
    d = decide(strict, scores={"calibration.ece": ("FAIL", 0)})
    assert d.decision == Decision.ROLLBACK
    assert "severe_miscalibration" in rule_ids(d)
    default = load_config(CONFIGS / "default.yaml")
    assert decide(default, scores={"calibration.ece": ("FAIL", 0)}).decision \
        == Decision.HOLD                                  # default only blocks


def test_same_inputs_give_the_same_decision(default_config):
    kwargs = dict(scores={LEAKAGE: ("FAIL", 10), "calibration.ece": ("WARN", 60)})
    assert decide(default_config, **kwargs) == decide(default_config, **kwargs)


def test_decision_records_always_satisfy_the_schema(default_config):
    cases = [{}, {"base": 60}, {"base": 30}, {"base": UNAVAILABLE},
             {"scores": {LEAKAGE: ("FAIL", 0)}},
             {"scores": {"calibration.ece": ("WARN", 50)}}]
    for kwargs in cases:
        d = decide(default_config, **kwargs)
        DecisionRecord.model_validate(d.model_dump())
        assert d.summary


def test_duplicate_results_are_rejected(default_config):
    results = make_results(default_config)
    card = compute_scorecard(results, default_config)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_rules(results + results[:1], card, default_config)


# --------------------------------- real fixtures against the manifest
def _decisions(config, root=ROOT):
    manifest = yaml.safe_load(
        (root / "fixtures" / "manifest.yaml").read_text(encoding="utf-8"))["fixtures"]
    out = {}
    for name, entry in manifest.items():
        paths = {k: root / entry[k] for k in ("model", "metadata", "eval_data")}
        if not all(p.exists() for p in paths.values()):
            pytest.skip("fixtures not generated (run scripts/make_fixtures.py)")
        candidate = load_candidate(CandidateBundle(
            model_path=paths["model"], eval_data_path=paths["eval_data"],
            metadata_path=paths["metadata"],
            train_row_hashes_path=root / "data" / "train_row_hashes.txt"))
        results = (run_training_validity_checks(candidate, config)
                   + run_behaviour_checks(candidate, config)
                   + run_operational_checks(candidate, config))
        card = compute_scorecard(results, config)
        out[name] = (entry, results, card, evaluate_rules(results, card, config))
    return out


@pytest.fixture(scope="module")
def scenarios():
    return _decisions(load_config(CONFIGS / "default.yaml"))


def test_every_fixture_gets_the_decision_the_manifest_expects(scenarios):
    """The manifest was written before the rules existed, from knowing which
    defect each fixture contains. Thresholds were NOT tuned to fit it."""
    for name, (entry, _, _, decision) in scenarios.items():
        expected = entry["expected_decision"]
        allowed = expected if isinstance(expected, list) else [expected]
        assert decision.decision.value in allowed, (
            f"{name}: got {decision.decision.value}, manifest says {allowed}; "
            f"rules: {rule_ids(decision)}")


def test_the_three_demonstration_outcomes_all_occur(scenarios):
    outcomes = {d.decision for *_, d in scenarios.values()}
    assert outcomes == {Decision.PROMOTE, Decision.HOLD, Decision.ROLLBACK}


def test_leaky_and_biased_models_are_critical_overrides_of_a_passing_score(scenarios):
    threshold = load_config(CONFIGS / "default.yaml").decision.promote_at
    for name, rule in (("leaky", "target_leakage"),
                       ("biased", "severe_selection_rate_disparity")):
        _, _, card, decision = scenarios[name]
        assert card.mqs >= threshold                     # the score alone would PROMOTE
        assert decision.decision == Decision.ROLLBACK
        assert decision.overridden_by_critical and rule in rule_ids(decision)


def test_corrupt_model_is_rejected_by_a_critical_rule_despite_its_mqs(scenarios):
    _, _, card, decision = scenarios["corrupt"]
    assert card.mqs > 90 and card.mqs_lower_bound < 40   # optimistic number
    assert decision.decision == Decision.ROLLBACK
    assert "model_not_loadable" in rule_ids(decision)


def test_missing_fairness_evidence_holds_a_model_that_looks_better(scenarios):
    _, _, missing_card, decision = scenarios["good_no_sensitive"]
    _, _, good_card, _ = scenarios["good"]
    assert missing_card.mqs > good_card.mqs              # the optimism of exclusion
    assert decision.decision == Decision.HOLD            # ...which the rules contain
    assert "required_evidence_missing" in rule_ids(decision)


def test_a_slow_model_holds_whenever_its_latency_check_does_not_pass(scenarios):
    _, results, _, decision = scenarios["slow"]
    latency = next(r for r in results if r.id == "operational.latency")
    blocked = latency.status != CheckStatus.PASS
    assert (decision.decision == Decision.HOLD) == blocked
    size = next(r for r in results if r.id == "operational.artifact_size")
    assert size.status == CheckStatus.WARN               # 113 MB, deterministic


def test_a_tightened_latency_limit_deterministically_holds_the_slow_model(
        make_config):
    config = make_config(lambda d: d["checks"]["operational.latency"].update(
        thresholds={"target": 1, "pass_at": 2, "fail_at": 1000, "floor": 5000}))
    _, results, _, decision = _decisions(config)["slow"]
    assert decision.decision == Decision.HOLD
    assert "must_pass:operational.latency" in rule_ids(decision)


def test_strict_profile_is_unattainable_for_the_baseline_model():
    """Finding: strict fairness limits reject even the 'good' baseline, whose
    selection-rate ratio (0.31) is below the strict fail_at (0.50)."""
    strict = load_config(CONFIGS / "strict.yaml")
    _, _, _, decision = _decisions(strict)["good"]
    assert decision.decision == Decision.ROLLBACK
    assert "severe_selection_rate_disparity" in rule_ids(decision)


# ------------------------------------------------ insufficient evidence (Step A)
def test_an_unscored_dimension_holds_even_when_overall_coverage_is_enough(make_config):
    config = make_config(lambda d: d["coverage"].update(min_overall=0.3))
    d = decide(config, per_dimension={MB: UNAVAILABLE})
    assert d.decision == Decision.HOLD
    rule = by_id(d)["mqs_insufficient_evidence"]
    assert rule.kind == RuleKind.EVIDENCE and rule.observed == ["model_behaviour"]
    assert "not reliable" in d.summary


def test_the_insufficient_evidence_rule_is_not_duplicated(default_config):
    d = decide(default_config, per_dimension={MB: UNAVAILABLE})
    assert "overall_coverage_below_minimum" in rule_ids(d)
    assert "mqs_insufficient_evidence" not in rule_ids(d)


def test_a_reliable_mqs_summary_has_no_warning(default_config):
    assert "not reliable" not in decide(default_config).summary
