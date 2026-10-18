"""How a Hidden Failure Group result affects the score and the decision."""
import pytest

from mqvg.config import ConfigError, load_config
from mqvg.explain import explain_failures
from mqvg.rules import evaluate_rules
from mqvg.schemas import (
    CheckStatus, CheckTreatment, Decision, Dimension, RuleKind,
)
from mqvg.scoring import compute_scorecard

from builders import ERROR, NA, UNAVAILABLE, make_results
from conftest import CONFIGS

CHECK = "behaviour.hidden_failure_groups"
MB = Dimension.MODEL_BEHAVIOUR

PASS = {"status": "PASS", "score": 100, "raw": 0.0}
WARN = {"status": "WARN", "score": 70, "raw": 0.12}
FAIL = {"status": "FAIL", "score": 10, "raw": 0.45}


def policy(make_config, consequence="hold", require_assessment=False, weight=None, **hidden):
    def tweak(d):
        d["checks"][CHECK]["enabled"] = True
        if weight is not None:
            d["checks"][CHECK]["weight"] = weight
        d["hidden_groups"].update(consequence=consequence,
                                  require_assessment=require_assessment, **hidden)
    return make_config(tweak)


def decide(config, hidden, **kwargs):
    results = make_results(config, scores={CHECK: hidden}, **kwargs)
    card = compute_scorecard(results, config)
    return results, card, evaluate_rules(results, card, config)


def ids(decision):
    return [r.rule_id for r in decision.triggered_rules]


# ------------------------------------------------ the default never changes
def test_the_default_profile_neither_runs_nor_polices_the_check(default_config):
    assert not default_config.checks[CHECK].enabled
    assert default_config.hidden_groups.consequence == "score_only"
    results = make_results(default_config)
    assert CHECK not in {r.id for r in results}
    decision = evaluate_rules(results, compute_scorecard(results, default_config), default_config)
    assert not [i for i in ids(decision) if "hidden" in i]


# --------------------------------------------------------------- score_only
def test_score_only_lowers_the_score_but_never_the_decision(make_config):
    config = policy(make_config, "score_only")
    _, clean_card, clean = decide(config, PASS, per_dimension={MB: 85})
    _, card, decision = decide(config, FAIL, per_dimension={MB: 85})
    assert card.mqs < clean_card.mqs                                # scoring is affected
    assert decision.decision == clean.decision == Decision.PROMOTE   # the decision is not
    assert not [i for i in ids(decision) if "hidden" in i]


# -------------------------------------------------------------------- hold
@pytest.mark.parametrize("hidden", [WARN, FAIL])
def test_hold_blocks_promotion_for_a_confirmed_group(make_config, hidden):
    _, _, decision = decide(policy(make_config, "hold"), hidden)
    assert decision.decision == Decision.HOLD
    rule = next(r for r in decision.triggered_rules if r.rule_id == "hidden_failure_groups_hold")
    assert rule.kind == RuleKind.BLOCKING and rule.effect == Decision.HOLD
    assert "hidden_groups.consequence is 'hold'" in rule.reason
    assert rule.check_ids == [CHECK] and rule.observed == hidden["raw"]


def test_hold_does_nothing_when_no_group_is_confirmed(make_config):
    _, _, decision = decide(policy(make_config, "hold"), PASS)
    assert decision.decision == Decision.PROMOTE
    assert not [i for i in ids(decision) if "hidden" in i]


# ---------------------------------------------------------------- critical
def test_critical_makes_only_a_severe_group_a_rollback(make_config):
    config = policy(make_config, "critical")
    _, _, warn = decide(config, WARN)
    assert warn.decision == Decision.HOLD                          # a lesser group only holds
    assert "hidden_failure_groups_critical" not in ids(warn)
    _, card, severe = decide(config, FAIL)
    assert severe.decision == Decision.ROLLBACK
    assert {"hidden_failure_groups_hold", "hidden_failure_groups_critical"} <= set(ids(severe))
    critical = next(r for r in severe.triggered_rules if r.rule_id == "hidden_failure_groups_critical")
    assert critical.kind == RuleKind.CRITICAL and critical.effect == Decision.ROLLBACK
    assert "association" in critical.reason and "not a proven cause" in critical.reason
    assert critical.threshold == {"fail_at": 0.25}


def test_a_critical_hidden_group_overrides_a_good_mqs_and_says_so(make_config):
    config = policy(make_config, "critical")
    _, card, decision = decide(config, FAIL, per_dimension={MB: 90})
    assert card.mqs > config.decision.promote_at
    assert decision.decision == Decision.ROLLBACK and decision.overridden_by_critical
    assert "overrides the score" in decision.summary


@pytest.mark.parametrize("consequence", ["score_only", "hold"])
def test_a_detected_subgroup_alone_never_forces_a_rollback(make_config, consequence):
    """Unless the policy is explicitly 'critical', the worst result cannot roll back."""
    _, _, decision = decide(policy(make_config, consequence), FAIL, per_dimension={MB: 90})
    assert decision.decision != Decision.ROLLBACK
    assert not decision.overridden_by_critical


def test_the_severity_of_a_rollback_comes_from_the_checks_own_threshold(make_config):
    # the status (and so the severity) is set by the configured thresholds
    config = policy(make_config, "critical")
    assert config.checks[CHECK].thresholds.fail_at == 0.25
    loose = make_config(lambda d: (d["checks"][CHECK].update(
        enabled=True, thresholds={"target": 0.0, "pass_at": 0.0, "fail_at": 0.9, "floor": 0.95}),
        d["hidden_groups"].update(consequence="critical")))
    assert loose.checks[CHECK].thresholds.fail_at == 0.9


# -------------------------------------------------- unavailable is not a pass
@pytest.mark.parametrize("missing", [UNAVAILABLE, ERROR])
def test_require_assessment_blocks_when_the_check_could_not_run(make_config, missing):
    config = policy(make_config, "hold", require_assessment=True)
    _, _, decision = decide(config, missing)
    assert decision.decision == Decision.HOLD
    rule = next(r for r in decision.triggered_rules
                if r.rule_id == "hidden_failure_groups_not_assessed")
    assert rule.kind == RuleKind.EVIDENCE and "not a pass" in rule.reason
    assert rule.observed == missing


@pytest.mark.parametrize("consequence", ["score_only", "hold", "critical"])
def test_unavailable_never_counts_as_a_pass_under_any_policy(make_config, consequence):
    config = policy(make_config, consequence, require_assessment=False)
    _, card, decision = decide(config, UNAVAILABLE, per_dimension={MB: 70})
    contribution = next(c for c in card.dimensions[MB].contributions if c.check_id == CHECK)
    assert contribution.treatment == CheckTreatment.EXCLUDED_UNAVAILABLE
    assert contribution.points == 0 and contribution.score is None
    assert card.dimensions[MB].provisional and card.dimensions[MB].coverage < 1
    assert card.overall_coverage < 1
    assert card.mqs_lower_bound < card.mqs                          # the worst case is visible
    assert not [i for i in ids(decision) if "hidden" in i]          # no consequence is invented


def test_unavailable_scores_below_a_pass_not_as_one(make_config):
    config = policy(make_config, "score_only")
    _, with_pass, _ = decide(config, PASS, per_dimension={MB: 70})
    _, unavailable, _ = decide(config, UNAVAILABLE, per_dimension={MB: 70})
    assert unavailable.dimensions[MB].score < with_pass.dimensions[MB].score


def test_not_assessed_without_require_assessment_does_not_block(make_config):
    config = policy(make_config, "hold", require_assessment=False)
    _, _, decision = decide(config, UNAVAILABLE)
    assert decision.decision == Decision.PROMOTE
    assert not [i for i in ids(decision) if "hidden" in i]


def test_not_applicable_triggers_nothing(make_config):
    _, _, decision = decide(policy(make_config, "critical", require_assessment=True), NA)
    assert not [i for i in ids(decision) if "hidden" in i]


# ----------------------------------------------------- weight and thresholds
def test_the_weight_is_configurable_and_scales_the_effect_on_the_mqs(make_config):
    lighter = decide(policy(make_config, "score_only", weight=0.5), FAIL)[1]
    default = decide(policy(make_config, "score_only"), FAIL)[1]
    heavier = decide(policy(make_config, "score_only", weight=5.0), FAIL)[1]
    assert heavier.mqs < default.mqs < lighter.mqs


def test_the_statistical_settings_are_configurable(make_config):
    config = policy(make_config, "hold", alpha=0.01, min_effect=0.2, folds=3,
                    kmeans_clusters=6, methods=["tree"], report_groups=5)
    h = config.hidden_groups
    assert (h.alpha, h.min_effect, h.folds, h.kmeans_clusters, h.methods, h.report_groups) == \
        (0.01, 0.2, 3, 6, ["tree"], 5)


# ------------------------------------------------------------ explanation
@pytest.mark.parametrize("consequence, phrase", [
    ("score_only", "affects the score only"), ("hold", "blocks promotion"),
    ("critical", "critical failure")])
def test_the_explanation_names_the_configured_policy(make_config, consequence, phrase):
    config = policy(make_config, consequence)
    results, card, decision = decide(config, FAIL)
    lines = explain_failures(results, card, decision, config)
    line = next(l for l in lines if l.startswith("Hidden failure group policy (configured)"))
    assert f"consequence '{consequence}'" in line and phrase in line


def test_the_explanation_has_no_policy_line_when_the_check_is_off(default_config):
    results = make_results(default_config)
    card = compute_scorecard(results, default_config)
    lines = explain_failures(results, card, evaluate_rules(results, card, default_config),
                             default_config)
    assert not any("Hidden failure group policy" in l for l in lines)


# ------------------------------------------------------ config and profiles
def test_a_policy_for_a_disabled_check_is_rejected(make_config):
    for hidden in ({"consequence": "hold"}, {"consequence": "critical"},
                   {"require_assessment": True}):
        with pytest.raises(ConfigError, match="disabled, so they would do nothing"):
            make_config(lambda d: d["hidden_groups"].update(**hidden))


def test_an_unknown_consequence_is_rejected(make_config):
    with pytest.raises(ConfigError, match="consequence"):
        policy(make_config, "rollback_always")


def test_the_shipped_profiles_set_the_documented_policies():
    expected = {"default": ("score_only", False, False),
                "hidden_groups_score_only": ("score_only", False, True),
                "hidden_groups": ("hold", True, True),
                "hidden_groups_critical": ("critical", True, True)}
    for name, (consequence, assessment, enabled) in expected.items():
        config = load_config(CONFIGS / f"{name}.yaml")
        h = config.hidden_groups
        assert (h.consequence, h.require_assessment) == (consequence, assessment), name
        assert config.checks[CHECK].enabled is enabled, name
        assert CHECK not in config.decision.must_pass_checks        # the policy replaces it


def test_the_variant_profiles_differ_from_hold_only_in_the_policy():
    hold = load_config(CONFIGS / "hidden_groups.yaml")
    for name in ("hidden_groups_score_only", "hidden_groups_critical"):
        other = load_config(CONFIGS / f"{name}.yaml")
        assert other.checks == hold.checks and other.decision == hold.decision
        assert other.critical_rules == hold.critical_rules and other.weights == hold.weights
