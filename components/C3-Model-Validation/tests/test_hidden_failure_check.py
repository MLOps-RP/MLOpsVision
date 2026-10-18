"""The behaviour.hidden_failure_groups check inside the whole gate."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from mqvg import synthetic as syn
from mqvg.catalogue import CHECKS_BY_ID
from mqvg.checks.behaviour import BEHAVIOUR_CHECKS, run_behaviour_checks
from mqvg.config import load_config
from mqvg.explain import CHECK_HINTS, explain_failures
from mqvg.loader import LoadedCandidate
from mqvg.pipeline import run_validation
from mqvg.reporting import render_markdown
from mqvg.rules import evaluate_rules
from mqvg.schemas import (
    CandidateBundle, CheckStatus, Decision, Dimension, EvidenceSource,
    TrainingMetadata,
)
from mqvg.scoring import compute_scorecard

import joblib

from conftest import CONFIGS, FIXTURES, ROOT

CHECK = "behaviour.hidden_failure_groups"


@pytest.fixture(scope="module")
def enabled_config():
    return load_config(CONFIGS / "hidden_groups.yaml")


@pytest.fixture(scope="module")
def default_cfg():
    return load_config(CONFIGS / "default.yaml")


def candidate(pattern, tmp_path_factory=None, n_eval=3000, seed=7, fit=syn.fit_logistic,
              mode="misspec", sensitive=True):
    train, evaluation, region = syn.make_pair(5000, n_eval, seed, pattern, mode)
    model = fit(train)
    meta = TrainingMetadata(
        target_column=syn.TARGET, positive_label=syn.POSITIVE,
        feature_columns=syn.FEATURES,
        sensitive_columns=[syn.SENSITIVE] if sensitive else None)
    from mqvg.schemas import CandidateBundle as Bundle
    from pathlib import Path
    bundle = Bundle(model_path=Path("m.joblib"), eval_data_path=Path("e.csv"))
    return LoadedCandidate(bundle=bundle, model=model, eval_df=evaluation, metadata=meta), region


def result(cand, config):
    return {r.id: r for r in run_behaviour_checks(cand, config)}[CHECK]


# ------------------------------------------------------------ integration
def test_the_check_is_in_the_catalogue_and_registered():
    spec = CHECKS_BY_ID[CHECK]
    assert spec.dimension == Dimension.MODEL_BEHAVIOUR and spec.direction == "lower_is_better"
    assert CHECK in BEHAVIOUR_CHECKS and CHECK in CHECK_HINTS


def test_it_is_off_in_the_default_profile_so_existing_scores_are_unchanged(default_cfg):
    cand, _ = candidate("two_feature")
    ids = {r.id for r in run_behaviour_checks(cand, default_cfg)}
    assert CHECK not in ids
    assert not default_cfg.checks[CHECK].enabled


def test_the_hidden_groups_profile_enables_it_and_changes_nothing_else(default_cfg,
                                                                        enabled_config):
    assert enabled_config.checks[CHECK].enabled
    for check_id, check in default_cfg.checks.items():
        if check_id != CHECK:
            assert enabled_config.checks[check_id] == check, check_id
    assert enabled_config.weights == default_cfg.weights
    assert enabled_config.decision == default_cfg.decision      # incl. must_pass_checks
    assert enabled_config.critical_rules == default_cfg.critical_rules
    # the consequence is the explicit policy, not a hidden edit of the default lists
    assert enabled_config.hidden_groups.consequence == "hold"
    assert enabled_config.hidden_groups.require_assessment
    assert default_cfg.hidden_groups.consequence == "score_only"
    assert not default_cfg.hidden_groups.require_assessment


def test_a_config_without_the_new_section_still_loads(tmp_path):
    import yaml
    data = yaml.safe_load((CONFIGS / "default.yaml").read_text(encoding="utf-8"))
    del data["hidden_groups"]
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_config(path).hidden_groups.alpha == 0.05      # defaults apply


# -------------------------------------------------------------- the result
def test_a_planted_group_gives_a_scored_warn_or_fail_with_full_details(enabled_config):
    cand, region = candidate("two_feature")
    r = result(cand, enabled_config)
    assert r.status in (CheckStatus.WARN, CheckStatus.FAIL) and r.score < 100
    assert r.evidence_source == EvidenceSource.MEASURED and r.raw_value >= 0.10
    d = r.details
    assert d["groups_confirmed"] >= 1 and d["groups_tested"] >= d["groups_confirmed"]
    top = d["groups"][0]
    assert top["confirmed"] and "segment = d" in top["rule"] + " ".join(
        g["rule"] for g in d["groups"])
    for field in ("method", "n_rows", "share", "observed_error_rate",
                  "expected_error_rate", "effect", "effect_ci", "p_value",
                  "adjusted_p_value", "confirmed"):
        assert field in top
    assert top["effect_ci"][0] < top["effect"] < top["effect_ci"][1]
    assert top["adjusted_p_value"] <= 0.05
    json.dumps(d)                                             # plain JSON types only
    assert "rows" in r.message and "held-out" in r.message and top["rule"] in r.message


def test_no_planted_group_gives_a_pass(enabled_config):
    cand, _ = candidate(None)
    r = result(cand, enabled_config)
    assert r.status == CheckStatus.PASS and r.raw_value == 0.0 and r.score == 100
    assert r.details["groups_confirmed"] == 0 and r.details["groups_tested"] > 0
    assert "does not prove none exists" in r.message


def test_a_flexible_model_failing_on_a_shifted_subgroup_is_found(enabled_config):
    cand, region = candidate("two_feature", mode="shift", fit=syn.fit_gradient_boosting)
    r = result(cand, enabled_config)
    assert r.status in (CheckStatus.WARN, CheckStatus.FAIL)


def test_the_target_is_never_searched(enabled_config):
    cand, _ = candidate("two_feature")
    cand.metadata = cand.metadata.model_copy(
        update={"feature_columns": syn.FEATURES + [syn.TARGET]})
    cand.model = None            # force the metadata path for feature_columns
    cand.model = syn.fit_logistic(syn.make_pair(2000, 100, 1, None)[0])
    r = result(cand, enabled_config)
    assert syn.TARGET not in r.details["features_used"]


# ------------------------------------------------------ privacy / features
def test_it_uses_the_models_features_and_not_other_columns_by_default(enabled_config):
    cand, _ = candidate("two_feature")
    r = result(cand, enabled_config)
    assert set(r.details["features_used"]) == set(syn.FEATURES)
    assert syn.SENSITIVE not in r.details["features_used"]    # the model did not use it


def test_features_can_be_excluded(enabled_config):
    config = enabled_config.model_copy(deep=True)
    config.hidden_groups.exclude_features = ["tenure_years"]
    cand, _ = candidate("two_feature")
    r = result(cand, config)
    assert "tenure_years" not in r.details["features_used"]


def test_extra_columns_can_be_added_and_missing_ones_are_reported(enabled_config):
    config = enabled_config.model_copy(deep=True)
    config.hidden_groups.extra_columns = [syn.SENSITIVE, "not_a_column"]
    cand, _ = candidate("two_feature")
    r = result(cand, config)
    assert syn.SENSITIVE in r.details["features_used"]
    assert r.details["extra_columns_missing"] == ["not_a_column"]


def test_excluding_every_feature_is_unavailable(enabled_config):
    config = enabled_config.model_copy(deep=True)
    config.hidden_groups.exclude_features = list(syn.FEATURES)
    cand, _ = candidate("two_feature")
    r = result(cand, config)
    assert r.status == CheckStatus.UNAVAILABLE and "No feature is left" in r.message


# ------------------------------------------------------- unavailable paths
def test_a_model_without_probabilities_is_unavailable(enabled_config):
    train, evaluation, _ = syn.make_pair(5000, 3000, 7, "two_feature")
    svc = Pipeline([("p", ColumnTransformer([
        ("n", StandardScaler(), syn.NUMERIC),
        ("c", OneHotEncoder(handle_unknown="ignore"), syn.CATEGORICAL)])),
        ("m", LinearSVC())]).fit(train[syn.FEATURES], train[syn.TARGET])
    cand, _ = candidate("two_feature")
    cand.model, cand.eval_df = svc, evaluation
    r = result(cand, enabled_config)
    assert r.status == CheckStatus.UNAVAILABLE and r.score is None
    assert "probability estimates" in r.message and "not probabilities" in r.message


def test_a_small_sample_is_unavailable(enabled_config):
    cand, _ = candidate("two_feature", n_eval=700)
    r = result(cand, enabled_config)
    assert r.status == CheckStatus.UNAVAILABLE and "only 700 evaluation rows" in r.message
    assert r.details["n_rows"] == 700


def test_a_miscalibrated_model_is_unavailable_not_a_pass(enabled_config):
    train, evaluation, _ = syn.make_pair(5000, 3000, 7, "two_feature")
    skewed = Pipeline([("p", ColumnTransformer([
        ("n", StandardScaler(), syn.NUMERIC),
        ("c", OneHotEncoder(handle_unknown="ignore"), syn.CATEGORICAL)])),
        ("m", LogisticRegression(max_iter=500, class_weight={syn.POSITIVE: 8, syn.NEGATIVE: 1}))]
    ).fit(train[syn.FEATURES], train[syn.TARGET])
    cand, _ = candidate("two_feature")
    cand.model, cand.eval_df = skewed, evaluation
    r = result(cand, enabled_config)
    assert r.status == CheckStatus.UNAVAILABLE and r.score is None
    assert "not calibrated" in r.message and r.details["ece"] > 0.05


def test_an_unavailable_prediction_stage_is_unavailable():
    config = load_config(CONFIGS / "hidden_groups.yaml")
    cand, _ = candidate("two_feature")
    cand.metadata = TrainingMetadata()                       # no target declared
    r = result(cand, config)
    assert r.status == CheckStatus.UNAVAILABLE and "target column is not declared" in r.message


# ---------------------------------------------------- scoring and decisions
def _run_all(cand, config):
    from mqvg.checks.operational import run_operational_checks
    from mqvg.checks.training_validity import run_training_validity_checks
    results = (run_training_validity_checks(cand, config) + run_behaviour_checks(cand, config)
               + run_operational_checks(cand, config))
    card = compute_scorecard(results, config)
    return results, card, evaluate_rules(results, card, config)


def test_the_check_contributes_to_the_behaviour_score_with_its_configured_weight(
        enabled_config):
    cand, _ = candidate("two_feature")
    results, card, _ = _run_all(cand, enabled_config)
    mb = card.dimensions[Dimension.MODEL_BEHAVIOUR]
    contribution = next(c for c in mb.contributions if c.check_id == CHECK)
    assert contribution.weight == 1.5 and contribution.treatment.value == "counted"
    assert sum(c.points for c in mb.contributions) == pytest.approx(mb.score)


def test_a_confirmed_group_blocks_promotion_under_the_hold_policy(enabled_config):
    cand, _ = candidate("two_feature")
    _, _, decision = _run_all(cand, enabled_config)
    assert decision.decision == Decision.HOLD
    assert "hidden_failure_groups_hold" in [r.rule_id for r in decision.triggered_rules]


def test_under_score_only_a_confirmed_group_does_not_block(enabled_config):
    config = enabled_config.model_copy(deep=True)
    config.hidden_groups.consequence = "score_only"
    config.hidden_groups.require_assessment = False
    cand, _ = candidate("two_feature")
    results, _, decision = _run_all(cand, config)
    assert {r.id: r for r in results}[CHECK].status in (CheckStatus.WARN, CheckStatus.FAIL)
    assert not [r for r in decision.triggered_rules if "hidden_failure" in r.rule_id]


def test_criticality_is_configurable_a_critical_rule_forces_rollback(enabled_config):
    import yaml
    data = enabled_config.model_dump(mode="json")
    data["critical_rules"].append({
        "id": "severe_hidden_failure_group", "name": "Severe hidden failure group",
        "description": None,
        "condition": {"check": CHECK, "status_in": ["WARN", "FAIL"],
                      "value_above": None, "value_below": None}})
    from mqvg.config import GateConfig
    config = GateConfig.model_validate(data)
    cand, _ = candidate("two_feature")
    _, _, decision = _run_all(cand, config)
    assert decision.decision == Decision.ROLLBACK
    assert "severe_hidden_failure_group" in [r.rule_id for r in decision.triggered_rules]


def test_a_clean_model_can_still_be_promoted_with_the_check_on(enabled_config):
    cand, _ = candidate(None)
    results, card, decision = _run_all(cand, enabled_config)
    assert {r.id: r for r in results}[CHECK].status == CheckStatus.PASS
    assert f"must_pass:{CHECK}" not in [r.rule_id for r in decision.triggered_rules]


def test_unavailable_hidden_groups_never_count_as_a_pass(enabled_config):
    cand, _ = candidate("two_feature", n_eval=700)
    results, card, decision = _run_all(cand, enabled_config)
    mb = card.dimensions[Dimension.MODEL_BEHAVIOUR]
    contribution = next(c for c in mb.contributions if c.check_id == CHECK)
    assert contribution.treatment.value == "excluded_unavailable" and contribution.points == 0
    assert decision.decision != Decision.PROMOTE              # must-pass and unavailable


# -------------------------------------------------- explanation and report
def _validation(cand, config):
    results, card, decision = _run_all(cand, config)
    from datetime import datetime, timezone
    from mqvg.schemas import CandidateSummary, ScenarioInfo, ValidationResult
    return ValidationResult(
        gate_version="t", run_id="r", created_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        candidate=CandidateSummary(model_path="m"), scenario=ScenarioInfo(),
        config_hash="h", config_snapshot=config.model_dump(mode="json"),
        checks=results, scorecard=card, decision=decision,
        explanations=explain_failures(results, card, decision, config))


def test_the_explanation_names_the_hidden_group(enabled_config):
    v = _validation(candidate("two_feature")[0], enabled_config)
    line = next(l for l in v.explanations if l.startswith("Hidden failure group"))
    assert "held-out rows" in line and "expected" in line
    assert any("Suggested action (behaviour.hidden_failure_groups)" in l for l in v.explanations)


def test_the_report_has_a_hidden_group_section_with_the_statistics(enabled_config):
    v = _validation(candidate("two_feature")[0], enabled_config)
    text = render_markdown(v)
    section = text.split("## Hidden failure groups")[1].split("## Evidence gaps")[0]
    top = next(c for c in v.checks if c.id == CHECK).details["groups"][0]
    assert top["rule"] in section and "Adjusted p" in section and "Excess (95% CI)" in section
    assert f"{top['observed_error_rate']:.1%}" in section
    assert "Caveats:" in section and "held-out" in section
    assert "## Score breakdown" in text.split("## Hidden failure groups")[0]


def test_the_report_says_when_nothing_was_found(enabled_config):
    section = render_markdown(_validation(candidate(None)[0], enabled_config)
                              ).split("## Hidden failure groups")[1]
    assert "0 confirmed" in section and "No group passed both" in section


def test_the_report_says_when_it_could_not_run(enabled_config):
    section = render_markdown(_validation(candidate("two_feature", n_eval=700)[0], enabled_config)
                              ).split("## Hidden failure groups")[1].split("## Evidence gaps")[0]
    assert "**Not assessed.**" in section and "only 700 evaluation rows" in section


def test_the_report_has_no_such_section_when_the_check_is_off(default_cfg):
    text = render_markdown(_validation(candidate("two_feature")[0], default_cfg))
    assert "Hidden failure groups" not in text


# ------------------------------------------------------ the whole pipeline
def test_determinism_through_the_pipeline(enabled_config, tmp_path):
    train, evaluation, _ = syn.make_pair(4000, 2500, 11, "two_feature")
    joblib.dump(syn.fit_logistic(train), tmp_path / "m.joblib")
    evaluation.to_csv(tmp_path / "e.csv", index=False)
    (tmp_path / "meta.json").write_text(json.dumps({
        "target_column": syn.TARGET, "positive_label": syn.POSITIVE,
        "feature_columns": syn.FEATURES, "sensitive_columns": [syn.SENSITIVE]}))
    bundle = CandidateBundle(model_path=tmp_path / "m.joblib", eval_data_path=tmp_path / "e.csv",
                             metadata_path=tmp_path / "meta.json")
    a = {c.id: c for c in run_validation(bundle, enabled_config).checks}[CHECK]
    b = {c.id: c for c in run_validation(bundle, enabled_config).checks}[CHECK]
    assert a.raw_value == b.raw_value and a.message == b.message
    assert a.details["groups"] == b.details["groups"]


# --------------------------------------------- the sample fixtures (if built)
def _fixture_result(name, config):
    model = FIXTURES / name / "model.joblib"
    if not model.exists():
        pytest.skip("hidden-group fixtures not generated (run scripts/make_hidden_group_fixtures.py)")
    return run_validation(CandidateBundle(
        candidate_id=name, model_path=model, eval_data_path=FIXTURES / name / "eval.csv",
        metadata_path=FIXTURES / name / "metadata.json",
        train_row_hashes_path=FIXTURES / name / "train_row_hashes.txt"), config)


def test_the_planted_fixture_is_flagged(enabled_config):
    r = _fixture_result("hidden_group", enabled_config)
    check = {c.id: c for c in r.checks}[CHECK]
    assert check.status in (CheckStatus.WARN, CheckStatus.FAIL)
    truth = json.loads((FIXTURES / "hidden_group" / "ground_truth.json").read_text())
    assert any(truth["rule_keywords"][0] in g["rule"] for g in check.details["groups"]
               if g["confirmed"])
    assert r.decision.decision == Decision.HOLD


def test_the_null_control_fixture_is_clean(enabled_config):
    r = _fixture_result("hidden_group_null", enabled_config)
    check = {c.id: c for c in r.checks}[CHECK]
    assert check.status == CheckStatus.PASS
    assert r.decision.decision == Decision.PROMOTE


def test_the_default_profile_does_not_see_the_planted_fixture(default_cfg):
    """The motivating gap: the existing checks promote a model with a hidden failure group."""
    r = _fixture_result("hidden_group", default_cfg)
    assert CHECK not in {c.id for c in r.checks}
    assert r.decision.decision == Decision.PROMOTE


def test_the_report_states_truthfully_whether_recalibration_happened(enabled_config):
    # no fold reaches the trigger
    never = enabled_config.model_copy(deep=True)
    never.hidden_groups.recalibration_trigger_ece = 0.5
    text = render_markdown(_validation(candidate("two_feature")[0], never))
    assert "No fold needed recalibration" in text and "were recalibrated" not in text

    # clearly over-confident probabilities: recalibration is triggered on both folds
    cand, _ = candidate("two_feature")
    base_model = cand.model

    class Overconfident:
        classes_ = base_model.classes_
        feature_names_in_ = np.array(syn.FEATURES)

        def predict(self, X):
            return base_model.predict(X)

        def predict_proba(self, X):
            q = np.clip(base_model.predict_proba(X), 1e-6, 1 - 1e-6)
            out = 1 / (1 + np.exp(-np.log(q / (1 - q)) * 1.6))
            return out / out.sum(axis=1, keepdims=True)

    cand.model = Overconfident()
    text = render_markdown(_validation(cand, enabled_config))
    assert "recalibrated (Platt map, discovery rows only) on 2 of 2 folds" in text

    off = enabled_config.model_copy(deep=True)
    off.hidden_groups.recalibrate = False
    assert "Recalibration is switched off." in render_markdown(
        _validation(candidate("two_feature")[0], off))


def test_the_report_always_agrees_with_the_recorded_folds(enabled_config):
    v = _validation(candidate("two_feature")[0], enabled_config)
    d = next(c for c in v.checks if c.id == CHECK).details
    text = render_markdown(v)
    folds = d["recalibrated_folds"]
    if any(folds):
        assert f"on {sum(folds)} of {len(folds)} folds" in text
    else:
        assert "No fold needed recalibration" in text
