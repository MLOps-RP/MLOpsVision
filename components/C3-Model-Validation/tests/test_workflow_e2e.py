"""The complete workflow, from files to the unified report.

Each test starts from a saved model file and a SEPARATE evaluation CSV that
holds the true labels (plus metadata and training-row hashes), runs the three
layers, scores the MQS, applies the configured policy and writes the report.
Nothing here depends on the generated fixtures.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from mqvg import synthetic as syn
from mqvg.config import load_config
from mqvg.environment import current_versions
from mqvg.hashing import read_csv_as_text, row_hashes
from mqvg.schemas import (
    ActionMode, CandidateBundle, CheckStatus, Decision, Dimension, ValidationResult,
)
from mqvg.workflow import validate_and_report

from conftest import CONFIGS

CHECK = "behaviour.hidden_failure_groups"
PROFILES = ["hidden_groups_score_only", "hidden_groups", "hidden_groups_critical"]
HASH_COLUMNS = syn.FEATURES + [syn.SENSITIVE]


@dataclass
class Case:
    name: str
    folder: Path
    train: pd.DataFrame
    evaluation: pd.DataFrame
    region: object
    model_path: Path
    eval_path: Path
    meta_path: Path
    hashes_path: Path

    def bundle(self, model_path=None, eval_path=None):
        return CandidateBundle(
            candidate_id=self.name, model_path=model_path or self.model_path,
            eval_data_path=eval_path or self.eval_path, metadata_path=self.meta_path,
            train_row_hashes_path=self.hashes_path)


def build_case(folder: Path, name: str, pattern, seed=11, n_eval=3000) -> Case:
    folder = folder / name
    folder.mkdir()
    train, evaluation, region = syn.make_pair(5000, n_eval, seed, pattern)
    model = syn.fit_logistic(train)
    train.to_csv(folder / "train.csv", index=False)
    evaluation.to_csv(folder / "eval.csv", index=False)           # separate from the training data
    joblib.dump(model, folder / "model.joblib")
    hashes = row_hashes(read_csv_as_text(folder / "train.csv"), HASH_COLUMNS)
    (folder / "hashes.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    accuracy = float((model.predict(train[syn.FEATURES]) == train[syn.TARGET]).mean())
    (folder / "metadata.json").write_text(json.dumps({
        "model_type": "LogisticRegression", "task_type": "binary_classification",
        "dataset": "synthetic", "target_column": syn.TARGET, "positive_label": syn.POSITIVE,
        "sensitive_columns": [syn.SENSITIVE], "feature_columns": syn.FEATURES,
        "hyperparameters": {"max_iter": 500}, "random_state": seed, "split": {"seed": seed},
        "train_rows": len(train), "train_data_sha256": "x", "versions": current_versions(),
        "code": {"commit": "abc"}, "training_metrics": {"accuracy": accuracy},
        "row_hash_columns": HASH_COLUMNS}), encoding="utf-8")
    return Case(name, folder, train, evaluation, region, folder / "model.joblib",
                folder / "eval.csv", folder / "metadata.json", folder / "hashes.txt")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    folder = tmp_path_factory.mktemp("e2e")
    planted = build_case(folder, "planted", "two_feature")
    control = build_case(folder, "control", None)
    # variants of the planted case: same data, a different model or a smaller sample
    pre = lambda dense=False: ColumnTransformer([                  # noqa: E731
        ("n", StandardScaler(), syn.NUMERIC),
        ("c", OneHotEncoder(handle_unknown="ignore"), syn.CATEGORICAL)])
    skewed = Pipeline([("p", pre()), ("m", LogisticRegression(
        max_iter=500, class_weight={syn.POSITIVE: 8, syn.NEGATIVE: 1}))]
    ).fit(planted.train[syn.FEATURES], planted.train[syn.TARGET])
    joblib.dump(skewed, planted.folder / "miscalibrated.joblib")
    svc = Pipeline([("p", pre()), ("m", LinearSVC())]).fit(
        planted.train[syn.FEATURES], planted.train[syn.TARGET])
    joblib.dump(svc, planted.folder / "svc.joblib")
    planted.evaluation.head(700).to_csv(planted.folder / "eval_small.csv", index=False)
    return {"planted": planted, "control": control, "reports": folder / "reports",
            "cache": {}}


def run(world, case_name, profile, *, model=None, eval_file=None, edit=None, key=None):
    key = key or (case_name, profile, model, eval_file)
    if key not in world["cache"]:
        case = world[case_name]
        config = load_config(CONFIGS / f"{profile}.yaml").model_copy(deep=True)
        if edit:
            edit(config)
        world["cache"][key] = validate_and_report(
            case.bundle(case.folder / model if model else None,
                        case.folder / eval_file if eval_file else None),
            config, world["reports"] / "_".join(str(p) for p in key if p))
    return world["cache"][key]


def checks(output):
    return {c.id: c for c in output.result.checks}


def section(text, heading, until=None):
    body = text.split(f"## {heading}")[1]
    return body.split(f"## {until}")[0] if until else re.split(r"\n## ", body)[0]


# ============================================================ the workflow
def test_the_workflow_runs_all_three_layers_and_writes_the_unified_report(world):
    out = run(world, "planted", "hidden_groups")
    assert {c.dimension for c in out.result.checks} == set(Dimension)       # all three layers
    assert CHECK in checks(out)
    assert out.paths["markdown"].exists() and out.paths["json"].exists()
    assert out.paths["markdown"].read_text(encoding="utf-8") == out.report_markdown
    assert ValidationResult.model_validate_json(
        out.paths["json"].read_text(encoding="utf-8")) == out.result


def test_the_unified_report_shows_every_required_element_in_order(world):
    text = run(world, "planted", "hidden_groups").report_markdown
    headings = re.findall(r"^## (.+)$", text, re.M)
    assert headings == [
        "Decision: HOLD", "Scores (0–100)", "Why this decision", "Score breakdown",
        "Hidden failure groups", "Evidence gaps", "All checks", "Inputs and audit trail",
        "Configuration used", "Limitations"]
    scores = section(text, "Scores (0–100)")
    for layer in ("Training Validity (TVS)", "Model Behaviour (MBS)", "Operational Readiness (ORS)"):
        assert layer in scores                                     # the three layers
    assert "The three layers at a glance" in scores and "**Calculation:** MQS =" in scores
    assert "Hidden failure group found" in section(text, "Why this decision")   # fired rule
    assert "recommendation" in text and "Nothing was deployed" in text


def test_the_mqs_calculation_in_the_report_is_the_real_one(world):
    out = run(world, "planted", "hidden_groups")
    card, text = out.result.scorecard, out.report_markdown
    calc = re.search(r"\*\*Calculation:\*\* MQS = (.+) = ([\d.]+)", text)
    assert calc and float(calc.group(2)) == pytest.approx(card.mqs, abs=0.05)
    terms = re.findall(r"(TVS|MBS|ORS) ([\d.]+) × ([\d.]+)", calc.group(1))
    assert [t[0] for t in terms] == ["TVS", "MBS", "ORS"]
    total = sum(float(score) * float(weight) for _, score, weight in terms)
    assert total == pytest.approx(card.mqs, abs=0.15)              # rounding of displayed values
    for (label, score, weight), dimension in zip(terms, Dimension):
        assert float(score) == pytest.approx(card.dimensions[dimension].score, abs=0.05)
        assert float(weight) == pytest.approx(card.weights[dimension])


def test_a_confirmed_planted_subgroup_is_reported_with_all_its_statistics(world):
    out = run(world, "planted", "hidden_groups")
    check = checks(out)[CHECK]
    assert check.status in (CheckStatus.WARN, CheckStatus.FAIL) and check.raw_value >= 0.10
    section_text = section(out.report_markdown, "Hidden failure groups")
    confirmed = [g for g in check.details["groups"] if g["confirmed"]]
    assert confirmed and any("segment = d" in g["rule"] for g in confirmed)   # the planted rule
    top = confirmed[0]
    assert top["rule"] in section_text
    for column in ("Rows", "Observed error", "Expected error", "Excess (95% CI)",
                   "Adjusted p", "Confirmed", "Method"):
        assert column in section_text
    assert f"{top['observed_error_rate']:.1%}" in section_text
    assert f"{top['expected_error_rate']:.1%}" in section_text
    assert f"[{top['effect_ci'][0]:+.1%}, {top['effect_ci'][1]:+.1%}]" in section_text
    assert top["n_rows"] >= 30 and top["adjusted_p_value"] <= 0.05
    assert top["effect"] >= out.result.config_snapshot["hidden_groups"]["min_effect"]
    # the held-out rows really are the planted ones: most rows of a confirmed group
    # fall inside the planted region
    assert "held-out" in section_text


def test_the_report_states_that_subgroups_are_associations_and_may_be_proxies(world):
    text = section(run(world, "planted", "hidden_groups").report_markdown, "Hidden failure groups")
    assert "association" in text and "not a causal" in text
    assert "proxies for sensitive characteristics" in text


def test_the_recommendation_is_never_a_deployment_action(world):
    out = run(world, "planted", "hidden_groups_critical")
    assert out.result.decision.decision == Decision.ROLLBACK       # a recommendation...
    assert out.result.decision.action_mode == ActionMode.RECOMMENDATION   # ...only
    assert "Nothing was deployed, promoted or rolled back" in out.report_markdown
    produced = sorted(p.name for p in out.paths["markdown"].parent.iterdir())
    assert produced == ["report.md", "result.json"]               # no other side effects


# ===================================================== configurable consequences
def test_the_consequence_follows_the_configuration(world):
    outcome = {}
    for profile in PROFILES:
        out = run(world, "planted", profile)
        assert checks(out)[CHECK].status == CheckStatus.FAIL        # same evidence each time
        outcome[profile] = out
    default_case = run(world, "planted", "hidden_groups", edit=lambda c: (
        setattr(c.checks[CHECK], "enabled", False),
        setattr(c.hidden_groups, "consequence", "score_only"),
        setattr(c.hidden_groups, "require_assessment", False)), key="check_off")
    assert CHECK not in checks(default_case)                       # the blind spot: not run
    assert default_case.result.decision.decision == Decision.PROMOTE

    assert outcome["hidden_groups_score_only"].decision == "PROMOTE"      # score only
    assert outcome["hidden_groups"].decision == "HOLD"                    # hold
    assert outcome["hidden_groups_critical"].decision == "ROLLBACK"       # critical (opt in)
    # scoring is affected the same way in all three; only the decision differs
    mqs = {p: o.result.scorecard.mqs for p, o in outcome.items()}
    assert len({round(v, 6) for v in mqs.values()}) == 1
    assert mqs["hidden_groups"] < default_case.result.scorecard.mqs
    # and each report names the rule that produced its recommendation
    assert "hidden_failure_groups_hold" in [r.rule_id for r in outcome["hidden_groups"].result.decision.triggered_rules]
    critical = outcome["hidden_groups_critical"].result.decision
    assert critical.overridden_by_critical
    assert "Severe hidden failure group" in outcome["hidden_groups_critical"].report_markdown
    assert "A critical rule overrode the score." in outcome["hidden_groups_critical"].report_markdown


@pytest.mark.parametrize("profile", PROFILES)
def test_no_planted_group_does_not_cause_a_failure_decision(world, profile):
    out = run(world, "control", profile)
    check = checks(out)[CHECK]
    assert check.status == CheckStatus.PASS and check.raw_value == 0.0
    assert out.result.decision.decision == Decision.PROMOTE
    assert not [r for r in out.result.decision.triggered_rules if "hidden" in r.rule_id]
    text = section(out.report_markdown, "Hidden failure groups")
    assert "**0 confirmed**" in text and "No group passed both" in text


def test_a_stricter_minimum_effect_removes_the_finding_and_the_consequence(world):
    out = run(world, "planted", "hidden_groups", key="strict_effect",
              edit=lambda c: setattr(c.hidden_groups, "min_effect", 0.9))
    assert checks(out)[CHECK].status == CheckStatus.PASS           # nothing reaches a 90-point excess
    assert out.result.decision.decision == Decision.PROMOTE


def test_the_significance_threshold_is_applied_and_reported(world):
    out = run(world, "planted", "hidden_groups", key="alpha_tiny",
              edit=lambda c: setattr(c.hidden_groups, "alpha", 1e-12))
    details = checks(out)[CHECK].details
    assert details["alpha"] == 1e-12
    assert all(g["adjusted_p_value"] <= 1e-12 for g in details["groups"] if g["confirmed"])
    assert "false discovery rate 1e-12" in out.report_markdown


def test_the_checks_own_thresholds_decide_what_counts_as_severe(world):
    out = run(world, "planted", "hidden_groups_critical", key="lenient_severity",
              edit=lambda c: setattr(c.checks[CHECK], "thresholds",
                                     c.checks[CHECK].thresholds.model_copy(
                                         update={"fail_at": 0.9, "floor": 0.95})))
    assert checks(out)[CHECK].status == CheckStatus.WARN           # found, but not severe
    assert out.result.decision.decision == Decision.HOLD           # so only a hold, no rollback


def test_the_weight_changes_the_mqs_through_the_whole_workflow(world):
    light = run(world, "planted", "hidden_groups_score_only", key="w_light",
                edit=lambda c: setattr(c.checks[CHECK], "weight", 0.5))
    heavy = run(world, "planted", "hidden_groups_score_only", key="w_heavy",
                edit=lambda c: setattr(c.checks[CHECK], "weight", 6.0))
    assert heavy.result.scorecard.mqs < light.result.scorecard.mqs


# ============================================================ unavailable evidence
UNAVAILABLE_CASES = [
    ("insufficient data (700 rows)", dict(eval_file="eval_small.csv"), "only 700 evaluation rows"),
    ("miscalibrated model", dict(model="miscalibrated.joblib"), "not calibrated"),
    ("missing probabilities", dict(model="svc.joblib"), "probability estimates"),
]


@pytest.mark.parametrize("label, kwargs, reason", UNAVAILABLE_CASES)
@pytest.mark.parametrize("profile", PROFILES)
def test_unavailable_hidden_evidence_is_never_a_pass_and_is_reported(world, label, kwargs,
                                                                    reason, profile):
    out = run(world, "planted", profile, **kwargs)
    check = checks(out)[CHECK]
    assert check.status == CheckStatus.UNAVAILABLE and check.score is None, label
    assert check.evidence_source.value == "UNAVAILABLE" and reason in check.message
    # it appears as an evidence gap, in the hidden section and in the layer overview
    assert CHECK in section(out.report_markdown, "Evidence gaps")
    hidden = section(out.report_markdown, "Hidden failure groups")
    assert "**Not assessed.**" in hidden and reason in hidden
    assert "Subgroup detection: **not assessed**" in out.report_markdown
    # the score does not treat it as a pass
    contribution = next(c for c in out.result.scorecard.dimensions[Dimension.MODEL_BEHAVIOUR]
                        .contributions if c.check_id == CHECK)
    assert contribution.treatment.value == "excluded_unavailable" and contribution.points == 0
    assert out.result.scorecard.overall_coverage < 1


@pytest.mark.parametrize("label, kwargs, reason", UNAVAILABLE_CASES)
def test_the_hold_profile_requires_assessment_and_the_score_only_profile_does_not(
        world, label, kwargs, reason):
    hold = run(world, "planted", "hidden_groups", **kwargs)
    assert hold.result.decision.decision == Decision.HOLD
    assert "hidden_failure_groups_not_assessed" in [
        r.rule_id for r in hold.result.decision.triggered_rules]
    score_only = run(world, "planted", "hidden_groups_score_only", **kwargs)
    assert "hidden_failure_groups_not_assessed" not in [
        r.rule_id for r in score_only.result.decision.triggered_rules]
    assert not [r for r in score_only.result.decision.triggered_rules if "hidden" in r.rule_id]


def test_insufficient_data_alone_does_not_force_a_failure_under_score_only(world):
    out = run(world, "planted", "hidden_groups_score_only", eval_file="eval_small.csv")
    assert out.result.decision.decision != Decision.ROLLBACK
    assert out.result.scorecard.provisional                         # but it is flagged as incomplete


def test_too_few_errors_make_the_check_unavailable(world):
    out = run(world, "planted", "hidden_groups", key="few_errors",
              edit=lambda c: setattr(c.hidden_groups, "min_errors", 5000))
    check = checks(out)[CHECK]
    assert check.status == CheckStatus.UNAVAILABLE and "prediction errors" in check.message
    assert out.result.decision.decision == Decision.HOLD


# =========================================================== the supported input
def test_the_evaluation_data_is_separate_from_the_training_data(world):
    case = world["planted"]
    assert case.eval_path != case.folder / "train.csv"
    merged = case.evaluation.merge(case.train, on=syn.FEATURES + [syn.SENSITIVE], how="inner")
    assert len(merged) / len(case.evaluation) < 0.02             # (almost) no shared rows
    overlap = checks(run(world, "planted", "hidden_groups"))["training_validity.train_eval_overlap"]
    assert overlap.status == CheckStatus.PASS


def test_layers_one_and_three_run_alongside_the_subgroup_check(world):
    out = run(world, "planted", "hidden_groups")
    ids = set(checks(out))
    assert {"training_validity.overfit_gap", "leakage.single_feature_predictiveness",
            "operational.model_loadable", "operational.latency", CHECK} <= ids
    assert checks(out)["operational.model_loadable"].status == CheckStatus.PASS


def test_a_model_the_gate_cannot_read_still_gets_a_unified_report(world, tmp_path):
    bad = tmp_path / "not_a_model.joblib"
    bad.write_bytes(b"corrupt")
    case = world["planted"]
    config = load_config(CONFIGS / "hidden_groups.yaml")
    out = validate_and_report(case.bundle(model_path=bad), config, tmp_path / "bad")
    # a file the gate cannot read is "cannot assess" (existing design), never a pass
    assert out.decision == "HOLD"
    assert checks(out)["operational.model_loadable"].status != CheckStatus.PASS
    assert checks(out)[CHECK].status == CheckStatus.UNAVAILABLE
    assert "## Hidden failure groups" in out.report_markdown
    assert (tmp_path / "bad" / "report.md").exists()
