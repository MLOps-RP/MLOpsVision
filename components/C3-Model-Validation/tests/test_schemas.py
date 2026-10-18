from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from mqvg.schemas import (
    ActionMode, CandidateBundle, CandidateSummary, CheckResult, CheckStatus,
    Decision, DecisionRecord, Dimension, DimensionScore, EvidenceSource,
    MissingPolicy, RuleKind, ScenarioInfo, ScenarioMode, ScoreCard,
    TrainingMetadata, TriggeredRule, ValidationResult,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


# ---------------------------------------------------------------- helpers
def make_check(**overrides) -> CheckResult:
    fields = dict(
        id="behaviour.f1", dimension=Dimension.MODEL_BEHAVIOUR, name="F1 score",
        status=CheckStatus.PASS, evidence_source=EvidenceSource.MEASURED,
        score=80.0, raw_value=0.8, message="F1 is 0.80",
    )
    fields.update(overrides)
    return CheckResult(**fields)


def make_dimension(dimension, score=80.0, coverage=1.0, provisional=False):
    return DimensionScore(dimension=dimension, score=score, coverage=coverage,
                          provisional=provisional, available_checks=2,
                          total_checks=2)


def make_scorecard(**overrides) -> ScoreCard:
    fields = dict(
        dimensions={d: make_dimension(d) for d in Dimension},
        weights={Dimension.TRAINING_VALIDITY: 0.3,
                 Dimension.MODEL_BEHAVIOUR: 0.5,
                 Dimension.OPERATIONAL_READINESS: 0.2},
        mqs=80.0, overall_coverage=1.0, provisional=False,
        missing_policy=MissingPolicy.EXCLUDE_AND_CAP,
    )
    fields.update(overrides)
    return ScoreCard(**fields)


def make_decision(**overrides) -> DecisionRecord:
    fields = dict(decision=Decision.PROMOTE, mqs=80.0,
                  summary="All requirements met")
    fields.update(overrides)
    return DecisionRecord(**fields)


def make_result(**overrides) -> ValidationResult:
    fields = dict(
        gate_version="0.1.0", run_id="run-1",
        created_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        candidate=CandidateSummary(model_path="model.joblib"),
        scenario=ScenarioInfo(), config_hash="abc", config_snapshot={},
        checks=[make_check()], scorecard=make_scorecard(),
        decision=make_decision(),
    )
    fields.update(overrides)
    return ValidationResult(**fields)


# ----------------------------------------------------------- CheckResult
def test_valid_check_result():
    check = make_check()
    assert check.has_evidence


def test_unavailable_check_has_no_score_and_no_evidence():
    check = make_check(status=CheckStatus.UNAVAILABLE, score=None,
                       evidence_source=EvidenceSource.UNAVAILABLE,
                       raw_value=None, message="No sensitive column")
    assert not check.has_evidence


@pytest.mark.parametrize("status", [CheckStatus.UNAVAILABLE,
                                    CheckStatus.NOT_APPLICABLE,
                                    CheckStatus.ERROR])
def test_missing_evidence_status_cannot_carry_a_score(status):
    with pytest.raises(ValidationError, match="must not have a score"):
        make_check(status=status, score=100.0,
                   evidence_source=EvidenceSource.UNAVAILABLE)


def test_unavailable_check_cannot_claim_measured_evidence():
    with pytest.raises(ValidationError, match="UNAVAILABLE"):
        make_check(status=CheckStatus.UNAVAILABLE, score=None,
                   evidence_source=EvidenceSource.MEASURED)


@pytest.mark.parametrize("status", [CheckStatus.PASS, CheckStatus.WARN,
                                    CheckStatus.FAIL])
def test_scored_status_requires_a_score(status):
    with pytest.raises(ValidationError, match="requires a score"):
        make_check(status=status, score=None)


def test_passing_check_cannot_have_unavailable_evidence():
    with pytest.raises(ValidationError, match="cannot have evidence_source"):
        make_check(evidence_source=EvidenceSource.UNAVAILABLE)


@pytest.mark.parametrize("score", [-0.1, 100.1])
def test_score_must_be_between_0_and_100(score):
    with pytest.raises(ValidationError):
        make_check(score=score)


@pytest.mark.parametrize("bad_id", ["f1", "Behaviour.F1", "a.b.c", "1a.b", ""])
def test_check_id_format(bad_id):
    with pytest.raises(ValidationError):
        make_check(id=bad_id)


def test_check_message_cannot_be_empty():
    with pytest.raises(ValidationError):
        make_check(message="")


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        make_check(unexpected="x")


# -------------------------------------------------------- DimensionScore
def test_partial_coverage_must_be_provisional():
    with pytest.raises(ValidationError, match="provisional"):
        make_dimension(Dimension.MODEL_BEHAVIOUR, coverage=0.5,
                       provisional=False)


def test_score_requires_nonzero_coverage():
    with pytest.raises(ValidationError, match="zero coverage"):
        make_dimension(Dimension.MODEL_BEHAVIOUR, coverage=0.0,
                       provisional=True)


def test_dimension_without_evidence_has_no_score():
    dim = DimensionScore(dimension=Dimension.MODEL_BEHAVIOUR, score=None,
                         coverage=0.0, provisional=True, available_checks=0,
                         total_checks=3)
    assert dim.score is None


def test_available_checks_cannot_exceed_total():
    with pytest.raises(ValidationError):
        DimensionScore(dimension=Dimension.MODEL_BEHAVIOUR, score=50,
                       coverage=1, provisional=False, available_checks=3,
                       total_checks=2)


# -------------------------------------------------------------- ScoreCard
def test_valid_scorecard():
    assert make_scorecard().mqs == 80.0


def test_weights_must_sum_to_one():
    with pytest.raises(ValidationError, match="sum to 1"):
        make_scorecard(weights={Dimension.TRAINING_VALIDITY: 0.5,
                                Dimension.MODEL_BEHAVIOUR: 0.5,
                                Dimension.OPERATIONAL_READINESS: 0.5})


def test_weights_must_cover_all_dimensions():
    with pytest.raises(ValidationError, match="all three"):
        make_scorecard(weights={Dimension.TRAINING_VALIDITY: 0.5,
                                Dimension.MODEL_BEHAVIOUR: 0.5})


def test_weights_cannot_be_negative():
    with pytest.raises(ValidationError, match="negative"):
        make_scorecard(weights={Dimension.TRAINING_VALIDITY: -0.2,
                                Dimension.MODEL_BEHAVIOUR: 0.7,
                                Dimension.OPERATIONAL_READINESS: 0.5})


def test_mqs_from_incomplete_dimensions_must_be_provisional():
    dims = {d: make_dimension(d) for d in Dimension}
    dims[Dimension.MODEL_BEHAVIOUR] = make_dimension(
        Dimension.MODEL_BEHAVIOUR, coverage=0.5, provisional=True)
    with pytest.raises(ValidationError, match="provisional"):
        make_scorecard(dimensions=dims, provisional=False)
    assert make_scorecard(dimensions=dims, provisional=True).provisional


# ----------------------------------------------------------- DecisionRecord
def rule(effect, kind=RuleKind.SCORE, check_ids=()):
    return TriggeredRule(rule_id="r1", name="rule", kind=kind, effect=effect,
                         reason="because", check_ids=list(check_ids))


def test_promote_cannot_have_blocking_rules():
    with pytest.raises(ValidationError, match="PROMOTE"):
        make_decision(decision=Decision.PROMOTE,
                      triggered_rules=[rule(Decision.HOLD)])


def test_rollback_needs_a_rollback_rule():
    with pytest.raises(ValidationError, match="ROLLBACK requires"):
        make_decision(decision=Decision.ROLLBACK,
                      triggered_rules=[rule(Decision.HOLD)])


def test_critical_override_must_be_rollback_with_critical_rule():
    with pytest.raises(ValidationError, match="only a ROLLBACK"):
        make_decision(decision=Decision.HOLD, overridden_by_critical=True,
                      triggered_rules=[rule(Decision.HOLD, RuleKind.CRITICAL)])
    with pytest.raises(ValidationError, match="critical rule"):
        make_decision(decision=Decision.ROLLBACK, overridden_by_critical=True,
                      triggered_rules=[rule(Decision.ROLLBACK, RuleKind.SCORE)])


def test_high_mqs_with_critical_failure_is_a_valid_rollback():
    decision = make_decision(
        decision=Decision.ROLLBACK, mqs=95.0, overridden_by_critical=True,
        triggered_rules=[rule(Decision.ROLLBACK, RuleKind.CRITICAL)])
    assert decision.action_mode == ActionMode.RECOMMENDATION


# -------------------------------------------------------- ValidationResult
def test_valid_validation_result_round_trips_through_json():
    result = make_result()
    restored = ValidationResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_duplicate_check_ids_are_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        make_result(checks=[make_check(), make_check()])


def test_simulated_evidence_requires_simulated_scenario():
    simulated = make_check(evidence_source=EvidenceSource.SIMULATED)
    with pytest.raises(ValidationError, match="SIMULATED"):
        make_result(checks=[simulated])
    scenario = ScenarioInfo(mode=ScenarioMode.SIMULATED,
                            simulated_inputs=["behaviour.f1"])
    assert make_result(checks=[simulated], scenario=scenario)


def test_rules_must_reference_existing_checks():
    decision = make_decision(
        decision=Decision.HOLD,
        triggered_rules=[rule(Decision.HOLD, check_ids=["nope.missing"])])
    with pytest.raises(ValidationError, match="unknown checks"):
        make_result(decision=decision)


# ------------------------------------------------------ ScenarioInfo/bundle
def test_simulated_scenario_must_list_inputs():
    with pytest.raises(ValidationError, match="must list"):
        ScenarioInfo(mode=ScenarioMode.SIMULATED)
    with pytest.raises(ValidationError, match="mode is 'real'"):
        ScenarioInfo(simulated_inputs=["x"])


def test_bundle_does_not_require_files_to_exist():
    bundle = CandidateBundle(model_path="nowhere/model.joblib",
                             eval_data_path="nowhere/eval.csv")
    assert bundle.metadata_path is None
    assert bundle.scenario.mode == ScenarioMode.REAL


# ---------------------------------------------------------- TrainingMetadata
def test_empty_metadata_is_valid_and_reports_all_missing():
    meta = TrainingMetadata()
    required = ["random_state", "split.seed", "train_data_sha256"]
    assert meta.missing_fields(required) == required


def test_missing_fields_treats_empty_values_as_missing_and_looks_in_nested():
    meta = TrainingMetadata(random_state=0, split={"seed": 7},
                            sensitive_columns=[])
    assert meta.missing_fields(["random_state", "split.seed",
                                "split.method", "sensitive_columns"]) == [
        "split.method", "sensitive_columns"]


def test_metadata_keeps_extra_fields():
    meta = TrainingMetadata.model_validate({"fixture_name": "x"})
    assert meta.model_dump()["fixture_name"] == "x"


@pytest.mark.skipif(not (FIXTURES / "good" / "metadata.json").exists(),
                    reason="fixtures not generated (run scripts/make_fixtures.py)")
def test_good_fixture_metadata_is_complete():
    meta = TrainingMetadata.model_validate_json(
        (FIXTURES / "good" / "metadata.json").read_text(encoding="utf-8"))
    required = ["random_state", "split.seed", "train_data_sha256",
                "hyperparameters", "versions", "training_metrics"]
    assert meta.missing_fields(required) == []


@pytest.mark.skipif(
    not (FIXTURES / "good_incomplete_metadata" / "metadata.json").exists(),
    reason="fixtures not generated (run scripts/make_fixtures.py)")
def test_incomplete_fixture_metadata_reports_missing_fields():
    meta = TrainingMetadata.model_validate_json(
        (FIXTURES / "good_incomplete_metadata" / "metadata.json").read_text(
            encoding="utf-8"))
    missing = meta.missing_fields(["random_state", "split.seed",
                                   "train_data_sha256", "hyperparameters",
                                   "versions", "training_metrics"])
    assert len(missing) == 6


@pytest.mark.skipif(not FIXTURES.exists(), reason="fixtures not generated")
def test_every_fixture_metadata_file_parses():
    files = list(FIXTURES.glob("*/metadata.json"))
    assert files
    for path in files:
        TrainingMetadata.model_validate_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- JSON schema
def test_json_schema_can_be_exported():
    schema = ValidationResult.model_json_schema()
    assert "checks" in schema["properties"]
