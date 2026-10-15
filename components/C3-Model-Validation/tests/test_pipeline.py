import json
import re
from datetime import datetime, timezone

import pytest

from mqvg import __version__, pipeline
from mqvg.catalogue import CATALOGUE
from mqvg.config import ConfigError, config_hash, load_config
from mqvg.loader import file_sha256
from mqvg.pipeline import run_validation
from mqvg.schemas import (
    CandidateBundle, CheckStatus, Decision, Dimension, ScenarioInfo,
    ScenarioMode, ValidationResult,
)

from conftest import CONFIGS, DATA, FIXTURES, ROOT

ORDER = [Dimension.TRAINING_VALIDITY, Dimension.MODEL_BEHAVIOUR,
         Dimension.OPERATIONAL_READINESS]


def enabled_catalogue_ids(config):
    return [s.id for d in ORDER for s in CATALOGUE
            if s.dimension == d and config.checks[s.id].enabled]


def statuses(result):
    return {c.id: c.status for c in result.checks}


# ------------------------------------------------------------- the result
def test_result_is_complete_and_valid(tiny_bundle, default_config):
    result = run_validation(tiny_bundle, default_config)
    assert isinstance(result, ValidationResult)
    assert result.gate_version == __version__
    assert re.fullmatch(r"[0-9a-f]{32}", result.run_id)
    assert result.created_at_utc.tzinfo is not None
    assert result.scenario.mode == ScenarioMode.REAL
    assert [c.id for c in result.checks] == enabled_catalogue_ids(default_config)
    assert result.decision.decision in set(Decision)
    assert result.scorecard.missing_policy == default_config.missing_policy


def test_audit_fields_record_the_inputs(tiny_bundle, tiny_files, default_config):
    result = run_validation(tiny_bundle, default_config)
    assert result.config_hash == config_hash(default_config)
    assert result.config_snapshot == default_config.model_dump(mode="json")
    assert result.candidate.model_path == str(tiny_files["model"])
    assert result.candidate.model_sha256 == file_sha256(tiny_files["model"])
    assert result.candidate.eval_data_sha256 == file_sha256(tiny_files["eval"])
    assert result.candidate.metadata_sha256 == file_sha256(tiny_files["metadata"])


def test_environment_has_a_machine_profile_and_stage_timings(tiny_bundle,
                                                             default_config):
    env = run_validation(tiny_bundle, default_config).environment
    assert "cpu_count" in env["machine"] and "versions" in env["machine"]
    assert set(env["timings_seconds"]) == {
        "load", "training_validity", "model_behaviour", "operational_readiness",
        "score_and_decide"}
    assert all(t >= 0 for t in env["timings_seconds"].values())


def test_result_survives_a_json_round_trip(tiny_bundle, default_config):
    result = run_validation(tiny_bundle, default_config)
    assert ValidationResult.model_validate_json(result.model_dump_json()) == result
    json.loads(result.model_dump_json())                 # plain valid JSON


def test_run_id_and_time_can_be_fixed_and_otherwise_differ(tiny_bundle,
                                                           default_config):
    when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    fixed = run_validation(tiny_bundle, default_config, run_id="abc",
                           created_at=when)
    assert fixed.run_id == "abc" and fixed.created_at_utc == when
    assert (run_validation(tiny_bundle, default_config).run_id
            != run_validation(tiny_bundle, default_config).run_id)


def test_candidate_id_and_scenario_are_carried_through(tiny_files, default_config):
    scenario = ScenarioInfo(mode=ScenarioMode.SIMULATED, name="demo",
                            simulated_inputs=["metadata"])
    bundle = CandidateBundle(
        candidate_id="cand-7", model_path=tiny_files["model"],
        eval_data_path=tiny_files["eval"], metadata_path=tiny_files["metadata"],
        scenario=scenario)
    result = run_validation(bundle, default_config)
    assert result.candidate.candidate_id == "cand-7"
    assert result.scenario == scenario


# ------------------------------------------------------------- configuration
def test_config_can_come_from_the_bundle(tiny_files):
    bundle = CandidateBundle(model_path=tiny_files["model"],
                             eval_data_path=tiny_files["eval"],
                             metadata_path=tiny_files["metadata"],
                             config_path=CONFIGS / "strict.yaml")
    assert run_validation(bundle).config_snapshot["profile"] == "strict"


def test_explicit_config_overrides_the_bundle_path(tiny_files, default_config):
    bundle = CandidateBundle(model_path=tiny_files["model"],
                             eval_data_path=tiny_files["eval"],
                             config_path=CONFIGS / "strict.yaml")
    assert run_validation(bundle, default_config).config_snapshot["profile"] == "default"


def test_no_config_at_all_is_an_error(tiny_bundle):
    with pytest.raises(ConfigError, match="no configuration"):
        run_validation(tiny_bundle)


def test_unusable_config_path_raises(tiny_files):
    bundle = CandidateBundle(model_path=tiny_files["model"],
                             eval_data_path=tiny_files["eval"],
                             config_path=ROOT / "configs" / "missing.yaml")
    with pytest.raises(ConfigError, match="not found"):
        run_validation(bundle)


# ------------------------------------------- bad candidates never raise
def test_missing_model_gives_a_rollback_not_an_exception(tiny_files, default_config):
    tiny_files["model"].unlink()
    result = run_validation(CandidateBundle(
        model_path=tiny_files["model"], eval_data_path=tiny_files["eval"],
        metadata_path=tiny_files["metadata"]), default_config)
    assert result.decision.decision == Decision.ROLLBACK
    assert "model_not_loadable" in [r.rule_id for r in result.decision.triggered_rules]
    assert statuses(result)["operational.model_loadable"] == CheckStatus.FAIL
    assert statuses(result)["operational.latency"] == CheckStatus.UNAVAILABLE


def test_corrupt_model_gives_a_rollback(tiny_files, default_config):
    tiny_files["model"].write_bytes(tiny_files["model"].read_bytes()[:40])
    result = run_validation(CandidateBundle(
        model_path=tiny_files["model"], eval_data_path=tiny_files["eval"],
        metadata_path=tiny_files["metadata"]), default_config)
    assert result.decision.decision == Decision.ROLLBACK


def test_missing_evaluation_data_gives_a_hold_with_unavailable_checks(
        tiny_files, default_config):
    tiny_files["eval"].unlink()
    result = run_validation(CandidateBundle(
        model_path=tiny_files["model"], eval_data_path=tiny_files["eval"],
        metadata_path=tiny_files["metadata"]), default_config)
    assert result.decision.decision == Decision.HOLD
    assert statuses(result)["behaviour.f1"] == CheckStatus.UNAVAILABLE
    assert "required_evidence_missing" in [
        r.rule_id for r in result.decision.triggered_rules]


def test_broken_metadata_is_a_finding(tiny_files, default_config):
    tiny_files["metadata"].write_text("{broken", encoding="utf-8")
    result = run_validation(CandidateBundle(
        model_path=tiny_files["model"], eval_data_path=tiny_files["eval"],
        metadata_path=tiny_files["metadata"]), default_config)
    assert statuses(result)["training_validity.metadata_completeness"] == CheckStatus.FAIL
    assert result.decision.decision != Decision.PROMOTE


def test_nothing_exists_still_gives_a_result(tmp_path, default_config):
    result = run_validation(CandidateBundle(
        model_path=tmp_path / "no.joblib", eval_data_path=tmp_path / "no.csv",
        metadata_path=tmp_path / "no.json"), default_config)
    assert result.decision.decision == Decision.ROLLBACK
    assert result.scorecard.mqs is None
    assert result.candidate.model_sha256 is None


# ------------------------------------------------------- crash isolation
def test_one_crashing_check_does_not_stop_the_run(tiny_bundle, default_config,
                                                  monkeypatch):
    from mqvg.checks import operational

    def boom(ctx):
        raise RuntimeError("kaput")

    monkeypatch.setitem(operational.OPERATIONAL_CHECKS, "operational.memory",
                        operational.guarded("operational.memory")(boom))
    result = run_validation(tiny_bundle, default_config)
    assert statuses(result)["operational.memory"] == CheckStatus.ERROR
    assert statuses(result)["operational.model_loadable"] == CheckStatus.PASS
    assert len(result.checks) == len(enabled_catalogue_ids(default_config))


def test_a_crashing_check_group_errors_only_its_own_dimension(
        tiny_bundle, default_config, monkeypatch):
    def broken(candidate, config):
        raise MemoryError("out of memory")

    monkeypatch.setattr(pipeline, "run_behaviour_checks", broken)
    result = run_validation(tiny_bundle, default_config)
    by_dimension = {}
    for check in result.checks:
        by_dimension.setdefault(check.dimension, set()).add(check.status)
    assert by_dimension[Dimension.MODEL_BEHAVIOUR] == {CheckStatus.ERROR}
    assert CheckStatus.ERROR not in by_dimension[Dimension.OPERATIONAL_READINESS]
    behaviour = [c for c in result.checks if c.dimension == Dimension.MODEL_BEHAVIOUR]
    assert "MemoryError" in behaviour[0].message
    assert len(result.checks) == len(enabled_catalogue_ids(default_config))
    # required behaviour evidence is missing, so the model cannot be promoted
    assert result.decision.decision != Decision.PROMOTE


# ------------------------------------------------------- reproducibility
def test_identical_inputs_give_identical_results(tiny_bundle, make_config):
    """Apart from run id, time and environment. Latency and memory depend on
    the machine, so they are switched off for an exact comparison."""
    def tweak(d):
        for check_id in ("operational.latency", "operational.memory"):
            d["checks"][check_id]["enabled"] = False
        d["decision"]["must_pass_checks"] = [
            c for c in d["decision"]["must_pass_checks"] if c != "operational.latency"]
    config = make_config(tweak)

    def comparable(result):
        data = result.model_dump(mode="json")
        for volatile in ("run_id", "created_at_utc", "environment"):
            data.pop(volatile)
        return data

    first = comparable(run_validation(tiny_bundle, config))
    second = comparable(run_validation(tiny_bundle, config))
    assert first == second


def test_the_evaluation_data_is_not_modified(tiny_bundle, tiny_files, default_config):
    before = file_sha256(tiny_files["eval"])
    run_validation(tiny_bundle, default_config)
    assert file_sha256(tiny_files["eval"]) == before


# ------------------------------------------- generated fixture models
def _fixture_bundle(name, eval_path=None, **overrides):
    model = FIXTURES / ("good" if name.startswith("good") else name) / "model.joblib"
    eval_path = eval_path or DATA / "eval.csv"
    if not model.exists() or not eval_path.exists():
        pytest.skip("fixtures not generated (run scripts/make_fixtures.py)")
    return CandidateBundle(
        candidate_id=name, model_path=model, eval_data_path=eval_path,
        metadata_path=FIXTURES / name / "metadata.json",
        train_row_hashes_path=ROOT / "data" / "train_row_hashes.txt", **overrides)


def test_good_fixture_is_promoted(default_config):
    result = run_validation(_fixture_bundle("good"), default_config)
    assert result.decision.decision == Decision.PROMOTE
    assert result.scorecard.overall_coverage == 1.0 and not result.scorecard.provisional
    assert result.candidate.model_sha256 == file_sha256(
        FIXTURES / "good" / "model.joblib")


def test_leaky_fixture_is_a_critical_override(default_config):
    result = run_validation(
        _fixture_bundle("leaky", FIXTURES / "leaky" / "eval.csv"), default_config)
    assert result.decision.decision == Decision.ROLLBACK
    assert result.decision.overridden_by_critical
    assert result.scorecard.mqs > default_config.decision.promote_at


def test_corrupt_fixture_is_rolled_back(default_config):
    result = run_validation(_fixture_bundle("corrupt"), default_config)
    assert result.decision.decision == Decision.ROLLBACK
    assert result.scorecard.mqs_lower_bound < 40


def test_missing_sensitive_columns_hold_the_model(default_config):
    result = run_validation(
        _fixture_bundle("good_no_sensitive",
                        FIXTURES / "good_no_sensitive" / "eval.csv"),
        default_config)
    assert result.decision.decision == Decision.HOLD
    assert statuses(result)["fairness.selection_rate_ratio"] == CheckStatus.UNAVAILABLE


def test_strict_profile_runs_through_the_same_pipeline():
    strict = load_config(CONFIGS / "strict.yaml")
    result = run_validation(_fixture_bundle("good"), strict)
    assert result.config_snapshot["profile"] == "strict"
    assert result.scorecard.missing_policy.value == "penalize"
