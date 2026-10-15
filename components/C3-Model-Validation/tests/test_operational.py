import time

import numpy as np
import pandas as pd
import pytest

from mqvg.catalogue import CATALOGUE
from mqvg.checks import operational
from mqvg.checks.base import CheckContext
from mqvg.checks.operational import OPERATIONAL_CHECKS, run_operational_checks
from mqvg.environment import current_versions
from mqvg.loader import load_candidate
from mqvg.schemas import (
    CandidateBundle, CheckStatus, Dimension, EvidenceSource, TrainingMetadata,
)

from conftest import DATA, FEATURES, FIXTURES

OPERATIONAL_IDS = [s.id for s in CATALOGUE
                   if s.dimension == Dimension.OPERATIONAL_READINESS]


# ------------------------------------------------------------ stub models
class _Stub:
    feature_names_in_ = np.array(FEATURES)

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class AcceptsAnything(_Stub):
    """Predicts for any input, however malformed."""


class Strict(_Stub):
    """Rejects bad input the way scikit-learn pipelines do."""

    def predict(self, X):
        if list(X.columns) != FEATURES:
            raise ValueError("columns do not match")
        if len(X) == 0:
            raise ValueError("0 samples")
        if not pd.api.types.is_numeric_dtype(X["age"]):
            raise ValueError("age must be numeric")
        return np.zeros(len(X), dtype=int)


class Flaky(Strict):
    def __init__(self):
        self.rng = np.random.default_rng()

    def predict(self, X):
        super().predict(X)
        return self.rng.integers(0, 2, len(X))


class FlakyProbabilities(Strict):
    def __init__(self):
        self.rng = np.random.default_rng()

    def predict_proba(self, X):
        return self.rng.random((len(X), 2))


class ShortOutput(Strict):
    def predict(self, X):
        super().predict(X)
        return np.zeros(1, dtype=int)


class AlwaysRaises(_Stub):
    def predict(self, X):
        raise RuntimeError("boom")


class Sleepy(Strict):
    def predict(self, X):
        time.sleep(0.01)
        return super().predict(X)


def by_id(results):
    return {r.id: r for r in results}


# ------------------------------------------------------------- registry
def test_registry_matches_the_catalogue():
    assert sorted(OPERATIONAL_CHECKS) == sorted(OPERATIONAL_IDS)


def test_run_returns_one_valid_result_per_enabled_check(
        tiny_bundle, default_config):
    results = run_operational_checks(load_candidate(tiny_bundle), default_config)
    assert [r.id for r in results] == OPERATIONAL_IDS
    assert all(r.dimension == Dimension.OPERATIONAL_READINESS for r in results)


def test_disabled_check_is_not_run(tiny_bundle, make_config):
    config = make_config(lambda d: d["checks"]["operational.memory"].update(
        enabled=False))
    ids = [r.id for r in run_operational_checks(load_candidate(tiny_bundle), config)]
    assert "operational.memory" not in ids and len(ids) == len(OPERATIONAL_IDS) - 1


def test_healthy_model_passes_every_check(tiny_bundle, make_config):
    config = make_config(lambda d: d["checks"]["operational.latency"].update(
        thresholds={"target": 100, "pass_at": 1000, "fail_at": 5000,
                    "floor": 10000}))
    results = run_operational_checks(load_candidate(tiny_bundle), config)
    assert {r.id: r.status for r in results} == {
        i: CheckStatus.PASS for i in OPERATIONAL_IDS}
    assert all(r.evidence_source in (EvidenceSource.MEASURED,
                                     EvidenceSource.DECLARED) for r in results)


def test_a_crashing_check_becomes_an_error_result(
        tiny_bundle, default_config, monkeypatch):
    def boom(ctx):
        raise RuntimeError("kaput")

    monkeypatch.setitem(
        OPERATIONAL_CHECKS, "operational.memory",
        operational.guarded("operational.memory")(boom))
    result = by_id(run_operational_checks(
        load_candidate(tiny_bundle), default_config))["operational.memory"]
    assert result.status == CheckStatus.ERROR and result.score is None
    assert "kaput" in result.message


# ------------------------------------------------------ model_loadable
def test_model_loadable_fails_when_file_is_missing(tiny_files, default_config):
    tiny_files["model"].unlink()
    c = load_candidate(CandidateBundle(model_path=tiny_files["model"],
                                       eval_data_path=tiny_files["eval"]))
    result = by_id(run_operational_checks(c, default_config))
    assert result["operational.model_loadable"].status == CheckStatus.FAIL
    assert result["operational.model_loadable"].score == 0


def test_unloadable_model_makes_dependent_checks_unavailable_not_passing(
        tiny_files, default_config):
    data = tiny_files["model"].read_bytes()
    tiny_files["model"].write_bytes(data[:50])
    c = load_candidate(CandidateBundle(
        model_path=tiny_files["model"], eval_data_path=tiny_files["eval"],
        metadata_path=tiny_files["metadata"]))
    result = by_id(run_operational_checks(c, default_config))
    for check_id in ("operational.input_schema", "operational.prediction_determinism",
                     "operational.malformed_input_handling", "operational.latency",
                     "operational.memory"):
        assert result[check_id].status == CheckStatus.UNAVAILABLE, check_id
        assert result[check_id].score is None
    assert result["operational.model_loadable"].status == CheckStatus.FAIL
    # measurable without loading the model:
    assert result["operational.artifact_size"].status == CheckStatus.PASS


# --------------------------------------------------------- input_schema
def test_input_schema_fails_when_eval_lacks_a_feature(tiny_bundle, default_config):
    c = load_candidate(tiny_bundle)
    c.eval_df = c.eval_df.drop(columns=["hours"])
    result = by_id(run_operational_checks(c, default_config))
    schema = result["operational.input_schema"]
    assert schema.status == CheckStatus.FAIL
    assert schema.details["missing_columns"] == ["hours"]
    # other model-running checks cannot run, and say so
    assert result["operational.latency"].status == CheckStatus.UNAVAILABLE


def test_input_schema_fails_when_model_raises(candidate_with, default_config):
    result = by_id(run_operational_checks(candidate_with(AlwaysRaises()),
                                          default_config))
    assert result["operational.input_schema"].status == CheckStatus.FAIL
    assert "boom" in result["operational.input_schema"].message


def test_input_schema_fails_on_wrong_prediction_count(candidate_with, default_config):
    result = by_id(run_operational_checks(candidate_with(ShortOutput()),
                                          default_config))
    assert result["operational.input_schema"].status == CheckStatus.FAIL


def test_input_schema_unavailable_without_eval_data(candidate_with, default_config):
    c = candidate_with(Strict(), eval_df=None, eval_error="no data")
    result = by_id(run_operational_checks(c, default_config))
    assert result["operational.input_schema"].status == CheckStatus.UNAVAILABLE


def test_input_schema_unavailable_when_features_unknown(candidate_with, default_config):
    class Anonymous:
        def predict(self, X):
            return np.zeros(len(X))

    result = by_id(run_operational_checks(candidate_with(Anonymous()),
                                          default_config))
    assert result["operational.input_schema"].status == CheckStatus.UNAVAILABLE
    assert "feature columns" in result["operational.input_schema"].message


# ------------------------------------------------- prediction_determinism
def test_nondeterministic_predictions_fail(candidate_with, default_config):
    result = by_id(run_operational_checks(candidate_with(Flaky()), default_config))
    assert result["operational.prediction_determinism"].status == CheckStatus.FAIL


def test_nondeterministic_probabilities_fail(candidate_with, default_config):
    result = by_id(run_operational_checks(candidate_with(FlakyProbabilities()),
                                          default_config))
    determinism = result["operational.prediction_determinism"]
    assert determinism.status == CheckStatus.FAIL
    assert determinism.details["probabilities_checked"]


# ---------------------------------------------- malformed_input_handling
def test_model_that_accepts_malformed_input_fails(candidate_with, default_config):
    result = by_id(run_operational_checks(candidate_with(AcceptsAnything()),
                                          default_config))
    malformed = result["operational.malformed_input_handling"]
    assert malformed.status == CheckStatus.FAIL
    assert malformed.details["cases"]["missing_column_rejected"] is False


def test_model_that_rejects_malformed_input_passes(candidate_with, default_config):
    result = by_id(run_operational_checks(candidate_with(Strict()), default_config))
    malformed = result["operational.malformed_input_handling"]
    assert malformed.status == CheckStatus.PASS
    assert set(malformed.details["cases"]) == {
        "missing_column_rejected", "non_numeric_value_rejected",
        "empty_input_handled"}


# ---------------------------------------------------------- artifact_size
def test_artifact_size_is_scored_from_the_file(tiny_bundle, make_config):
    config = make_config(lambda d: d["checks"]["operational.artifact_size"].update(
        thresholds={"target": 0.0, "pass_at": 0.001, "fail_at": 0.002,
                    "floor": 1.0}))
    c = load_candidate(tiny_bundle)
    result = by_id(run_operational_checks(c, config))["operational.artifact_size"]
    assert result.raw_value == pytest.approx(c.model_size_bytes / 1024 / 1024)
    assert result.status == CheckStatus.FAIL  # a few KB is above 0.002 MB
    assert "KB" in result.message


# ---------------------------------------------------------------- latency
def test_latency_measures_real_time_and_records_the_machine(
        candidate_with, make_config):
    config = make_config(lambda d: d["checks"]["operational.latency"].update(
        thresholds={"target": 1, "pass_at": 5, "fail_at": 5000, "floor": 10000}))
    config.evaluation.latency_repeats = 10
    config.evaluation.latency_warmup_runs = 2
    result = by_id(run_operational_checks(candidate_with(Sleepy()),
                                          config))["operational.latency"]
    assert result.raw_value >= 10                      # each call sleeps 10 ms
    assert result.details["calls"] == 10
    assert result.details["p50_ms"] <= result.details["p95_ms"] <= result.details["max_ms"]
    assert result.status == CheckStatus.WARN           # slower than pass_at=5
    assert "cpu_count" in result.details["machine"]


# ----------------------------------------------------------------- memory
@pytest.mark.parametrize("delta, status", [
    (50.0, CheckStatus.PASS), (800.0, CheckStatus.WARN),
    (3000.0, CheckStatus.FAIL)])
def test_memory_is_scored(candidate_with, default_config, delta, status):
    c = candidate_with(Strict(), memory_delta_mb=delta)
    assert by_id(run_operational_checks(c, default_config))[
        "operational.memory"].status == status


def test_memory_unavailable_when_not_measured(candidate_with, default_config):
    c = candidate_with(Strict(), memory_delta_mb=None)
    result = by_id(run_operational_checks(c, default_config))["operational.memory"]
    assert result.status == CheckStatus.UNAVAILABLE


# ------------------------------------------------------ dependency_match
def versions_candidate(candidate_with, versions):
    return candidate_with(Strict(), metadata=TrainingMetadata(versions=versions))


def test_dependency_match_is_declared_evidence(candidate_with, default_config):
    c = versions_candidate(candidate_with, current_versions())
    result = by_id(run_operational_checks(c, default_config))["operational.dependency_match"]
    assert result.raw_value == 1.0 and result.status == CheckStatus.PASS
    assert result.evidence_source == EvidenceSource.DECLARED


def test_dependency_mismatch_lowers_the_fraction(candidate_with, default_config):
    declared = current_versions() | {"scikit_learn": "0.0.1", "numpy": "0.0.1"}
    c = versions_candidate(candidate_with, declared)
    result = by_id(run_operational_checks(c, default_config))["operational.dependency_match"]
    assert result.raw_value == pytest.approx(1 - 2 / len(declared))
    assert "scikit_learn" in result.message


def test_python_patch_version_difference_is_compatible(candidate_with, default_config):
    major, minor, *_ = current_versions()["python"].split(".")
    declared = {"python": f"{major}.{minor}.0"}
    c = versions_candidate(candidate_with, declared)
    result = by_id(run_operational_checks(c, default_config))["operational.dependency_match"]
    assert result.raw_value == 1.0


def test_dependency_match_unavailable_without_declared_versions(
        candidate_with, default_config):
    for metadata in (None, TrainingMetadata(), TrainingMetadata(versions={})):
        c = candidate_with(Strict(), metadata=metadata)
        result = by_id(run_operational_checks(c, default_config))[
            "operational.dependency_match"]
        assert result.status == CheckStatus.UNAVAILABLE
        assert result.evidence_source == EvidenceSource.UNAVAILABLE


# ------------------------------------------------ generated fixture models
def _fixture_results(name, eval_path, config):
    model = FIXTURES / name / "model.joblib"
    if not model.exists() or not eval_path.exists():
        pytest.skip("fixtures not generated (run scripts/make_fixtures.py)")
    c = load_candidate(CandidateBundle(
        model_path=model, eval_data_path=eval_path,
        metadata_path=FIXTURES / name / "metadata.json"))
    return by_id(run_operational_checks(c, config))


def test_good_fixture_is_operationally_ready(default_config):
    results = _fixture_results("good", DATA / "eval.csv", default_config)
    for check_id in ("operational.model_loadable", "operational.input_schema",
                     "operational.prediction_determinism",
                     "operational.malformed_input_handling",
                     "operational.artifact_size", "operational.dependency_match"):
        assert results[check_id].status == CheckStatus.PASS, check_id


def test_corrupt_fixture_fails_to_load(default_config):
    results = _fixture_results("corrupt", DATA / "eval.csv", default_config)
    assert results["operational.model_loadable"].status == CheckStatus.FAIL
    assert results["operational.input_schema"].status == CheckStatus.UNAVAILABLE


def test_slow_fixture_is_flagged_for_size(default_config):
    results = _fixture_results("slow", DATA / "eval.csv", default_config)
    assert results["operational.artifact_size"].status == CheckStatus.WARN
