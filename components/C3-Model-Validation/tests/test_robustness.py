import numpy as np
import pandas as pd
import pytest

from mqvg.checks.behaviour import run_behaviour_checks
from mqvg.loader import load_candidate
from mqvg.schemas import (
    CandidateBundle, CheckStatus, EvidenceSource, TrainingMetadata,
)

from conftest import DATA, FIXTURES
from stubs import NEG, POS

ROBUSTNESS = "robustness.noise_degradation"


class AgeThreshold:
    """Predicts positive when age exceeds 50: sensitive to noise on age."""
    feature_names_in_ = np.array(["age", "workclass"])

    def predict(self, X):
        return np.where(X["age"].to_numpy(dtype=float) > 50, POS, NEG)


class Constant:
    """Never looks at the numeric features, so noise cannot matter."""
    feature_names_in_ = np.array(["age", "workclass"])

    def predict(self, X):
        return np.full(len(X), POS)


class CategoricalOnly:
    feature_names_in_ = np.array(["workclass"])

    def predict(self, X):
        return np.where(X["workclass"] == "Gov", POS, NEG)


@pytest.fixture
def enabled(make_config):
    def build(**evaluation):
        def tweak(d):
            d["checks"][ROBUSTNESS]["enabled"] = True
            d["evaluation"].update(evaluation)
        return make_config(tweak)
    return build


def frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 80, n)
    return pd.DataFrame({"age": age,
                         "workclass": rng.choice(["Gov", "Private"], n),
                         "target": np.where(age > 50, POS, NEG)})


def result(candidate_with, model, config, eval_df=None):
    meta = TrainingMetadata(target_column="target", positive_label=POS)
    c = candidate_with(model, eval_df=frame() if eval_df is None else eval_df,
                       metadata=meta)
    return {r.id: r for r in run_behaviour_checks(c, config)}[ROBUSTNESS]


def test_disabled_by_default(candidate_with, default_config):
    meta = TrainingMetadata(target_column="target", positive_label=POS)
    c = candidate_with(AgeThreshold(), eval_df=frame(), metadata=meta)
    assert ROBUSTNESS not in {r.id for r in run_behaviour_checks(c, default_config)}


def test_noise_sensitive_model_loses_f1(candidate_with, enabled):
    r = result(candidate_with, AgeThreshold(), enabled(robustness_noise_scale=0.3))
    assert r.details["clean_f1"] == pytest.approx(1.0)
    assert 0 < r.raw_value < 1
    assert r.details["mean_noisy_f1"] < 1.0
    assert r.evidence_source == EvidenceSource.MEASURED


def test_more_noise_hurts_more(candidate_with, enabled):
    small = result(candidate_with, AgeThreshold(), enabled(robustness_noise_scale=0.05))
    large = result(candidate_with, AgeThreshold(), enabled(robustness_noise_scale=0.5))
    assert large.raw_value > small.raw_value


def test_noise_insensitive_model_does_not_drop(candidate_with, enabled):
    # predicts everything positive: F1 is the same however the age is perturbed
    r = result(candidate_with, Constant(), enabled())
    assert r.raw_value == pytest.approx(0.0)
    assert r.status == CheckStatus.PASS


def test_result_is_reproducible_and_depends_on_the_seed(candidate_with, enabled):
    a = result(candidate_with, AgeThreshold(), enabled(robustness_noise_scale=0.3))
    b = result(candidate_with, AgeThreshold(), enabled(robustness_noise_scale=0.3))
    other = result(candidate_with, AgeThreshold(),
                   enabled(robustness_noise_scale=0.3, random_seed=7))
    assert a.raw_value == b.raw_value
    assert a.details["noisy_f1_per_repeat"] == b.details["noisy_f1_per_repeat"]
    assert other.details["noisy_f1_per_repeat"] != a.details["noisy_f1_per_repeat"]


def test_repeats_come_from_the_config(candidate_with, enabled):
    r = result(candidate_with, AgeThreshold(), enabled(robustness_repeats=3))
    assert len(r.details["noisy_f1_per_repeat"]) == 3


def test_candidate_data_is_not_modified(candidate_with, enabled):
    data = frame()
    before = data.copy()
    result(candidate_with, AgeThreshold(), enabled(robustness_noise_scale=0.5), data)
    pd.testing.assert_frame_equal(data, before)


def test_only_numeric_features_are_perturbed(candidate_with, enabled):
    details = result(candidate_with, AgeThreshold(), enabled()).details
    assert details["perturbed_columns"] == ["age"]
    assert details["not_perturbed_columns"] == ["workclass"]


def test_status_follows_configured_thresholds(candidate_with, make_config):
    def tweak(d):
        d["checks"][ROBUSTNESS]["enabled"] = True
        d["evaluation"]["robustness_noise_scale"] = 0.3
        d["checks"][ROBUSTNESS]["thresholds"] = {
            "target": 0.0, "pass_at": 0.0001, "fail_at": 0.9, "floor": 1.0}

    r = result(candidate_with, AgeThreshold(), make_config(tweak))
    assert r.status == CheckStatus.WARN


# ----------------------------------------------- not applicable / unavailable
def test_no_numeric_features_is_not_applicable(candidate_with, enabled):
    r = result(candidate_with, CategoricalOnly(), enabled())
    assert r.status == CheckStatus.NOT_APPLICABLE and r.score is None


def test_constant_numeric_feature_cannot_be_perturbed(candidate_with, enabled):
    data = frame().assign(age=40)
    data["target"] = [POS, NEG] * 150
    r = result(candidate_with, Constant(), enabled(), data)   # F1 > 0, age never varies
    assert r.status == CheckStatus.NOT_APPLICABLE


def test_zero_clean_f1_is_unavailable(candidate_with, enabled):
    data = frame().assign(target=POS)        # model predicts by age; truth is all POS
    data["age"] = 10                          # model predicts NEG everywhere -> F1 0
    r = result(candidate_with, AgeThreshold(), enabled(), data)
    assert r.status == CheckStatus.UNAVAILABLE and r.score is None


def test_missing_target_is_unavailable(candidate_with, enabled):
    c = candidate_with(AgeThreshold(), eval_df=frame(), metadata=TrainingMetadata())
    r = {x.id: x for x in run_behaviour_checks(c, enabled())}[ROBUSTNESS]
    assert r.status == CheckStatus.UNAVAILABLE


# ------------------------------------------------ generated fixture models
def test_good_fixture_is_robust_to_mild_noise(enabled):
    model = FIXTURES / "good" / "model.joblib"
    if not model.exists() or not (DATA / "eval.csv").exists():
        pytest.skip("fixtures not generated (run scripts/make_fixtures.py)")
    c = load_candidate(CandidateBundle(
        model_path=model, eval_data_path=DATA / "eval.csv",
        metadata_path=FIXTURES / "good" / "metadata.json"))
    r = {x.id: x for x in run_behaviour_checks(c, enabled())}[ROBUSTNESS]
    assert r.status == CheckStatus.PASS
    assert 0 <= r.raw_value < 0.05
