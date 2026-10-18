import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from mqvg.checks.behaviour import run_behaviour_checks
from mqvg.checks.calibration import reliability_summary
from mqvg.loader import load_candidate
from mqvg.schemas import (
    CandidateBundle, CheckStatus, EvidenceSource, TrainingMetadata,
)

from conftest import DATA, FIXTURES
from stubs import NEG, POS, DecisionOnly, Scripted, ScriptedNoProba

ECE = "calibration.ece"

# Hand-computed with 2 bins.
#   bin [0, .5): scores .1 .2 .3 .4 .1 (mean .22), labels 1 1 0 0 0 (rate .4) -> gap .18
#   bin [.5, 1]: scores .6 .7 .8 .9 .9 (mean .78), labels 1 0 1 1 0 (rate .6) -> gap .18
#   ECE = .5*.18 + .5*.18 = .18
#   Brier = (.81+.64+.09+.16+.01 + .16+.49+.04+.01+.81) / 10 = .322
SCORES = [0.1, 0.2, 0.3, 0.4, 0.1, 0.6, 0.7, 0.8, 0.9, 0.9]
LABELS = [1, 1, 0, 0, 0, 1, 0, 1, 1, 0]


@pytest.fixture
def two_bins(make_config):
    return make_config(lambda d: d["evaluation"].update(ece_bins=2))


def build(candidate_with, scores=SCORES, labels=LABELS, model=None):
    n = len(labels)
    frame = pd.DataFrame({
        "age": np.arange(n) + 20, "hours": np.arange(n) + 30,
        "workclass": ["Private"] * n,
        "target": [POS if t else NEG for t in labels],
    })
    model = model or Scripted([POS if s >= 0.5 else NEG for s in scores], scores)
    meta = TrainingMetadata(target_column="target", positive_label=POS)
    return candidate_with(model, eval_df=frame, metadata=meta)


def ece(candidate, config):
    return {r.id: r for r in run_behaviour_checks(candidate, config)}[ECE]


# ----------------------------------------------------------- correctness
def test_ece_matches_hand_computed_value(candidate_with, two_bins):
    result = ece(build(candidate_with), two_bins)
    assert result.raw_value == pytest.approx(0.18)
    assert result.evidence_source == EvidenceSource.MEASURED


def test_details_hold_the_reliability_table_and_brier_score(candidate_with, two_bins):
    details = ece(build(candidate_with), two_bins).details
    assert details["brier_score"] == pytest.approx(0.322)
    assert details["max_calibration_error"] == pytest.approx(0.18)
    low, high = details["reliability"]
    assert (low["n"], high["n"]) == (5, 5)
    assert low["mean_score"] == pytest.approx(0.22)
    assert low["observed_rate"] == pytest.approx(0.4)
    assert high["mean_score"] == pytest.approx(0.78)
    assert details["mean_score"] == pytest.approx(0.5)
    assert details["observed_positive_rate"] == pytest.approx(0.5)


def test_message_reports_ece_and_direction(candidate_with, two_bins):
    message = ece(build(candidate_with), two_bins).message
    assert "0.180" in message and "2 bins" in message


def test_boundary_scores_land_in_the_end_bins(candidate_with, make_config):
    summary = reliability_summary(np.array([False, True]), np.array([0.0, 1.0]), 10)
    assert summary["reliability"][0]["n"] == 1 and summary["reliability"][9]["n"] == 1
    assert summary["ece"] == pytest.approx(0.0)         # 0.0 -> 0, 1.0 -> 1: perfect


def test_empty_bins_are_listed_with_no_values():
    summary = reliability_summary(np.array([True, False]), np.array([0.05, 0.95]), 10)
    empty = [row for row in summary["reliability"] if row["n"] == 0]
    assert len(empty) == 8 and all(r["gap"] is None for r in empty)


def test_number_of_bins_comes_from_the_config(candidate_with, make_config):
    config = make_config(lambda d: d["evaluation"].update(ece_bins=5))
    details = ece(build(candidate_with), config).details
    assert details["bins"] == 5 and len(details["reliability"]) == 5


def test_perfectly_calibrated_scores_have_zero_ece():
    summary = reliability_summary(
        np.array([True] * 8 + [False] * 2 + [True] * 2 + [False] * 8),
        np.array([0.8] * 10 + [0.2] * 10), 10)
    assert summary["ece"] == pytest.approx(0.0)


# ------------------------------------------- independent reference: sklearn
def test_agrees_with_scikit_learn_on_random_data():
    rng = np.random.default_rng(3)
    scores = rng.random(2000)
    labels = rng.random(2000) < scores ** 2          # deliberately miscalibrated
    summary = reliability_summary(labels, scores, 10)

    true_rate, mean_score = calibration_curve(labels, scores, n_bins=10,
                                              strategy="uniform")
    counts, _ = np.histogram(scores, bins=np.linspace(0, 1, 11))
    reference = float(np.sum(counts[counts > 0] / len(scores)
                             * np.abs(true_rate - mean_score)))
    assert summary["ece"] == pytest.approx(reference)
    assert summary["brier_score"] == pytest.approx(brier_score_loss(labels, scores))


# -------------------------------------------------------------- statuses
def test_status_follows_configured_thresholds(candidate_with, make_config):
    def tweak(d):
        d["evaluation"]["ece_bins"] = 2
        d["checks"][ECE]["thresholds"] = {
            "target": 0.0, "pass_at": 0.1, "fail_at": 0.3, "floor": 0.5}

    result = ece(build(candidate_with), make_config(tweak))
    assert result.status == CheckStatus.WARN             # 0.18 between 0.1 and 0.3


def test_a_single_class_evaluation_set_still_has_a_calibration_error(
        candidate_with, two_bins):
    result = ece(build(candidate_with, labels=[1] * 10), two_bins)
    assert result.status != CheckStatus.UNAVAILABLE and result.raw_value > 0


# ----------------------------------------------- unavailable, not passing
def test_model_without_scores_is_unavailable(candidate_with, two_bins):
    model = ScriptedNoProba([POS if s >= 0.5 else NEG for s in SCORES])
    result = ece(build(candidate_with, model=model), two_bins)
    assert result.status == CheckStatus.UNAVAILABLE and result.score is None
    assert "no probability scores" in result.message


def test_decision_function_scores_are_not_probabilities(candidate_with, two_bins):
    model = DecisionOnly([POS if s >= 0.5 else NEG for s in SCORES], SCORES)
    result = ece(build(candidate_with, model=model), two_bins)
    assert result.status == CheckStatus.UNAVAILABLE
    assert "not probabilities" in result.message


@pytest.mark.parametrize("bad", [1.2, -0.1, np.nan])
def test_invalid_probabilities_are_unavailable(candidate_with, two_bins, bad):
    scores = SCORES[:-1] + [bad]
    result = ece(build(candidate_with, scores=scores), two_bins)
    assert result.status == CheckStatus.UNAVAILABLE
    assert "[0, 1]" in result.message


def test_no_target_column_is_unavailable(candidate_with, two_bins):
    c = build(candidate_with)
    c.metadata = TrainingMetadata()
    assert ece(c, two_bins).status == CheckStatus.UNAVAILABLE


# ------------------------------------------------ generated fixture models
def _fixture(name):
    model = FIXTURES / name / "model.joblib"
    if not model.exists() or not (DATA / "eval.csv").exists():
        pytest.skip("fixtures not generated (run scripts/make_fixtures.py)")
    return load_candidate(CandidateBundle(
        model_path=model, eval_data_path=DATA / "eval.csv",
        metadata_path=FIXTURES / name / "metadata.json"))


def test_good_fixture_is_well_calibrated(default_config):
    result = ece(_fixture("good"), default_config)
    assert result.status == CheckStatus.PASS
    assert result.raw_value == pytest.approx(0.011, abs=1e-3)
    assert "close to the observed rate" in result.message


def test_miscalibrated_fixture_fails_but_ranks_well(default_config):
    c = _fixture("miscalibrated")
    results = {r.id: r for r in run_behaviour_checks(c, default_config)}
    assert results[ECE].status == CheckStatus.FAIL
    assert results[ECE].raw_value == pytest.approx(0.240, abs=1e-3)
    assert "over-confident" in results[ECE].message
    assert results["behaviour.roc_auc"].status == CheckStatus.PASS


def test_overfit_fixture_is_caught_by_calibration(default_config):
    """Overfitting that F1 misses shows up as poor calibration."""
    c = _fixture("overfit")
    results = {r.id: r for r in run_behaviour_checks(c, default_config)}
    assert results["behaviour.f1"].status == CheckStatus.PASS
    assert results[ECE].status == CheckStatus.FAIL
