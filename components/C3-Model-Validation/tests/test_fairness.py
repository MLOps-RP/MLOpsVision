import json

import numpy as np
import pandas as pd
import pytest
from fairlearn.metrics import (
    MetricFrame, demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference,
)
from sklearn.metrics import f1_score

from mqvg.checks.behaviour import run_behaviour_checks
from mqvg.checks.fairness import GroupStats, get_group_analysis
from mqvg.checks.base import CheckContext
from mqvg.loader import load_candidate
from mqvg.schemas import (
    CandidateBundle, CheckStatus, EvidenceSource, TrainingMetadata,
)

from conftest import DATA, FIXTURES
from stubs import NEG, POS, Scripted

RATIO = "fairness.selection_rate_ratio"
ODDS = "fairness.equalized_odds"
GAP = "behaviour.subgroup_f1_gap"

# ---- hand-computed example: two groups of 8 ------------------------------
# A: tp=3 fn=1 fp=1 tn=3 -> selection .5,   tpr .75, fpr .25, F1 .75
# B: tp=1 fn=1 fp=0 tn=6 -> selection .125, tpr .5,  fpr 0,   F1 2/3
GROUPS = ["A"] * 8 + ["B"] * 8
TRUE = [1, 1, 1, 1, 0, 0, 0, 0] + [1, 1, 0, 0, 0, 0, 0, 0]
PRED = [1, 1, 1, 0, 1, 0, 0, 0] + [1, 0, 0, 0, 0, 0, 0, 0]
HAND = {RATIO: 0.125 / 0.5, ODDS: 0.25, GAP: 0.75 - 2 / 3}


@pytest.fixture
def small_groups(make_config):
    """Default config but allowing groups of 2 rows."""
    return make_config(lambda d: d["evaluation"].update(min_group_size=2))


def build(candidate_with, groups=GROUPS, true=TRUE, pred=PRED, *, extra=None,
          sensitive=("sex",), column="sex", model=None):
    n = len(true)
    frame = pd.DataFrame({
        "age": np.arange(n) + 20, "hours": np.arange(n) + 30,
        "workclass": ["Private"] * n,
        "target": [POS if t else NEG for t in true],
        column: groups,
    })
    for name, values in (extra or {}).items():
        frame[name] = values
    model = model or Scripted([POS if p else NEG for p in pred])
    meta = TrainingMetadata(target_column="target", positive_label=POS,
                            sensitive_columns=list(sensitive))
    return candidate_with(model, eval_df=frame, metadata=meta)


def run(candidate, config):
    return {r.id: r for r in run_behaviour_checks(candidate, config)}


# ------------------------------------------------------------- correctness
@pytest.mark.parametrize("check_id", [RATIO, ODDS, GAP])
def test_matches_hand_computed_values(candidate_with, small_groups, check_id):
    result = run(build(candidate_with), small_groups)[check_id]
    assert result.raw_value == pytest.approx(HAND[check_id])
    assert result.evidence_source == EvidenceSource.MEASURED


def test_group_details_are_reported(candidate_with, small_groups):
    details = run(build(candidate_with), small_groups)[RATIO].details
    a = details["columns"]["sex"]["groups"]["A"]
    assert (a["n"], a["tp"], a["fp"], a["fn"], a["tn"]) == (8, 3, 1, 1, 3)
    assert a["selection_rate"] == .5 and a["tpr"] == .75 and a["fpr"] == .25
    per = details["per_column"]["sex"]
    assert per["parity_difference"] == pytest.approx(0.375)
    assert (per["lowest_group"], per["highest_group"]) == ("B", "A")


def test_message_names_the_groups_and_rates(candidate_with, small_groups):
    message = run(build(candidate_with), small_groups)[RATIO].message
    assert "B 12.5%" in message and "A 50.0%" in message and "'sex'" in message


def test_group_stats_properties():
    s = GroupStats(n=8, tp=3, fp=1, fn=1, tn=3)
    assert (s.selection_rate, s.tpr, s.fpr, s.f1) == (0.5, 0.75, 0.25, 0.75)
    assert GroupStats(n=4, tp=0, fp=0, fn=0, tn=4).tpr is None       # no positives
    assert GroupStats(n=4, tp=0, fp=0, fn=0, tn=4).f1 is None
    assert GroupStats(n=4, tp=2, fp=0, fn=2, tn=0).fpr is None       # no negatives
    assert GroupStats(n=4, tp=0, fp=0, fn=2, tn=2).f1 == 0.0


# --------------------------------------- independent reference: Fairlearn
def test_agrees_with_fairlearn_on_random_data(candidate_with, make_config):
    rng = np.random.default_rng(7)
    groups = rng.choice(["g1", "g2", "g3"], 700, p=[0.5, 0.3, 0.2])
    true = rng.random(700) < np.where(groups == "g1", 0.4, 0.25)
    pred = rng.random(700) < np.where(groups == "g1", 0.45, 0.2)
    config = make_config(lambda d: d["evaluation"].update(min_group_size=10))
    results = run(build(candidate_with, groups, true.astype(int),
                        pred.astype(int)), config)

    kwargs = dict(sensitive_features=groups)
    assert results[RATIO].raw_value == pytest.approx(
        demographic_parity_ratio(true, pred, **kwargs))
    assert results[RATIO].details["per_column"]["sex"][
        "parity_difference"] == pytest.approx(
        demographic_parity_difference(true, pred, **kwargs))
    assert results[ODDS].raw_value == pytest.approx(
        equalized_odds_difference(true, pred, **kwargs))
    f1_gap = MetricFrame(metrics=f1_score, y_true=true, y_pred=pred,
                         sensitive_features=groups).difference()
    assert results[GAP].raw_value == pytest.approx(f1_gap)


# ----------------------------------------------------- small groups
def test_small_groups_are_excluded_and_listed(candidate_with, make_config):
    config = make_config(lambda d: d["evaluation"].update(min_group_size=5))
    groups = GROUPS[:-2] + ["C", "C"]            # C has only 2 rows
    result = run(build(candidate_with, groups), config)[RATIO]
    col = result.details["columns"]["sex"]
    assert col["excluded_small_groups"] == {"C": 2}
    assert sorted(col["groups"]) == ["A", "B"]


def test_tiny_group_cannot_distort_the_result(candidate_with, make_config):
    config = make_config(lambda d: d["evaluation"].update(min_group_size=5))
    base = run(build(candidate_with), config)[RATIO].raw_value
    # replace the last two rows of B by a 2-row group "C"
    groups = GROUPS[:-2] + ["C", "C"]
    noisy = run(build(candidate_with, groups,
                      TRUE[:-2] + [1, 1], PRED[:-2] + [1, 1]), config)[RATIO]
    assert noisy.details["columns"]["sex"]["excluded_small_groups"] == {"C": 2}
    assert noisy.status != CheckStatus.UNAVAILABLE and base == pytest.approx(0.25)


def test_column_with_fewer_than_two_usable_groups_is_unavailable(
        candidate_with, make_config):
    config = make_config(lambda d: d["evaluation"].update(min_group_size=9))
    result = run(build(candidate_with), config)[RATIO]
    assert result.status == CheckStatus.UNAVAILABLE and result.score is None
    assert "fewer than two groups" in result.message


# ------------------------------------------------- unavailable, not passing
def test_no_sensitive_columns_declared(candidate_with, small_groups):
    c = build(candidate_with, sensitive=())
    for check_id in (RATIO, ODDS, GAP):
        result = run(c, small_groups)[check_id]
        assert result.status == CheckStatus.UNAVAILABLE and result.score is None
        assert "No sensitive columns are declared" in result.message


def test_declared_column_missing_from_eval_data(candidate_with, small_groups):
    c = build(candidate_with, sensitive=("sex", "race"))
    result = run(c, small_groups)[RATIO]
    assert result.status != CheckStatus.UNAVAILABLE       # 'sex' still works
    assert "Not evaluated: 'race'" in result.message
    assert result.details["skipped_columns"] == {
        "race": "column is not in the evaluation data"}


def test_all_declared_columns_missing(candidate_with, small_groups):
    c = build(candidate_with, sensitive=("race",))
    result = run(c, small_groups)[ODDS]
    assert result.status == CheckStatus.UNAVAILABLE
    assert "'race'" in result.message


def test_bundle_override_supplies_sensitive_columns(candidate_with, small_groups,
                                                    tmp_path):
    c = build(candidate_with, sensitive=())
    c.bundle = CandidateBundle(model_path=tmp_path / "m", eval_data_path=tmp_path / "e",
                               sensitive_columns=["sex"])
    assert run(c, small_groups)[RATIO].raw_value == pytest.approx(HAND[RATIO])


def test_unavailable_predictions_make_fairness_unavailable(candidate_with,
                                                           small_groups):
    c = build(candidate_with)
    c.metadata = TrainingMetadata(sensitive_columns=["sex"])   # no target column
    result = run(c, small_groups)[RATIO]
    assert result.status == CheckStatus.UNAVAILABLE
    assert "target column is not declared" in result.message


# ------------------------------------------------------------- data quirks
def test_rows_missing_the_group_value_are_left_out_and_counted(
        candidate_with, small_groups):
    groups = list(GROUPS)
    groups[0] = None                       # a row of A disappears
    result = run(build(candidate_with, groups), small_groups)[RATIO]
    col = result.details["columns"]["sex"]
    assert col["rows_missing_group"] == 1
    assert col["groups"]["A"]["n"] == 7


def test_groups_stay_aligned_when_label_rows_are_dropped(candidate_with,
                                                         small_groups):
    c = build(candidate_with)
    c.eval_df["target"] = c.eval_df["target"].astype(object)
    c.eval_df.loc[0, "target"] = None      # row 0 (group A, a true positive) dropped
    col = run(c, small_groups)[RATIO].details["columns"]["sex"]
    assert col["groups"]["A"]["n"] == 7 and col["groups"]["B"]["n"] == 8
    assert col["groups"]["A"]["tp"] == 2   # was 3 before the row was dropped


def test_numeric_group_labels_are_supported(candidate_with, small_groups):
    groups = [1] * 8 + [2] * 8
    result = run(build(candidate_with, groups), small_groups)[RATIO]
    assert result.raw_value == pytest.approx(HAND[RATIO])
    assert sorted(result.details["columns"]["sex"]["groups"]) == ["1", "2"]


def test_group_without_positives_leaves_true_positive_rate_undefined(
        candidate_with, small_groups):
    true = [1, 1, 1, 1, 0, 0, 0, 0] + [0] * 8           # B has no positives
    pred = [1, 1, 1, 0, 1, 0, 0, 0] + [0] * 8
    results = run(build(candidate_with, GROUPS, true, pred), small_groups)
    odds = results[ODDS]
    assert odds.details["per_column"]["sex"]["tpr_difference"] is None
    assert odds.raw_value == pytest.approx(0.25)         # fpr: A .25 vs B 0
    assert results[GAP].status == CheckStatus.UNAVAILABLE  # B's F1 is undefined


def test_equalized_odds_unavailable_when_rates_undefined(candidate_with,
                                                         small_groups):
    true = [1] * 8 + [0] * 8              # A has only positives, B only negatives
    pred = [1] * 8 + [0] * 8
    result = run(build(candidate_with, GROUPS, true, pred), small_groups)[ODDS]
    assert result.status == CheckStatus.UNAVAILABLE and "undefined" in result.message


def test_model_with_no_positive_predictions_gives_flagged_vacuous_ratio(
        candidate_with, small_groups):
    result = run(build(candidate_with, pred=[0] * 16), small_groups)[RATIO]
    assert result.raw_value == 1.0
    assert result.details["per_column"]["sex"]["vacuous_no_positive_predictions"]
    assert "vacuous" in result.message


def test_worst_column_is_reported(candidate_with, small_groups):
    # 'age_band' splits the rows so that selection rates are nearly equal
    band = ["x", "y"] * 8
    c = build(candidate_with, sensitive=("sex", "band"), extra={"band": band})
    result = run(c, small_groups)[RATIO]
    assert result.details["worst_column"] == "sex"
    assert set(result.details["evaluated_columns"]) == {"sex", "band"}


# ---------------------------------------------------- caching and statuses
def test_group_analysis_is_computed_once(candidate_with, small_groups):
    c = build(candidate_with)
    ctx = CheckContext(candidate=c, config=small_groups)
    first, _ = get_group_analysis(ctx)
    run(c, small_groups)
    second, _ = get_group_analysis(ctx)
    assert first is second


def test_statuses_follow_configured_thresholds(candidate_with, make_config):
    def tweak(d):
        d["evaluation"]["min_group_size"] = 2
        d["checks"][RATIO]["thresholds"] = {
            "target": 0.9, "pass_at": 0.2, "fail_at": 0.1, "floor": 0.0}

    result = run(build(candidate_with), make_config(tweak))[RATIO]
    assert result.status == CheckStatus.PASS             # 0.25 >= pass_at 0.2


# --------------------------------------------------- generated fixture models
def _fixture(name, eval_path):
    model = FIXTURES / name / "model.joblib"
    if not model.exists() or not eval_path.exists():
        pytest.skip("fixtures not generated (run scripts/make_fixtures.py)")
    return load_candidate(CandidateBundle(
        model_path=model, eval_data_path=eval_path,
        metadata_path=FIXTURES / name / "metadata.json"))


def test_good_fixture_matches_the_scripts_recorded_fairness(default_config):
    c = _fixture("good", DATA / "eval.csv")
    recorded = json.loads((FIXTURES / "good" / "metadata.json").read_text())[
        "fairness"]
    per = run(c, default_config)[RATIO].details["per_column"]
    assert per["sex"]["parity_difference"] == pytest.approx(
        recorded["demographic_parity_difference_by_sex"], abs=1e-9)
    assert per["sex"]["ratio"] == pytest.approx(
        recorded["selection_rate_ratio_by_sex"], abs=1e-9)
    assert per["race"]["ratio"] == pytest.approx(
        recorded["selection_rate_ratio_by_race"], abs=1e-9)


def test_good_fixture_gets_warnings_not_failures(default_config):
    results = run(_fixture("good", DATA / "eval.csv"), default_config)
    assert results[RATIO].status == CheckStatus.WARN
    assert results[RATIO].raw_value == pytest.approx(0.3113, abs=1e-3)


def test_biased_fixture_fails_every_fairness_check(default_config):
    results = run(_fixture("biased", DATA / "eval.csv"), default_config)
    for check_id in (RATIO, ODDS, GAP):
        assert results[check_id].status == CheckStatus.FAIL, check_id
    assert results[RATIO].raw_value < 0.01


def test_no_sensitive_fixture_is_unavailable(default_config):
    c = _fixture("good", FIXTURES / "good_no_sensitive" / "eval.csv")
    for check_id in (RATIO, ODDS, GAP):
        result = run(c, default_config)[check_id]
        assert result.status == CheckStatus.UNAVAILABLE and result.score is None
