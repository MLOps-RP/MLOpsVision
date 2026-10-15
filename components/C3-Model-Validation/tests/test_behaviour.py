import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from mqvg.catalogue import CATALOGUE
from mqvg.checks.behaviour import (
    BEHAVIOUR_CHECKS, classification_metrics, run_behaviour_checks,
)
from mqvg.checks.predictions import get_predictions
from mqvg.loader import LoadedCandidate, load_candidate
from mqvg.schemas import (
    CandidateBundle, CheckStatus, Dimension, EvidenceSource, TaskType,
    TrainingMetadata,
)

from conftest import DATA, FEATURES, FIXTURES
from stubs import DecisionOnly, Scripted, ScriptedNoProba

NEG, POS = "<=50K", ">50K"
IMPLEMENTED = ["behaviour.f1", "behaviour.roc_auc", "behaviour.precision",
               "behaviour.recall"]

# Hand-computed example (10 rows). 1 = positive.
TRUE = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
PRED = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
SCORE = [0.9, 0.8, 0.7, 0.4, 0.6, 0.55, 0.3, 0.2, 0.1, 0.05]
# tp=3 fp=2 fn=1 tn=4 -> precision 3/5, recall 3/4, F1 = 2*.6*.75/1.35
EXPECTED = {"precision": 0.6, "recall": 0.75, "f1": 2 * 0.6 * 0.75 / 1.35,
            # 22 of the 4*6=24 positive/negative pairs are ranked correctly
            "roc_auc": 22 / 24}


def frame(true=TRUE, pos=POS, neg=NEG):
    n = len(true)
    return pd.DataFrame({
        "age": np.arange(n) + 20, "hours": np.arange(n) + 30,
        "workclass": ["Private"] * n,
        "target": [pos if t else neg for t in true],
    })


def make(candidate_with, model=None, true=TRUE, *, target="target", pos=POS,
         task=None, **overrides):
    model = model or Scripted([POS if p else NEG for p in PRED], SCORE)
    meta = TrainingMetadata(target_column=target, positive_label=pos,
                            task_type=task)
    c = candidate_with(model, eval_df=frame(true), metadata=meta, **overrides)
    return c


def run(candidate, config):
    return {r.id: r for r in run_behaviour_checks(candidate, config)}


# ----------------------------------------------------------- correctness
@pytest.mark.parametrize("check_id, key", [
    ("behaviour.precision", "precision"), ("behaviour.recall", "recall"),
    ("behaviour.f1", "f1"), ("behaviour.roc_auc", "roc_auc")])
def test_metrics_match_hand_computed_values(candidate_with, default_config,
                                            check_id, key):
    result = run(make(candidate_with), default_config)[check_id]
    assert result.raw_value == pytest.approx(EXPECTED[key])
    assert result.evidence_source == EvidenceSource.MEASURED


def test_metrics_agree_with_scikit_learn(candidate_with, default_config):
    results = run(make(candidate_with), default_config)
    assert results["behaviour.precision"].raw_value == pytest.approx(
        precision_score(TRUE, PRED))
    assert results["behaviour.recall"].raw_value == pytest.approx(
        recall_score(TRUE, PRED))
    assert results["behaviour.f1"].raw_value == pytest.approx(f1_score(TRUE, PRED))
    assert results["behaviour.roc_auc"].raw_value == pytest.approx(
        roc_auc_score(TRUE, SCORE))


def test_details_carry_confusion_matrix_and_baseline(candidate_with, default_config):
    details = run(make(candidate_with), default_config)["behaviour.f1"].details
    assert details["confusion"] == {"tp": 3, "fp": 2, "fn": 1, "tn": 4}
    assert details["n_rows"] == 10
    assert details["prevalence"] == pytest.approx(0.4)
    assert details["baseline_always_positive"]["f1"] == pytest.approx(0.8 / 1.4)
    assert details["positive_label"] == POS


def test_status_follows_the_configured_thresholds(candidate_with, make_config):
    def set_f1(d):
        d["checks"]["behaviour.f1"]["thresholds"] = {
            "target": 0.9, "pass_at": 0.8, "fail_at": 0.5, "floor": 0.0}

    result = run(make(candidate_with), make_config(set_f1))["behaviour.f1"]
    assert result.status == CheckStatus.WARN       # F1 0.667 is between 0.5 and 0.8
    assert result.score == pytest.approx(EXPECTED["f1"] / 0.9 * 100)


def test_message_mentions_the_baseline_except_for_recall(candidate_with,
                                                          default_config):
    results = run(make(candidate_with), default_config)
    assert "baseline" in results["behaviour.f1"].message
    assert "baseline" not in results["behaviour.recall"].message


# ---------------------------------------------------------- label handling
def test_positive_label_can_be_the_models_first_class(candidate_with,
                                                      default_config):
    # classes_ = [POS, NEG]: the positive class is column 0 of predict_proba,
    # so that column must hold P(positive) = SCORE.
    model = Scripted([POS if p else NEG for p in PRED],
                     [1 - s for s in SCORE], classes=(POS, NEG))
    results = run(make(candidate_with, model), default_config)
    assert results["behaviour.f1"].raw_value == pytest.approx(EXPECTED["f1"])
    assert results["behaviour.roc_auc"].raw_value == pytest.approx(EXPECTED["roc_auc"])


def test_numeric_labels_work(candidate_with, default_config):
    model = Scripted(PRED, SCORE, classes=(0, 1))
    c = candidate_with(model, eval_df=frame().assign(target=TRUE),
                       metadata=TrainingMetadata(target_column="target",
                                                 positive_label=1))
    assert run(c, default_config)["behaviour.f1"].raw_value == pytest.approx(
        EXPECTED["f1"])


def test_string_and_integer_labels_are_compared_consistently(
        candidate_with, default_config):
    # metadata says "1" (text), the CSV and model use the integer 1
    model = Scripted(PRED, SCORE, classes=(0, 1))
    c = candidate_with(model, eval_df=frame().assign(target=TRUE),
                       metadata=TrainingMetadata(target_column="target",
                                                 positive_label="1"))
    assert run(c, default_config)["behaviour.f1"].raw_value == pytest.approx(
        EXPECTED["f1"])


def test_undeclared_positive_label_is_inferred_and_flagged(
        candidate_with, default_config):
    c = make(candidate_with, pos=None)
    f1 = run(c, default_config)["behaviour.f1"]
    assert f1.raw_value == pytest.approx(EXPECTED["f1"])
    assert "inferred" in f1.details["positive_label_source"]


def test_undeclared_label_cannot_be_inferred_without_classes(
        candidate_with, default_config):
    model = Scripted([POS if p else NEG for p in PRED], SCORE)
    del model.classes_
    result = run(make(candidate_with, model, pos=None), default_config)["behaviour.f1"]
    assert result.status == CheckStatus.UNAVAILABLE
    assert "positive label" in result.message


def test_declared_label_absent_from_data_is_unavailable_with_values(
        candidate_with, default_config):
    result = run(make(candidate_with, pos=">50k"), default_config)["behaviour.f1"]
    assert result.status == CheckStatus.UNAVAILABLE
    assert POS in result.message and "does not appear" in result.message


def test_bundle_overrides_metadata(candidate_with, default_config, tmp_path):
    bundle = CandidateBundle(model_path=tmp_path / "m", eval_data_path=tmp_path / "e",
                             target_column="target", positive_label=POS)
    c = candidate_with(Scripted([POS if p else NEG for p in PRED], SCORE),
                       eval_df=frame(), bundle=bundle)       # no metadata at all
    assert run(c, default_config)["behaviour.f1"].raw_value == pytest.approx(
        EXPECTED["f1"])


# ------------------------------------------------ unavailable, never a pass
def test_undeclared_target_makes_all_metrics_unavailable(candidate_with,
                                                         default_config):
    c = candidate_with(Scripted([POS] * 10), eval_df=frame())   # no metadata
    for check_id in IMPLEMENTED:
        result = run(c, default_config)[check_id]
        assert result.status == CheckStatus.UNAVAILABLE and result.score is None
        assert "target column is not declared" in result.message


def test_target_column_missing_from_eval_data(candidate_with, default_config):
    result = run(make(candidate_with, target="income"), default_config)["behaviour.f1"]
    assert result.status == CheckStatus.UNAVAILABLE
    assert "'income'" in result.message


def test_model_not_loaded(candidate_with, default_config):
    c = make(candidate_with, model=None)
    c.model, c.model_error = None, "cannot load model: EOFError"
    result = run(c, default_config)["behaviour.f1"]
    assert result.status == CheckStatus.UNAVAILABLE
    assert "EOFError" in result.message


def test_one_class_only_is_unavailable(candidate_with, default_config):
    # all positive: the metric itself is undefined
    result = run(make(candidate_with, true=[1] * 10), default_config)["behaviour.f1"]
    assert result.status == CheckStatus.UNAVAILABLE and "undefined" in result.message
    # all negative: indistinguishable from a misspelled label, and says both
    result = run(make(candidate_with, true=[0] * 10), default_config)["behaviour.f1"]
    assert result.status == CheckStatus.UNAVAILABLE
    assert "misspelled" in result.message and "no positive examples" in result.message


def test_multiclass_target_is_unsupported(candidate_with, default_config):
    c = make(candidate_with)
    c.eval_df["target"] = ["a", "b", "c"] * 3 + ["a"]
    result = run(c, default_config)["behaviour.f1"]
    assert result.status == CheckStatus.UNAVAILABLE and "multiclass" in result.message


@pytest.mark.parametrize("task", [TaskType.MULTICLASS_CLASSIFICATION,
                                  TaskType.REGRESSION])
def test_declared_non_binary_task_is_unsupported(candidate_with, default_config, task):
    result = run(make(candidate_with, task=task), default_config)["behaviour.f1"]
    assert result.status == CheckStatus.UNAVAILABLE
    assert "not supported" in result.message


def test_model_that_fails_to_predict_is_unavailable(candidate_with, default_config):
    class Raises(Scripted):
        def predict(self, X):
            raise RuntimeError("boom")

    result = run(make(candidate_with, Raises([0] * 10)), default_config)["behaviour.f1"]
    assert result.status == CheckStatus.UNAVAILABLE and "boom" in result.message


def test_wrong_prediction_count_is_unavailable(candidate_with, default_config):
    class Short(Scripted):
        def predict(self, X):
            return np.array([POS])

    result = run(make(candidate_with, Short([0] * 10)), default_config)["behaviour.f1"]
    assert result.status == CheckStatus.UNAVAILABLE and "1 predictions" in result.message


# -------------------------------------------------------- missing values
def test_rows_with_missing_labels_are_dropped_and_counted(candidate_with,
                                                           default_config):
    c = make(candidate_with)
    c.eval_df["target"] = c.eval_df["target"].astype(object)
    c.eval_df.loc[4, "target"] = None            # a tn/fp row disappears
    f1 = run(c, default_config)["behaviour.f1"]
    assert f1.details["n_rows"] == 9
    assert f1.details["dropped_rows_missing_label"] == 1
    # row 4 was a false positive, so precision rises to 3/4
    assert run(c, default_config)["behaviour.precision"].raw_value == pytest.approx(0.75)


def test_all_labels_missing_is_unavailable(candidate_with, default_config):
    c = make(candidate_with)
    c.eval_df["target"] = np.nan
    assert run(c, default_config)["behaviour.f1"].status == CheckStatus.UNAVAILABLE


# ------------------------------------------------ precision / scores edge
def test_no_positive_predictions_gives_flagged_zero_precision(candidate_with,
                                                               default_config):
    model = Scripted([NEG] * 10, SCORE)
    results = run(make(candidate_with, model), default_config)
    precision = results["behaviour.precision"]
    assert precision.raw_value == 0.0 and precision.details["precision_undefined"]
    assert "undefined" in precision.message
    assert results["behaviour.recall"].raw_value == 0.0
    assert results["behaviour.f1"].raw_value == 0.0


def test_model_without_scores_still_gets_label_metrics(candidate_with, default_config):
    model = ScriptedNoProba([POS if p else NEG for p in PRED])
    results = run(make(candidate_with, model), default_config)
    assert results["behaviour.f1"].raw_value == pytest.approx(EXPECTED["f1"])
    auc = results["behaviour.roc_auc"]
    assert auc.status == CheckStatus.UNAVAILABLE and "no probability" in auc.message


def test_decision_function_is_used_when_there_is_no_probability(candidate_with,
                                                                 default_config):
    model = DecisionOnly([POS if p else NEG for p in PRED], SCORE)
    auc = run(make(candidate_with, model), default_config)["behaviour.roc_auc"]
    assert auc.raw_value == pytest.approx(EXPECTED["roc_auc"])
    assert auc.details["score_source"] == "decision_function"


# ----------------------------------------------------- shared predictions
def test_predictions_are_computed_once_and_shared(candidate_with, default_config):
    model = Scripted([POS if p else NEG for p in PRED], SCORE)
    c = make(candidate_with, model)
    run(c, default_config)
    get_predictions(c)
    assert model.predict_calls == 1


def test_classification_metrics_helper_is_consistent(candidate_with):
    c = make(candidate_with)
    predictions, _ = get_predictions(c)
    metrics = classification_metrics(predictions)
    assert sum(metrics["confusion"].values()) == 10


# ---------------------------------------------------------- run / registry
def test_every_behaviour_check_is_registered():
    expected = [s.id for s in CATALOGUE if s.dimension == Dimension.MODEL_BEHAVIOUR]
    assert sorted(BEHAVIOUR_CHECKS) == sorted(expected)


def test_disabled_behaviour_check_is_not_run(candidate_with, make_config):
    config = make_config(lambda d: d["checks"]["behaviour.recall"].update(
        enabled=False))
    assert "behaviour.recall" not in run(make(candidate_with), config)


# ------------------------------------------------ generated fixture models
def _fixture_candidate(name, eval_path):
    model = FIXTURES / name / "model.joblib"
    if not model.exists() or not eval_path.exists():
        pytest.skip("fixtures not generated (run scripts/make_fixtures.py)")
    return load_candidate(CandidateBundle(
        model_path=model, eval_data_path=eval_path,
        metadata_path=FIXTURES / name / "metadata.json"))


def test_gate_measurement_matches_the_fixture_scripts_record(default_config):
    """Two independent code paths must agree on the good model."""
    import json
    c = _fixture_candidate("good", DATA / "eval.csv")
    recorded = json.loads((FIXTURES / "good" / "metadata.json").read_text())[
        "evaluation_metrics"]
    results = run(c, default_config)
    for key in ("f1", "precision", "recall", "roc_auc"):
        assert results[f"behaviour.{key}"].raw_value == pytest.approx(
            recorded[key], abs=1e-9)


def test_leaky_model_looks_excellent_on_behaviour_alone(default_config):
    """Why leakage needs its own check: performance cannot reveal it."""
    c = _fixture_candidate("leaky", FIXTURES / "leaky" / "eval.csv")
    results = run(c, default_config)
    assert results["behaviour.f1"].status == CheckStatus.PASS
    assert results["behaviour.f1"].raw_value > 0.9
    assert results["behaviour.roc_auc"].raw_value > 0.99


def test_incomplete_metadata_fixture_still_has_the_target(default_config):
    c = _fixture_candidate("good", DATA / "eval.csv")
    meta = TrainingMetadata.model_validate_json(
        (FIXTURES / "good_incomplete_metadata" / "metadata.json").read_text())
    c.metadata = meta
    assert run(c, default_config)["behaviour.f1"].status == CheckStatus.PASS
