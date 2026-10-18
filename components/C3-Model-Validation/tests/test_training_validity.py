import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest

from mqvg.catalogue import CATALOGUE
from mqvg.checks.training_validity import (
    TRAINING_VALIDITY_CHECKS, run_training_validity_checks, single_feature_auc,
)
from mqvg.hashing import read_csv_as_text, row_hashes
from mqvg.loader import load_candidate
from mqvg.schemas import (
    CandidateBundle, CheckStatus, Dimension, EvidenceSource, TrainingMetadata,
)

from conftest import DATA, FIXTURES, ROOT
from stubs import NEG, POS, Scripted

COMPLETE = "training_validity.metadata_completeness"
REPRO = "training_validity.reproducibility_info"
GAP = "training_validity.overfit_gap"
OVERLAP = "training_validity.train_eval_overlap"
LEAK = "leakage.single_feature_predictiveness"


def run(candidate, config):
    return {r.id: r for r in run_training_validity_checks(candidate, config)}


class Features:
    """A model that only declares which feature columns it expects."""

    def __init__(self, columns):
        self.feature_names_in_ = np.array(columns)

    def predict(self, X):
        return np.full(len(X), NEG)


# ----------------------------------------------------- metadata presence
FULL = dict(model_type="LR", task_type="binary_classification",
            target_column="target", positive_label=POS,
            sensitive_columns=["sex"], feature_columns=["age"],
            hyperparameters={"C": 1}, training_metrics={"accuracy": 0.9},
            train_rows=100, dataset="demo",
            random_state=1, split={"seed": 1}, versions={"python": "3.12"},
            train_data_sha256="abc", code={"commit": "deadbeef"})


def test_complete_metadata_scores_one(candidate_with, default_config):
    c = candidate_with(Scripted([0] * 10), metadata=TrainingMetadata(**FULL))
    results = run(c, default_config)
    for check_id in (COMPLETE, REPRO):
        assert results[check_id].raw_value == 1.0
        assert results[check_id].status == CheckStatus.PASS
        assert results[check_id].evidence_source == EvidenceSource.MEASURED


def test_missing_fields_are_counted_and_named(candidate_with, default_config):
    data = {k: v for k, v in FULL.items()
            if k not in ("hyperparameters", "training_metrics", "split")}
    results = run(candidate_with(Scripted([0] * 10),
                                 metadata=TrainingMetadata(**data)), default_config)
    assert results[COMPLETE].raw_value == pytest.approx(8 / 10)
    assert results[COMPLETE].details["missing"] == ["hyperparameters",
                                                    "training_metrics"]
    assert results[REPRO].raw_value == pytest.approx(4 / 5)   # split.seed missing
    assert "split.seed" in results[REPRO].message


def test_empty_values_count_as_missing(candidate_with, default_config):
    data = dict(FULL, sensitive_columns=[], hyperparameters={}, dataset="")
    result = run(candidate_with(Scripted([0] * 10),
                                metadata=TrainingMetadata(**data)),
                 default_config)[COMPLETE]
    assert result.raw_value == pytest.approx(7 / 10)


def test_falsy_but_present_values_are_not_missing(candidate_with, default_config):
    data = dict(FULL, random_state=0, positive_label=False, train_rows=0)
    results = run(candidate_with(Scripted([0] * 10),
                                 metadata=TrainingMetadata(**data)), default_config)
    assert results[COMPLETE].raw_value == 1.0 and results[REPRO].raw_value == 1.0


def test_no_metadata_scores_zero_and_fails(candidate_with, default_config):
    results = run(candidate_with(Scripted([0] * 10)), default_config)
    for check_id in (COMPLETE, REPRO):
        assert results[check_id].raw_value == 0.0
        assert results[check_id].status == CheckStatus.FAIL
        assert "no training metadata was provided" in results[check_id].message


def test_unparseable_metadata_explains_why(candidate_with, default_config):
    c = candidate_with(Scripted([0] * 10), metadata=None,
                       metadata_provided=True,
                       metadata_error="cannot parse metadata: JSONDecodeError")
    result = run(c, default_config)[COMPLETE]
    assert result.status == CheckStatus.FAIL
    assert "JSONDecodeError" in result.message


def test_required_field_list_comes_from_the_config(candidate_with, make_config):
    config = make_config(lambda d: d["metadata"].update(required_fields=["dataset"]))
    result = run(candidate_with(Scripted([0] * 10),
                                metadata=TrainingMetadata(dataset="x")),
                 config)[COMPLETE]
    assert result.raw_value == 1.0


# ----------------------------------------------------------- overfit gap
# Hand example from the behaviour tests: tp=3 fp=2 fn=1 tn=4 -> eval accuracy 0.7,
# eval F1 = 2*3/(6+2+1) = 2/3.
TRUE = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
PRED = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]


def gap_candidate(candidate_with, training_metrics, **meta):
    frame = pd.DataFrame({"age": np.arange(10) + 20, "hours": np.arange(10) + 30,
                          "workclass": ["Private"] * 10,
                          "target": [POS if t else NEG for t in TRUE]})
    metadata = TrainingMetadata(target_column="target", positive_label=POS,
                                training_metrics=training_metrics, **meta)
    return candidate_with(Scripted([POS if p else NEG for p in PRED]),
                          eval_df=frame, metadata=metadata)


def test_gap_is_declared_train_minus_measured_eval(candidate_with, default_config):
    result = run(gap_candidate(candidate_with, {"accuracy": 0.95}),
                 default_config)[GAP]
    assert result.raw_value == pytest.approx(0.25)
    assert result.status == CheckStatus.FAIL
    assert result.evidence_source == EvidenceSource.DECLARED
    assert result.details["metric"] == "accuracy"
    assert result.details["measured_eval_value"] == pytest.approx(0.7)
    assert "not verified" in result.message


def test_f1_is_used_when_accuracy_is_not_declared(candidate_with, default_config):
    result = run(gap_candidate(candidate_with, {"f1": 0.9}), default_config)[GAP]
    assert result.details["metric"] == "f1"
    assert result.raw_value == pytest.approx(0.9 - 2 / 3)


def test_accuracy_is_preferred_when_both_are_declared(candidate_with, default_config):
    result = run(gap_candidate(candidate_with, {"accuracy": 0.8, "f1": 0.9}),
                 default_config)[GAP]
    assert result.details["metric"] == "accuracy"


def test_eval_better_than_train_gives_a_negative_gap_that_passes(
        candidate_with, default_config):
    result = run(gap_candidate(candidate_with, {"accuracy": 0.6}),
                 default_config)[GAP]
    assert result.raw_value == pytest.approx(-0.1)
    assert result.status == CheckStatus.PASS


@pytest.mark.parametrize("declared", [None, {}, {"auc": 0.9}])
def test_gap_unavailable_without_usable_training_metrics(candidate_with,
                                                         default_config, declared):
    result = run(gap_candidate(candidate_with, declared), default_config)[GAP]
    assert result.status == CheckStatus.UNAVAILABLE and result.score is None
    assert result.evidence_source == EvidenceSource.UNAVAILABLE


def test_gap_unavailable_for_an_out_of_range_declared_value(candidate_with,
                                                            default_config):
    result = run(gap_candidate(candidate_with, {"accuracy": 1.5}),
                 default_config)[GAP]
    assert result.status == CheckStatus.UNAVAILABLE and "[0, 1]" in result.message


def test_gap_unavailable_when_predictions_are_unavailable(candidate_with,
                                                          default_config):
    c = gap_candidate(candidate_with, {"accuracy": 0.9})
    c.metadata = TrainingMetadata(training_metrics={"accuracy": 0.9})  # no target
    result = run(c, default_config)[GAP]
    assert result.status == CheckStatus.UNAVAILABLE
    assert "target column is not declared" in result.message


def test_gap_unavailable_without_any_metadata(candidate_with, default_config):
    assert run(candidate_with(Scripted([0] * 10)),
               default_config)[GAP].status == CheckStatus.UNAVAILABLE


# ------------------------------------------------------ train/eval overlap
@pytest.fixture
def overlap_files(tmp_path):
    """100 unique evaluation rows; the first `k` are written as 'training' hashes."""
    frame = pd.DataFrame({"id": range(100), "age": np.arange(100) % 50 + 20,
                          "workclass": ["Private", "Gov"] * 50,
                          "target": [POS, NEG] * 50})
    eval_path = tmp_path / "eval.csv"
    frame.to_csv(eval_path, index=False)

    def build(k=20, columns=("id", "age", "workclass"), declare="bundle",
              file_text=None):
        hashes = row_hashes(read_csv_as_text(eval_path), ["id", "age", "workclass"])[:k]
        hash_path = tmp_path / "train_hashes.txt"
        hash_path.write_text(
            file_text if file_text is not None else "\n".join(hashes) + "\n",
            encoding="utf-8")
        meta = {"target_column": "target", "positive_label": POS,
                "feature_columns": ["age", "workclass"]}
        if columns is not None:
            meta["row_hash_columns"] = list(columns)
        if declare == "metadata":
            meta["train_row_hashes_file"] = hash_path.name   # relative to metadata
        (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        model = tmp_path / "model.joblib"
        model.write_bytes(b"x")                        # not loadable; not needed
        return load_candidate(CandidateBundle(
            model_path=model, eval_data_path=eval_path,
            metadata_path=tmp_path / "metadata.json",
            train_row_hashes_path=hash_path if declare == "bundle" else None))
    return build


def test_overlap_fraction_is_exact(overlap_files, default_config):
    result = run(overlap_files(k=20), default_config)[OVERLAP]
    assert result.raw_value == pytest.approx(0.20)
    assert result.status == CheckStatus.FAIL
    assert result.details["overlap_rows"] == 20
    assert result.evidence_source == EvidenceSource.DECLARED


def test_disjoint_sets_have_zero_overlap(overlap_files, default_config):
    result = run(overlap_files(k=0, file_text="deadbeef\n"), default_config)[OVERLAP]
    assert result.raw_value == 0.0 and result.status == CheckStatus.PASS


def test_hash_file_declared_in_metadata_resolves_next_to_it(overlap_files,
                                                            default_config):
    result = run(overlap_files(k=10, declare="metadata"), default_config)[OVERLAP]
    assert result.raw_value == pytest.approx(0.10)


def test_internal_duplicates_are_reported_as_context(tmp_path, default_config):
    frame = pd.DataFrame({"a": [1, 1, 1, 2, 3, 4, 5, 6, 7, 8] * 5,
                          "target": [POS, NEG] * 25})
    eval_path = tmp_path / "e.csv"
    frame.to_csv(eval_path, index=False)
    hash_path = tmp_path / "h.txt"
    hash_path.write_text("nothing\n", encoding="utf-8")
    meta = tmp_path / "m.json"
    meta.write_text(json.dumps({"row_hash_columns": ["a"]}), encoding="utf-8")
    c = load_candidate(CandidateBundle(
        model_path=tmp_path / "none", eval_data_path=eval_path,
        metadata_path=meta, train_row_hashes_path=hash_path))
    result = run(c, default_config)[OVERLAP]
    assert result.details["eval_internal_duplicate_fraction"] == pytest.approx(
        1 - 8 / 50)
    assert "genuine duplicate records" in result.message


def test_overlap_unavailable_cases(overlap_files, tmp_path, default_config):
    # no hash file anywhere
    c = overlap_files()
    c.bundle = c.bundle.model_copy(update={"train_row_hashes_path": None})
    assert run(c, default_config)[OVERLAP].status == CheckStatus.UNAVAILABLE
    # declared hash file missing
    c = overlap_files(declare="metadata")
    (tmp_path / "train_hashes.txt").unlink()
    result = run(c, default_config)[OVERLAP]
    assert result.status == CheckStatus.UNAVAILABLE and "not found" in result.message
    # explicit path missing
    c = overlap_files()
    c.bundle = c.bundle.model_copy(
        update={"train_row_hashes_path": tmp_path / "gone.txt"})
    assert "not found" in run(c, default_config)[OVERLAP].message
    # empty hash file
    assert "empty" in run(overlap_files(file_text="\n"), default_config)[
        OVERLAP].message
    # metadata does not say which columns were hashed
    result = run(overlap_files(columns=None), default_config)[OVERLAP]
    assert result.status == CheckStatus.UNAVAILABLE
    assert "row_hash_columns" in result.message
    # eval data lacks a hashed column
    result = run(overlap_files(columns=("id", "missing_col")), default_config)[OVERLAP]
    assert result.status == CheckStatus.UNAVAILABLE and "missing_col" in result.message


def test_overlap_unavailable_without_evaluation_data(candidate_with, default_config):
    c = candidate_with(Scripted([0] * 10), eval_df=None, eval_error="gone")
    assert run(c, default_config)[OVERLAP].status == CheckStatus.UNAVAILABLE


# ------------------------------------------- hashing convention is shared
def _scripts_module():
    path = ROOT / "scripts" / "fixture_utils.py"
    spec = importlib.util.spec_from_file_location("fixture_utils_ref", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def test_row_hashing_matches_the_training_side_convention():
    reference = _scripts_module()
    frame = pd.DataFrame({"a": ["x", "y,z", "", "n/a"], "b": [1, 2, 3, 4],
                          "c": ['quote"d', "ünïcode", " padded ", "plain"]})
    assert row_hashes(frame, ["a", "b", "c"]) == reference.row_hashes(
        frame, ["a", "b", "c"])


# ----------------------------------------------------------- leakage probe
def leak_frame(n=1000, seed=1, leak=True):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.3
    frame = pd.DataFrame({
        "signal": rng.normal(0, 1, n) + y * 0.8,        # honest, moderate signal
        "noise_cat": rng.choice(["a", "b", "c", "d"], n),
        "noise_num": rng.normal(0, 1, n),
        "target": np.where(y, POS, NEG),
    })
    if leak:
        flip = rng.random(n) < 0.03
        frame["leak"] = np.where(flip, ~y, y).astype(int)
    return frame


def leak_candidate(candidate_with, frame, **meta_overrides):
    columns = [c for c in frame.columns if c != "target"]
    meta = TrainingMetadata(target_column="target", positive_label=POS,
                            feature_columns=columns, **meta_overrides)
    return candidate_with(Features(columns), eval_df=frame, metadata=meta)


def test_a_label_derived_feature_is_flagged(candidate_with, default_config):
    result = run(leak_candidate(candidate_with, leak_frame()), default_config)[LEAK]
    assert result.details["top_feature"] == "leak"
    assert result.raw_value > 0.95
    assert result.status == CheckStatus.FAIL
    assert result.evidence_source == EvidenceSource.MEASURED


def test_honest_features_are_not_flagged(candidate_with, default_config):
    frame = leak_frame(leak=False)
    result = run(leak_candidate(candidate_with, frame), default_config)[LEAK]
    assert result.status == CheckStatus.PASS
    assert result.details["top_feature"] == "signal"
    assert result.details["all_features"]["noise_cat"] < 0.6
    assert result.details["all_features"]["noise_num"] < 0.6


def test_leakage_probe_is_reproducible(candidate_with, default_config):
    a = run(leak_candidate(candidate_with, leak_frame()), default_config)[LEAK]
    b = run(leak_candidate(candidate_with, leak_frame()), default_config)[LEAK]
    assert a.details["all_features"] == b.details["all_features"]


def test_leakage_probe_needs_no_loaded_model(candidate_with, default_config):
    c = leak_candidate(candidate_with, leak_frame())
    c.model, c.model_error = None, "cannot load model: EOFError"
    result = run(c, default_config)[LEAK]
    assert result.details["top_feature"] == "leak"       # features from metadata


def test_large_evaluation_sets_are_sampled(candidate_with, make_config):
    config = make_config(lambda d: d["evaluation"].update(leakage_max_rows=200))
    result = run(leak_candidate(candidate_with, leak_frame()), config)[LEAK]
    assert result.details["sampled"] and result.details["rows_used"] == 200


def test_odd_columns_are_handled(candidate_with, default_config):
    frame = leak_frame(leak=False)
    frame["constant"] = 7
    frame["all_missing"] = np.nan
    frame["flag"] = frame["target"] == POS            # a bool feature equal to the label
    frame["with_nan"] = frame["noise_cat"].where(frame.index % 5 != 0)
    result = run(leak_candidate(candidate_with, frame), default_config)[LEAK]
    aucs = result.details["all_features"]
    assert aucs["constant"] == 0.5 and aucs["all_missing"] == 0.5
    assert aucs["flag"] > 0.99 and result.details["top_feature"] == "flag"
    assert aucs["with_nan"] < 0.6


def test_one_unusable_column_does_not_stop_the_rest(candidate_with, default_config):
    frame = leak_frame(leak=False)
    frame["listy"] = [[i] for i in range(len(frame))]       # unhashable values
    result = run(leak_candidate(candidate_with, frame), default_config)[LEAK]
    assert "listy" in result.details["failed_features"]
    assert result.details["top_feature"] == "signal"


def test_single_feature_auc_of_a_perfect_feature():
    y = np.array([True, False] * 100)
    assert single_feature_auc(pd.Series(y.astype(int)), y, 5, 0) > 0.99
    assert single_feature_auc(pd.Series(["same"] * 200), y, 5, 0) == 0.5


def test_leakage_unavailable_cases(candidate_with, default_config):
    frame = leak_frame()
    # too few of one class
    few = frame.copy()
    few["target"] = [POS] * 3 + [NEG] * (len(few) - 3)
    assert "Fewer than" in run(leak_candidate(candidate_with, few),
                               default_config)[LEAK].message
    # target undeclared
    c = leak_candidate(candidate_with, frame)
    c.metadata = TrainingMetadata(feature_columns=["signal"])
    assert run(c, default_config)[LEAK].status == CheckStatus.UNAVAILABLE
    # feature columns unknown
    c = candidate_with(Scripted([0] * 10), eval_df=frame.drop(columns="leak"),
                       metadata=TrainingMetadata(target_column="target",
                                                 positive_label=POS))
    c.model = object()                      # no feature_names_in_, no metadata list
    assert "feature columns are unknown" in run(c, default_config)[LEAK].message
    # a declared feature is absent from the evaluation data
    c = leak_candidate(candidate_with, frame)
    c.eval_df = c.eval_df.drop(columns="noise_num")
    assert "lacks feature columns" in run(c, default_config)[LEAK].message


# ----------------------------------------------------------- run / registry
def test_registry_matches_the_catalogue():
    expected = [s.id for s in CATALOGUE if s.dimension == Dimension.TRAINING_VALIDITY]
    assert sorted(TRAINING_VALIDITY_CHECKS) == sorted(expected)


def test_run_returns_enabled_checks_in_catalogue_order(candidate_with, default_config,
                                                      make_config):
    ids = [r.id for r in run_training_validity_checks(
        candidate_with(Scripted([0] * 10)), default_config)]
    assert ids == [s.id for s in CATALOGUE
                   if s.dimension == Dimension.TRAINING_VALIDITY]
    config = make_config(lambda d: d["checks"][REPRO].update(enabled=False))
    assert REPRO not in {r.id for r in run_training_validity_checks(
        candidate_with(Scripted([0] * 10)), config)}


# ------------------------------------------------ generated fixture models
def _fixture(name, eval_path=None, metadata=None):
    model = FIXTURES / ("good" if name.startswith("good") else name) / "model.joblib"
    eval_path = eval_path or DATA / "eval.csv"
    if not model.exists() or not eval_path.exists():
        pytest.skip("fixtures not generated (run scripts/make_fixtures.py)")
    return load_candidate(CandidateBundle(
        model_path=model, eval_data_path=eval_path,
        metadata_path=FIXTURES / name / "metadata.json",
        train_row_hashes_path=ROOT / "data" / "train_row_hashes.txt"))


def test_good_fixture_passes_every_training_validity_check(default_config):
    results = run(_fixture("good"), default_config)
    assert {r.status for r in results.values()} == {CheckStatus.PASS}


def test_gate_overlap_matches_the_count_recorded_by_the_scripts(default_config):
    recorded = json.loads((FIXTURES / "good" / "metadata.json").read_text())[
        "train_eval_exact_row_overlap"]["rows"]
    result = run(_fixture("good"), default_config)[OVERLAP]
    assert result.details["overlap_rows"] == recorded


def test_good_fixture_honest_best_feature_is_well_below_the_threshold(default_config):
    result = run(_fixture("good"), default_config)[LEAK]
    assert result.details["top_feature"] == "relationship"
    assert 0.74 < result.raw_value < 0.80


def test_leaky_fixture_is_flagged(default_config):
    results = run(_fixture("leaky", FIXTURES / "leaky" / "eval.csv"), default_config)
    assert results[LEAK].status == CheckStatus.FAIL
    assert results[LEAK].details["top_feature"] == "prior_high_income_flag"


def test_overfit_fixture_is_flagged_by_the_declared_gap(default_config):
    results = run(_fixture("overfit"), default_config)
    assert results[GAP].status == CheckStatus.FAIL
    assert results[GAP].raw_value == pytest.approx(0.185, abs=1e-3)


def test_incomplete_metadata_fixture_loses_declared_evidence(default_config):
    results = run(_fixture("good_incomplete_metadata"), default_config)
    assert results[COMPLETE].status == CheckStatus.WARN
    assert results[REPRO].status == CheckStatus.FAIL
    for check_id in (GAP, OVERLAP):
        assert results[check_id].status == CheckStatus.UNAVAILABLE
        assert results[check_id].score is None
    assert results[LEAK].status == CheckStatus.PASS      # needs no metadata


# ----------------------------------------- text-like features (Step A)
from mqvg.checks.training_validity import is_free_text_like  # noqa: E402


@pytest.mark.parametrize("values, expected", [
    (["a sentence with several words here"] * 10, True),     # free text
    ([f"id{i}" for i in range(120)], True),                  # identifier-like
    (["x", "y", "z"] * 50, False),                           # ordinary category
    ([1, 2, 3] * 40, False),                                 # numbers
    ([None] * 10, False),                                    # nothing there
    ([[1], [2], [3]] * 20, False),                           # not strings
    (["single-word"] * 5 + [None] * 5, False),
])
def test_free_text_detection(values, expected):
    assert is_free_text_like(pd.Series(values, dtype=object)) is expected


def test_features_that_are_all_free_text_are_not_assessed_not_passed(
        candidate_with, default_config):
    rng = np.random.default_rng(0)
    words = np.array("alpha beta gamma delta epsilon zeta eta theta".split())
    frame = pd.DataFrame({
        "text": [" ".join(rng.choice(words, 6)) for _ in range(300)],
        "target": rng.choice([POS, NEG], 300)})
    meta = TrainingMetadata(target_column="target", positive_label=POS,
                            feature_columns=["text"])
    c = candidate_with(Features(["text"]), eval_df=frame, metadata=meta)
    result = run(c, default_config)[LEAK]
    assert result.status == CheckStatus.UNAVAILABLE and result.score is None
    assert result.details["not_assessed_columns"] == ["text"]
    assert "free text" in result.message


def test_a_text_column_is_skipped_while_other_features_are_still_checked(
        candidate_with, default_config):
    frame = leak_frame()
    rng = np.random.default_rng(1)
    words = np.array("alpha beta gamma delta epsilon zeta eta theta".split())
    frame["comment"] = [" ".join(rng.choice(words, 6)) for _ in range(len(frame))]
    result = run(leak_candidate(candidate_with, frame), default_config)[LEAK]
    assert result.details["top_feature"] == "leak"
    assert result.details["not_assessed_columns"] == ["comment"]
    assert "Not assessed" in result.message and "comment" not in result.details["all_features"]
