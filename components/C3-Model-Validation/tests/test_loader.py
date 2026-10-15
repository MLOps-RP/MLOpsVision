import joblib
import pytest

from mqvg.loader import file_sha256, load_candidate
from mqvg.schemas import CandidateBundle

from conftest import FEATURES


def test_loads_all_parts(tiny_bundle):
    c = load_candidate(tiny_bundle)
    assert c.model is not None and c.model_error is None
    assert c.eval_df is not None and len(c.eval_df) == 400
    assert c.metadata is not None and c.metadata_error is None
    assert c.model_sha256 and c.eval_sha256 and c.metadata_sha256
    assert c.model_size_bytes > 0
    assert c.memory_delta_mb is not None and c.memory_delta_mb >= 0


def test_feature_columns_come_from_the_model(tiny_bundle):
    assert load_candidate(tiny_bundle).feature_columns == FEATURES


def test_feature_columns_fall_back_to_metadata(tiny_bundle, candidate_with):
    class NoNames:
        def predict(self, X):
            return X.index.to_numpy()

    from mqvg.schemas import TrainingMetadata
    c = candidate_with(NoNames(), metadata=TrainingMetadata(
        feature_columns=["age"], target_column="target"))
    assert c.feature_columns == ["age"]
    assert c.target_column == "target"


def test_feature_columns_unknown_without_either(candidate_with):
    class NoNames:
        def predict(self, X):
            return X

    assert candidate_with(NoNames()).feature_columns is None


def test_missing_model_file(tiny_files):
    tiny_files["model"].unlink()
    c = load_candidate(CandidateBundle(model_path=tiny_files["model"],
                                       eval_data_path=tiny_files["eval"]))
    assert c.model is None and "not found" in c.model_error
    assert c.eval_df is not None  # other parts still load


def test_corrupt_model_file(tiny_files):
    data = tiny_files["model"].read_bytes()
    tiny_files["model"].write_bytes(data[: len(data) // 3])
    c = load_candidate(CandidateBundle(model_path=tiny_files["model"],
                                       eval_data_path=tiny_files["eval"]))
    assert c.model is None and "cannot load model" in c.model_error
    assert c.model_size_bytes is not None  # size is still measurable


def test_garbage_model_file(tiny_files):
    tiny_files["model"].write_bytes(b"this is not a pickle")
    c = load_candidate(CandidateBundle(model_path=tiny_files["model"],
                                       eval_data_path=tiny_files["eval"]))
    assert c.model is None and c.model_error


def test_loaded_object_without_predict_is_rejected(tiny_files):
    joblib.dump({"not": "a model"}, tiny_files["model"])
    c = load_candidate(CandidateBundle(model_path=tiny_files["model"],
                                       eval_data_path=tiny_files["eval"]))
    assert c.model is None and "no predict method" in c.model_error


def test_missing_eval_data(tiny_files):
    tiny_files["eval"].unlink()
    c = load_candidate(CandidateBundle(model_path=tiny_files["model"],
                                       eval_data_path=tiny_files["eval"]))
    assert c.eval_df is None and "not found" in c.eval_error
    assert c.model is not None


def test_empty_eval_data(tiny_files):
    tiny_files["eval"].write_text("age,hours\n", encoding="utf-8")
    c = load_candidate(CandidateBundle(model_path=tiny_files["model"],
                                       eval_data_path=tiny_files["eval"]))
    assert c.eval_df is None and "no rows" in c.eval_error


def test_unreadable_eval_data(tiny_files):
    tiny_files["eval"].write_bytes(b"")
    c = load_candidate(CandidateBundle(model_path=tiny_files["model"],
                                       eval_data_path=tiny_files["eval"]))
    assert c.eval_df is None and c.eval_error


def test_metadata_not_provided(tiny_files):
    c = load_candidate(CandidateBundle(model_path=tiny_files["model"],
                                       eval_data_path=tiny_files["eval"]))
    assert not c.metadata_provided and c.metadata is None
    assert c.metadata_error is None


def test_metadata_file_missing(tiny_files):
    tiny_files["metadata"].unlink()
    c = load_candidate(CandidateBundle(model_path=tiny_files["model"],
                                       eval_data_path=tiny_files["eval"],
                                       metadata_path=tiny_files["metadata"]))
    assert c.metadata_provided and c.metadata is None
    assert "not found" in c.metadata_error


def test_metadata_invalid_json(tiny_files):
    tiny_files["metadata"].write_text("{broken", encoding="utf-8")
    c = load_candidate(CandidateBundle(model_path=tiny_files["model"],
                                       eval_data_path=tiny_files["eval"],
                                       metadata_path=tiny_files["metadata"]))
    assert c.metadata is None and "cannot parse metadata" in c.metadata_error


def test_metadata_wrong_field_type(tiny_files):
    tiny_files["metadata"].write_text('{"train_rows": "many"}', encoding="utf-8")
    c = load_candidate(CandidateBundle(model_path=tiny_files["model"],
                                       eval_data_path=tiny_files["eval"],
                                       metadata_path=tiny_files["metadata"]))
    assert c.metadata is None and c.metadata_error


def test_yaml_metadata_is_supported(tiny_files):
    yaml_path = tiny_files["metadata"].with_suffix(".yaml")
    yaml_path.write_text("target_column: target\nfeature_columns: [age]\n",
                         encoding="utf-8")
    c = load_candidate(CandidateBundle(model_path=tiny_files["model"],
                                       eval_data_path=tiny_files["eval"],
                                       metadata_path=yaml_path))
    assert c.metadata.target_column == "target"


def test_hash_is_content_based(tiny_files):
    first = file_sha256(tiny_files["eval"])
    assert first == file_sha256(tiny_files["eval"])
    tiny_files["eval"].write_text("changed", encoding="utf-8")
    assert first != file_sha256(tiny_files["eval"])


@pytest.mark.parametrize("name", ["good", "corrupt"])
def test_generated_fixtures_load_as_expected(name):
    from conftest import DATA, FIXTURES
    model = FIXTURES / name / "model.joblib"
    if not model.exists() or not (DATA / "eval.csv").exists():
        pytest.skip("fixtures not generated (run scripts/make_fixtures.py)")
    c = load_candidate(CandidateBundle(model_path=model,
                                       eval_data_path=DATA / "eval.csv"))
    assert (c.model is not None) == (name == "good")
