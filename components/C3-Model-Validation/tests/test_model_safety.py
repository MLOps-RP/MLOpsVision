"""The model-file safety scan: it must find dangerous code WITHOUT running it."""
import os
import pickle

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from mqvg import model_safety as ms
from mqvg.config import ModelSafetyConfig
from mqvg.loader import BLOCKED_REVIEW, BLOCKED_UNSAFE, load_candidate
from mqvg.schemas import CandidateBundle

from conftest import FIXTURES

POLICY = ModelSafetyConfig()


class Payload:
    """Unpickling this calls os.mkdir(marker): proof that code ran."""

    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (os.mkdir, (self.marker,))


class CustomModel:
    """A user's own model class: importable here, but not on the allowlist."""
    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


# ----------------------------------------------------------------- classification
@pytest.mark.parametrize("reference, expected", [
    ("sklearn.linear_model._logistic.LogisticRegression", ms.CLEAN),
    ("numpy.core.multiarray._reconstruct", ms.CLEAN),
    ("joblib.numpy_pickle.NumpyArrayWrapper", ms.CLEAN),
    ("collections.OrderedDict", ms.CLEAN),
    ("builtins.list", ms.CLEAN),
    ("os.system", ms.DANGEROUS), ("nt.system", ms.DANGEROUS), ("posix.system", ms.DANGEROUS),
    ("subprocess.Popen", ms.DANGEROUS), ("socket.socket", ms.DANGEROUS),
    ("builtins.eval", ms.DANGEROUS), ("builtins.__import__", ms.DANGEROUS),
    ("builtins.getattr", ms.DANGEROUS),
    ("numpy.load", ms.DANGEROUS), ("joblib.load", ms.DANGEROUS),
    ("pandas.io.pickle.read_pickle", ms.DANGEROUS), ("importlib.import_module", ms.DANGEROUS),
    ("builtins.print", ms.UNRECOGNISED), ("mypackage.MyModel", ms.UNRECOGNISED),
    ("__main__.Model", ms.UNRECOGNISED),
])
def test_references_are_classified(reference, expected):
    assert ms.classify(reference, POLICY) == expected


def test_prefixes_match_whole_module_names_only():
    # 'sklearn' is allowed but 'sklearnx' is not; 'os' is blocked but 'osmium' is only unknown
    assert ms.classify("sklearnx.Model", POLICY) == ms.UNRECOGNISED
    assert ms.classify("osmium.thing", POLICY) == ms.UNRECOGNISED
    assert ms.classify("os.path.join", POLICY) == ms.DANGEROUS      # submodules are covered


def test_blocked_beats_allowed():
    policy = ModelSafetyConfig(allowed_module_prefixes=["os", "sklearn"])
    assert ms.classify("os.system", policy) == ms.DANGEROUS


def test_the_policy_is_configurable():
    policy = ModelSafetyConfig(allowed_module_prefixes=["mypackage"])
    assert ms.classify("mypackage.MyModel", policy) == ms.CLEAN
    assert ms.classify("sklearn.tree.DecisionTreeClassifier", policy) == ms.UNRECOGNISED


# ------------------------------------------------------------------------- scanning
@pytest.mark.parametrize("name", ["good", "overfit", "slow", "biased"])
def test_sample_models_scan_clean(name):
    path = FIXTURES / name / "model.joblib"
    if not path.exists():
        pytest.skip("fixture models not generated")
    report = ms.scan_model_file(path, POLICY)
    assert report.severity == ms.CLEAN and not report.dangerous and not report.unrecognised
    assert any(r.startswith("sklearn.") for r in report.references)


@pytest.mark.parametrize("compress", [0, 3])
def test_models_with_string_categories_scan_clean(tmp_path, compress):
    """OneHotEncoder stores object arrays, which joblib writes as a nested pickle."""
    pipe = make_pipeline(OneHotEncoder(), LogisticRegression())
    pipe.fit([["a"], ["b"], ["a"], ["b"]], [0, 1, 0, 1])
    path = tmp_path / "m.joblib"
    joblib.dump(pipe, path, compress=compress)
    report = ms.scan_model_file(path, POLICY)
    assert report.severity == ms.CLEAN, report.unrecognised


def test_a_plain_pickle_is_scanned_too(tmp_path):
    path = tmp_path / "m.pkl"
    path.write_bytes(pickle.dumps(LogisticRegression()))
    assert ms.scan_model_file(path, POLICY).severity == ms.CLEAN


@pytest.mark.parametrize("how", ["joblib", "compressed", "pickle"])
def test_dangerous_code_is_found_and_not_run(tmp_path, how):
    marker = tmp_path / "pwned"
    path = tmp_path / "evil.bin"
    if how == "pickle":
        path.write_bytes(pickle.dumps(Payload(marker)))
    else:
        joblib.dump(Payload(marker), path, compress=3 if how == "compressed" else 0)
    report = ms.scan_model_file(path, POLICY)
    assert report.severity == ms.DANGEROUS
    assert any(r.endswith(".mkdir") for r in report.dangerous)
    assert not marker.exists(), "the scan executed the payload"


def test_dangerous_code_hidden_in_an_object_array_is_found_and_not_run(tmp_path):
    marker = tmp_path / "pwned"
    path = tmp_path / "evil_array.joblib"
    joblib.dump({"categories": np.array([Payload(marker)], dtype=object)}, path)
    report = ms.scan_model_file(path, POLICY)
    assert report.severity == ms.DANGEROUS and not marker.exists()


def test_the_payloads_are_real_so_the_tests_above_prove_something(tmp_path):
    """Control: really loading the same file does run the payload."""
    marker = tmp_path / "pwned"
    path = tmp_path / "evil.joblib"
    joblib.dump(Payload(marker), path)
    assert not marker.exists()
    joblib.load(path)
    assert marker.exists()


def test_unrecognised_code_is_reported_but_not_dangerous(tmp_path):
    path = tmp_path / "custom.joblib"
    joblib.dump(CustomModel(), path)
    report = ms.scan_model_file(path, POLICY)
    assert report.severity == ms.UNRECOGNISED and not report.dangerous
    assert any(r.endswith("CustomModel") for r in report.unrecognised)


def test_an_unparseable_file_is_a_scan_error_not_a_pass(tmp_path):
    truncated = tmp_path / "t.joblib"
    full = tmp_path / "f.joblib"
    joblib.dump(LogisticRegression(), full)
    truncated.write_bytes(full.read_bytes()[:40])
    garbage = tmp_path / "g.joblib"
    garbage.write_bytes(b"this is not a pickle")
    for path in (truncated, garbage):
        with pytest.raises(ms.ScanError) as error:
            ms.scan_model_file(path, POLICY)
        assert error.value.parse


# --------------------------------------------------------------------- the loader
def bundle(model, tmp_path):
    return CandidateBundle(candidate_id="t", model_path=model,
                           eval_data_path=tmp_path / "missing.csv")


def test_a_dangerous_file_is_never_loaded(tmp_path):
    marker = tmp_path / "pwned"
    path = tmp_path / "evil.joblib"
    joblib.dump(Payload(marker), path)
    candidate = load_candidate(bundle(path, tmp_path), POLICY)
    assert candidate.model is None and candidate.model_failure == BLOCKED_UNSAFE
    assert candidate.safety_scan.severity == ms.DANGEROUS
    assert "NOT loaded" in candidate.model_error
    assert not marker.exists()


def test_unrecognised_code_is_held_for_review_by_default(tmp_path):
    path = tmp_path / "custom.joblib"
    joblib.dump(CustomModel(), path)
    candidate = load_candidate(bundle(path, tmp_path), POLICY)
    assert candidate.model is None and candidate.model_failure == BLOCKED_REVIEW
    assert "allowed_module_prefixes" in candidate.model_error


def test_refuse_at_dangerous_loads_unrecognised_code(tmp_path):
    path = tmp_path / "custom.joblib"
    joblib.dump(CustomModel(), path)
    candidate = load_candidate(bundle(path, tmp_path), ModelSafetyConfig(refuse_at="dangerous"))
    assert isinstance(candidate.model, CustomModel)
    assert candidate.safety_scan.severity == ms.UNRECOGNISED


def test_trusting_a_module_lets_its_model_load(tmp_path):
    path = tmp_path / "custom.joblib"
    joblib.dump(CustomModel(), path)
    policy = ModelSafetyConfig(allowed_module_prefixes=[CustomModel.__module__, "numpy", "joblib"])
    candidate = load_candidate(bundle(path, tmp_path), policy)
    assert isinstance(candidate.model, CustomModel) and candidate.safety_scan.severity == ms.CLEAN


def test_refuse_at_never_reports_but_loads_and_runs_the_code(tmp_path):
    """Documents what 'never' means: the scan is informational and the code does run."""
    marker = tmp_path / "pwned"
    path = tmp_path / "evil.joblib"
    joblib.dump(Payload(marker), path)
    candidate = load_candidate(bundle(path, tmp_path), ModelSafetyConfig(refuse_at="never"))
    assert candidate.safety_scan.severity == ms.DANGEROUS
    assert marker.exists()


def test_a_corrupt_file_is_not_loaded_and_is_a_corrupt_failure(tmp_path):
    path = tmp_path / "bad.joblib"
    path.write_bytes(b"corrupt")
    candidate = load_candidate(bundle(path, tmp_path), POLICY)
    assert candidate.model is None and candidate.model_failure == "corrupt"
    assert "not loaded" in candidate.model_error and candidate.safety_error


def test_without_a_safety_policy_loading_is_unchanged(tmp_path):
    path = tmp_path / "m.joblib"
    joblib.dump(LogisticRegression(), path)
    candidate = load_candidate(bundle(path, tmp_path))
    assert candidate.model is not None
    assert candidate.safety_scan is None and candidate.safety_error is None


def test_an_unsupported_format_is_not_scanned(tmp_path):
    path = tmp_path / "m.onnx"
    path.write_bytes(b"x")
    candidate = load_candidate(bundle(path, tmp_path), POLICY)
    assert candidate.model_failure == "unsupported_format" and candidate.safety_scan is None
