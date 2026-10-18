"""End-to-end tests with models the gate did not build.

Each scenario trains or fabricates a model in a temporary folder and validates
it through the full pipeline. They document exactly what the gate supports
(scikit-learn-style tabular binary classifiers loaded with joblib/pickle) and
what it declines, and check that declining is always safe: an input the gate
cannot assess is UNAVAILABLE (or a clear FAIL for a broken artifact) and never
a pass.
"""
import json
import pickle
import shutil
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from mqvg.checks.predictions import _normalized, normalize_label
from mqvg.environment import current_versions
from mqvg.hashing import read_csv_as_text, row_hashes
from mqvg.pipeline import run_validation
from mqvg.schemas import CandidateBundle, CheckStatus, Decision

import foreign

warnings.filterwarnings("ignore")

NUM = ["age", "income_k", "hours"]
CAT = ["segment", "region"]
FEATS = NUM + CAT
POS, NEG = "approved", "denied"
NO_EVIDENCE = (CheckStatus.UNAVAILABLE, CheckStatus.ERROR, CheckStatus.NOT_APPLICABLE)
BEHAVIOUR = ["behaviour.f1", "behaviour.roc_auc", "behaviour.precision",
             "behaviour.recall", "calibration.ece"]


def make_frame(n, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n), "income_k": rng.normal(50, 15, n).round(1),
        "hours": rng.integers(10, 60, n),
        "segment": rng.choice(list("abc"), n),
        "region": rng.choice(["n", "s", "e", "w"], n),
        "sex": rng.choice(["Female", "Male"], n)})
    z = (0.04 * (df["age"] - 40) + 0.03 * (df["income_k"] - 50)
         + 0.02 * (df["hours"] - 35) + 0.6 * (df["segment"] == "a")
         + rng.normal(0, 1, n))
    df["outcome"] = np.where(z > 0.3, POS, NEG)
    return df


def preprocessor(dense=False):
    return ColumnTransformer([
        ("n", StandardScaler(), NUM),
        ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense), CAT)])


class World:
    """Trains the foreign models once and validates scenarios on demand."""

    def __init__(self, folder, config):
        self.dir, self.config = folder, config
        self.train, self.eval = make_frame(1500, 1), make_frame(900, 2)
        self.eval_csv = folder / "eval.csv"
        self.eval.to_csv(self.eval_csv, index=False)
        train_csv = folder / "train.csv"
        self.train.to_csv(train_csv, index=False)
        self.hash_cols = FEATS + ["sex"]
        self.hashes = folder / "train_hashes.txt"
        self.hashes.write_text(
            "\n".join(row_hashes(read_csv_as_text(train_csv), self.hash_cols)) + "\n",
            encoding="utf-8")
        self.cache = {}

    # ---- helpers -------------------------------------------------------
    def meta(self, model=None, **overrides):
        accuracy = 0.8
        if model is not None:
            try:
                accuracy = float((model.predict(self.train[FEATS])
                                  == self.train["outcome"]).mean())
            except Exception:  # noqa: BLE001
                pass
        meta = {"model_type": "sklearn", "task_type": "binary_classification",
                "target_column": "outcome", "positive_label": POS,
                "sensitive_columns": ["sex"], "feature_columns": FEATS,
                "hyperparameters": {"seed": 1}, "training_metrics": {"accuracy": accuracy},
                "train_rows": len(self.train), "dataset": "synthetic",
                "random_state": 1, "split": {"seed": 1},
                "versions": current_versions(), "train_data_sha256": "x",
                "code": {"commit": "abc"}, "row_hash_columns": self.hash_cols}
        meta.update(overrides)
        return {k: v for k, v in meta.items() if v is not None}

    def save(self, name, model, suffix=".joblib"):
        path = self.dir / f"{name}{suffix}"
        joblib.dump(model, path)
        return path

    def validate(self, key, model_path, *, meta="complete", eval_path=None,
                 hashes=True, **bundle):
        if key in self.cache:
            return self.cache[key]
        meta_path = None
        if meta is not None:
            meta_path = self.dir / f"{key}.meta.json"
            meta_path.write_text(json.dumps(
                self.meta() if meta == "complete" else meta), encoding="utf-8")
        result = run_validation(CandidateBundle(
            candidate_id=key, model_path=model_path,
            eval_data_path=eval_path or self.eval_csv, metadata_path=meta_path,
            train_row_hashes_path=self.hashes if hashes else None, **bundle),
            self.config)
        self.cache[key] = result
        return result


@pytest.fixture(scope="module")
def w(tmp_path_factory, default_config_module):
    world = World(tmp_path_factory.mktemp("external"), default_config_module)
    # the main model: an ordinary scikit-learn pipeline fitted on a DataFrame
    world.lr = Pipeline([("p", preprocessor()), ("m", LogisticRegression(max_iter=500))]
                        ).fit(world.train[FEATS], world.train["outcome"])
    world.lr_path = world.save("lr", world.lr)
    return world


@pytest.fixture(scope="module")
def default_config_module():
    from conftest import CONFIGS
    from mqvg.config import load_config
    return load_config(CONFIGS / "default.yaml")


def status(result):
    return {c.id: c.status for c in result.checks}


def check(result, check_id):
    return next(c for c in result.checks if c.id == check_id)


# ============================================================= works ===
def test_an_ordinary_sklearn_pipeline_is_fully_evaluated(w):
    r = w.validate("happy", w.lr_path, meta=w.meta(w.lr))
    assert not [c.id for c in r.checks if c.status in NO_EVIDENCE]
    assert r.scorecard.mqs_sufficient_evidence and r.scorecard.overall_coverage == 1.0
    assert r.decision.decision in (Decision.PROMOTE, Decision.HOLD)
    assert check(r, "behaviour.f1").evidence_source.value == "MEASURED"


@pytest.mark.parametrize("name, model_factory, suffix", [
    ("hgb", lambda w: Pipeline([("p", preprocessor(dense=True)),
                                ("m", HistGradientBoostingClassifier(max_iter=40))]), ".joblib"),
    ("svc", lambda w: Pipeline([("p", preprocessor()), ("m", LinearSVC())]), ".joblib"),
    ("tree", lambda w: Pipeline([("p", preprocessor()),
                                 ("m", DecisionTreeClassifier(max_depth=5))]), ".joblib"),
])
def test_other_sklearn_families_are_evaluated(w, name, model_factory, suffix):
    model = model_factory(w).fit(w.train[FEATS], w.train["outcome"])
    r = w.validate(name, w.save(name, model, suffix), meta=w.meta(model))
    assert status(r)["operational.model_loadable"] == CheckStatus.PASS
    assert check(r, "behaviour.f1").raw_value > 0.5
    assert status(r)["fairness.selection_rate_ratio"] != CheckStatus.UNAVAILABLE


def test_a_margin_classifier_has_auc_but_no_calibration(w):
    r = w.validate("svc", w.dir / "svc.joblib")        # trained in the test above
    assert status(r)["behaviour.roc_auc"] != CheckStatus.UNAVAILABLE
    assert status(r)["calibration.ece"] == CheckStatus.UNAVAILABLE
    assert "not probabilities" in check(r, "calibration.ece").message


def test_a_plain_pickle_file_is_accepted(w):
    path = w.dir / "plain.pkl"
    with open(path, "wb") as handle:
        pickle.dump(w.lr, handle)
    r = w.validate("plain_pickle", path, meta=w.meta(w.lr))
    assert status(r)["operational.model_loadable"] == CheckStatus.PASS
    assert check(r, "behaviour.f1").raw_value == pytest.approx(
        check(w.validate("happy", w.lr_path), "behaviour.f1").raw_value)


def test_a_bare_estimator_fitted_on_numpy_needs_the_feature_columns(w):
    bare = LogisticRegression(max_iter=500).fit(
        w.train[NUM].to_numpy(), w.train["outcome"])
    path = w.save("bare", bare)
    with_cols = w.validate("bare_with", path, meta=w.meta(bare, feature_columns=NUM))
    assert check(with_cols, "behaviour.f1").status != CheckStatus.UNAVAILABLE
    without = w.validate("bare_without", path, meta=w.meta(bare, feature_columns=None))
    for check_id in ("operational.input_schema", "behaviour.f1", "calibration.ece"):
        assert status(without)[check_id] == CheckStatus.UNAVAILABLE
    assert "feature columns" in check(without, "behaviour.f1").message
    assert not without.scorecard.mqs_sufficient_evidence
    assert without.decision.decision == Decision.HOLD


def test_labels_that_differ_only_as_int_versus_float_agree(w):
    y = (w.train["outcome"] == POS).astype(float)         # model classes_ = [0.0, 1.0]
    model = Pipeline([("p", preprocessor()), ("m", LogisticRegression(max_iter=500))]
                     ).fit(w.train[FEATS], y)
    eval_int = w.eval.assign(outcome=(w.eval["outcome"] == POS).astype(int))
    eval_path = w.dir / "eval_int.csv"
    eval_int.to_csv(eval_path, index=False)
    r = w.validate("float_vs_int", w.save("floaty", model), eval_path=eval_path,
                   meta=w.meta(model, positive_label=1))
    assert check(r, "behaviour.f1").status != CheckStatus.UNAVAILABLE
    assert check(r, "behaviour.f1").raw_value > 0.5


# =========================================== label-space mismatch (A1) ===
def encoded_model(w):
    codes = (w.train["outcome"] == POS).astype(int)       # the model predicts 0 / 1
    return Pipeline([("p", preprocessor()), ("m", LogisticRegression(max_iter=500))]
                    ).fit(w.train[FEATS], codes)


@pytest.mark.parametrize("key, overrides", [
    ("mismatch_declared", {}),                            # label declared as text
    ("mismatch_undeclared", {"positive_label": None}),    # label not declared
])
def test_a_model_and_target_with_different_labels_gives_no_metric(w, key, overrides):
    model = encoded_model(w)
    r = w.validate(key, w.save(key, model), meta=w.meta(model, **overrides))
    for check_id in BEHAVIOUR + ["fairness.selection_rate_ratio",
                                 "fairness.equalized_odds"]:
        c = check(r, check_id)
        assert c.status == CheckStatus.UNAVAILABLE and c.score is None, check_id
    assert "Label mismatch" in check(r, "behaviour.f1").message
    assert "different label encodings" in check(r, "behaviour.f1").message
    # a data problem is not a model verdict: HOLD, never a ROLLBACK by score
    assert r.decision.decision == Decision.HOLD
    assert not r.scorecard.mqs_sufficient_evidence


def test_a_target_with_different_words_than_the_model_gives_no_metric(w):
    yes_no = w.eval.assign(outcome=w.eval["outcome"].map({POS: "yes", NEG: "no"}))
    path = w.dir / "eval_yesno.csv"
    yes_no.to_csv(path, index=False)
    r = w.validate("yes_no", w.lr_path, eval_path=path,
                   meta=w.meta(w.lr, positive_label="yes"))
    f1 = check(r, "behaviour.f1")
    assert f1.status == CheckStatus.UNAVAILABLE and "Label mismatch" in f1.message
    assert r.decision.decision == Decision.HOLD


def test_normalize_label():
    assert [normalize_label(v) for v in (1, 1.0, "1", "1.0", True, "yes", 1.5, "nan")] \
        == ["1", "1", "1", "1", "True", "yes", "1.5", "nan"]
    assert list(_normalized(np.array([0.0, 1.0, 1.0]))) == ["0", "1", "1"]


# ================================================ load failures (A3) ===
def test_unsupported_formats_are_unavailable_and_never_executed(w, tmp_path):
    marker = tmp_path / "ran.txt"
    for suffix in (".onnx", ".h5", ".pt", ".keras", ".pb"):
        path = w.dir / f"model{suffix}"
        with open(path, "wb") as handle:                  # a pickle that runs code
            pickle.dump(foreign.CodeOnLoad(marker), handle)
        r = w.validate(f"unsupported{suffix}", path)
        loadable = check(r, "operational.model_loadable")
        assert loadable.status == CheckStatus.UNAVAILABLE and loadable.score is None
        assert loadable.details["failure_kind"] == "unsupported_format"
        assert "says nothing about the model's quality" in loadable.message
        assert r.decision.decision == Decision.HOLD       # not a ROLLBACK
        assert "model_not_loadable" not in [x.rule_id for x in r.decision.triggered_rules]
        assert "required_evidence_missing" in [x.rule_id for x in r.decision.triggered_rules]
    assert not marker.exists()                            # the file was never loaded


def test_a_pickle_with_a_supported_suffix_does_run_its_code(w, tmp_path):
    """Documents the security limitation: loading a .pkl executes it."""
    marker = tmp_path / "ran_pkl.txt"
    path = w.dir / "evil.pkl"
    with open(path, "wb") as handle:
        pickle.dump(foreign.CodeOnLoad(marker), handle)
    r = w.validate("code_on_load", path)
    assert marker.exists()
    assert check(r, "operational.model_loadable").status == CheckStatus.FAIL


def test_a_model_needing_an_uninstalled_module_is_unavailable(w, tmp_path_factory):
    folder = tmp_path_factory.mktemp("private")
    (folder / "private_model_mod.py").write_text(
        "import numpy as np\n"
        "class Mine:\n"
        "    classes_ = np.array(['approved', 'denied'])\n"
        "    feature_names_in_ = np.array(['age'])\n"
        "    def predict(self, X):\n"
        "        return np.where(X['age'] > 40, 'approved', 'denied')\n",
        encoding="utf-8")
    sys.path.insert(0, str(folder))
    try:
        import private_model_mod
        path = w.save("private", private_model_mod.Mine())
    finally:
        sys.path.remove(str(folder))
        sys.modules.pop("private_model_mod", None)
    shutil.rmtree(folder)                                 # now it cannot be imported
    r = w.validate("missing_module", path)
    loadable = check(r, "operational.model_loadable")
    assert loadable.status == CheckStatus.UNAVAILABLE
    assert loadable.details["failure_kind"] == "missing_dependency"
    assert "private_model_mod" in loadable.message
    assert r.decision.decision == Decision.HOLD


def test_corrupt_and_non_model_files_still_fail_and_roll_back(w):
    truncated = w.dir / "truncated.joblib"
    truncated.write_bytes(w.lr_path.read_bytes()[:60])
    text = w.dir / "notes.txt"
    text.write_text("this is not a model", encoding="utf-8")
    dictionary = w.save("dict", {"not": "a model"})
    for key, path, kind in (("truncated", truncated, "corrupt"),
                            ("textfile", text, "corrupt"),
                            ("dict", dictionary, "not_a_model")):
        r = w.validate(key, path)
        loadable = check(r, "operational.model_loadable")
        assert loadable.status == CheckStatus.FAIL
        assert loadable.details["failure_kind"] == kind
        assert r.decision.decision == Decision.ROLLBACK


def test_a_missing_model_file_fails(w):
    r = w.validate("nofile", w.dir / "nothing.joblib")
    assert check(r, "operational.model_loadable").details["failure_kind"] == "not_found"
    assert r.decision.decision == Decision.ROLLBACK


# ================================================== input forms ===
def test_a_numpy_only_model_cannot_be_evaluated_and_is_not_promoted(w):
    path = w.save("numpy_only", foreign.NumpyOnlyModel())
    r = w.validate("numpy_only", path, meta=w.meta(feature_columns=["age"]))
    assert status(r)["operational.input_schema"] == CheckStatus.FAIL
    assert status(r)["behaviour.f1"] == CheckStatus.UNAVAILABLE
    assert r.decision.decision != Decision.PROMOTE


@pytest.mark.parametrize("key, column_edit", [
    ("missing_column", lambda df: df.drop(columns=["segment"])),
    ("renamed_column", lambda df: df.rename(columns={"age": "Age"})),
    ("text_in_number", lambda df: df.assign(age=df["age"].astype(str) + "yrs")),
])
def test_incompatible_evaluation_schemas_are_failures_not_passes(w, key, column_edit):
    path = w.dir / f"{key}.csv"
    column_edit(w.eval).to_csv(path, index=False)
    r = w.validate(key, w.lr_path, eval_path=path)
    assert status(r)["operational.input_schema"] == CheckStatus.FAIL
    for check_id in BEHAVIOUR:
        assert status(r)[check_id] == CheckStatus.UNAVAILABLE, check_id
    assert r.decision.decision == Decision.HOLD and not r.scorecard.mqs_sufficient_evidence


def test_a_tab_separated_evaluation_file_is_not_silently_misread(w):
    path = w.dir / "eval.tsv"
    w.eval.to_csv(path, index=False, sep="\t")
    r = w.validate("tsv", w.lr_path, eval_path=path)
    assert status(r)["operational.input_schema"] == CheckStatus.FAIL
    assert status(r)["behaviour.f1"] == CheckStatus.UNAVAILABLE


# ============================================== tasks that are not binary ===
def test_a_multiclass_model_is_declined_with_a_clear_message(w):
    model = DecisionTreeClassifier(max_depth=4).fit(w.train[NUM], w.train["segment"])
    r = w.validate("multiclass", w.save("multi", model), meta={
        "target_column": "segment", "feature_columns": NUM})
    f1 = check(r, "behaviour.f1")
    assert f1.status == CheckStatus.UNAVAILABLE and "multiclass" in f1.message
    assert status(r)["operational.model_loadable"] == CheckStatus.PASS
    assert not r.scorecard.mqs_sufficient_evidence
    assert "reliable" in r.decision.summary
    assert r.decision.decision == Decision.HOLD


def test_a_regression_model_is_recognised_as_regression(w):
    model = RandomForestRegressor(n_estimators=15, random_state=0).fit(
        w.train[["age", "hours"]], w.train["income_k"])
    r = w.validate("regression", w.save("reg", model), meta={
        "target_column": "income_k", "feature_columns": ["age", "hours"]})
    f1 = check(r, "behaviour.f1")
    assert f1.status == CheckStatus.UNAVAILABLE
    assert "regression" in f1.message and "multiclass" not in f1.message
    declared = w.validate("regression_declared", w.dir / "reg.joblib", meta={
        "target_column": "income_k", "task_type": "regression",
        "feature_columns": ["age", "hours"]})
    assert "regression" in check(declared, "behaviour.f1").message


# ============================================================== text ===
def text_world(w):
    rng = np.random.default_rng(3)
    good = "great excellent love wonderful amazing good happy perfect".split()
    bad = "terrible awful hate poor bad broken refund boring".split()
    filler = "the a product item order arrived box price service day".split()

    def build(n):
        rows = []
        for _ in range(n):
            positive = rng.random() < 0.45
            words = (list(rng.choice(good if positive else bad, 3))
                     + list(rng.choice(filler, 6)))
            rng.shuffle(words)
            rows.append((" ".join(words), POS if positive else NEG))
        return pd.DataFrame(rows, columns=["text", "outcome"])
    return build(1500), build(800)


def test_a_text_model_is_scored_but_leakage_is_honestly_not_assessed(w):
    train, evaluation = text_world(w)
    path_eval = w.dir / "text_eval.csv"
    evaluation.to_csv(path_eval, index=False)
    model = Pipeline([
        ("t", ColumnTransformer([("tf", TfidfVectorizer(), "text")])),
        ("m", LogisticRegression(max_iter=300))]).fit(train[["text"]], train["outcome"])
    r = w.validate("text_ok", w.save("text_ok", model), eval_path=path_eval,
                   meta=w.meta(model, feature_columns=["text"], sensitive_columns=None,
                               row_hash_columns=None), hashes=False)
    assert check(r, "behaviour.f1").status != CheckStatus.UNAVAILABLE
    leak = check(r, "leakage.single_feature_predictiveness")
    assert leak.status == CheckStatus.UNAVAILABLE and leak.score is None
    assert "free text" in leak.message
    assert r.decision.decision == Decision.HOLD           # required leakage evidence


def test_a_text_vectorizer_fitted_on_plain_strings_is_declined(w):
    train, evaluation = text_world(w)
    path_eval = w.dir / "text_eval2.csv"
    evaluation.to_csv(path_eval, index=False)
    model = Pipeline([("t", TfidfVectorizer()), ("m", LogisticRegression(max_iter=300))]
                     ).fit(train["text"], train["outcome"])
    r = w.validate("text_series", w.save("text_series", model), eval_path=path_eval,
                   meta=w.meta(model, feature_columns=["text"], sensitive_columns=None,
                               row_hash_columns=None), hashes=False)
    assert status(r)["operational.input_schema"] == CheckStatus.FAIL
    f1 = check(r, "behaviour.f1")
    assert f1.status == CheckStatus.UNAVAILABLE
    assert "different input form" in f1.message
    assert r.decision.decision != Decision.PROMOTE


# ======================================== missing metadata / evidence ===
def test_no_metadata_at_all_gives_an_unreliable_mqs_and_a_hold(w):
    r = w.validate("no_meta", w.lr_path, meta=None, hashes=False)
    assert status(r)["operational.model_loadable"] == CheckStatus.PASS
    assert status(r)["behaviour.f1"] == CheckStatus.UNAVAILABLE
    assert not r.scorecard.mqs_sufficient_evidence
    assert r.scorecard.insufficient_evidence_reason
    assert r.decision.decision == Decision.HOLD
    assert "not reliable" in r.decision.summary


def test_undeclared_sensitive_columns_make_fairness_unavailable(w):
    r = w.validate("no_sensitive", w.lr_path, meta=w.meta(w.lr, sensitive_columns=None))
    assert status(r)["fairness.selection_rate_ratio"] == CheckStatus.UNAVAILABLE
    assert r.decision.decision == Decision.HOLD
    assert "required_evidence_missing" in [x.rule_id for x in r.decision.triggered_rules]


def test_bundle_overrides_stand_in_for_missing_metadata(w):
    r = w.validate("bundle_override", w.lr_path, meta={"feature_columns": FEATS},
                   target_column="outcome", positive_label=POS,
                   sensitive_columns=["sex"], hashes=False)
    assert check(r, "behaviour.f1").status != CheckStatus.UNAVAILABLE
    assert check(r, "fairness.selection_rate_ratio").status != CheckStatus.UNAVAILABLE


# ============================== the invariant behind every scenario ===
def test_no_scenario_ever_scores_a_check_that_has_no_evidence(w):
    assert len(w.cache) > 20                              # the scenarios above ran
    for key, result in w.cache.items():
        for c in result.checks:
            if c.status in NO_EVIDENCE:
                assert c.score is None, (key, c.id)
                assert c.evidence_source.value == "UNAVAILABLE", (key, c.id)


def test_every_unreliable_or_declined_scenario_is_kept_from_promotion(w):
    for key in ("bare_without", "mismatch_declared", "mismatch_undeclared", "yes_no",
                "multiclass", "regression", "missing_column", "renamed_column",
                "text_in_number", "tsv", "text_ok", "text_series", "no_meta",
                "numpy_only", "missing_module", "unsupported.onnx", "truncated"):
        assert w.cache[key].decision.decision != Decision.PROMOTE, key
