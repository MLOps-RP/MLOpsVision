import optuna
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from c2_automl.agents.agent2_hpo import run_defaults, split_budget
from c2_automl.config import load_default_config, load_yaml
from c2_automl.schemas import C1Output, TaskProfile, TaskType
from c2_automl.tools.hpo import study_to_records, tune_family
from c2_automl.tools.search_space import load_base_space, suggest_params

REGISTRY = load_yaml("model_registry.yaml")


@pytest.fixture(scope="module")
def small_data():
    X, y = make_classification(n_samples=300, n_features=6, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(6)]), pd.Series(y)


# ---------- search spaces ----------
def test_every_registry_family_has_a_search_space():
    for family in REGISTRY["families"]:
        assert load_base_space(family), family


def test_suggested_params_stay_inside_ranges():
    space = load_base_space("xgboost")
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    study.optimize(lambda t: len(suggest_params(t, space)), n_trials=20)
    for t in study.trials:
        for name, r in space.items():
            assert r["low"] <= t.params[name] <= r["high"]
        assert isinstance(t.params["max_depth"], int)


def test_load_base_space_returns_a_copy():
    space = load_base_space("xgboost")
    space["max_depth"]["high"] = 999
    assert load_base_space("xgboost")["max_depth"]["high"] != 999


# ---------- budget (Level 3) ----------
def test_split_budget_equal_and_exact():
    b = split_budget(10, ["a", "b", "c"])
    assert sum(b.values()) == 10 and max(b.values()) - min(b.values()) <= 1


def test_split_budget_with_weights():
    b = split_budget(30, ["a", "b"], weights={"a": 2, "b": 1})
    assert b == {"a": 20, "b": 10}


def test_split_budget_gives_every_family_at_least_one_trial():
    b = split_budget(3, ["a", "b", "c"], weights={"a": 100, "b": 1, "c": 1})
    assert sum(b.values()) == 3 and min(b.values()) >= 1


def test_split_budget_too_small_raises():
    with pytest.raises(ValueError):
        split_budget(2, ["a", "b", "c"])


# ---------- tuning (Level 2) ----------
def test_tune_family_records_every_trial(small_data, tmp_path):
    X, y = small_data
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    study = tune_family("logistic_regression", load_base_space("logistic_regression"), X, y,
                        TaskType.BINARY, "roc_auc", n_trials=4, folds=3, method="tpe",
                        study_name="test", storage=storage)
    rows = study_to_records(study, "logistic_regression", "tpe", 1, "run1")
    assert len(rows) == 4
    assert all(r["state"] == "COMPLETE" for r in rows)
    assert all(0.5 < r["mean_val"] <= 1.0 for r in rows)
    # the study was saved and can be loaded again from SQLite
    assert len(optuna.load_study(study_name="test", storage=storage).trials) == 4


def test_enqueued_params_are_tried_first(small_data):
    X, y = small_data
    study = tune_family("logistic_regression", load_base_space("logistic_regression"), X, y,
                        TaskType.BINARY, "roc_auc", n_trials=2, folds=3,
                        enqueue=[{"C": 0.5}])
    assert study.trials[0].params["C"] == 0.5


# ---------- defaults (Level 1) ----------
def test_defaults_use_same_row_format_as_hpo(small_data):
    X, y = small_data
    cfg = {**load_default_config(), "cv_folds": 3}
    c1 = C1Output(dataset_id="t", task_type=TaskType.BINARY, target="y", data_dir=".",
                  feature_types={}, fqs={})
    profile = TaskProfile(n_rows=300, n_features=6, n_categorical=0, categorical_ratio=0,
                          size_class="small", metric="roc_auc")
    rows = run_defaults(["logistic_regression"], X, y, c1, profile, cfg, run_id="run1")
    study = tune_family("logistic_regression", load_base_space("logistic_regression"), X, y,
                        TaskType.BINARY, "roc_auc", n_trials=1, folds=3)
    hpo_rows = study_to_records(study, "logistic_regression", "tpe", 1, "run1")
    assert set(rows[0]) == set(hpo_rows[0])
