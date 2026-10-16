"""Step 4 tests: Agent 2 Level 4 (adaptive ranges, warm start) and Agent 1 Level 4 (loop control)."""
import random

import optuna
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from c2_automl.agents.agent1_generation import decide_next_round, round_budget
from c2_automl.config import load_yaml
from c2_automl.schemas import PlanDecision, TaskType
from c2_automl.tools.hpo import study_to_records, tune_family
from c2_automl.tools.search_space import (adapt_space, clip_to_space, in_space, load_base_space,
                                          range_edge_hits)

RULES = load_yaml("space_adaptation.yaml")
BINARY = {"task_type": "binary", "imbalance_ratio": 0.05}


def valid(space: dict) -> bool:
    for name, r in space.items():
        if r["type"] == "cat":
            continue
        lo, hi = RULES["limits"].get(name, [float("-inf"), float("inf")])
        if not (lo <= r["low"] < r["high"] <= hi):
            return False
        if r["type"] == "int" and not (isinstance(r["low"], int) and isinstance(r["high"], int)):
            return False
    return True


# ---------- Agent 2 Level 4: adapt_space ----------
def test_shrink_depth_lowers_max_depth_high():
    space = load_base_space("xgboost")
    new, changes = adapt_space(space, ["shrink_depth_range"], "xgboost", RULES)
    assert new["max_depth"]["high"] < space["max_depth"]["high"]
    assert any("max_depth.high" in c for c in changes)


def test_adapt_space_does_not_modify_its_input():
    space = load_base_space("xgboost")
    adapt_space(space, ["shrink_depth_range"], "xgboost", RULES)
    assert space == load_base_space("xgboost")


def test_increase_regularization_moves_reg_lambda_up():
    space = load_base_space("lightgbm")
    new, _ = adapt_space(space, ["increase_regularization"], "lightgbm", RULES)
    assert new["reg_lambda"]["low"] > space["reg_lambda"]["low"]
    assert new["min_child_samples"]["low"] > space["min_child_samples"]["low"]


def test_logistic_regression_regularization_lowers_C():
    space = load_base_space("logistic_regression")
    new, _ = adapt_space(space, ["increase_regularization"], "logistic_regression", RULES)
    assert new["C"]["high"] < space["C"]["high"]


@pytest.mark.parametrize("family, param", [("random_forest", "class_weight"),
                                           ("lightgbm", "class_weight"),
                                           ("catboost", "auto_class_weights"),
                                           ("xgboost", "scale_pos_weight")])
def test_enable_class_weights_adds_a_parameter(family, param):
    new, _ = adapt_space(load_base_space(family), ["enable_class_weights"], family, RULES, context=BINARY)
    assert param in new


def test_xgboost_scale_pos_weight_uses_imbalance_ratio():
    new, _ = adapt_space(load_base_space("xgboost"), ["enable_class_weights"], "xgboost", RULES,
                         context=BINARY)
    assert new["scale_pos_weight"]["high"] == pytest.approx(20.0)     # 1 / 0.05


def test_xgboost_scale_pos_weight_skipped_for_multiclass():
    new, _ = adapt_space(load_base_space("xgboost"), ["enable_class_weights"], "xgboost", RULES,
                         context={"task_type": "multiclass", "imbalance_ratio": 0.05})
    assert "scale_pos_weight" not in new


def test_unknown_and_family_level_actions_are_ignored():
    space = load_base_space("random_forest")
    new, changes = adapt_space(space, ["drop_family", "refer_features_to_c1"], "random_forest", RULES)
    assert new == space and changes == []


def test_repeated_random_actions_always_give_valid_ranges():
    rng = random.Random(0)
    actions = [a for a in RULES["actions"] if a != "enable_class_weights"]
    for family in ["random_forest", "xgboost", "lightgbm", "catboost", "logistic_regression"]:
        space = load_base_space(family)
        for _ in range(30):
            space, _ = adapt_space(space, rng.sample(actions, 2), family, RULES)
            assert valid(space), (family, space)


def test_range_edge_hits_detects_both_sides():
    space = {"a": {"type": "int", "low": 1, "high": 100}, "b": {"type": "float", "low": 0.0, "high": 1.0}}
    assert set(range_edge_hits({"a": 99, "b": 0.02}, space, 0.1)) == {("a", "high"), ("b", "low")}


def test_edge_rule_extends_untouched_param_only():
    space = load_base_space("xgboost")
    best = {"max_depth": 10, "n_estimators": 600}           # both at the high edge
    new, _ = adapt_space(space, ["shrink_depth_range"], "xgboost", RULES, best_params=best)
    assert new["max_depth"]["high"] < 10                     # diagnosis wins over the edge rule
    assert new["n_estimators"]["high"] > 600                 # untouched: extended


# ---------- warm start ----------
def test_in_space_and_clip_to_space():
    space = {"d": {"type": "int", "low": 3, "high": 6}, "c": {"type": "cat", "choices": ["x"]}}
    assert in_space({"d": 4, "c": "x"}, space)
    assert not in_space({"d": 9, "c": "x"}, space)
    assert clip_to_space({"d": 9, "c": "y"}, space) == {"d": 6}


@pytest.fixture(scope="module")
def small_data():
    X, y = make_classification(n_samples=300, n_features=6, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(6)]), pd.Series(y)


def test_warm_start_trials_are_used_but_not_counted(small_data):
    X, y = small_data
    space = load_base_space("logistic_regression")
    prior = [{"params": {"C": 0.5}, "mean_val": 0.9, "state": "COMPLETE"},
             {"params": {"C": 5000.0}, "mean_val": 0.8, "state": "COMPLETE"}]   # outside range: skipped
    study = tune_family("logistic_regression", space, X, y, TaskType.BINARY, "roc_auc",
                        n_trials=2, folds=3, prior_trials=prior)
    warm = [t for t in study.trials if t.user_attrs.get("warm_start")]
    assert len(warm) == 1
    assert len(study_to_records(study, "logistic_regression", "tpe", 2, "r")) == 2


# ---------- Agent 1 Level 4: loop control ----------
def plan(families=("a", "b"), stop=False):
    return PlanDecision(families_next_round=list(families), actions={}, stop=stop)


def test_round_budget_splits_remaining_over_remaining_rounds():
    assert round_budget(90, 0, 1, 3, 3) == 30
    assert round_budget(90, 30, 2, 3, 3) == 30
    assert round_budget(90, 60, 3, 3, 3) == 30
    assert round_budget(90, 89, 3, 3, 3) == 0          # not enough for one trial per family


@pytest.mark.parametrize("kwargs, expected", [
    (dict(plan=plan(stop=True)), "weak_features"),
    (dict(round_no=3), "max_rounds"),
    (dict(trials_used=89), "budget_used"),
    (dict(best_history=[0.80, 0.8001]), "converged"),
    (dict(best_history=[0.80, 0.81]), None),
    (dict(best_history=[0.80]), None),
])
def test_decide_next_round(kwargs, expected):
    args = dict(plan=plan(), round_no=2, best_history=[0.80, 0.81], trials_used=30,
                total_trials=90, max_rounds=3, min_improvement=0.0005)
    args.update(kwargs)
    reason, entry = decide_next_round(**args)
    assert reason == expected
    assert entry.decision["continue"] == (expected is None)
