import pytest

from c2_automl.agents.agent1_generation import (diagnose_family, eligible_families,
                                                plan_next_round, select_initial_families)
from c2_automl.config import load_default_config, load_yaml
from c2_automl.schemas import C1Output, DiagnosisLabel, TaskProfile, TaskType

CFG = load_default_config()
REGISTRY = load_yaml("model_registry.yaml")
RULES = load_yaml("diagnosis_rules.yaml")


def make_c1(task_type=TaskType.BINARY):
    return C1Output(dataset_id="t", task_type=task_type, target="y", data_dir=".",
                    feature_types={}, fqs={})


def make_profile(n_rows=5000, categorical_ratio=0.0):
    size = "small" if n_rows < 1000 else "large" if n_rows > 200000 else "medium"
    return TaskProfile(n_rows=n_rows, n_features=10, n_categorical=int(10 * categorical_ratio),
                       categorical_ratio=categorical_ratio, size_class=size, metric="roc_auc")


def make_signals(**overrides):
    s = {"mean_train": 0.90, "mean_val": 0.89, "gap": 0.01, "cv_std": 0.01, "baseline": 0.5,
         "lift_over_baseline": 0.39, "gap_to_best_family": 0.0, "mean_fqs": 0.4,
         "low_fqs_fraction": 0.1, "imbalance_ratio": 0.5, "minority_recall": 0.8}
    s.update(overrides)
    return s


# ---------- Level 1 ----------
def test_eligible_families_binary_excludes_ridge():
    fams = eligible_families("binary", REGISTRY)
    assert "ridge" not in fams and "xgboost" in fams


def test_blocked_family_removed():
    assert "catboost" not in eligible_families("binary", REGISTRY, blocked=["catboost"])


# ---------- Level 2 ----------
def test_small_dataset_avoids_lightgbm():
    fams, _ = select_initial_families(make_profile(n_rows=500), make_c1(), REGISTRY, CFG)
    assert "random_forest" in fams and "lightgbm" not in fams


def test_many_categoricals_adds_catboost():
    fams, entry = select_initial_families(make_profile(categorical_ratio=0.5), make_c1(), REGISTRY, CFG)
    assert "catboost" in fams
    assert "categorical" in entry.rationale


def test_regression_uses_ridge_baseline():
    fams, _ = select_initial_families(make_profile(n_rows=500), make_c1(TaskType.REGRESSION),
                                      REGISTRY, CFG)
    assert "ridge" in fams and "logistic_regression" not in fams


# ---------- Level 3 ----------
@pytest.mark.parametrize("overrides, expected", [
    ({"mean_train": 1.0, "gap": 0.15}, DiagnosisLabel.OVERFITTING),
    ({"mean_train": 0.70, "gap": 0.0}, DiagnosisLabel.UNDERFITTING),
    ({"imbalance_ratio": 0.03, "minority_recall": 0.2}, DiagnosisLabel.IMBALANCE),
    ({"lift_over_baseline": 0.01}, DiagnosisLabel.WEAK_FEATURES),
    ({"gap_to_best_family": 0.08}, DiagnosisLabel.UNSUITABLE_FAMILY),
    ({}, DiagnosisLabel.HEALTHY),
])
def test_diagnosis_rules(overrides, expected):
    d = diagnose_family("xgboost", make_signals(**overrides), RULES)
    assert expected in d.labels
    assert len(d.reasons) == len(d.labels)


def test_plan_drops_unsuitable_and_sends_actions():
    over = diagnose_family("random_forest", make_signals(mean_train=1.0, gap=0.15), RULES)
    bad = diagnose_family("logistic_regression", make_signals(gap_to_best_family=0.1), RULES)
    plan, _ = plan_next_round([over, bad], RULES, round_no=1)
    assert plan.families_next_round == ["random_forest"]
    assert "shrink_depth_range" in plan.actions["random_forest"]


def test_plan_refers_weak_features_to_c1_and_stops():
    weak = [diagnose_family(f, make_signals(lift_over_baseline=0.0), RULES) for f in ["a", "b"]]
    plan, _ = plan_next_round(weak, RULES, round_no=1)
    assert plan.stop and any("weak_features" in r for r in plan.referrals_to_c1)
