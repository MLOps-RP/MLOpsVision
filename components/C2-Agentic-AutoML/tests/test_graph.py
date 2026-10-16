import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from c2_automl.agents.agent3_selection import select_best
from c2_automl.graph import build_graph
from c2_automl.schemas import C2Output, TaskType
from c2_automl.tools.holdout import fit_final_model, holdout_metrics

EXPECTED_NODES = ["load_input", "profile_task", "agent1_select", "agent2_hpo", "evaluate",
                  "agent1_diagnose", "agent2_adapt", "agent3_select", "package_output"]


# ---------- graph structure ----------
def test_graph_has_all_nodes():
    nodes = set(build_graph().get_graph().nodes)
    assert set(EXPECTED_NODES) <= nodes


# ---------- Agent 3 Level 1 ----------
def test_select_best_picks_highest_cv_score():
    candidates = {"a": {"mean_val": 0.80, "params": {}}, "b": {"mean_val": 0.85, "params": {"x": 1}}}
    winner, entry = select_best(candidates, round_no=1)
    assert winner == "b"
    assert "0.0500 ahead of a" in entry.rationale


def test_select_best_without_candidates_raises():
    with pytest.raises(ValueError):
        select_best({}, round_no=1)


def test_holdout_metrics_binary():
    X, y = make_classification(n_samples=200, n_features=5, random_state=0)
    X, y = pd.DataFrame(X), pd.Series(y)
    model = fit_final_model("logistic_regression", {}, X[:150], y[:150], TaskType.BINARY, seed=0)
    m = holdout_metrics(model, X[150:], y[150:], TaskType.BINARY)
    assert set(m) == {"accuracy", "roc_auc", "f1", "precision", "recall"}


# ---------- end-to-end (small synthetic dataset, temporary folders) ----------
@pytest.fixture
def tmp_project(tmp_path, monkeypatch):
    """Redirect data and outputs to a temp folder so the test never touches real files."""
    import c2_automl.agents.agent2_hpo as agent2
    import c2_automl.config as config
    import c2_automl.io.dummy_c1 as dummy_c1
    monkeypatch.setattr(dummy_c1, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(agent2, "OUTPUT_DIR", tmp_path / "outputs")

    X, y = make_classification(n_samples=300, n_features=6, weights=[0.7], random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    dummy_c1.build_dummy_c1(df, "target", TaskType.BINARY, dataset_id="tiny")
    return tmp_path


def test_single_round_workflow(tmp_project):
    final = build_graph().invoke({"dataset_id": "tiny", "total_trials": 6, "folds": 3,
                                  "overrides": {"budget": {"max_rounds": 1}}})

    out = C2Output(**final["output"])                  # output matches the contract
    assert out.total_trials == 6 and out.rounds == 1 and out.stop_reason == "max_rounds"
    assert out.model_family in final["families"]
    assert "roc_auc" in out.metrics_holdout
    for path in (out.selected_model_path, out.search_history_path, out.reasoning_log_path):
        assert Path(path).exists()

    agents = {e["agent"] for e in json.loads(Path(out.reasoning_log_path).read_text())}
    assert agents == {"agent1", "agent2", "agent3"}


def test_adaptive_loop_runs_several_rounds(tmp_project):
    final = build_graph().invoke({"dataset_id": "tiny", "total_trials": 12, "folds": 3,
                                  "overrides": {"budget": {"max_rounds": 2},
                                                "adaptive": {"min_improvement": -1}}})
    out = C2Output(**final["output"])
    assert out.rounds == 2 and out.stop_reason == "max_rounds"
    assert out.total_trials == 12                       # never more than the budget
    assert len(out.score_by_round) == 2
    assert out.score_by_round[1] >= out.score_by_round[0]   # best-so-far never gets worse
    assert {r["round"] for r in final["trial_rows"]} == {1, 2}
    steps = [e["step"] for e in final["reasoning_log"]]
    assert "adapt_spaces" in steps and steps.count("decide_next_round") == 2
