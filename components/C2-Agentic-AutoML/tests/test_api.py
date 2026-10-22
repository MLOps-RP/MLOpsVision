"""Tests for the dashboard API (FastAPI). Uses a temporary data/output folder."""
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification

from c2_automl.schemas import TaskType


@pytest.fixture
def client(tmp_path, monkeypatch):
    import c2_automl.agents.agent2_hpo as agent2
    import c2_automl.api as api
    import c2_automl.config as config
    import c2_automl.io.dummy_c1 as dummy_c1
    monkeypatch.setattr(dummy_c1, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(agent2, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(api, "_RUNS", {})

    X, y = make_classification(n_samples=300, n_features=6, weights=[0.7], random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    dummy_c1.build_dummy_c1(df, "target", TaskType.BINARY, dataset_id="tiny")
    return TestClient(api.app)


def test_health_config_notebooks(client):
    assert client.get("/api/health").json()["status"] == "ok"
    cfg = client.get("/api/config").json()
    assert {"default", "diagnosis_rules", "space_adaptation"} <= set(cfg)
    nbs = client.get("/api/notebooks").json()["notebooks"]
    assert len(nbs) == 4 and all(nb["url"].endswith(".ipynb") for nb in nbs)


def test_datasets(client):
    items = client.get("/api/datasets").json()
    assert [d["dataset_id"] for d in items] == ["tiny"]
    detail = client.get("/api/datasets/tiny").json()
    assert detail["profile"]["n_rows"] == 240 and len(detail["fqs"]) == 6
    assert client.get("/api/datasets/missing").status_code == 404


def test_invalid_ids_are_rejected(client):
    assert client.get("/api/datasets/..%2Fsecret").status_code in (400, 404)
    assert client.get("/api/runs/bad%20id").status_code == 400
    assert client.get("/api/runs/x/files/secret.txt").status_code == 400


def test_upload_csv(client, tmp_path):
    X, y = make_classification(n_samples=120, n_features=4, random_state=1)
    df = pd.DataFrame(X, columns=list("abcd"))
    df["label"] = y
    r = client.post("/api/datasets", data={"target": "label", "task_type": "binary", "dataset_id": "up"},
                    files={"file": ("up.csv", df.to_csv(index=False), "text/csv")})
    assert r.status_code == 200, r.text
    bad = client.post("/api/datasets", data={"target": "nope", "dataset_id": "up2"},
                      files={"file": ("up2.csv", df.to_csv(index=False), "text/csv")})
    assert bad.status_code == 400


def test_run_end_to_end_through_api(client):
    r = client.post("/api/runs", json={"dataset_id": "tiny", "total_trials": 6, "folds": 3, "max_rounds": 2})
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    for _ in range(120):
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] != "running":
            break
        time.sleep(0.5)
    assert run["status"] == "completed", run.get("error")
    nodes = [e["node"] for e in run["events"]]
    assert nodes[0] == "load_input" and nodes[-1] == "package_output" and "agent2_adapt" in nodes
    assert len(client.get(f"/api/runs/{run_id}/trials").json()) == 6
    assert client.get(f"/api/runs/{run_id}/reasoning").json()
    ev = client.get(f"/api/runs/{run_id}/evaluation").json()
    assert ev["supported"] and 0.5 < ev["selected"]["auc"] <= 1
    assert client.get(f"/api/runs/{run_id}/files/c2_output.json").status_code == 200
    assert client.get("/api/runs").json()[0]["run_id"] == run_id
