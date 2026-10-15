"""Shared test fixtures.

Most tests use a tiny synthetic model built here, so they run without the
generated fixtures in fixtures/ (which are gitignored binaries).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mqvg.config import GateConfig, load_config
from mqvg.environment import current_versions
from mqvg.loader import LoadedCandidate
from mqvg.schemas import CandidateBundle

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
FIXTURES = ROOT / "fixtures"
DATA = ROOT / "data"

FEATURES = ["age", "hours", "workclass"]


@pytest.fixture(scope="session")
def tiny_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame({
        "age": rng.integers(18, 70, n),
        "hours": rng.integers(10, 60, n),
        "workclass": rng.choice(["Private", "Gov", "Self"], n),
        "sex": rng.choice(["Female", "Male"], n),
    })
    signal = frame["age"] / 70 + frame["hours"] / 60 + rng.normal(0, 0.3, n)
    frame["target"] = np.where(signal > 1.0, ">50K", "<=50K")
    return frame


@pytest.fixture(scope="session")
def tiny_model(tiny_frame) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), ["age", "hours"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["workclass"]),
    ])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=500))])
    return model.fit(tiny_frame[FEATURES], tiny_frame["target"])


@pytest.fixture
def tiny_files(tmp_path, tiny_frame, tiny_model) -> dict[str, Path]:
    paths = {"model": tmp_path / "model.joblib",
             "eval": tmp_path / "eval.csv",
             "metadata": tmp_path / "metadata.json"}
    joblib.dump(tiny_model, paths["model"])
    tiny_frame.to_csv(paths["eval"], index=False)
    paths["metadata"].write_text(json.dumps({
        "model_type": "LogisticRegression", "task_type": "binary_classification",
        "target_column": "target", "feature_columns": FEATURES,
        "versions": current_versions(),
    }), encoding="utf-8")
    return paths


@pytest.fixture
def tiny_bundle(tiny_files) -> CandidateBundle:
    return CandidateBundle(model_path=tiny_files["model"],
                           eval_data_path=tiny_files["eval"],
                           metadata_path=tiny_files["metadata"])


@pytest.fixture(scope="session")
def default_config() -> GateConfig:
    return load_config(CONFIGS / "default.yaml")


@pytest.fixture
def make_config(tmp_path):
    """Build a config from the default with `mutate(dict)` applied."""
    def build(mutate=None) -> GateConfig:
        data = yaml.safe_load((CONFIGS / "default.yaml").read_text(encoding="utf-8"))
        if mutate:
            mutate(data)
        path = tmp_path / "custom.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return load_config(path)
    return build


@pytest.fixture
def candidate_with(tiny_frame, tmp_path):
    """Build a LoadedCandidate around any in-memory model (no pickling)."""
    def build(model, **overrides) -> LoadedCandidate:
        bundle = CandidateBundle(model_path=tmp_path / "m.joblib",
                                 eval_data_path=tmp_path / "e.csv")
        fields = dict(bundle=bundle, model=model, eval_df=tiny_frame.copy())
        fields.update(overrides)
        return LoadedCandidate(**fields)
    return build
