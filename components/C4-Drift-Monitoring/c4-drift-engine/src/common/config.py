"""Project paths and config loading, shared by every module."""
from pathlib import Path
import json

import yaml

# Project root = the folder that contains src/, configs/, data/ ...
ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = ROOT / "configs"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SCENARIOS = ROOT / "data" / "scenarios"
ARTIFACTS = ROOT / "artifacts"          # local copy of baselines / thresholds per model version
OUTPUTS = ROOT / "outputs"              # what C4 produces (reports, audit DB, window logs)
RESULTS = ROOT / "results"              # tables and charts for the slides

MODEL_NAME = "income-classifier"
EXPERIMENT_NAME = "c4-drift"
LABEL = "y"


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml as a dictionary."""
    with open(CONFIG_DIR / f"{name}.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def feature_lists() -> tuple[list[str], list[str]]:
    cfg = load_config("monitoring")
    return cfg["numeric_features"], cfg["categorical_features"]


def all_features() -> list[str]:
    num, cat = feature_lists()
    return num + cat


def model_dir(version) -> Path:
    """Local folder holding baseline / thresholds for one model version."""
    d = ARTIFACTS / f"model_v{version}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_to_builtin)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_builtin(o):
    """Make numpy numbers JSON-serialisable."""
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
    except ImportError:
        pass
    return str(o)
