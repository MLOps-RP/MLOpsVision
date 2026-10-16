from pathlib import Path

import yaml

# src/c2_automl/config.py -> parents[2] is the component root folder
ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


def load_default_config() -> dict:
    return load_yaml("default.yaml")
