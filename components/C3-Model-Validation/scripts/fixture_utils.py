"""Shared helpers for the fixture scripts.

These scripts only prepare sample assets for testing the validation gate.
They are not part of the gate and must never be imported from src/.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import subprocess

import joblib
import numpy as np
import pandas as pd
import sklearn
from fairlearn.datasets import fetch_adult
from fairlearn.metrics import demographic_parity_difference, selection_rate
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
SPLIT_RATIOS = {"train": 0.60, "eval": 0.20, "spare": 0.20}
SENSITIVE_COLUMNS = ["sex", "race"]
TARGET_COLUMN = "target"
POSITIVE_LABEL = ">50K"
NEGATIVE_LABEL = "<=50K"
DATA_SOURCE = "UCI Adult Income via fairlearn.datasets.fetch_adult (OpenML)"

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIXTURES_DIR = PROJECT_DIR / "fixtures"
TRAIN_ROW_HASHES_PATH = DATA_DIR / "train_row_hashes.txt"


def load_adult_splits():
    """Return (X, y, train_idx, eval_idx, spare_idx) using the fixed seed.

    Splits are 60/20/20, stratified on the target. Every script must use
    this function so that all fixtures share one train/eval split.
    """
    adult = fetch_adult(as_frame=True)
    X = adult.data.copy()
    y = adult.target.copy()
    labels = set(y.unique())
    if labels != {POSITIVE_LABEL, NEGATIVE_LABEL}:
        raise ValueError(f"Unexpected target labels {labels}; fixtures assume "
                         f"{POSITIVE_LABEL!r} and {NEGATIVE_LABEL!r}.")

    train_idx, temp_idx = train_test_split(
        X.index, test_size=0.40, random_state=SEED, stratify=y
    )
    eval_idx, spare_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=SEED, stratify=y.loc[temp_idx]
    )
    return X, y, train_idx, eval_idx, spare_idx


def row_hashes(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """SHA-256 of each row's feature values (the label is not included).

    Rows are rendered through to_csv so that an in-memory frame and the same
    data re-read from CSV (as strings) produce identical hashes. The gate's
    train/eval overlap check must use this same convention.
    """
    text = frame.loc[:, columns].astype(str).replace("nan", "").to_csv(
        index=False, header=False, lineterminator="\n"
    )
    return [hashlib.sha256(line.encode("utf-8")).hexdigest()
            for line in text.split("\n")[:-1]]


def write_text_lf(path: Path, text: str) -> None:
    """Write UTF-8 text with LF endings on every OS."""
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_csv_lf(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def read_csv_as_text(path: Path) -> pd.DataFrame:
    """Read a CSV without type inference so row hashes stay comparable."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def training_data_sha256(X_train: pd.DataFrame, y_train: pd.Series) -> str:
    frame = X_train.copy()
    frame[TARGET_COLUMN] = y_train
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


SOURCE_PATHS = ["scripts", "src", "configs"]


def code_commit() -> dict:
    """Git commit of the repo and whether the SOURCE code had uncommitted
    changes. Generated outputs (data/, fixtures/) are deliberately excluded,
    otherwise the flag would always be true right after generation."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        existing = [p for p in SOURCE_PATHS if (PROJECT_DIR / p).exists()]
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--", *existing], cwd=PROJECT_DIR,
            capture_output=True, text=True, check=True,
        ).stdout.strip())
        return {"commit": sha, "uncommitted_source_changes": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_source_changes": None}


def common_metadata(feature_columns: list[str], train_rows: int,
                    eval_rows: int, train_hash: str,
                    row_hash_columns: list[str] | None = None,
                    excluded_sensitive_features: list[str] | None = None) -> dict:
    """Metadata fields shared by every fixture."""
    return {
        "dataset": "UCI Adult Income",
        "data_source": DATA_SOURCE,
        "task_type": "binary_classification",
        "target_column": TARGET_COLUMN,
        "positive_label": POSITIVE_LABEL,
        "sensitive_columns": SENSITIVE_COLUMNS,
        "excluded_sensitive_features": (
            SENSITIVE_COLUMNS if excluded_sensitive_features is None
            else excluded_sensitive_features),
        "feature_columns": feature_columns,
        "random_state": SEED,
        "split": {
            "method": "random_stratified",
            "stratify_on": TARGET_COLUMN,
            "seed": SEED,
            "ratios": SPLIT_RATIOS,
        },
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "train_data_sha256": train_hash,
        "train_row_hashes_file": "data/train_row_hashes.txt",
        "row_hash_columns": row_hash_columns,
        "code": code_commit(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def build_preprocessor(X: pd.DataFrame, dense: bool = False) -> ColumnTransformer:
    """Impute/scale numeric columns and impute/one-hot categorical columns."""
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=not dense)),
        ]), categorical),
    ])


def _binary(labels) -> np.ndarray:
    return (np.asarray(labels) == POSITIVE_LABEL).astype(int)


def measure_model(model, X_train, y_train, eval_df: pd.DataFrame,
                  feature_columns: list[str]) -> dict:
    """Measured metrics recorded in fixture metadata.

    Evaluation uses the eval frame as read from CSV, exactly as the gate
    will see it. Fairness is computed on the eval sensitive columns.
    """
    train_pred = model.predict(X_train[feature_columns])
    pred = model.predict(eval_df[feature_columns])
    y_true = eval_df[TARGET_COLUMN]
    score = model.predict_proba(eval_df[feature_columns])[
        :, list(model.classes_).index(POSITIVE_LABEL)
    ]
    out = {
        "training_metrics": {
            "accuracy": float(accuracy_score(y_train, train_pred)),
            "f1": float(f1_score(y_train, train_pred, pos_label=POSITIVE_LABEL)),
        },
        "evaluation_metrics": {
            "accuracy": float(accuracy_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, pos_label=POSITIVE_LABEL)),
            "recall": float(recall_score(y_true, pred, pos_label=POSITIVE_LABEL)),
            "f1": float(f1_score(y_true, pred, pos_label=POSITIVE_LABEL)),
            "roc_auc": float(roc_auc_score(_binary(y_true), score)),
        },
    }
    if all(c in eval_df.columns for c in SENSITIVE_COLUMNS):
        fairness = {}
        for col in SENSITIVE_COLUMNS:
            groups = eval_df[col]
            fairness[f"demographic_parity_difference_by_{col}"] = float(
                demographic_parity_difference(
                    _binary(y_true), _binary(pred), sensitive_features=groups)
            )
            rates = {
                g: float(selection_rate(_binary(y_true)[groups == g],
                                        _binary(pred)[groups == g]))
                for g in sorted(groups.unique())
            }
            fairness[f"selection_rate_by_{col}"] = rates
            top = max(rates.values())
            fairness[f"selection_rate_ratio_by_{col}"] = (
                min(rates.values()) / top if top > 0 else None)
        out["fairness"] = fairness
    return out


def library_versions() -> dict:
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }


def save_fixture(name: str, model, metadata: dict) -> Path:
    out = FIXTURES_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    write_text_lf(out / "metadata.json",
                  json.dumps(metadata, indent=2, default=str) + "\n")
    return out
