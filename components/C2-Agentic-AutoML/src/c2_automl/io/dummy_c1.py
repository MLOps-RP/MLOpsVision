"""Simulates C1's output so C2 can be built before C1 is ready.

Only this file knows how C1 data is produced. When C1 is ready, replace it
with a real C1 client and keep the same C1Output contract.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split

from ..config import DATA_DIR, load_default_config
from ..schemas import C1Output, TaskType

# Public datasets from OpenML (downloaded automatically, then cached)
OPENML_DATASETS = {
    "adult": {"name": "adult", "version": 2, "task": TaskType.BINARY},
    "diabetes": {"name": "diabetes", "version": 1, "task": TaskType.BINARY},
}


def load_table(path: str) -> pd.DataFrame:
    """Read a CSV or Excel file (for Kaggle datasets you download yourself)."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _feature_types(X: pd.DataFrame) -> dict[str, str]:
    return {c: "numeric" if pd.api.types.is_numeric_dtype(X[c]) else "categorical"
            for c in X.columns}


def _basic_clean(X_train: pd.DataFrame, X_test: pd.DataFrame, feature_types: dict[str, str]):
    """Very simple cleaning. The real, adaptive cleaning is C1's job."""
    X_train, X_test = X_train.copy(), X_test.copy()
    for c, kind in feature_types.items():
        if kind == "numeric":
            median = X_train[c].median()
            X_train[c] = X_train[c].fillna(median)
            X_test[c] = X_test[c].fillna(median)
        else:
            train_vals = X_train[c].astype("object").fillna("missing").astype(str)
            test_vals = X_test[c].astype("object").fillna("missing").astype(str)
            categories = sorted(train_vals.unique())
            # unseen test categories become -1
            X_train[c] = pd.Categorical(train_vals, categories=categories).codes.astype("int32")
            X_test[c] = pd.Categorical(test_vals, categories=categories).codes.astype("int32")
    return X_train, X_test


def _encode_target(y: pd.Series, task_type: TaskType, positive_class: Optional[str]):
    """Binary: positive class -> 1 (default = minority class). Multiclass: 0..k-1."""
    if task_type == TaskType.REGRESSION:
        return y.astype(float), None
    y = y.astype(str)
    if task_type == TaskType.BINARY:
        positive_class = positive_class or y.value_counts().idxmin()
        return (y == positive_class).astype(int), positive_class
    classes = sorted(y.unique())
    return y.map({c: i for i, c in enumerate(classes)}).astype(int), None


def make_imbalanced(X: pd.DataFrame, y: pd.Series, minority_share: float, seed: int):
    """Downsample class 1 so it is only `minority_share` of all rows (for testing imbalance)."""
    rng = np.random.default_rng(seed)
    majority_idx = y.index[y == 0]
    minority_idx = y.index[y == 1]
    n_keep = int(minority_share / (1 - minority_share) * len(majority_idx))
    keep = rng.choice(minority_idx, size=min(n_keep, len(minority_idx)), replace=False)
    idx = majority_idx.append(pd.Index(keep))
    return X.loc[idx], y.loc[idx]


def build_dummy_c1(df: pd.DataFrame, target: str, task_type: TaskType, dataset_id: str,
                   positive_class: Optional[str] = None,
                   minority_share: Optional[float] = None) -> C1Output:
    cfg = load_default_config()
    seed = cfg["seed"]
    out_dir = DATA_DIR / "c1_output" / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.dropna(subset=[target]).reset_index(drop=True)
    X, y = df.drop(columns=[target]), df[target]
    y, positive_class = _encode_target(y, task_type, positive_class)
    if minority_share is not None:
        X, y = make_imbalanced(X, y, minority_share, seed)

    stratify = None if task_type == TaskType.REGRESSION else y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=stratify, random_state=seed)

    feature_types = _feature_types(X_train)
    X_train, X_test = _basic_clean(X_train, X_test, feature_types)

    # Fake Feature Quality Score = mutual information, scaled to 0..1
    discrete = [feature_types[c] == "categorical" for c in X_train.columns]
    mi_fn = mutual_info_regression if task_type == TaskType.REGRESSION else mutual_info_classif
    mi = mi_fn(X_train, y_train, discrete_features=discrete, random_state=seed)
    fqs = {c: round(float(v / (mi.max() + 1e-12)), 4) for c, v in zip(X_train.columns, mi)}

    X_train.to_parquet(out_dir / "X_train.parquet")
    X_test.to_parquet(out_dir / "X_test.parquet")
    y_train.to_frame(target).to_parquet(out_dir / "y_train.parquet")
    y_test.to_frame(target).to_parquet(out_dir / "y_test.parquet")

    class_distribution = None
    if task_type != TaskType.REGRESSION:
        class_distribution = {str(k): round(float(v), 4)
                              for k, v in y_train.value_counts(normalize=True).items()}

    c1 = C1Output(
        dataset_id=dataset_id, task_type=task_type, target=target, data_dir=str(out_dir),
        feature_types=feature_types, fqs=fqs, class_distribution=class_distribution,
        positive_class=positive_class,
        decision_log=["DUMMY C1: median imputation, ordinal encoding, mutual-information FQS"],
    )
    (out_dir / "manifest.json").write_text(c1.model_dump_json(indent=2))
    return c1


def build_from_openml(key: str, dataset_id: Optional[str] = None,
                      minority_share: Optional[float] = None) -> C1Output:
    spec = OPENML_DATASETS[key]
    data = fetch_openml(name=spec["name"], version=spec["version"], as_frame=True)
    df = data.frame
    target = data.target.name
    return build_dummy_c1(df, target, spec["task"], dataset_id or key,
                          minority_share=minority_share)


def load_c1_output(dataset_id: str) -> C1Output:
    path = DATA_DIR / "c1_output" / dataset_id / "manifest.json"
    return C1Output.model_validate_json(path.read_text())


def load_split(c1: C1Output, split: str = "train") -> tuple[pd.DataFrame, pd.Series]:
    """split = "train" or "test". C2 must only tune on "train"."""
    d = Path(c1.data_dir)
    X = pd.read_parquet(d / f"X_{split}.parquet")
    y = pd.read_parquet(d / f"y_{split}.parquet")[c1.target]
    return X, y
