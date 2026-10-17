import pandas as pd

from ..schemas import C1Output, TaskProfile, TaskType


def profile_task(c1: C1Output, X: pd.DataFrame, y: pd.Series, cfg: dict) -> TaskProfile:
    """Describe the dataset so Agent 1 can choose suitable model families."""
    p = cfg["profile"]
    n_rows, n_features = X.shape
    n_categorical = sum(1 for t in c1.feature_types.values() if t == "categorical")

    if n_rows < p["small_rows"]:
        size_class = "small"
    elif n_rows > p["large_rows"]:
        size_class = "large"
    else:
        size_class = "medium"

    n_classes = imbalance_ratio = None
    if c1.task_type != TaskType.REGRESSION:
        counts = y.value_counts()
        n_classes = int(len(counts))
        imbalance_ratio = round(float(counts.min() / counts.max()), 4)

    return TaskProfile(
        n_rows=n_rows, n_features=n_features, n_categorical=n_categorical,
        categorical_ratio=round(n_categorical / max(n_features, 1), 4),
        size_class=size_class, metric=cfg["metrics"][c1.task_type.value],
        n_classes=n_classes, imbalance_ratio=imbalance_ratio,
    )
