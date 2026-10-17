"""Evidence signals for diagnosis (trusted tool). Pure numbers, no decisions."""
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from ..schemas import TaskType


def compute_signals(result: dict, y: pd.Series, task_type: TaskType, baseline: float,
                    best_val_all_families: float, fqs: dict[str, float]) -> dict[str, float]:
    fqs_values = np.array(list(fqs.values())) if fqs else np.array([1.0])
    signals = {
        "mean_train": result["mean_train"],
        "mean_val": result["mean_val"],
        "gap": result["mean_train"] - result["mean_val"],
        "cv_std": result["cv_std"],
        "baseline": baseline,
        "lift_over_baseline": result["mean_val"] - baseline,
        "gap_to_best_family": best_val_all_families - result["mean_val"],
        "mean_fqs": float(fqs_values.mean()),
        "low_fqs_fraction": float((fqs_values < 0.1).mean()),
    }
    if task_type != TaskType.REGRESSION:
        signals.update(_class_signals(y, result["oof_pred"]))
    return {k: round(float(v), 4) for k, v in signals.items()}


def _class_signals(y: pd.Series, oof_pred: np.ndarray) -> dict[str, float]:
    counts = y.value_counts()
    minority_class = counts.idxmin()
    labels = list(counts.index)
    recalls = recall_score(y, oof_pred, labels=labels, average=None, zero_division=0)
    minority_recall = float(recalls[labels.index(minority_class)])
    return {
        "imbalance_ratio": float(counts.min() / counts.max()),
        "minority_recall": minority_recall,
        "min_class_recall": float(np.min(recalls)),
    }
