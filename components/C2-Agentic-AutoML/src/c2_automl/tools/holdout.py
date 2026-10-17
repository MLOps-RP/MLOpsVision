"""Final fit + one-time hold-out evaluation (trusted tool). Only Agent 3 calls this, once per run."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, precision_score,
                             r2_score, recall_score, roc_auc_score, root_mean_squared_error)

from ..schemas import TaskType
from .models import make_model


def holdout_metrics(model, X_test: pd.DataFrame, y_test: pd.Series, task_type: TaskType) -> dict:
    if task_type == TaskType.REGRESSION:
        pred = model.predict(X_test)
        return {"rmse": float(root_mean_squared_error(y_test, pred)),
                "mae": float(mean_absolute_error(y_test, pred)),
                "r2": float(r2_score(y_test, pred))}

    pred = np.ravel(model.predict(X_test))
    proba = model.predict_proba(X_test)
    metrics = {"accuracy": float(accuracy_score(y_test, pred))}
    if task_type == TaskType.BINARY:
        metrics.update({
            "roc_auc": float(roc_auc_score(y_test, proba[:, 1])),
            "f1": float(f1_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred)),
        })
    else:
        metrics.update({
            "f1_macro": float(f1_score(y_test, pred, average="macro")),
            "roc_auc_ovr": float(roc_auc_score(y_test, proba, multi_class="ovr")),
        })
    return metrics


def fit_final_model(family: str, params: dict, X_train: pd.DataFrame, y_train: pd.Series,
                    task_type: TaskType, seed: int):
    """Refit the chosen configuration on ALL training data."""
    model = make_model(family, task_type, params, seed)
    model.fit(X_train, y_train)
    return model


def save_model(model, path: Path) -> float:
    """Save with joblib; returns the file size in KB (a simple complexity measure)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return round(path.stat().st_size / 1024, 1)
