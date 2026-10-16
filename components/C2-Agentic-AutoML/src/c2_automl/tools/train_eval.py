"""Cross-validated training (trusted tool). Agent 1 reads these results; it never trains models itself."""
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from ..schemas import TaskType
from .models import make_model


def make_splitter(task_type: TaskType, folds: int, seed: int):
    if task_type == TaskType.REGRESSION:
        return KFold(n_splits=folds, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def cross_validate_family(family: str, X: pd.DataFrame, y: pd.Series, task_type: TaskType,
                          metric: str, folds: int = 5, seed: int = 42,
                          params: Optional[dict] = None,
                          on_fold: Optional[Callable[[int, list[float]], None]] = None) -> dict:
    """Train one model family with CV. Returns train/validation scores and out-of-fold predictions.

    on_fold(fold_index, val_scores_so_far) is called after each fold. Agent 2 uses it to let
    Optuna stop (prune) a bad trial early by raising optuna.TrialPruned.
    """
    scorer = get_scorer(metric)
    splitter = make_splitter(task_type, folds, seed)
    train_scores, val_scores, fit_times = [], [], []
    oof_pred = np.zeros(len(y), dtype=y.dtype)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        model = make_model(family, task_type, params, seed)
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        start = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_times.append(time.perf_counter() - start)
        train_scores.append(scorer(model, X_tr, y_tr))
        val_scores.append(scorer(model, X_va, y_va))
        oof_pred[val_idx] = np.ravel(model.predict(X_va))
        if on_fold is not None:
            on_fold(fold, val_scores)

    return {
        "family": family,
        "params": params or {},
        "train_scores": [float(s) for s in train_scores],
        "val_scores": [float(s) for s in val_scores],
        "mean_train": float(np.mean(train_scores)),
        "mean_val": float(np.mean(val_scores)),
        "cv_std": float(np.std(val_scores)),
        "fit_time_s": float(np.sum(fit_times)),
        "oof_pred": oof_pred,
    }


def baseline_score(X: pd.DataFrame, y: pd.Series, task_type: TaskType, metric: str,
                   folds: int = 5, seed: int = 42) -> float:
    """Score of a 'no-skill' model. Used to measure lift (are the features useful at all?)."""
    dummy = DummyRegressor() if task_type == TaskType.REGRESSION else DummyClassifier(strategy="prior")
    scores = cross_val_score(dummy, X, y, scoring=metric, cv=make_splitter(task_type, folds, seed))
    return float(np.mean(scores))
