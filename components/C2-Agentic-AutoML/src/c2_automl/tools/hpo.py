"""Optuna runner (trusted tool). Tunes ONE model family inside a given search space."""
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import RandomSampler, TPESampler

from ..schemas import TaskType
from .search_space import suggest_params
from .train_eval import cross_validate_family

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_sampler(method: str, seed: int) -> optuna.samplers.BaseSampler:
    if method == "tpe":
        return TPESampler(seed=seed, multivariate=True, n_startup_trials=10)
    if method == "random":
        return RandomSampler(seed=seed)
    raise ValueError(f"Unknown sampler: {method}")


def tune_family(family: str, space: dict, X: pd.DataFrame, y: pd.Series, task_type: TaskType,
                metric: str, n_trials: int, folds: int = 5, seed: int = 42,
                method: str = "tpe", pruning: bool = True, study_name: Optional[str] = None,
                storage: Optional[str] = None, enqueue: Optional[list[dict]] = None) -> optuna.Study:
    """Run one Optuna study. Every trial is a full CV run; evidence is stored as trial user attributes."""
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1) if pruning else NopPruner()
    study = optuna.create_study(direction="maximize", sampler=make_sampler(method, seed),
                                pruner=pruner, study_name=study_name, storage=storage)
    for params in enqueue or []:
        study.enqueue_trial(params)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, space)

        def on_fold(fold: int, val_scores: list[float]) -> None:
            trial.set_user_attr("val_scores", [float(s) for s in val_scores])
            trial.report(float(np.mean(val_scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result = cross_validate_family(family, X, y, task_type, metric, folds, seed,
                                       params=params, on_fold=on_fold)
        trial.set_user_attr("mean_train", result["mean_train"])
        trial.set_user_attr("cv_std", result["cv_std"])
        trial.set_user_attr("fit_time_s", result["fit_time_s"])
        return result["mean_val"]

    study.optimize(objective, n_trials=n_trials)
    return study


def study_to_records(study: optuna.Study, family: str, method: str, round_no: int,
                     run_id: str) -> list[dict]:
    """Turn every trial (complete AND pruned) into a flat row: this is the search history."""
    rows = []
    for t in study.trials:
        val_scores = t.user_attrs.get("val_scores", [])
        mean_train = t.user_attrs.get("mean_train")
        mean_val = t.value if t.value is not None else (float(np.mean(val_scores)) if val_scores else None)
        rows.append({
            "run_id": run_id, "method": method, "round": round_no, "family": family,
            "trial": t.number, "state": t.state.name, "params": t.params,
            "mean_val": mean_val, "mean_train": mean_train,
            "gap": None if mean_train is None or mean_val is None else mean_train - mean_val,
            "cv_std": t.user_attrs.get("cv_std"), "folds_done": len(val_scores),
            "fit_time_s": t.user_attrs.get("fit_time_s"),
            "duration_s": t.duration.total_seconds() if t.duration else None,
        })
    return rows
