"""Hyperparameter search spaces (trusted tool). Spaces are plain dicts so they can be adapted later."""
import copy

import optuna

from ..config import load_yaml


def load_base_space(family: str) -> dict:
    """The static search space for one family (from configs/search_spaces.yaml)."""
    return copy.deepcopy(load_yaml("search_spaces.yaml")[family])


def suggest_params(trial: optuna.Trial, space: dict) -> dict:
    """Ask Optuna for one value per hyperparameter inside the given ranges."""
    params = {}
    for name, r in space.items():
        if r["type"] == "int":
            params[name] = trial.suggest_int(name, r["low"], r["high"], log=r.get("log", False))
        elif r["type"] == "float":
            params[name] = trial.suggest_float(name, r["low"], r["high"], log=r.get("log", False))
        elif r["type"] == "cat":
            params[name] = trial.suggest_categorical(name, r["choices"])
        else:
            raise ValueError(f"Unknown range type for {name}: {r['type']}")
    return params
