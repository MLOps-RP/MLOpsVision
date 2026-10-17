"""Model factory: the only place that creates model objects (trusted tool)."""
from typing import Optional

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from ..schemas import TaskType


def make_model(family: str, task_type: TaskType, params: Optional[dict] = None, seed: int = 42):
    is_clf = task_type != TaskType.REGRESSION
    p = dict(params or {})

    if family == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(**{"max_iter": 2000, **p}))
    if family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**p))
    if family == "random_forest":
        cls = RandomForestClassifier if is_clf else RandomForestRegressor
        return cls(**{"n_jobs": -1, "random_state": seed, **p})
    if family == "xgboost":
        cls = XGBClassifier if is_clf else XGBRegressor
        return cls(**{"n_jobs": -1, "random_state": seed, "verbosity": 0, **p})
    if family == "lightgbm":
        cls = LGBMClassifier if is_clf else LGBMRegressor
        return cls(**{"n_jobs": -1, "random_state": seed, "verbose": -1, **p})
    if family == "catboost":
        cls = CatBoostClassifier if is_clf else CatBoostRegressor
        return cls(**{"random_seed": seed, "verbose": 0, "allow_writing_files": False, **p})
    raise ValueError(f"Unknown model family: {family}")
