"""Assemble preprocess -> (resample) -> model as one imblearn Pipeline (no leakage by construction)."""
import pandas as pd
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from c1.preprocessing.balancing import make_sampler, uses_class_weight
from c1.preprocessing.imputation import make_preprocessor, split_columns


def make_model(name: str, class_weight: bool, seed: int):
    cw = "balanced" if class_weight else None
    if name == "lgbm":
        return LGBMClassifier(n_estimators=100, class_weight=cw, random_state=seed, n_jobs=1, verbose=-1)
    if name == "logreg":
        return LogisticRegression(max_iter=1000, class_weight=cw, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, class_weight=cw, random_state=seed, n_jobs=1)
    raise ValueError(name)


def build_pipeline(imputer: str, balancer: str, model: str, X_train: pd.DataFrame, seed: int = 0) -> Pipeline:
    pre, n_onehot = make_preprocessor(imputer, X_train)
    steps = [("pre", pre)]
    # SMOTE-NC needs >=1 numeric column; all-categorical data gets no resampling instead of a crash
    sampler = None if balancer == "smote_nc" and not split_columns(X_train)[0] else make_sampler(balancer, n_onehot, seed)
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", make_model(model, uses_class_weight(balancer), seed)))
    return Pipeline(steps)
