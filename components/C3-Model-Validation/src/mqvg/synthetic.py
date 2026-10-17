"""Synthetic datasets with a PLANTED hidden failure group.

Used by the tests, the fixtures and the detection study, so that the true
failure region is known and detection can be scored. This is evaluation
tooling, not part of the validation checks.

A dataset has numeric and categorical features and a binary outcome driven by
a smooth rule. A pattern names a region of feature space; inside it the
outcome is flipped with probability `strength`, so a model that learned the
general rule fails there. Two ways to plant it:

* ``misspec``: the region is planted in the training data too. A model that
  cannot represent the region (for example logistic regression, which has no
  interactions) fails there anyway.
* ``shift``: the region is planted in the evaluation data only (a concept
  shift in a subpopulation). Any model fails there, however flexible.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

POSITIVE, NEGATIVE = "approved", "denied"
NUMERIC = ["age", "income_k", "tenure_years"]
CATEGORICAL = ["segment", "region", "channel"]
FEATURES = NUMERIC + CATEGORICAL
SENSITIVE = "sex"
TARGET = "outcome"

Pattern = Callable[[pd.DataFrame], np.ndarray]

# Each pattern names a region; sizes are approximate shares of the rows.
PATTERNS: dict[str, Pattern] = {
    # two features: tenure above 6 years in segment d (about 9%)
    "two_feature": lambda d: ((d["tenure_years"] > 6) & (d["segment"] == "d")).to_numpy(),
    # one numeric threshold: low income (about 9%)
    "single_threshold": lambda d: (d["income_k"] < 30).to_numpy(),
    # two categorical features (about 6%)
    "categorical_pair": lambda d: ((d["segment"] == "c") & (d["region"] == "e")).to_numpy(),
    # an oblique region: income and tenure both high (about 10%); not axis-aligned
    "oblique": lambda d: (((d["income_k"] - 50) / 15 + (d["tenure_years"] - 5) / 3) > 1.8).to_numpy(),
    # a small group (about 2%)
    "small_group": lambda d: ((d["tenure_years"] > 9) & (d["segment"] == "d")).to_numpy(),
}

# Patterns added AFTER the detector's design choices were made on the ones
# above, so that the final study includes regions it was never tuned on.
HELD_OUT_PATTERNS: dict[str, Pattern] = {
    # three categorical conditions at once (about 3%)
    "three_way_categorical": lambda d: ((d["segment"] == "a") & (d["region"] == "w")
                                        & (d["channel"] == "u")).to_numpy(),
    # young and lower income (about 7%): two numeric thresholds
    "young_low_income": lambda d: ((d["age"] < 28) & (d["income_k"] < 45)).to_numpy(),
}
PATTERNS.update(HELD_OUT_PATTERNS)

DEFAULT_STRENGTH = 0.8
SUBTLE_STRENGTH = 0.4


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_dataset(n: int, seed: int, pattern: str | None = None,
                 strength: float = DEFAULT_STRENGTH) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (frame, region_mask). The mask is all False when `pattern` is None."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "age": rng.integers(18, 70, n),
        "income_k": rng.normal(50, 15, n).round(1),
        "tenure_years": rng.normal(5, 3, n).clip(0, None).round(2),
        "segment": rng.choice(list("abcd"), n),
        "region": rng.choice(["n", "s", "e", "w"], n),
        "channel": rng.choice(["u", "v"], n),
        SENSITIVE: rng.choice(["Female", "Male"], n),
    })
    logit = (1.0 * (frame["income_k"] - 50) / 15 + 0.6 * (frame["tenure_years"] - 5) / 3
             - 0.4 * (frame["age"] - 44) / 15 + 0.5 * (frame["segment"] == "b"))
    outcome = rng.random(n) < _sigmoid(logit.to_numpy())
    region = np.zeros(n, bool)
    if pattern is not None:
        region = PATTERNS[pattern](frame)
        flip = region & (rng.random(n) < strength)
        outcome = np.where(flip, ~outcome, outcome)
    frame[TARGET] = np.where(outcome, POSITIVE, NEGATIVE)
    return frame, region


def make_pair(n_train: int, n_eval: int, seed: int, pattern: str | None,
              mode: str = "misspec", strength: float = DEFAULT_STRENGTH):
    """Training data, evaluation data and the evaluation-set region mask."""
    if mode not in ("misspec", "shift"):
        raise ValueError("mode must be 'misspec' or 'shift'")
    train, _ = make_dataset(n_train, seed, pattern if mode == "misspec" else None, strength)
    evaluation, region = make_dataset(n_eval, seed + 10_000, pattern, strength)
    return train, evaluation, region


def _preprocessor(dense: bool) -> ColumnTransformer:
    return ColumnTransformer([
        ("n", StandardScaler(), NUMERIC),
        ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense), CATEGORICAL)])


def fit_logistic(train: pd.DataFrame) -> Pipeline:
    """Linear model: cannot represent interactions, so it fails in planted regions."""
    return Pipeline([("p", _preprocessor(False)), ("m", LogisticRegression(max_iter=500))]
                    ).fit(train[FEATURES], train[TARGET])


def fit_gradient_boosting(train: pd.DataFrame) -> Pipeline:
    return Pipeline([("p", _preprocessor(True)),
                     ("m", HistGradientBoostingClassifier(max_iter=80, random_state=0))]
                    ).fit(train[FEATURES], train[TARGET])
