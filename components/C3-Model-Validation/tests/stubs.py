"""Stub models for tests: they return prescribed predictions, no training."""
import numpy as np
import pandas as pd

from conftest import FEATURES

NEG, POS = "<=50K", ">50K"


class Scripted:
    """A model that returns prescribed predictions, looked up by row index."""
    feature_names_in_ = np.array(FEATURES)

    def __init__(self, pred, score=None, classes=(NEG, POS), label=lambda v: v):
        self.pred = pd.Series(pred)
        self.score = None if score is None else pd.Series(score, dtype=float)
        self.classes_ = np.array(classes)
        self.label = label
        self.predict_calls = 0

    def predict(self, X):
        self.predict_calls += 1
        return np.array([self.label(v) for v in self.pred.loc[X.index]])

    def predict_proba(self, X):
        if self.score is None:
            raise AttributeError("no scores")
        s = self.score.loc[X.index].to_numpy()
        return np.column_stack([1 - s, s])


class ScriptedNoProba(Scripted):
    def __getattribute__(self, name):
        if name == "predict_proba":
            raise AttributeError(name)
        return super().__getattribute__(name)


class DecisionOnly(ScriptedNoProba):
    def decision_function(self, X):
        return self.score.loc[X.index].to_numpy() - 0.5
