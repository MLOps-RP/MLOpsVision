"""Awkward 'foreign' model classes for the external-model tests.

They live in an importable module so joblib can pickle them by reference.
"""
import numpy as np


class NumpyOnlyModel:
    """predict() understands only numpy arrays, not DataFrames."""
    classes_ = np.array(["approved", "denied"])

    def predict(self, X):
        return np.where(X[:, 0] > 40, "approved", "denied")


class CodeOnLoad:
    """A pickle that runs code when it is loaded (writes a marker file)."""

    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (open, (self.marker, "w"))
