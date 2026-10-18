"""Exploration helper: print the shape, columns and labels of the Adult dataset.

Not used by make_fixtures.py.
"""
from fairlearn.datasets import fetch_adult

print("Loading UCI Adult dataset...")

adult = fetch_adult(as_frame=True)

print("Dataset loaded successfully!")
print("Dataset shape:", adult.data.shape)
print("Columns:", adult.data.columns.tolist())
print("Target values:")
print(adult.target.value_counts())
