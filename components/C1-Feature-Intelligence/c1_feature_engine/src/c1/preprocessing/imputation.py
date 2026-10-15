"""Candidate imputers (FR2). All return a mixed-type -> numeric-matrix preprocessor."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

IMPUTERS = {
    "mean": lambda: SimpleImputer(strategy="mean"),
    "median": lambda: SimpleImputer(strategy="median"),
    "median_ind": lambda: SimpleImputer(strategy="median", add_indicator=True),
    "knn": lambda: KNNImputer(n_neighbors=5),
    "mice": lambda: IterativeImputer(max_iter=5, random_state=0),
}


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(numeric, categorical) columns, dropping all-NaN ones. Call on the TRAIN fold only."""
    X = X.loc[:, X.notna().any()]
    num = X.select_dtypes("number").columns.tolist()
    return num, [c for c in X.columns if c not in num]


def make_preprocessor(imputer: str, X_train: pd.DataFrame) -> tuple[ColumnTransformer, int]:
    """Returns (preprocessor, n_onehot). Output = [one-hot block | numeric block], so the one-hot
    block is the first n_onehot columns (SMOTE-NC relies on this). Categories are fixed from X_train
    so that block's width never changes between inner folds."""
    num, cat = split_columns(X_train)
    cats = [sorted(X_train[c].dropna().unique()) for c in cat]
    num_pipe = Pipeline([("imp", IMPUTERS[imputer]()), ("sc", StandardScaler())])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(categories=cats, handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([("cat", cat_pipe, cat), ("num", num_pipe, num)])
    return pre, sum(len(c) for c in cats)
