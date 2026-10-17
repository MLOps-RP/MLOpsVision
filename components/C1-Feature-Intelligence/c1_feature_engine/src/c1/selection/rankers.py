"""Three rankers (FR5). Each returns a Series (index = feature name, higher = more important).
Run on TRAIN rows, on real (non-resampled) data; LightGBM is class-weighted instead (PLAN principle 2)."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE, mutual_info_classif


def prep(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Numeric matrix (cats -> codes, NaN -> median) + boolean 'is categorical' mask."""
    Z = X.copy()
    cat = np.array([not pd.api.types.is_numeric_dtype(Z[c]) for c in Z.columns])
    for c in Z.columns[cat]:
        Z[c] = Z[c].astype("category").cat.codes.astype(float)
    Z = Z.fillna(Z.median()).fillna(0)
    return Z.to_numpy(dtype=float), cat


def _lgbm(seed, n=100):
    return LGBMClassifier(n_estimators=n, class_weight="balanced", random_state=seed, n_jobs=1, verbose=-1)


def mi_rank(X: pd.DataFrame, y, seed: int = 0) -> pd.Series:
    Z, cat = prep(X)
    return pd.Series(mutual_info_classif(Z, y, discrete_features=cat, random_state=seed), index=X.columns)


def shap_rank(X: pd.DataFrame, y, seed: int = 0) -> pd.Series:
    """mean |SHAP| from LightGBM's native TreeSHAP (pred_contrib) - no extra dependency."""
    Z, _ = prep(X)
    m = _lgbm(seed).fit(Z, y)
    f = Z.shape[1]
    c = m.booster_.predict(Z, pred_contrib=True).reshape(len(Z), -1, f + 1)[:, :, :f]
    return pd.Series(np.abs(c).mean(axis=(0, 1)), index=X.columns)


def rfe_rank(X: pd.DataFrame, y, seed: int = 0) -> pd.Series:
    Z, _ = prep(X)
    r = RFE(_lgbm(seed, 50), n_features_to_select=1, step=0.25).fit(Z, y)
    return pd.Series(-r.ranking_.astype(float), index=X.columns)       # rank 1 (kept longest) -> highest score


def rank_all(X: pd.DataFrame, y, seed: int = 0, max_rows: int = 3000) -> pd.DataFrame:
    y = np.asarray(y)
    if len(X) > max_rows:                                               # row sampling for scoring (NFR5)
        idx = np.random.default_rng(seed).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], y[idx]
    return pd.DataFrame({"mi": mi_rank(X, y, seed), "shap": shap_rank(X, y, seed), "rfe": rfe_rank(X, y, seed)})
