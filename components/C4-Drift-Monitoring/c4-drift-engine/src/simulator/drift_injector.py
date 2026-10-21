"""
Drift injector: one function per drift type.

Every function receives ONE window of production rows (features + true label 'y'),
a strength factor between 0 and 1 (used for gradual drift), the scenario settings
and a random generator, and returns the modified window.
"""
import numpy as np
import pandas as pd

from src.common.config import LABEL


def _pick(rng: np.random.Generator, n: int, share: float) -> np.ndarray:
    """Boolean mask selecting about `share` of n rows."""
    return rng.random(n) < share


def covariate_numeric(w: pd.DataFrame, strength: float, s: dict, rng) -> pd.DataFrame:
    """
    Shift numeric inputs (e.g. age + 10) for a share of rows. Labels stay unchanged.
    Values are kept inside the column's normal range, so this is real drift,
    not a data-quality error.
    """
    w = w.copy()
    mask = _pick(rng, len(w), s["share"] * strength)
    for feature, shift in s["shifts"].items():
        lo, hi = w[feature].min(), w[feature].max()
        w.loc[mask, feature] = (w.loc[mask, feature] + shift).clip(lo, hi)
    return w


def covariate_categorical(w: pd.DataFrame, strength: float, s: dict, rng) -> pd.DataFrame:
    """Replace a categorical value for a share of rows (the population mix changes)."""
    w = w.copy()
    mask = _pick(rng, len(w), s["share"] * strength)
    w.loc[mask, s["feature"]] = rng.choice(s["new_values"], size=int(mask.sum()))
    return w


def concept(w: pd.DataFrame, strength: float, s: dict, rng) -> pd.DataFrame:
    """Inputs untouched; the true label is flipped for part of a segment (the rule changes)."""
    w = w.copy()
    seg = s.get("segment", "all")
    if seg == "all":
        in_segment = np.ones(len(w), dtype=bool)
    else:
        in_segment = np.ones(len(w), dtype=bool)
        for col, value in seg.items():
            in_segment &= (w[col] == value).to_numpy()
    flip = in_segment & _pick(rng, len(w), s["flip_prob"] * strength)
    w.loc[flip, LABEL] = 1 - w.loc[flip, LABEL]
    return w


def label_shift(w: pd.DataFrame, strength: float, s: dict, rng, pool: pd.DataFrame) -> pd.DataFrame:
    """
    Change the class mix: resample rows so that `target_positive_share` of them are >50K.
    Rows themselves are real and unchanged, only their proportions change.
    """
    base_share = pool[LABEL].mean()
    target = base_share + (s["target_positive_share"] - base_share) * strength
    n = len(w)
    n_pos = int(round(n * target))
    pos = pool[pool[LABEL] == 1]
    neg = pool[pool[LABEL] == 0]
    take_pos = pos.sample(n_pos, replace=n_pos > len(pos), random_state=int(rng.integers(1e9)))
    take_neg = neg.sample(n - n_pos, replace=(n - n_pos) > len(neg), random_state=int(rng.integers(1e9)))
    out = pd.concat([take_pos, take_neg]).sample(frac=1, random_state=int(rng.integers(1e9)))
    out["window"] = w["window"].iloc[0]
    return out.reset_index(drop=True)


def data_quality(w: pd.DataFrame, strength: float, s: dict, rng) -> pd.DataFrame:
    """Upstream bug: some values go missing and the rest arrive in the wrong unit."""
    w = w.copy()
    f = s["feature"]
    nulls = _pick(rng, len(w), s["null_share"] * strength)
    w[f] = w[f].astype(float)
    w.loc[~nulls, f] = w.loc[~nulls, f] * s["multiply"]
    w.loc[nulls, f] = np.nan
    return w


INJECTORS = {
    "covariate_numeric": covariate_numeric,
    "covariate_categorical": covariate_categorical,
    "concept": concept,
    "data_quality": data_quality,
}
