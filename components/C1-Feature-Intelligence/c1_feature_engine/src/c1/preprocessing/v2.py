"""Gap 1 v2: wider, model-aware preprocessing search.
Stages (greedy, inner CV, anchor always evaluated): clean -> encoder -> scaler -> imputer -> balancer -> threshold."""
import time

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, StandardScaler, TargetEncoder

from c1.preprocessing.balancing import BALANCERS, make_sampler, uses_class_weight
from c1.preprocessing.build import make_model
from c1.preprocessing.imputation import IMPUTERS, split_columns
from c1.preprocessing.selector import shortlist

POOL = {"clean": ["none", "clean"], "encoder": ["onehot", "ordinal", "target"], "scaler": ["standard", "power", "none"],
        "imputer": sorted(IMPUTERS), "balancer": BALANCERS, "threshold": ["0.5", "tuned"]}
ANCHOR = {"clean": "none", "encoder": "onehot", "scaler": "standard", "imputer": "median", "balancer": "none", "threshold": "0.5"}
STAGES = list(POOL)
LINEAR = {"logreg"}
_RARE_MIN, _MAX_LEVELS = 5, 50


class Cleaner(BaseEstimator, TransformerMixin):
    """Static (train-derived) column drops + rare-level grouping."""
    def __init__(self, drop=(), rare=None):
        self.drop, self.rare = drop, rare

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(columns=[c for c in self.drop if c in X.columns])
        for c, keep in (self.rare or {}).items():
            if c in X.columns:
                X[c] = X[c].where(X[c].isin(keep) | X[c].isna(), "__rare__")
        return X


def clean_plan(X: pd.DataFrame) -> tuple[list[str], dict]:
    n, drop, rare = len(X), [], {}
    for c in X.columns:
        s = X[c]
        nu = s.nunique(dropna=True)
        numeric = pd.api.types.is_numeric_dtype(s)
        idlike = (not numeric and nu / n > 0.9) or (numeric and nu / n > 0.99 and bool((s.dropna() % 1 == 0).all())
                                                  and bool(s.dropna().is_monotonic_increasing))
        if nu <= 1 or s.isna().mean() > 0.5 or idlike:
            drop.append(c)
        elif not numeric:
            vc = s.value_counts()
            keep = vc[vc >= max(_RARE_MIN, 0.01 * n)].index.tolist()
            if len(keep) < len(vc):
                rare[c] = keep
    return drop, rare


def extra_profile(X: pd.DataFrame, model: str, y) -> dict:
    drop, rare = clean_plan(X)
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    card = [X[c].nunique() for c in cat]
    return {"n_junk_cols": len(drop), "n_rare_cols": len(rare), "max_card": max(card, default=0),
            "n_cat": len(cat), "is_linear": int(model in LINEAR), "is_binary": int(len(np.unique(y)) == 2)}


def _encoder(kind, cats):
    if kind == "onehot":
        return OneHotEncoder(categories=cats, handle_unknown="ignore", sparse_output=False)
    if kind == "ordinal":
        return OrdinalEncoder(categories=cats, handle_unknown="use_encoded_value", unknown_value=-1)
    return TargetEncoder(random_state=0)


def build_v2(opts: dict, model: str, X_train: pd.DataFrame, seed: int = 0) -> Pipeline:
    o = {**ANCHOR, **opts}
    steps = []
    if o["clean"] == "clean":
        drop, rare = clean_plan(X_train)
        cl = Cleaner(drop, rare)
        steps.append(("clean", cl))
        X_train = cl.transform(X_train)
    num, cat = split_columns(X_train)
    cats = [X_train[c].value_counts().loc[lambda v: v >= _RARE_MIN].index.sort_values()[:_MAX_LEVELS].tolist() or [""]
            for c in cat]
    sc = {"standard": StandardScaler, "power": lambda: PowerTransformer(standardize=True)}.get(o["scaler"])
    num_pipe = Pipeline([("imp", IMPUTERS[o["imputer"]]())] + ([("sc", sc())] if sc else []))
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", _encoder(o["encoder"], cats))])
    # categorical block FIRST so SMOTE-NC's leading-columns convention holds (target encoding -> plain numeric)
    n_cat = sum(len(c) for c in cats) if o["encoder"] == "onehot" else (len(cat) if o["encoder"] == "ordinal" else 0)
    blocks = ([("cat", cat_pipe, cat)] if cat else []) + ([("num", num_pipe, num)] if num else [])
    steps.append(("pre", ColumnTransformer(blocks)))
    bal = o["balancer"]
    sampler = None if bal == "smote_nc" and not num else make_sampler(bal, n_cat, seed)
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", make_model(model, uses_class_weight(bal), seed)))
    return Pipeline(steps)


class ThresholdedModel:
    """Fit pipeline; optionally pick the macro-F1-optimal threshold on OOF probabilities (binary only)."""
    def __init__(self, pipe, tuned=False, seed=0):
        self.pipe, self.tuned, self.seed, self.t = pipe, tuned, seed, 0.5

    def fit(self, X, y):
        if self.tuned:
            oof = cross_val_predict(clone(self.pipe), X, y, cv=StratifiedKFold(3, shuffle=True, random_state=self.seed),
                                    method="predict_proba")[:, 1]
            self.t = best_threshold(oof, np.asarray(y) == np.unique(y)[1])
        self.pipe.fit(X, y)
        self.classes_ = self.pipe.classes_
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[(p[:, 1] >= self.t).astype(int)] if self.tuned else self.classes_[p.argmax(1)]


_GRID = np.linspace(0.05, 0.95, 37)


def _f1(p, pos, t):
    return f1_score(pos, p >= t, average="macro", zero_division=0)


def best_threshold(p, pos) -> float:
    s = np.array([_f1(p, pos, t) for t in _GRID])
    return float(_GRID[s.argmax()])


def crossfit_f1(p, pos, seed=0, reps=3) -> float:
    """Honest score for 'tuned': choose t on one random half, score on the other (both ways, reps times)."""
    rng, out = np.random.default_rng(seed), []
    for _ in range(reps):
        idx = rng.permutation(len(p))
        a, b = idx[: len(p) // 2], idx[len(p) // 2:]
        for u, v in ((a, b), (b, a)):
            out.append(_f1(p[v], pos[v], best_threshold(p[u], pos[u])))
    return float(np.mean(out))


class SelectorV2:
    def __init__(self, rules: dict, model="lgbm", inner_folds=3, metric="f1_macro", cap=3, seed=0, mode="rules",
                 stages=STAGES):
        self.rules, self.model, self.k, self.metric, self.cap = rules, model, inner_folds, metric, cap
        self.seed, self.mode, self.stages = seed, mode, stages

    def _score(self, opts, X, y):
        cv = StratifiedKFold(self.k, shuffle=True, random_state=self.seed)
        try:
            s = cross_val_score(build_v2(opts, self.model, X, self.seed), X, y, cv=cv, scoring=self.metric,
                                error_score=np.nan)
        except Exception:
            return float("-inf")
        return float(np.nanmean(s)) if not np.isnan(s).all() else float("-inf")

    def _threshold_scores(self, opts, X, y):
        cv = StratifiedKFold(self.k, shuffle=True, random_state=self.seed)
        try:
            p = cross_val_predict(build_v2(opts, self.model, X, self.seed), X, y, cv=cv, method="predict_proba")[:, 1]
        except Exception:
            return {"0.5": 0.0, "tuned": float("-inf")}
        pos = np.asarray(y) == np.unique(y)[1]
        return {"0.5": _f1(p, pos, 0.5), "tuned": crossfit_f1(p, pos, self.seed)}

    def _cands(self, stage, p):
        pool = POOL[stage]
        if self.mode == "brute":
            return list(pool)
        c, _ = shortlist(self.rules[{"imputer": "imputation", "balancer": "balancing"}.get(stage, stage)], p, self.cap)
        if self.mode == "random":
            rng = np.random.default_rng(self.seed + STAGES.index(stage))
            c = list(rng.choice(pool, min(len(c), len(pool)), replace=False)) if c else []
        return c

    def select(self, X, y, prof) -> dict:
        t0, y = time.time(), np.asarray(y)
        p = {**prof.model_dump(), **extra_profile(X, self.model, y)}
        opts, scores = dict(ANCHOR), {}
        for st in self.stages:
            if st == "encoder" and p["n_cat"] == 0 or st == "imputer" and p["missing_ratio"] == 0:
                continue
            if st == "threshold" and not p["is_binary"]:
                continue
            c = self._cands(st, p)
            if ANCHOR[st] not in c:
                c = [ANCHOR[st]] + c
            if len(c) == 1:
                continue
            if st == "threshold":
                sc = self._threshold_scores(opts, X, y)
            else:
                sc = {v: self._score({**opts, st: v}, X, y) for v in c}
            scores[st] = sc
            opts[st] = max(sc, key=lambda v: (sc[v], v == ANCHOR[st]))
        return {"opts": opts, "scores": scores, "runtime_s": time.time() - t0}


def fit_final(opts: dict, model: str, X, y, seed=0) -> ThresholdedModel:
    return ThresholdedModel(build_v2(opts, model, X, seed), opts["threshold"] == "tuned", seed).fit(X, y)
