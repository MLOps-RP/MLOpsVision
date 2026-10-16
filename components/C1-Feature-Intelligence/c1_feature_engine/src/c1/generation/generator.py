"""Candidate pool with lineage (FR4). fit/transform on the TRAIN fold only; originals are always kept,
generated columns are deduplicated (constant / near-identical) and capped."""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold

from c1.generation.transforms import BINARY, UNARY, apply_bins, quantile_edges, smoothed_target_map
from c1.preprocessing.imputation import split_columns


class FeatureGenerator:
    def __init__(self, top_n: int = 8, cap: int = 100, max_card: int = 20, corr_thr: float = 0.999,
                 mi_rows: int = 3000, seed: int = 0):
        self.top_n, self.cap, self.max_card, self.thr, self.mi_rows, self.seed = top_n, cap, max_card, corr_thr, mi_rows, seed

    # ---- fit -------------------------------------------------------------------------------
    def _fit_stats(self, X: pd.DataFrame, y: np.ndarray):
        num, cat = split_columns(X)
        sub = X.sample(min(len(X), self.mi_rows), random_state=self.seed) if len(X) > self.mi_rows else X
        ys = y[X.index.get_indexer(sub.index)]
        if num:
            Z = sub[num].fillna(sub[num].median()).fillna(0)
            mi = pd.Series(mutual_info_classif(Z, ys, random_state=self.seed), index=num)
            self.top_ = mi.sort_values(ascending=False).index[: self.top_n].tolist()
        else:
            self.top_ = []
        self.orig_ = X.columns.tolist()
        self.cat_ = cat
        self.low_card_ = [c for c in cat if X[c].nunique() <= self.max_card][:3]
        self.ok_ = {c: {n: bool(ok(X[c].dropna())) if X[c].notna().any() else False for n, (_, ok) in UNARY.items()}
                    for c in self.top_}
        self.edges_ = {c: quantile_edges(X[c].dropna()) for c in self.top_ if X[c].notna().any()}
        self.pos_label_ = pd.Series(y).value_counts().idxmin()      # one-vs-rest target for TE (minority class)
        pos = (y == self.pos_label_).astype(float)
        self.prior_ = float(pos.mean())
        self.freq_ = {c: X[c].value_counts(normalize=True) for c in cat}
        self.te_ = {c: smoothed_target_map(X[c], pos, self.prior_) for c in cat}
        self.grp_ = {(n, c): X.groupby(c)[n].agg(["mean", "std"]) for c in self.low_card_ for n in self.top_[:3]}

    def _build(self, X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        out, lin = {}, {}

        def add(name, s, op, parents):
            out[name] = s.replace([np.inf, -np.inf], np.nan).astype(float)
            lin[name] = {"op": op, "parents": parents}

        for c in self.top_:
            for n, (f, _) in UNARY.items():
                if self.ok_[c][n]:
                    add(f"{n}({c})", f(X[c]), n, [c])
            if c in self.edges_:
                add(f"bin({c})", apply_bins(X[c], self.edges_[c]), "quantile_bin", [c])
        for a, b in combinations(self.top_, 2):
            for n, f in BINARY.items():
                add(f"{n}({a},{b})", f(X[a], X[b]), n, [a, b])
        for c in self.cat_:
            add(f"freq({c})", X[c].map(self.freq_[c]).fillna(0.0), "freq_enc", [c])
            add(f"te({c})", X[c].map(self.te_[c]).fillna(self.prior_), "target_enc", [c])
        for (n, c), g in self.grp_.items():
            add(f"{n}__mean_by__{c}", X[c].map(g["mean"]), "grp_mean", [n, c])
            add(f"{n}__std_by__{c}", X[c].map(g["std"]), "grp_std", [n, c])
        return pd.DataFrame(out, index=X.index), lin

    def _oof_te(self, X: pd.DataFrame, y: np.ndarray) -> dict[str, pd.Series]:
        """Out-of-fold target encoding for the training rows, so TE never sees its own label."""
        pos = (y == self.pos_label_).astype(float)
        res = {c: pd.Series(self.prior_, index=X.index) for c in self.cat_}
        k = int(min(5, pd.Series(y).value_counts().min()))
        if k < 2:
            return res
        for tr, va in StratifiedKFold(k, shuffle=True, random_state=self.seed).split(X, y):
            for c in self.cat_:
                m = smoothed_target_map(X[c].iloc[tr], pos[tr], self.prior_)
                res[c].iloc[va] = X[c].iloc[va].map(m).fillna(self.prior_).to_numpy()
        return res

    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        y = np.asarray(y)
        self._fit_stats(X, y)
        G, lin = self._build(X)
        for c, s in self._oof_te(X, y).items():
            G[f"te({c})"] = s
        # dedup against originals + earlier generated (greedy, original order)
        base = X[[c for c in self.orig_ if c in split_columns(X)[0]]]
        allc = pd.concat([base, G], axis=1)
        sub = allc.sample(min(len(allc), self.mi_rows), random_state=self.seed)
        Z = sub.fillna(sub.median()).fillna(0)
        std = Z.std().to_numpy()
        C = np.abs(np.nan_to_num(np.corrcoef(Z.to_numpy(), rowvar=False)))
        names, nb, keep = allc.columns.tolist(), base.shape[1], []
        for j in range(nb, len(names)):
            if std[j] == 0 or any(C[j, i] > self.thr for i in list(range(nb)) + keep):
                continue
            keep.append(j)
        kept = [names[j] for j in keep]
        if len(kept) > self.cap:                                  # ponytail: cap by MI; ranking proper is rankers.py
            ys = y[X.index.get_indexer(sub.index)]
            mi = pd.Series(mutual_info_classif(Z[kept], ys, random_state=self.seed), index=kept)
            top = set(mi.nlargest(self.cap).index)
            kept = [k for k in kept if k in top]
        self.keep_ = kept
        self.lineage_ = {c: {"op": "raw", "parents": []} for c in self.orig_} | {k: lin[k] for k in kept}
        return pd.concat([X, G[kept]], axis=1)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        G, _ = self._build(X)
        return pd.concat([X, G[self.keep_]], axis=1)
