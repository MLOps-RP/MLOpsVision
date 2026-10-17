"""FeatureIntelligenceEngine: profile -> strategy selection -> generation -> P/S/M/R -> FQS -> GA subset."""
import math
import time

import numpy as np
import pandas as pd
import yaml

from c1.generation.generator import FeatureGenerator
from c1.preprocessing.build import build_pipeline
from c1.preprocessing.selector import StrategySelector, load_rules
from c1.profiling.profiler import profile
from c1.selection.aggregate import aggregate
from c1.selection.evolutionary import ga_select
from c1.selection.fqs import feature_scores, fqs, missing_ratio
from c1.selection.rankers import prep, rank_all
from c1.selection.redundancy import redundancy
from c1.selection.stability import selection_frequency


class FeatureIntelligenceEngine:
    def __init__(self, cfg: dict | str = "configs/default.yaml", weights: dict | None = None, seed: int = 0,
                 use_selector: bool = True, use_generation: bool = True, max_genes: int = 60, boot: int = 10,
                 ga: dict | None = None):
        self.cfg = yaml.safe_load(open(cfg)) if isinstance(cfg, str) else cfg
        self.weights, self.seed, self.use_selector, self.use_generation = weights, seed, use_selector, use_generation
        self.max_genes, self.boot, self.ga = max_genes, boot, ga or {}

    def fit(self, X: pd.DataFrame, y):
        t0, y, c = time.time(), np.asarray(y), self.cfg
        self.profile_ = profile(X, y)
        if self.use_selector:
            sel = StrategySelector(load_rules(c["rules"]), "lgbm", c["inner_folds"], c["metric"], c["max_candidates"],
                                   c["anchor_imputer"], c["anchor_balancer"], self.seed)
            self.decision_ = sel.select(X, y, self.profile_)
            self.imputer_, self.balancer_ = self.decision_.imputer, self.decision_.balancer
        else:
            self.imputer_, self.balancer_ = c["anchor_imputer"], "class_weight"
        self.gen_ = FeatureGenerator(seed=self.seed) if self.use_generation else None
        G = self.gen_.fit_transform(X, y) if self.gen_ else X
        self.lineage_ = self.gen_.lineage_ if self.gen_ else {n: {"op": "raw", "parents": []} for n in X.columns}
        self.k0_ = max(2, math.ceil(X.shape[1] / 2))
        P = aggregate(rank_all(G, y, self.seed))
        S = selection_frequency(G, y, self.k0_, self.boot, self.seed)
        R = redundancy(G, P.sort_values(ascending=False).index.tolist(), seed=self.seed)
        M = missing_ratio(X, self.lineage_)
        self.components_ = pd.DataFrame({"P": P, "S": S, "M": M, "R": R}).loc[G.columns]
        self.G_, self.y_ = G, y
        self.select()
        self.fit_s_ = time.time() - t0
        return self

    def select(self, weights: dict | None = None, use_ga: bool = True, k: int | None = None) -> list[str]:
        """(Re)choose the subset from cached components, so ablations don't recompute rankings."""
        w = weights if weights is not None else self.weights
        q = fqs(self.components_, w)
        cand = q.nlargest(self.max_genes).index.tolist()
        if use_ga:
            Z, _ = prep(self.G_[cand])
            mask, self.ga_history_ = ga_select(Z, self.y_, q[cand].to_numpy(), self.k0_, seed=self.seed, **self.ga)
            chosen = [c for c, m in zip(cand, mask) if m]
        else:
            chosen = q.nlargest(k or self.k0_).index.tolist()
        self.selected_ = chosen
        self.feature_scores_ = feature_scores(self.components_, self.lineage_, w, chosen)
        return chosen

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        G = self.gen_.transform(X) if self.gen_ else X
        return G[self.selected_]

    def model_pipeline(self, model: str, X_sel: pd.DataFrame):
        """Final training pipeline using the chosen imputer/balancer (resampling only inside fit)."""
        return build_pipeline(self.imputer_, self.balancer_, model, X_sel, self.seed)
