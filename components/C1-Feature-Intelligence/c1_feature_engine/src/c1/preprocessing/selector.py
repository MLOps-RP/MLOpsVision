"""Strategy Selector (Gap 1): rule shortlist from the profile -> inner-CV pick among the shortlist."""
import operator
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, cross_val_score

from c1.preprocessing.balancing import BALANCERS
from c1.preprocessing.build import build_pipeline
from c1.preprocessing.imputation import IMPUTERS
from c1.schemas import DatasetProfile, StrategyDecision

_OPS = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge, "eq": operator.eq}


def load_rules(path: str) -> dict:
    return yaml.safe_load(open(path))


def shortlist(rules: list[dict], prof: DatasetProfile, cap: int) -> tuple[list[str], list[str]]:
    """Returns (candidates, names of fired rules). Rule order = priority."""
    p, cands, fired = (prof if isinstance(prof, dict) else prof.model_dump()), [], []
    for r in rules:
        if all(_OPS[op](p[k], v) for k, cond in r["when"].items() for op, v in cond.items()):
            fired.append(r["name"])
            cands += [c for c in r["propose"] if c not in cands]
    return cands[:cap], fired


class StrategySelector:
    def __init__(self, rules: dict, model: str = "lgbm", inner_folds: int = 3, metric: str = "f1_macro",
                 max_candidates: int = 3, anchor_imputer: str = "median", anchor_balancer: str = "none",
                 seed: int = 0, mode: str = "rules"):
        self.mode = mode   # rules | random (same shortlist size, profile ignored) | brute (all candidates): Gap 1 ablation
        self.rules, self.model, self.inner_folds, self.metric = rules, model, inner_folds, metric
        self.cap, self.a_imp, self.a_bal, self.seed = max_candidates, anchor_imputer, anchor_balancer, seed

    def _score(self, imp: str, bal: str, X: pd.DataFrame, y: np.ndarray) -> float:
        cv = StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.seed)
        pipe = build_pipeline(imp, bal, self.model, X, self.seed)
        # error_score=nan: a sampler that cannot run on this data just loses the comparison
        try:
            s = cross_val_score(pipe, X, y, cv=cv, scoring=self.metric, error_score=np.nan)
        except ValueError:                # sklearn raises when EVERY fold fails (e.g. ADASYN on this data)
            return float("-inf")
        return float(np.nanmean(s)) if not np.isnan(s).all() else float("-inf")

    def _pick(self, cands: list[str], anchor: str, evaluate) -> tuple[str, dict[str, float], bool]:
        if anchor not in cands:
            cands = [anchor] + cands          # anchor guarantees we never lose to the plain default by construction
        if len(cands) == 1:
            return cands[0], {}, False
        scores = {c: evaluate(c) for c in cands}
        return max(scores, key=lambda c: (scores[c], c == anchor)), scores, True  # ties -> anchor

    def select(self, X: pd.DataFrame, y, prof: DatasetProfile) -> StrategyDecision:
        t0, y = time.time(), np.asarray(y)
        imp_c, imp_f = shortlist(self.rules["imputation"], prof, self.cap)
        bal_c, bal_f = shortlist(self.rules["balancing"], prof, self.cap)
        if self.mode != "rules":
            rng = np.random.default_rng(self.seed)
            pick = lambda pool, n: list(rng.choice(sorted(pool), min(n, len(pool)), replace=False)) if n else []
            if self.mode == "random":
                imp_c, bal_c = pick(IMPUTERS, len(imp_c)), pick(BALANCERS, len(bal_c))
            else:
                imp_c, bal_c = sorted(IMPUTERS), sorted(BALANCERS)
        if prof.missing_ratio == 0:                       # nothing to impute: skip stage entirely
            imp_c = [self.a_imp]
        stage1_bal = "class_weight" if prof.imbalance_ratio >= 1.5 else "none"
        imp, imp_s, cv1 = self._pick(imp_c, self.a_imp, lambda c: self._score(c, stage1_bal, X, y))
        bal, bal_s, cv2 = self._pick(bal_c, self.a_bal, lambda c: self._score(imp, c, X, y))
        return StrategyDecision(imputer=imp, balancer=bal, imputer_scores=imp_s, balancer_scores=bal_s,
                                fired_rules=imp_f + bal_f, used_cv=cv1 or cv2, runtime_s=time.time() - t0)
