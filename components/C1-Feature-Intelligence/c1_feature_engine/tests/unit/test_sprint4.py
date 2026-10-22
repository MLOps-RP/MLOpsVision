import joblib
import numpy as np

from c1.pipeline import FeatureIntelligenceEngine
from c1.selection.stability import nogueira
from tests.unit.test_sprint3 import _data


def test_nogueira():
    same = np.tile([1, 0, 1, 0, 0], (4, 1)).astype(bool)
    assert nogueira(same) == 1.0
    rnd = np.random.default_rng(0).random((30, 40)) < 0.3
    assert nogueira(rnd) < 0.2


def test_engine_end_to_end(tmp_path):
    X, y = _data(300)
    cfg = {"seed": 0, "inner_folds": 3, "metric": "f1_macro", "max_candidates": 3, "anchor_imputer": "median",
           "anchor_balancer": "none", "rules": "configs/rules.yaml"}
    e = FeatureIntelligenceEngine(cfg, boot=3, ga={"pop": 6, "gens": 2}).fit(X, y)
    assert e.selected_ and set(e.selected_) <= set(e.components_.index)
    c = e.components_.drop(columns="M").to_numpy()
    assert ((c >= 0) & (c <= 1)).all()
    Xs = e.transform(X)
    assert list(Xs.columns) == e.selected_
    e.select(use_ga=False, k=2)
    assert len(e.selected_) == 2
    assert e.feature_scores_[0].fqs >= e.feature_scores_[-1].fqs
    joblib.dump(e, tmp_path / "e.joblib")
    e.model_pipeline("lgbm", e.transform(X)).fit(e.transform(X), y)
