import numpy as np
import pandas as pd

from c1.generation.generator import FeatureGenerator
from c1.selection.aggregate import aggregate
from c1.selection.rankers import rank_all


def _data(n=400, seed=0):
    r = np.random.default_rng(seed)
    X = pd.DataFrame({"a": r.normal(size=n), "b": r.gamma(2, size=n), "c": r.normal(size=n),
                      "cat": r.choice(list("xyz"), n)})
    X.loc[r.random(n) < 0.1, "a"] = np.nan
    y = ((X["b"] / 2 + X["c"].fillna(0)) + r.normal(scale=.3, size=n) > 1.2).astype(int).to_numpy()
    return X, y


def test_generator_consistent_and_capped():
    X, y = _data()
    g = FeatureGenerator(cap=10, seed=0)
    Xtr = g.fit_transform(X.iloc[:300], y[:300])
    Xte = g.transform(X.iloc[300:])
    assert list(Xtr.columns) == list(Xte.columns) and list(Xtr.columns[:4]) == list(X.columns)
    assert Xtr.shape[1] - X.shape[1] <= 10
    assert set(Xtr.columns) <= set(g.lineage_)
    assert not np.isinf(Xte.select_dtypes("number").to_numpy()).any()


def test_te_is_out_of_fold():
    X, y = _data()
    g = FeatureGenerator(cap=500, seed=0)
    Xtr = g.fit_transform(X, y)
    full = X["cat"].map(g.te_["cat"])
    assert not np.allclose(Xtr["te(cat)"], full)          # OOF differs from in-sample map


def test_rankers_and_aggregate():
    X, y = _data()
    s = rank_all(X, y)
    assert s.shape == (4, 3) and np.isfinite(s.to_numpy()).all()
    P = aggregate(s)
    assert P.between(0, 1).all() and P.idxmax() in ("b", "c")
    assert aggregate(s, "borda").between(0, 1).all()
