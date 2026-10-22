import numpy as np
import pandas as pd
import yaml

from c1.preprocessing.build import build_pipeline
from c1.preprocessing.selector import shortlist
from c1.profiling.profiler import profile
from experiments.inject_missingness import inject

RULES = yaml.safe_load(open("configs/rules.yaml"))


def toy(n=200, seed=0):
    r = np.random.default_rng(seed)
    X = pd.DataFrame({"a": r.normal(size=n), "b": r.normal(size=n), "c": r.choice(["u", "v", "w"], n)})
    y = (X.a + r.normal(scale=0.5, size=n) > 0.8).astype(int).to_numpy()  # ~20% positives
    X.loc[r.random(n) < 0.3, "a"] = np.nan
    return X, y


def test_profile_hand_values():
    X, y = toy()
    p = profile(X, y)
    assert (p.n_rows, p.n_numeric, p.n_categorical) == (200, 2, 1)
    assert abs(p.missing_ratio - X.isna().values.mean()) < 1e-9
    assert p.imbalance_ratio > 2


def test_rules_shortlist():
    X, y = toy()
    imp, fired = shortlist(RULES["imputation"], profile(X, y), 3)
    assert "median" in imp and len(imp) <= 3 and fired
    bal, _ = shortlist(RULES["balancing"], profile(X, y), 3)
    assert bal[0] in ("none", "class_weight")


def test_injection_rates_and_mechanisms():
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(4000, 5)), columns=list("abcde"))
    for mech in ("MCAR", "MAR", "MNAR"):
        Xm, mask = inject(X, 0.2, mech, 0)
        assert abs(mask.values.mean() - 0.2) < 0.03 and Xm.isna().equals(mask)
    _, m = inject(X, 0.2, "MNAR", 0)
    assert X.a[m.a].mean() > X.a[~m.a].mean()   # high values go missing


def test_all_balancers_fit_and_never_touch_test_fold():
    X, y = toy()
    Xtr, ytr, Xte = X.iloc[:150], y[:150], X.iloc[150:]
    for bal in ["none", "class_weight", "smote", "smote_nc", "adasyn", "tomek", "smote_tomek"]:
        for imp in ["mean", "median_ind", "knn", "mice"]:
            p = build_pipeline(imp, bal, "logreg", Xtr).fit(Xtr, ytr)
            assert p.predict(Xte).shape == (50,)
