"""Tune Gap 2 selection on DEV datasets only, then confirm once on held-out. Uses results/gap2_cache (cache_gap2.py).
Usage: python -m experiments.tune_gap2 [--confirm]"""
import argparse
import itertools
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon

from c1.preprocessing.build import build_pipeline
from c1.selection.fqs import fqs
from c1.selection.greedy import corr_matrix, greedy_select
from c1.selection.redundancy import mean_pairwise
from c1.selection.stability import nogueira
from experiments.baselines import metrics
from experiments.cache_gap2 import OUT
from experiments.datasets import REGISTRY

WEIGHTS = {"PS": (.5, .5, 0), "P": (1, 0, 0), "PSm": (.4, .3, .15), "S+": (.3, .7, 0), "P+": (.7, .3, 0)}
GRID = [{"w": w, "lam": lam, "thr": thr} for w, lam, thr in itertools.product(WEIGHTS, (.25, .5, 1.0), (.8, .9, 1.01))]


def pick(eng, cfg, pool=60):
    """Greedy selection from cached components. rel = wP*P + wS*S - wM*M (set-aware R replaces the P-order R)."""
    wP, wS, wM = WEIGHTS[cfg["w"]]
    c = eng.components_
    rel = (wP * c["P"] + wS * c["S"] - wM * c["M"])
    cand = rel.nlargest(pool).index.tolist()
    C = corr_matrix(eng.G_, cand)
    return greedy_select(rel[cand], C, len(eng.selected_), cfg["lam"], cfg["thr"])


def stab(sets):
    U = sorted(set().union(*sets))
    return nogueira(np.array([[u in s for u in U] for s in sets])) if len(U) > 1 else 1.0


def evaluate(name, variants):
    warnings.filterwarnings("ignore")
    folds, rows, sets = joblib.load(OUT / f"{name}.joblib"), [], {}
    for v, f in variants.items():
        for i, fd in enumerate(folds):
            eng = fd["eng"]
            sel = eng.selected_ if f is None else pick(eng, f)
            a, b = eng.G_[sel], fd["Gte"][sel]
            m = metrics(build_pipeline(eng.imputer_, eng.balancer_, "lgbm", a, 42).fit(a, fd["ytr"]), b, fd["yte"])
            rows.append({"dataset": name, "variant": v, "fold": i, "f1": m["f1_macro"], "auc": m["roc_auc"], "k": len(sel),
                         "red": mean_pairwise(a, sel), "gen": np.mean([eng.lineage_[c]["op"] != "raw" for c in sel])})
            sets.setdefault(v, []).append(set(sel))
    return rows, {v: stab(s) for v, s in sets.items()}


def run(names, variants):
    res = Parallel(n_jobs=6)(delayed(evaluate)(n, variants) for n in names)
    df = pd.DataFrame([r for rs, _ in res for r in rs])
    st = pd.DataFrame([{"dataset": n, "variant": v, "stab": s} for n, (_, d) in zip(names, res) for v, s in d.items()])
    t = df.groupby("variant").agg(f1=("f1", "mean"), auc=("auc", "mean"), k=("k", "mean"), red=("red", "mean"), gen=("gen", "mean"))
    t["stab"] = st.groupby("variant").stab.mean()
    return df, t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", help="variant key 'w|lam|thr' to evaluate on held-out vs the current GA")
    a = ap.parse_args()
    key = lambda g: f"{g['w']}|{g['lam']}|{g['thr']}"
    if not a.confirm:
        dev = [n for n, (_, s) in REGISTRY.items() if s == "dev"]
        df, t = run(dev, {"ga_default": None} | {key(g): g for g in GRID})
        print(t.sort_values("f1", ascending=False).round(3).to_string())
    else:
        w, lam, thr = a.confirm.split("|")
        g = {"w": w, "lam": float(lam), "thr": float(thr)}
        test = [n for n, (_, s) in REGISTRY.items() if s == "test"]
        df, t = run(test, {"ga_default": None, "tuned": g})
        print(t.round(3).to_string())
        d = df.groupby(["dataset", "variant"]).f1.mean().unstack()
        print("f1 diff", (d.tuned - d.ga_default).mean().round(4), "p=", wilcoxon(d.tuned - d.ga_default).pvalue.round(3))
