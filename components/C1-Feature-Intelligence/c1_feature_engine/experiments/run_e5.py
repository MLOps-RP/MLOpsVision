"""E5: FQS ablation (drop each term), weight sensitivity, with/without GA. Engine is fit once per fold; variants only re-run
FQS + subset search on the cached P/S/M/R. Model: lgbm, C1's chosen prep.
Usage: python -m experiments.run_e5 [--datasets a,b] [--seeds 1] [--jobs 4]"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from c1.pipeline import FeatureIntelligenceEngine
from c1.selection.redundancy import mean_pairwise
from experiments.baselines import metrics
from experiments.datasets import REGISTRY, load
from experiments.run_e4 import stability

VARIANTS = {  # name: (weights, use_ga)
    "full": (None, True),
    "no_P": ({"P": 0}, True), "no_S": ({"S": 0}, True), "no_M": ({"M": 0}, True), "no_R": ({"R": 0}, True),
    "no_GA(topk)": (None, False),
    "w_equal": ({"P": .25, "S": .25, "M": .25, "R": .25}, True),
    "w_P_heavy": ({"P": .7, "S": .1, "M": .1, "R": .1}, True),
    "w_S_R_heavy": ({"P": .2, "S": .3, "M": .1, "R": .4}, True),
}


def one(name, seeds, cfg, model="lgbm"):
    from c1.preprocessing.build import build_pipeline
    warnings.filterwarnings("ignore")
    X, y = load(name, cfg["max_rows"], cfg["seed"])
    rows = []
    for s in range(seeds):
        seed = cfg["seed"] + s
        for fold, (tr, te) in enumerate(StratifiedKFold(cfg["outer_folds"], shuffle=True, random_state=seed).split(X, y)):
            Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]
            eng = FeatureIntelligenceEngine(cfg, seed=seed).fit(Xtr, ytr)
            Gte, k_full = eng.gen_.transform(Xte), len(eng.selected_)
            for v, (w, ga) in VARIANTS.items():
                sel = eng.selected_ if v == "full" else eng.select(w, ga, k_full)
                a, b = eng.G_[sel], Gte[sel]
                pipe = build_pipeline(eng.imputer_, eng.balancer_, model, a, seed).fit(a, ytr)
                rows.append({"dataset": name, "split": REGISTRY[name][1], "seed": seed, "fold": fold, "variant": v,
                             **metrics(pipe, b, yte), "n_features": len(sel), "redundancy": mean_pairwise(a, sel),
                             "features": json.dumps(sel)})
    return rows


def summarise(df):
    from scipy.stats import wilcoxon
    out, V = [], list(VARIANTS)
    for label, g in [("ALL", df), ("HELD-OUT (test)", df[df.split == "test"])]:
        if g.empty:
            continue
        t = g.groupby(["dataset", "variant"])["f1_macro"].mean().unstack()[V]
        tab = g.groupby("variant").agg(f1=("f1_macro", "mean"), roc_auc=("roc_auc", "mean"), n_feat=("n_features", "mean"),
                                       redundancy=("redundancy", "mean")).loc[V]
        tab["stability"] = [stability(g[g.variant == v].rename(columns={"variant": "arm"})) for v in V]
        d = t.sub(t["full"], axis=0)
        tab["diff_vs_full"] = d.mean()
        tab["p_vs_full"] = [wilcoxon(t[v] - t["full"]).pvalue if (t[v] != t["full"]).any() else 1.0 for v in V]
        out += [f"## {label} ({len(t)} datasets, lgbm)", tab.round(3).to_markdown(), ""]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(REGISTRY))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="results/e5")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/default.yaml"))
    res = Parallel(n_jobs=a.jobs)(delayed(one)(n, a.seeds, cfg) for n in a.datasets.split(","))
    df = pd.DataFrame([r for rs in res for r in rs])
    df.to_csv(a.out + "_folds.csv", index=False)
    md = summarise(df)
    open(a.out + "_summary.md", "w").write(md)
    print(md)
