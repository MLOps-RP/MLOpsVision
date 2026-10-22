"""Gap 1 ablation: does the PROFILE matter? rules shortlist vs random shortlist (same size) vs brute force (all candidates).
Same outer CV as E1. Usage: python -m experiments.run_ablation [--models lgbm,logreg] [--seeds 1] [--jobs 4]"""
import argparse
import time
import warnings

import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from c1.preprocessing.build import build_pipeline
from c1.preprocessing.selector import StrategySelector, load_rules
from c1.profiling.profiler import profile
from experiments.baselines import FIXED, metrics
from experiments.datasets import REGISTRY, load

MODES = ["rules", "random", "brute"]


def one(name, models, seeds, cfg, rules):
    warnings.filterwarnings("ignore")
    X, y = load(name, cfg["max_rows"], cfg["seed"])
    rows = []
    for s in range(seeds):
        seed = cfg["seed"] + s
        for fold, (tr, te) in enumerate(StratifiedKFold(cfg["outer_folds"], shuffle=True, random_state=seed).split(X, y)):
            Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]
            prof = profile(Xtr, ytr)
            for model in models:
                for mode in MODES:
                    t0 = time.time()
                    d = StrategySelector(rules, model, cfg["inner_folds"], cfg["metric"], cfg["max_candidates"],
                                         cfg["anchor_imputer"], cfg["anchor_balancer"], seed, mode).select(Xtr, ytr, prof)
                    try:
                        m = metrics(build_pipeline(d.imputer, d.balancer, model, Xtr, seed).fit(Xtr, ytr), Xte, yte)
                    except Exception:   # sampler that worked in inner folds can still fail on the full train fold (ADASYN)
                        d.balancer = cfg["anchor_balancer"] + "(fallback)"
                        m = metrics(build_pipeline(d.imputer, cfg["anchor_balancer"], model, Xtr, seed).fit(Xtr, ytr), Xte, yte)
                    rows.append({"dataset": name, "split": REGISTRY[name][1], "seed": seed, "fold": fold, "model": model,
                                 "method": mode, **m, "imputer": d.imputer, "balancer": d.balancer, "runtime_s": time.time() - t0})
    return rows


def summarise(df):
    from scipy.stats import wilcoxon
    out = []
    for model, g in df.groupby("model"):
        for label, gg in [("ALL", g), ("HELD-OUT (test)", g[g.split == "test"]), ("DEV", g[g.split == "dev"])]:
            t = gg.groupby(["dataset", "method"])["f1_macro"].mean().unstack()[MODES]
            out += [f"## {model} / {label} ({len(t)} datasets)", t.round(3).to_markdown(),
                    f"mean: {t.mean().round(4).to_dict()}", f"runtime_s: {gg.groupby('method').runtime_s.mean().round(2).to_dict()}"]
            for a, b in [("rules", "random"), ("rules", "brute")]:
                d = t[a] - t[b]
                p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
                out.append(f"- {a} vs {b}: diff {d.mean():+.4f}, W/T/L {(d>0).sum()}/{(d==0).sum()}/{(d<0).sum()}, p={p:.3f}")
            out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lgbm,logreg")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/default.yaml"))
    rules = load_rules(cfg["rules"])
    res = Parallel(n_jobs=a.jobs)(delayed(one)(n, a.models.split(","), a.seeds, cfg, rules) for n in REGISTRY)
    df = pd.DataFrame([r for rs in res for r in rs])
    df.to_csv("results/ablation_folds.csv", index=False)
    md = summarise(df)
    open("results/ablation_summary.md", "w").write(md)
    print(md)
