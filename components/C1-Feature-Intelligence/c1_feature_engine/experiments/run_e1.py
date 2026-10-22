"""E1: adaptive Strategy Selector vs fixed strategies. Outer stratified CV, selector runs on train fold only.
Usage: python -m experiments.run_e1 [--datasets a,b] [--models lgbm,logreg] [--seeds 1] [--jobs 4]"""
import argparse
import warnings

import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from c1.preprocessing.selector import load_rules
from experiments.baselines import FIXED, run_split
from experiments.datasets import REGISTRY, load

METHODS = [*FIXED, "adaptive"]
KEY = "f1_macro"


def one(name: str, models: list[str], seeds: int, cfg: dict, rules: dict) -> list[dict]:
    warnings.filterwarnings("ignore")
    X, y = load(name, cfg["max_rows"], cfg["seed"])
    rows = []
    for s in range(seeds):
        seed = cfg["seed"] + s
        for fold, (tr, te) in enumerate(StratifiedKFold(cfg["outer_folds"], shuffle=True, random_state=seed).split(X, y)):
            for model in models:
                for m in METHODS:
                    r = run_split(m, X.iloc[tr], y[tr], X.iloc[te], y[te], model, cfg, rules, seed)
                    rows.append({"dataset": name, "seed": seed, "fold": fold, "model": model, "method": m, **r})
    return rows


def summarise(df: pd.DataFrame) -> str:
    from scipy.stats import wilcoxon
    out = []
    for model, g in df.groupby("model"):
        t = g.groupby(["dataset", "method"])[KEY].mean().unstack()[METHODS]
        out += [f"## {model}: mean {KEY} per dataset", t.round(3).to_markdown(), "",
                f"mean rank (1=best): {t.rank(axis=1, ascending=False).mean().round(2).to_dict()}", ""]
        for b in FIXED:
            d = t["adaptive"] - t[b]
            p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
            out.append(f"- adaptive vs {b}: mean diff {d.mean():+.4f}, wins/ties/losses "
                       f"{(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}, Wilcoxon p={p:.3f}")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(REGISTRY))
    ap.add_argument("--models", default="lgbm")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="results/e1")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/default.yaml"))
    rules = load_rules(cfg["rules"])
    res = Parallel(n_jobs=a.jobs)(delayed(one)(n, a.models.split(","), a.seeds, cfg, rules) for n in a.datasets.split(","))
    df = pd.DataFrame([r for rs in res for r in rs])
    df.to_csv(a.out + "_folds.csv", index=False)
    md = summarise(df)
    open(a.out + "_summary.md", "w").write(md)
    print(md)
