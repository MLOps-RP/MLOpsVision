"""Gap 1 v2 study: wider model-aware search, 3 models, natural + controlled-stress data, rules vs random vs brute
vs fixed baselines. One CSV per (dataset, condition) in results/g1v2/ (resumable).
Usage: python -m experiments.run_g1v2 [--datasets a,b] [--models lgbm,logreg,rf] [--seeds 3] [--conds natural,stress] [--jobs 8]"""
import argparse
import time
import warnings
from pathlib import Path

import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from c1.preprocessing.v2 import ANCHOR, SelectorV2, fit_final
from c1.profiling.profiler import profile
from experiments.baselines import metrics
from experiments.datasets import ALL, load
from experiments.stress import stress

OUT = Path("results/g1v2")
METHODS = ["B0", "B_thr", "B_prac", "rules", "random", "brute"]


def fixed(method: str, model: str) -> dict:
    if method == "B0":
        return dict(ANCHOR)
    if method == "B_thr":
        return {**ANCHOR, "threshold": "tuned"}
    return {**ANCHOR, "clean": "clean", "balancer": "class_weight", "threshold": "tuned",   # B_prac = common best practice
            "scaler": "power" if model == "logreg" else "standard"}


def one(name, cond, models, seeds, cfg, rules):
    try:
        return _one(name, cond, models, seeds, cfg, rules)
    except Exception as e:   # a dataset that fails to load/run must not kill the sweep
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{name}__{cond}.err").write_text(repr(e))
        return None


def _one(name, cond, models, seeds, cfg, rules):
    warnings.filterwarnings("ignore")
    f = OUT / f"{name}__{cond}.csv"
    if f.exists():
        return str(f)
    X0, y0 = load(name, cfg["max_rows"], cfg["seed"])
    rows = []
    for s in range(seeds):
        seed = cfg["seed"] + s
        X, y = stress(X0, y0, seed) if cond == "stress" else (X0, y0)
        for fold, (tr, te) in enumerate(StratifiedKFold(cfg["outer_folds"], shuffle=True, random_state=seed).split(X, y)):
            Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]
            prof = profile(Xtr, ytr)
            for model in models:
                for method in METHODS:
                    t0, scores = time.time(), ""
                    try:
                        if method in ("rules", "random", "brute"):
                            d = SelectorV2(rules, model, cfg["inner_folds"], cfg["metric"], cfg["max_candidates"], seed, method).select(Xtr, ytr, prof)
                            opts, scores = d["opts"], str(d["scores"])
                        else:
                            opts = fixed(method, model)
                        m = metrics(fit_final(opts, model, Xtr, ytr, seed), Xte, yte)
                    except Exception as e:            # selected combo failing on the full train fold -> anchor
                        opts, m = dict(ANCHOR), metrics(fit_final(ANCHOR, model, Xtr, ytr, seed), Xte, yte)
                        opts["note"] = "fallback:" + type(e).__name__
                    rows.append({"dataset": name, "split": ALL[name][1], "cond": cond, "seed": seed, "fold": fold, "model": model,
                                 "method": method, **m, "opts": str(opts), "stage_scores": scores, "runtime_s": time.time() - t0,
                                 "n_rows": len(Xtr), "minority_share": float(min((ytr == c).mean() for c in set(ytr)))})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(f, index=False)
    return str(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(ALL))
    ap.add_argument("--models", default="lgbm,logreg,rf")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--conds", default="natural,stress")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/default.yaml"))
    rules = yaml.safe_load(open("configs/rules_all.yaml"))
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(n, c) for n in a.datasets.split(",") for c in a.conds.split(",")]
    Parallel(n_jobs=a.jobs, verbose=5)(delayed(one)(n, c, a.models.split(","), a.seeds, cfg, rules) for n, c in tasks)
