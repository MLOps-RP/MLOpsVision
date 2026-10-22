"""Fit the engine once per outer fold and cache P/S/M/R + frames, so Gap 2 selection variants can be tuned cheaply.
Usage: python -m experiments.cache_gap2 [--datasets a,b] [--jobs 4]"""
import argparse
import warnings
from pathlib import Path

import joblib
import yaml
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from c1.pipeline import FeatureIntelligenceEngine
from experiments.datasets import REGISTRY, load

OUT = Path("results/gap2_cache")


def one(name, cfg):
    warnings.filterwarnings("ignore")
    X, y = load(name, cfg["max_rows"], cfg["seed"])
    seed, folds = cfg["seed"], []
    for fold, (tr, te) in enumerate(StratifiedKFold(cfg["outer_folds"], shuffle=True, random_state=seed).split(X, y)):
        eng = FeatureIntelligenceEngine(cfg, seed=seed).fit(X.iloc[tr], y[tr])
        folds.append({"eng": eng, "Gte": eng.gen_.transform(X.iloc[te]), "yte": y[te], "ytr": y[tr]})
    joblib.dump(folds, OUT / f"{name}.joblib")
    return name


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(REGISTRY))
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open("configs/default.yaml"))
    todo = [n for n in a.datasets.split(",") if not (OUT / f"{n}.joblib").exists()]
    print(Parallel(n_jobs=a.jobs)(delayed(one)(n, cfg) for n in todo))
