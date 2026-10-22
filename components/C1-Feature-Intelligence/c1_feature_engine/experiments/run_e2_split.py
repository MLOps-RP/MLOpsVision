"""E2 split: which part of the adaptive downstream gain is the 'was missing' indicator? Fixed imputers incl. median_ind
vs median/mean/knn/mice, downstream f1 under injected missingness. Usage: python -m experiments.run_e2_split [--datasets a,b]"""
import argparse
import warnings

import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from c1.preprocessing.build import build_pipeline
from experiments.baselines import metrics
from experiments.datasets import load
from experiments.inject_missingness import inject
from experiments.run_e2 import DEFAULT, MECHS, RATES

IMPS = ["mean", "median", "median_ind", "knn", "mice"]


def one(name, cfg):
    warnings.filterwarnings("ignore")
    X, y = load(name, 3000, cfg["seed"])
    X = X.select_dtypes("number").dropna(axis=1)
    rows = []
    for mech in MECHS:
        for rate in RATES:
            Xm, _ = inject(X, rate, mech, cfg["seed"])
            for imp in IMPS:
                for model in ["lgbm", "logreg"]:
                    f = [metrics(build_pipeline(imp, "none", model, Xm.iloc[tr], cfg["seed"]).fit(Xm.iloc[tr], y[tr]), Xm.iloc[te], y[te])["f1_macro"]
                         for tr, te in StratifiedKFold(3, shuffle=True, random_state=cfg["seed"]).split(Xm, y)]
                    rows.append({"dataset": name, "mechanism": mech, "rate": rate, "imputer": imp, "model": model, "f1": sum(f) / 3})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=DEFAULT)
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/default.yaml"))
    df = pd.DataFrame([r for rs in Parallel(n_jobs=4)(delayed(one)(n, cfg) for n in a.datasets.split(",")) for r in rs])
    df.to_csv("results/e2_split.csv", index=False)
    md = "## E2 split: downstream f1 by imputer (mean over datasets)\n" + df.pivot_table(index=["model", "mechanism", "rate"], columns="imputer", values="f1").round(3).to_markdown()
    open("results/e2_split_summary.md", "w").write(md)
    print(md)
