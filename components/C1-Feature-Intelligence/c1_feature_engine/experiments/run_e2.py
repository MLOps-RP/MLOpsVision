"""E2: imputation under injected missingness. Ground-truth RMSE (standardised units) for the numeric imputers,
plus downstream f1 for fixed imputers vs the adaptive selector's pick (selected on train fold only).
Usage: python -m experiments.run_e2 [--datasets a,b]"""
import argparse
import warnings

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from c1.preprocessing.imputation import IMPUTERS
from c1.preprocessing.selector import load_rules
from experiments.baselines import run_split
from experiments.datasets import load
from experiments.inject_missingness import inject

DEFAULT = "diabetes,kc1,spambase,phoneme,blood-transfusion,pc1,ozone-level,ilpd"
RATES, MECHS = [0.05, 0.2, 0.4], ["MCAR", "MAR", "MNAR"]
NUMERIC_IMPUTERS = ["mean", "median", "knn", "mice"]


def one(name: str, cfg: dict, rules: dict) -> list[dict]:
    warnings.filterwarnings("ignore")
    X, y = load(name, 3000, cfg["seed"])             # smaller cap: MICE/KNN x 9 conditions
    X = X.select_dtypes("number").dropna(axis=1)     # complete numeric columns = clean ground truth
    Z = (X - X.mean()) / X.std().replace(0, 1)
    rows = []
    for mech in MECHS:
        for rate in RATES:
            Xm, mask = inject(X, rate, mech, cfg["seed"])
            Zm = (Xm - X.mean()) / X.std().replace(0, 1)
            for imp in NUMERIC_IMPUTERS:
                est = IMPUTERS[imp]()
                Zi = est.fit_transform(Zm)[:, : Z.shape[1]]  # indicator cols (if any) are appended after
                err = (Zi - Z.to_numpy())[mask.to_numpy()]
                rows.append({"dataset": name, "mechanism": mech, "rate": rate, "method": imp, "rmse": float(np.sqrt((err ** 2).mean()))})
            drop_cfg = {**cfg, "outer_folds": 3}
            for m in ["B0_median", "B1_mean", "adaptive"]:
                f1 = []
                for tr, te in StratifiedKFold(3, shuffle=True, random_state=cfg["seed"]).split(Xm, y):
                    f1.append(run_split(m, Xm.iloc[tr], y[tr], Xm.iloc[te], y[te], "lgbm", drop_cfg, rules, cfg["seed"])["f1_macro"])
                rows.append({"dataset": name, "mechanism": mech, "rate": rate, "method": f"f1:{m}", "f1_macro": float(np.mean(f1))})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=DEFAULT)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/default.yaml"))
    rules = load_rules(cfg["rules"])
    df = pd.DataFrame([r for rs in Parallel(n_jobs=a.jobs)(delayed(one)(n, cfg, rules) for n in a.datasets.split(",")) for r in rs])
    df.to_csv("results/e2.csv", index=False)
    rm = df[df.rmse.notna()].pivot_table(index=["mechanism", "rate"], columns="method", values="rmse")
    f1 = df[df.f1_macro.notna()].pivot_table(index=["mechanism", "rate"], columns="method", values="f1_macro")
    md = "## E2 mean RMSE (standardised)\n" + rm.round(3).to_markdown() + "\n\n## E2 mean downstream f1_macro\n" + f1.round(3).to_markdown()
    open("results/e2_summary.md", "w").write(md)
    print(md)
