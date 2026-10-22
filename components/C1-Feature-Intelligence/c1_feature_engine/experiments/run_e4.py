"""E4: full C1 vs baselines, size-matched. Per outer fold everything is fit on the train fold.
  B0 all originals, median+none | B2 all originals, median+SMOTE | B3 MI top-k originals | B4 SHAP top-k originals
  (B3/B4: k = size C1 chose, median+class_weight) | C1_fixedprep: C1 features, median+class_weight | C1_full: C1 features + adaptive prep
Usage: python -m experiments.run_e4 [--datasets a,b] [--models lgbm,logreg] [--seeds 1] [--jobs 4]"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from c1.pipeline import FeatureIntelligenceEngine
from c1.preprocessing.build import build_pipeline
from c1.selection.rankers import mi_rank, shap_rank
from c1.selection.redundancy import mean_pairwise
from c1.selection.stability import nogueira
from experiments.baselines import metrics
from experiments.datasets import REGISTRY, load

ARMS = ["B0_none", "B2_smote", "B3_mi_topk", "B4_shap_topk", "C1_fixedprep", "C1_full"]


def one(name, models, seeds, cfg):
    warnings.filterwarnings("ignore")
    X, y = load(name, cfg["max_rows"], cfg["seed"])
    rows = []
    for s in range(seeds):
        seed = cfg["seed"] + s
        for fold, (tr, te) in enumerate(StratifiedKFold(cfg["outer_folds"], shuffle=True, random_state=seed).split(X, y)):
            Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]
            eng = FeatureIntelligenceEngine(cfg, seed=seed).fit(Xtr, ytr)
            sel = eng.selected_
            k, Gtr, Gte = max(2, len(sel)), eng.G_, eng.gen_.transform(Xte)
            t0 = time.time()
            mi = mi_rank(Xtr, ytr, seed).nlargest(k).index.tolist()
            t_mi = time.time() - t0
            t0 = time.time()
            sh = shap_rank(Xtr, ytr, seed).nlargest(k).index.tolist()
            t_sh = time.time() - t0
            allc = Xtr.columns.tolist()
            # arm: (train frame, test frame, imputer, balancer, cols, seconds)
            arms = {"B0_none": (Xtr, Xte, "median", "none", allc, 0.0),
                    "B2_smote": (Xtr, Xte, "median", "smote", allc, 0.0),
                    "B3_mi_topk": (Xtr[mi], Xte[mi], "median", "class_weight", mi, t_mi),
                    "B4_shap_topk": (Xtr[sh], Xte[sh], "median", "class_weight", sh, t_sh),
                    "C1_fixedprep": (Gtr[sel], Gte[sel], "median", "class_weight", sel, eng.fit_s_),
                    "C1_full": (Gtr[sel], Gte[sel], eng.imputer_, eng.balancer_, sel, eng.fit_s_)}
            for model in models:
                for arm, (a, b, imp, bal, cols, secs) in arms.items():
                    pipe = build_pipeline(imp, bal, model, a, seed).fit(a, ytr)
                    rows.append({"dataset": name, "split": REGISTRY[name][1], "seed": seed, "fold": fold, "model": model,
                                 "arm": arm, **metrics(pipe, b, yte), "n_features": len(cols),
                                 "redundancy": mean_pairwise(a, cols), "select_s": secs,
                                 "gen_share": float(np.mean([eng.lineage_.get(c, {"op": "raw"})["op"] != "raw" for c in cols])),
                                 "features": json.dumps(cols), "imputer": imp, "balancer": bal})
    return rows


def stability(g: pd.DataFrame) -> float:
    """Nogueira across the outer folds, per dataset/seed, averaged."""
    vals = []
    for _, h in g.groupby(["dataset", "seed"]):
        sets = [set(json.loads(f)) for f in h.features]
        U = sorted(set().union(*sets))
        if len(sets) > 1 and len(U) > 1:
            vals.append(nogueira(np.array([[u in s for u in U] for s in sets])))
    return float(np.mean(vals)) if vals else np.nan


def summarise(df):
    from scipy.stats import wilcoxon
    out = []
    for model, g in df.groupby("model"):
        for label, gg in [("ALL", g), ("HELD-OUT (test)", g[g.split == "test"])]:
            if gg.empty:
                continue
            t = gg.groupby(["dataset", "arm"])["f1_macro"].mean().unstack()[ARMS]
            out += [f"## {model} / {label} ({len(t)} datasets): mean f1_macro", t.round(3).to_markdown(), "",
                    f"mean rank (1=best): {t.rank(axis=1, ascending=False).mean().round(2).to_dict()}", ""]
            tab = gg.groupby("arm").agg(f1=("f1_macro", "mean"), roc_auc=("roc_auc", "mean"), n_feat=("n_features", "mean"),
                                         redundancy=("redundancy", "mean"), gen_share=("gen_share", "mean"),
                                         select_s=("select_s", "mean")).loc[ARMS]
            tab["stability"] = [stability(gg[gg.arm == a]) for a in ARMS]
            out += [tab.round(3).to_markdown(), ""]
            for a, b in [("C1_full", "B0_none"), ("C1_full", "B3_mi_topk"), ("C1_full", "B4_shap_topk"),
                         ("C1_fixedprep", "B3_mi_topk"), ("C1_fixedprep", "B4_shap_topk"), ("C1_full", "C1_fixedprep")]:
                d = t[a] - t[b]
                p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
                out.append(f"- {a} vs {b}: diff {d.mean():+.4f}, W/T/L {(d>0).sum()}/{(d==0).sum()}/{(d<0).sum()}, p={p:.3f}")
            out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(REGISTRY))
    ap.add_argument("--models", default="lgbm,logreg")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="results/e4")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/default.yaml"))
    res = Parallel(n_jobs=a.jobs)(delayed(one)(n, a.models.split(","), a.seeds, cfg) for n in a.datasets.split(","))
    df = pd.DataFrame([r for rs in res for r in rs])
    df.to_csv(a.out + "_folds.csv", index=False)
    md = summarise(df)
    open(a.out + "_summary.md", "w").write(md)
    print(md)
