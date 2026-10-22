"""E3: does feature generation help? Arms (all: median impute + class-weight, same model; outer CV, everything fit on train fold):
  A orig      original features
  B pool      original + all generated candidates
  C gen_topk  top-k of the pool by aggregated P (MI+SHAP+RFE)
  D orig_topk top-k of the ORIGINALS by aggregated P      (k = ceil(n_orig/2); C vs D isolates generation, C vs A shows selection+generation)
Usage: python -m experiments.run_e3 [--datasets a,b] [--models lgbm,logreg] [--seeds 1] [--jobs 4]"""
import argparse
import math
import time
import warnings

import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from c1.generation.generator import FeatureGenerator
from c1.preprocessing.build import build_pipeline
from c1.selection.aggregate import aggregate
from c1.selection.rankers import rank_all
from experiments.baselines import metrics
from experiments.datasets import REGISTRY, load

ARMS = ["A_orig", "B_pool", "C_gen_topk", "D_orig_topk"]
KEY = "f1_macro"


def one(name: str, models: list[str], seeds: int, cfg: dict) -> list[dict]:
    warnings.filterwarnings("ignore")
    X, y = load(name, cfg["max_rows"], cfg["seed"])
    rows = []
    for s in range(seeds):
        seed = cfg["seed"] + s
        for fold, (tr, te) in enumerate(StratifiedKFold(cfg["outer_folds"], shuffle=True, random_state=seed).split(X, y)):
            Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]
            t0 = time.time()
            gen = FeatureGenerator(seed=seed)
            Gtr, Gte = gen.fit_transform(Xtr, ytr), None
            Gte = gen.transform(Xte)
            P = aggregate(rank_all(Gtr, ytr, seed))
            P0 = aggregate(rank_all(Xtr, ytr, seed))
            k = max(2, math.ceil(X.shape[1] / 2))
            sel_c = P.nlargest(k).index.tolist()
            sel_d = P0.nlargest(k).index.tolist()
            prep_s = time.time() - t0
            arms = {"A_orig": (Xtr, Xte), "B_pool": (Gtr, Gte), "C_gen_topk": (Gtr[sel_c], Gte[sel_c]),
                    "D_orig_topk": (Xtr[sel_d], Xte[sel_d])}
            n_gen_sel = sum(gen.lineage_[c]["op"] != "raw" for c in sel_c)
            for model in models:
                for arm, (a, b) in arms.items():
                    pipe = build_pipeline("median", "class_weight", model, a, seed).fit(a, ytr)
                    rows.append({"dataset": name, "seed": seed, "fold": fold, "model": model, "arm": arm,
                                 **metrics(pipe, b, yte), "n_features": a.shape[1], "n_pool": Gtr.shape[1] - X.shape[1],
                                 "gen_selected": n_gen_sel if arm == "C_gen_topk" else 0, "prep_s": prep_s})
    return rows


def summarise(df: pd.DataFrame) -> str:
    from scipy.stats import wilcoxon
    out = []
    for model, g in df.groupby("model"):
        t = g.groupby(["dataset", "arm"])[KEY].mean().unstack()[ARMS]
        out += [f"## {model}: mean {KEY} per dataset", t.round(3).to_markdown(), "",
                f"mean {KEY}: {t.mean().round(4).to_dict()}",
                f"mean rank (1=best): {t.rank(axis=1, ascending=False).mean().round(2).to_dict()}", ""]
        for a, b in [("C_gen_topk", "A_orig"), ("C_gen_topk", "D_orig_topk"), ("B_pool", "A_orig"), ("D_orig_topk", "A_orig")]:
            d = t[a] - t[b]
            p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
            out.append(f"- {a} vs {b}: mean diff {d.mean():+.4f}, wins/ties/losses "
                       f"{(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}, Wilcoxon p={p:.3f}")
        c = g[g.arm == "C_gen_topk"]
        out += ["", f"features: A={g[g.arm=='A_orig'].n_features.mean():.0f}, B={g[g.arm=='B_pool'].n_features.mean():.0f}, "
                f"C=D={c.n_features.mean():.0f}; generated share of C's top-k: {(c.gen_selected / c.n_features).mean():.0%}; "
                f"gen+rank time/fold {g.prep_s.mean():.1f}s", ""]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(REGISTRY))
    ap.add_argument("--models", default="lgbm,logreg")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="results/e3")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/default.yaml"))
    res = Parallel(n_jobs=a.jobs)(delayed(one)(n, a.models.split(","), a.seeds, cfg) for n in a.datasets.split(","))
    df = pd.DataFrame([r for rs in res for r in rs])
    df.to_csv(a.out + "_folds.csv", index=False)
    md = summarise(df)
    open(a.out + "_summary.md", "w").write(md)
    print(md)
