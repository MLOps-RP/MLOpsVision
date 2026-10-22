"""2% minority-class study: prepare data, then run the unchanged gap-1 v2 flow (experiments.run_g1v2) per seed.
Only addition: minority-class metrics (precision/recall/F1, confusion counts, minority PR-AUC), computed from the
same fitted model and test fold that run_g1v2 already scores. Selector, baselines, folds, seeds are untouched."""
import json, platform, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_recall_fscore_support

HERE = Path(__file__).parent
CFG = dict(datasets=["spambase", "phoneme", "mushroom", "kr-vs-kp", "steel-plates"], share=0.02, floor=40,
           seeds=[42, 43, 44], models=["lgbm", "logreg", "rf"], jobs=8)


def ext_metrics(pipe, Xte, yte):
    from experiments.baselines import metrics
    m = metrics(pipe, Xte, yte)
    cls, cnt = np.unique(yte, return_counts=True)
    pos = cls[cnt.argmin()]                              # positive = minority class (fewest rows), all datasets
    pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)
    j = list(pipe.classes_).index(pos)
    p, r, f, _ = precision_recall_fscore_support(yte, pred, labels=[pos], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(yte == pos, pred == pos, labels=[False, True]).ravel()
    return {**m, "pos_label": int(pos), "min_precision": p[0], "min_recall": r[0], "min_f1": f[0],
            "pr_auc_min": average_precision_score(yte == pos, proba[:, j]), "accuracy": accuracy_score(yte, pred),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def task(name, seed, out):
    import experiments.run_g1v2 as g
    g.metrics = ext_metrics
    cfg = yaml.safe_load(open("configs/default.yaml"))
    cfg.update(seed=seed, out=out)
    rules = yaml.safe_load(open("configs/rules_all.yaml"))
    g.one(name, "imb2", CFG["models"], 1, cfg, rules)   # 1 seed per task -> task seed = `seed`
    return name, seed


def prepare():
    from experiments.datasets import load
    from experiments.stress import subsample_minority
    rows = []
    for n in CFG["datasets"]:
        X0, y0 = load(n, 10000, 42)
        for s in CFG["seeds"]:
            X, y = subsample_minority(X0, y0, CFG["share"], s, CFG["floor"])
            d = X.assign(target=y)
            d.to_csv(HERE / "data/prepared_datasets" / f"{n}__imb2__seed{s}.csv", index=False)
            c0, c1 = pd.Series(y0).value_counts().sort_index(), pd.Series(y).value_counts().sort_index()
            rows.append(dict(dataset=n, seed=s, n_rows_original=len(y0), n_rows_prepared=len(y), n_features=X.shape[1],
                             original_counts=json.dumps(c0.to_dict(), default=int), prepared_counts=json.dumps(c1.to_dict(), default=int),
                             minority_label=int(c1.idxmin()), original_minority_share=c0.min() / c0.sum(),
                             prepared_minority_share=c1.min() / c1.sum(), duplicate_rows=int(d.duplicated().sum()),
                             floor_applied=bool(c1.min() == CFG["floor"]), preprocessing="stratified row cap 10000 (seed 42); minority rows randomly dropped; no resampling, no synthetic rows"))
    pd.DataFrame(rows).to_csv(HERE / "data/dataset_manifest.csv", index=False)


def main():
    t0 = time.time()
    (HERE / "config/experiment_config.json").write_text(json.dumps(
        {**CFG, "folds": 5, "inner_folds": 3, "metric": "f1_macro", "methods": ["B0", "B_thr", "B_prac", "rules", "random", "brute"],
         "condition": "imb2 = experiments.stress.subsample_minority(share=0.02, floor=40)", "positive_class": "minority class (fewest rows)",
         "python": platform.python_version(), "platform": platform.platform(),
         "versions": {m: __import__(m).__version__ for m in ["numpy", "pandas", "sklearn", "scipy", "lightgbm", "imblearn"]}}, indent=2), encoding="utf-8")
    prepare()
    Parallel(n_jobs=CFG["jobs"], verbose=5)(delayed(task)(n, s, str(HERE / f"results/raw/seed{s}")) for n in CFG["datasets"] for s in CFG["seeds"])
    fs = sorted((HERE / "results/raw").glob("seed*/*.csv"))
    pd.concat([pd.read_csv(f) for f in fs]).to_csv(HERE / "results/raw_fold_results.csv", index=False)
    print(f"run done: {len(fs)} CSVs in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
