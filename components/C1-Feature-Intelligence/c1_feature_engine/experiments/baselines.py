"""Fixed-strategy baselines (B0-B2) + the adaptive method, evaluated on one train/test split."""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from c1.preprocessing.build import build_pipeline
from c1.preprocessing.selector import StrategySelector
from c1.profiling.profiler import profile

FIXED = {  # name: (imputer, balancer)
    "B0_median": ("median", "none"),
    "B1_mean": ("mean", "none"),
    "B2_smote": ("median", "smote"),
    "B2_classweight": ("median", "class_weight"),
}


def metrics(pipe, Xte, yte) -> dict:
    pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)
    classes = pipe.classes_
    out = {"f1_macro": f1_score(yte, pred, average="macro"), "bal_acc": balanced_accuracy_score(yte, pred)}
    try:
        if len(classes) == 2:
            out["roc_auc"] = roc_auc_score(yte, proba[:, 1])
            out["pr_auc"] = average_precision_score(yte, proba[:, 1])
        else:
            out["roc_auc"] = roc_auc_score(yte, proba, multi_class="ovr", labels=classes)
            out["pr_auc"] = average_precision_score(label_binarize(yte, classes=classes), proba, average="macro")
    except ValueError:  # test fold missing a class
        out.setdefault("roc_auc", np.nan), out.setdefault("pr_auc", np.nan)
    return out


def run_split(method: str, Xtr: pd.DataFrame, ytr, Xte: pd.DataFrame, yte, model: str, cfg: dict, rules: dict,
              seed: int) -> dict:
    t0, dec = time.time(), None
    if method == "adaptive":
        sel = StrategySelector(rules, model, cfg["inner_folds"], cfg["metric"], cfg["max_candidates"],
                               cfg["anchor_imputer"], cfg["anchor_balancer"], seed)
        dec = sel.select(Xtr, ytr, profile(Xtr, ytr))
        imp, bal = dec.imputer, dec.balancer
    else:
        imp, bal = FIXED[method]
    pipe = build_pipeline(imp, bal, model, Xtr, seed).fit(Xtr, ytr)
    return {**metrics(pipe, Xte, yte), "imputer": imp, "balancer": bal, "runtime_s": time.time() - t0,
            "fired_rules": ",".join(dec.fired_rules) if dec else ""}
