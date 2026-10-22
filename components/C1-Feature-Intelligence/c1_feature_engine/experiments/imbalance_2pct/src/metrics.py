"""
metrics.py  -  how ONE fitted pipeline is scored on ONE untouched test fold.

The base metrics (macro-F1, balanced accuracy, ROC-AUC, PR-AUC of classes_[1]) come from the shared
`experiments.baselines.metrics`. At 2% minority share those are not enough: accuracy is ~98% for a model
that never predicts the minority class. So this study adds explicit MINORITY-class metrics, all computed
from the SAME fitted model and the SAME test fold (no extra fitting, no change to the method under test).
"""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_fscore_support)


def ext_metrics(pipe, Xte, yte):
    """Return base metrics + minority-class precision/recall/F1, minority PR-AUC, accuracy, confusion counts."""
    from experiments.baselines import metrics        # f1_macro, bal_acc, roc_auc, pr_auc (legacy columns)

    m = metrics(pipe, Xte, yte)

    # --- 1. Decide which class is "positive": always the RARER class in the test fold, on every dataset.
    #        (Datasets label the minority as 0 or 1 differently, so we cannot hard-code a label.)
    cls, cnt = np.unique(yte, return_counts=True)
    pos = cls[cnt.argmin()]

    # --- 2. Hard predictions and class probabilities from the already-fitted pipeline.
    pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)
    j = list(pipe.classes_).index(pos)               # column of `proba` that belongs to the minority class

    # --- 3. Minority precision / recall / F1 (labels=[pos] restricts the report to that single class).
    #        zero_division=0: a method that predicts no minority rows gets precision 0, not NaN.
    p, r, f, _ = precision_recall_fscore_support(yte, pred, labels=[pos], zero_division=0)

    # --- 4. Confusion counts with the minority as positive. labels=[False, True] fixes the order so that
    #        ravel() always yields (tn, fp, fn, tp), even if a fold has no predicted positives.
    tn, fp, fn, tp = confusion_matrix(yte == pos, pred == pos, labels=[False, True]).ravel()

    return {
        **m,
        "pos_label": int(pos),                                        # kept so analysis can verify it equals the manifest's minority label
        "min_precision": p[0], "min_recall": r[0], "min_f1": f[0],
        # PR-AUC of the MINORITY probability. The legacy `pr_auc` column uses classes_[1], which can be the
        # MAJORITY class -> do not use the legacy column for conclusions.
        "pr_auc_min": average_precision_score(yte == pos, proba[:, j]),
        "accuracy": accuracy_score(yte, pred),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,                       # summed later into the pooled confusion matrices
    }
