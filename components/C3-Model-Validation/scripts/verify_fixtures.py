"""Confirm each fixture really contains the defect it is meant to contain.

The bounds below are sanity checks on the *fixtures*, chosen to be clearly
met by an intentional defect. They are not validation thresholds for the
gate; those belong in the gate's configuration and need their own
justification.
"""
import json
import statistics
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from fixture_utils import (
    DATA_DIR, FIXTURES_DIR, POSITIVE_LABEL, SENSITIVE_COLUMNS, TARGET_COLUMN,
)

EXPECTED_FIXTURES = [
    "good", "leaky", "overfit", "biased", "miscalibrated", "slow",
    "corrupt", "good_incomplete_metadata", "good_no_sensitive",
]
REQUIRED_METADATA = ["random_state", "split", "train_data_sha256",
                     "hyperparameters", "versions", "training_metrics"]


def meta(name):
    return json.loads((FIXTURES_DIR / name / "metadata.json").read_text("utf-8"))


def eval_frame(name):
    path = FIXTURES_DIR / name / "eval.csv"
    return pd.read_csv(path if path.exists() else DATA_DIR / "eval.csv")


def load_model(name):
    return joblib.load(FIXTURES_DIR / name / "model.joblib")


def ece(name, bins=10):
    """Expected calibration error of the positive-class probability."""
    model, frame = load_model(name), eval_frame(name)
    cols = meta(name)["feature_columns"]
    p = model.predict_proba(frame[cols])[:, list(model.classes_).index(POSITIVE_LABEL)]
    y = (frame[TARGET_COLUMN] == POSITIVE_LABEL).to_numpy().astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return sum(
        (idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean())
        for b in range(bins) if (idx == b).any()
    )


def single_row_latency_ms(name, repeats=30):
    model, frame = load_model(name), eval_frame(name)
    row = frame[meta(name)["feature_columns"]].iloc[[0]]
    model.predict(row)  # warm-up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(row)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def min_selection_ratio(name, column="sex"):
    rates = list(meta(name)["fairness"][f"selection_rate_by_{column}"].values())
    top = max(rates)
    return min(rates) / top if top > 0 else 0.0


def recomputed_f1(name):
    """F1 recomputed from the saved model, to compare with the metadata."""
    model, frame = load_model(name), eval_frame(name)
    pred = model.predict(frame[meta(name)["feature_columns"]])
    return f1_score(frame[TARGET_COLUMN], pred, pos_label=POSITIVE_LABEL)


results = []


def check(name, description, passed, detail):
    results.append(passed)
    print(f"[{'OK' if passed else 'FAIL'}] {name:26s} {description}: {detail}")


# every fixture must exist before anything else is measured
missing_files = [
    f"{name}/metadata.json"
    for name in EXPECTED_FIXTURES
    if not (FIXTURES_DIR / name / "metadata.json").exists()
]
for name in EXPECTED_FIXTURES:
    check(name, "metadata.json present",
          (FIXTURES_DIR / name / "metadata.json").exists(), "")
if missing_files:
    print(f"\nMissing fixture files: {missing_files}. Run make_fixtures.py.")
    sys.exit(1)

good, leaky, overfit = meta("good"), meta("leaky"), meta("overfit")

# the saved good model must reproduce the metrics written to its metadata
f1_now, f1_saved = recomputed_f1("good"), good["evaluation_metrics"]["f1"]
check("good", "metadata F1 matches recomputed F1",
      abs(f1_now - f1_saved) < 1e-9, f"{f1_now:.6f} vs {f1_saved:.6f}")

# good: a reasonable baseline
f1 = good["evaluation_metrics"]["f1"]
gap = good["training_metrics"]["accuracy"] - good["evaluation_metrics"]["accuracy"]
check("good", "eval F1 reasonable and small train/eval gap",
      f1 > 0.6 and abs(gap) < 0.02, f"F1={f1:.3f}, acc gap={gap:.3f}")

# leaky: far better than any honest model on this data
lf1 = leaky["evaluation_metrics"]["f1"]
check("leaky", "eval F1 much higher than good (leakage signature)",
      lf1 >= f1 + 0.15, f"{lf1:.3f} vs {f1:.3f}")

# overfit: large train/eval gap
ogap = overfit["training_metrics"]["accuracy"] - overfit["evaluation_metrics"]["accuracy"]
check("overfit", "train accuracy exceeds eval accuracy by >= 0.10",
      ogap >= 0.10, f"gap={ogap:.3f}")

# biased: the selection rate for one group collapses
gr, br = min_selection_ratio("good"), min_selection_ratio("biased")
check("biased", "worst/best selection-rate ratio far below good",
      br <= gr / 3, f"{br:.3f} vs {gr:.3f}")

# miscalibrated: similar ranking quality, much worse calibration
g_ece, m_ece = ece("good"), ece("miscalibrated")
auc_diff = abs(meta("miscalibrated")["evaluation_metrics"]["roc_auc"]
               - good["evaluation_metrics"]["roc_auc"])
check("miscalibrated", "ECE >= 3x good and AUC within 0.02",
      m_ece >= 3 * g_ece and auc_diff <= 0.02,
      f"ECE {m_ece:.3f} vs {g_ece:.3f}, AUC diff {auc_diff:.3f}")

# slow: much slower single-row prediction
g_ms, s_ms = single_row_latency_ms("good"), single_row_latency_ms("slow")
check("slow", "single-row latency >= 5x good",
      s_ms >= 5 * g_ms, f"{s_ms:.1f} ms vs {g_ms:.1f} ms")

# corrupt: must fail to load
try:
    load_model("corrupt")
    check("corrupt", "model fails to load", False, "loaded without error")
except Exception as exc:
    check("corrupt", "model fails to load", True, type(exc).__name__)

# incomplete metadata: required fields missing, present for good
missing = [k for k in REQUIRED_METADATA if k not in meta("good_incomplete_metadata")]
check("good_incomplete_metadata", "required fields missing",
      len(missing) >= 5 and all(k in good for k in REQUIRED_METADATA),
      f"{len(missing)} missing")

# no sensitive columns in eval data
cols = set(eval_frame("good_no_sensitive").columns)
check("good_no_sensitive", "eval.csv has no sensitive columns",
      not cols & set(SENSITIVE_COLUMNS), "")

if not all(results):
    print("\nSome fixtures do not contain their intended defect.")
    sys.exit(1)
print("\nAll fixtures verified.")
