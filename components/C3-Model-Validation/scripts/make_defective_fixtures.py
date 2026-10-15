"""Create fixtures with known, deliberately injected defects.

Each model is a plain scikit-learn Pipeline (no custom classes), so the saved
files load anywhere scikit-learn is installed. The defect is the only thing
that differs from the 'good' fixture; see fixtures/manifest.yaml for what the
gate is expected to find.
"""
import copy
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from fixture_utils import (
    DATA_DIR, FIXTURES_DIR, NEGATIVE_LABEL, POSITIVE_LABEL, SEED,
    SENSITIVE_COLUMNS, TARGET_COLUMN, build_preprocessor, common_metadata,
    library_versions, load_adult_splits, measure_model, save_fixture,
    training_data_sha256, write_csv_lf, write_text_lf,
)

GOOD_DIR = FIXTURES_DIR / "good"
assert (GOOD_DIR / "model.joblib").exists(), "run train_good_model.py first"

X, y, train_idx, eval_idx, _ = load_adult_splits()
ORIGINAL_COLUMNS = list(X.columns)

# Evaluation frame exactly as the gate will read it (from CSV).
eval_df = pd.read_csv(DATA_DIR / "eval.csv")
assert len(eval_df) == len(eval_idx)


def train_and_save(name, estimator, X_train, y_train, eval_frame, *,
                   hyperparameters, excluded_sensitive, save_eval_copy=False):
    """Fit a pipeline, record measured metadata, and save the fixture."""
    feature_columns = list(X_train.columns)
    model = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", estimator),
    ])
    model.fit(X_train, y_train)

    metadata = {
        "fixture_name": name,
        **common_metadata(
            feature_columns=feature_columns,
            train_rows=len(X_train),
            eval_rows=len(eval_frame),
            train_hash=training_data_sha256(X_train, y_train),
            row_hash_columns=ORIGINAL_COLUMNS,
            excluded_sensitive_features=excluded_sensitive,
        ),
        "model_type": type(estimator).__name__,
        "hyperparameters": hyperparameters,
        **measure_model(model, X_train, y_train, eval_frame, feature_columns),
        "versions": library_versions(),
    }
    out = save_fixture(name, model, metadata)
    if save_eval_copy:
        write_csv_lf(eval_frame, out / "eval.csv")

    m = metadata["evaluation_metrics"]
    fairness = metadata.get("fairness", {})
    gap = fairness.get("demographic_parity_difference_by_sex")
    ratio = fairness.get("selection_rate_ratio_by_sex")
    print(f"{name:14s} eval F1={m['f1']:.3f} acc={m['accuracy']:.3f} "
          f"train acc={metadata['training_metrics']['accuracy']:.3f} "
          f"sex gap={gap if gap is None else round(gap, 3)} "
          f"sex ratio={ratio if ratio is None else round(ratio, 3)}")
    return model


def write_json(path, data):
    write_text_lf(path, json.dumps(data, indent=2) + "\n")


X_train = X.drop(columns=SENSITIVE_COLUMNS).loc[train_idx]
y_train = y.loc[train_idx]

# --- leaky: a feature derived from the label (target leakage) -------------
def leak_feature(labels: pd.Series, seed: int) -> np.ndarray:
    """The true label as 0/1, flipped for 3% of rows so it is not exact.

    Each call has its own generator, so the values do not depend on the
    order in which train and eval are generated.
    """
    flag = (np.asarray(labels) == POSITIVE_LABEL).astype(int)
    flip = np.random.default_rng(seed).random(len(flag)) < 0.03
    return np.where(flip, 1 - flag, flag)


X_leaky_train = X_train.copy()
X_leaky_train["prior_high_income_flag"] = leak_feature(y_train, SEED)
eval_leaky = eval_df.copy()
eval_leaky["prior_high_income_flag"] = leak_feature(
    eval_leaky[TARGET_COLUMN], SEED + 1)
train_and_save(
    "leaky", LogisticRegression(max_iter=1000, random_state=SEED),
    X_leaky_train, y_train, eval_leaky,
    hyperparameters={"max_iter": 1000, "random_state": SEED},
    excluded_sensitive=SENSITIVE_COLUMNS,
    save_eval_copy=True,
)

# --- overfit: unconstrained decision tree ----------------------------------
train_and_save(
    "overfit", DecisionTreeClassifier(random_state=SEED),
    X_train, y_train, eval_df,
    hyperparameters={"max_depth": None, "random_state": SEED},
    excluded_sensitive=SENSITIVE_COLUMNS,
)

# --- biased: label bias against one group, sensitive features used --------
# Its train_data_sha256 differs from the other fixtures on purpose: it was
# trained on the manipulated labels.
X_with_sens = X.loc[train_idx]
y_biased = y_train.copy()
female_pos = (X_with_sens["sex"] == "Female") & (y_train == POSITIVE_LABEL)
drop = female_pos & (np.random.default_rng(SEED).random(len(y_train)) < 0.95)
y_biased[drop] = NEGATIVE_LABEL
train_and_save(
    "biased", LogisticRegression(max_iter=1000, random_state=SEED),
    X_with_sens, y_biased, eval_df,
    hyperparameters={"max_iter": 1000, "random_state": SEED},
    excluded_sensitive=[],
)

# --- miscalibrated: class weighting inflates predicted probabilities -----
# Ranking quality (AUC) stays close to the good model, but the probabilities
# no longer match observed frequencies.
CLASS_WEIGHT = {POSITIVE_LABEL: 8, NEGATIVE_LABEL: 1}
train_and_save(
    "miscalibrated",
    LogisticRegression(max_iter=1000, random_state=SEED,
                       class_weight=CLASS_WEIGHT),
    X_train, y_train, eval_df,
    hyperparameters={"max_iter": 1000, "random_state": SEED,
                     "class_weight": CLASS_WEIGHT},
    excluded_sensitive=SENSITIVE_COLUMNS,
)

# --- slow: very large forest (slow single-row prediction, big artifact) ---
train_and_save(
    "slow", RandomForestClassifier(
        n_estimators=1000, max_depth=14, n_jobs=1, random_state=SEED),
    X_train, y_train, eval_df,
    hyperparameters={"n_estimators": 1000, "max_depth": 14, "n_jobs": 1,
                     "random_state": SEED},
    excluded_sensitive=SENSITIVE_COLUMNS,
)

# --- artifact-level variants that reuse the good model --------------------
good_meta = json.loads((GOOD_DIR / "metadata.json").read_text(encoding="utf-8"))
good_model_bytes = (GOOD_DIR / "model.joblib").read_bytes()

# corrupt: the good model file truncated part-way through
corrupt_dir = FIXTURES_DIR / "corrupt"
corrupt_dir.mkdir(parents=True, exist_ok=True)
(corrupt_dir / "model.joblib").write_bytes(
    good_model_bytes[: int(len(good_model_bytes) * 0.4)])
corrupt_meta = copy.deepcopy(good_meta)
corrupt_meta["fixture_name"] = "corrupt"
write_json(corrupt_dir / "metadata.json", corrupt_meta)
print("corrupt        good model truncated to 40% of its bytes")

# good_incomplete_metadata: same model, much of the training metadata missing
incomplete_dir = FIXTURES_DIR / "good_incomplete_metadata"
incomplete_dir.mkdir(parents=True, exist_ok=True)
REMOVED = ["random_state", "split", "train_data_sha256", "train_row_hashes_file",
           "row_hash_columns", "hyperparameters", "versions", "code",
           "training_metrics", "trained_at_utc", "data_source"]
incomplete_meta = {k: v for k, v in good_meta.items() if k not in REMOVED}
incomplete_meta["fixture_name"] = "good_incomplete_metadata"
write_json(incomplete_dir / "metadata.json", incomplete_meta)
print(f"good_incomplete_metadata  removed {len(REMOVED)} metadata fields "
      "(uses the good model)")

# good_no_sensitive: same model, evaluation data without sensitive columns
no_sens_dir = FIXTURES_DIR / "good_no_sensitive"
no_sens_dir.mkdir(parents=True, exist_ok=True)
write_csv_lf(eval_df.drop(columns=SENSITIVE_COLUMNS), no_sens_dir / "eval.csv")
no_sens_meta = copy.deepcopy(good_meta)
no_sens_meta["fixture_name"] = "good_no_sensitive"
write_json(no_sens_dir / "metadata.json", no_sens_meta)
print("good_no_sensitive  eval.csv has no sex/race columns (uses the good model)")
