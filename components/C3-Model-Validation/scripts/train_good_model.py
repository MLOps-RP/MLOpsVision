"""Train the 'good' baseline: logistic regression without sensitive features."""
import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fixture_utils import (
    DATA_DIR, SEED, SENSITIVE_COLUMNS, TARGET_COLUMN, build_preprocessor,
    common_metadata, library_versions, load_adult_splits, measure_model,
    read_csv_as_text, row_hashes, save_fixture, training_data_sha256,
)

print("Loading Adult dataset...")
X, y, train_idx, eval_idx, _ = load_adult_splits()

# Sensitive columns stay in eval.csv for fairness evaluation but are not
# model input features.
X_model = X.drop(columns=SENSITIVE_COLUMNS)
X_train, y_train = X_model.loc[train_idx], y.loc[train_idx]

# The saved eval.csv must be exactly the eval rows of this split, in order.
# Comparing row hashes catches a stale file from a different split, and also
# proves the hashing convention survives the CSV round trip.
saved_eval = pd.read_csv(DATA_DIR / "eval.csv")
saved_hashes = row_hashes(read_csv_as_text(DATA_DIR / "eval.csv"), list(X.columns))
expected_hashes = row_hashes(X.loc[eval_idx], list(X.columns))
assert saved_hashes == expected_hashes, (
    "data/eval.csv does not match the current split; run prepare_data.py"
)
assert TARGET_COLUMN in saved_eval.columns
print("Evaluation dataset verified:", len(saved_eval), "rows")

pipeline = Pipeline([
    ("preprocessor", build_preprocessor(X_train)),
    ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
])

print("Training the good model using training data only...")
pipeline.fit(X_train, y_train)

# Measured exactly as for every other fixture, on the eval CSV the gate reads.
measured = measure_model(
    pipeline, X_train, y_train, saved_eval, list(X_train.columns)
)

# Exact duplicate records occur naturally in Adult, so some eval rows match a
# training row. This is the baseline the gate's overlap check must allow for.
train_hashes = set(
    (DATA_DIR / "train_row_hashes.txt").read_text(encoding="utf-8").split()
)
overlap_rows = sum(h in train_hashes for h in saved_hashes)

metadata = {
    "fixture_name": "good",
    **common_metadata(
        feature_columns=list(X_train.columns),
        train_rows=len(X_train),
        eval_rows=len(saved_eval),
        train_hash=training_data_sha256(X.loc[train_idx], y_train),
        row_hash_columns=list(X.columns),
    ),
    "model_type": "LogisticRegression",
    "hyperparameters": {"max_iter": 1000, "random_state": SEED},
    "train_eval_exact_row_overlap": {
        "rows": overlap_rows,
        "fraction_of_eval": overlap_rows / len(saved_hashes),
        "note": "Adult contains naturally duplicated records.",
    },
    **measured,
    "versions": library_versions(),
}

out_dir = save_fixture("good", pipeline, metadata)

print("\nGood model trained and saved successfully.")
print(f"Fixture: {out_dir}")
print("Training metrics:", measured["training_metrics"])
print("Evaluation metrics:", measured["evaluation_metrics"])
print("Fairness:", json.dumps(
    {k: round(v, 4) for k, v in measured["fairness"].items()
     if k.startswith(("demographic", "selection_rate_ratio"))}))
