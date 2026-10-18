import json

from fixture_utils import (
    DATA_DIR, SENSITIVE_COLUMNS, SEED, SPLIT_RATIOS, TARGET_COLUMN,
    TRAIN_ROW_HASHES_PATH, load_adult_splits, row_hashes, training_data_sha256,
    write_csv_lf, write_text_lf,
)

DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Adult dataset...")
X, y, train_idx, eval_idx, spare_idx = load_adult_splits()
X_train, y_train = X.loc[train_idx], y.loc[train_idx]

# Evaluation and spare CSVs keep the sensitive columns and the true label.
eval_df = X.loc[eval_idx].copy()
eval_df[TARGET_COLUMN] = y.loc[eval_idx]
eval_path = DATA_DIR / "eval.csv"
write_csv_lf(eval_df, eval_path)

spare_df = X.loc[spare_idx].copy()
spare_df[TARGET_COLUMN] = y.loc[spare_idx]
write_csv_lf(spare_df, DATA_DIR / "spare.csv")

# Per-row hashes of the training split. The raw training data is not kept,
# but these let the gate detect train/eval overlap.
train_hashes = row_hashes(X_train, list(X.columns))
write_text_lf(TRAIN_ROW_HASHES_PATH, "\n".join(train_hashes) + "\n")

train_hash = training_data_sha256(X_train, y_train)

metadata = {
    "dataset": "UCI Adult Income",
    "random_state": SEED,
    "split_ratios": SPLIT_RATIOS,
    "row_counts": {
        "train": len(train_idx),
        "eval": len(eval_idx),
        "spare": len(spare_idx),
    },
    "sensitive_columns": SENSITIVE_COLUMNS,
    "excluded_from_baseline_model_features": SENSITIVE_COLUMNS,
    "train_data_sha256": train_hash,
    "train_row_hashes_file": "data/train_row_hashes.txt",
    "unique_train_row_hashes": len(set(train_hashes)),
}
write_text_lf(DATA_DIR / "split_metadata.json", json.dumps(metadata, indent=2) + "\n")

print("\nDataset split completed successfully!")
print(f"Train rows: {len(train_idx)}")
print(f"Evaluation rows: {len(eval_idx)}")
print(f"Spare rows: {len(spare_idx)}")
print(f"Training row hashes: {TRAIN_ROW_HASHES_PATH}")
print(f"Training data SHA-256: {train_hash}")
