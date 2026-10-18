"""Create the hidden-failure-group fixtures: one planted, one null control.

Run from the C3-Model-Validation folder:

    python scripts/make_hidden_group_fixtures.py

Creates (existing fixtures are not touched):

* fixtures/hidden_group/       a logistic regression that fails inside a region
                               (segment d AND tenure_years > 6, about 9% of rows)
                               that no sensitive column describes;
* fixtures/hidden_group_null/  the same data process with NO planted region.

Each folder holds model.joblib, eval.csv, metadata.json, train_row_hashes.txt
and ground_truth.json (the planted rows, for scoring detection). model.joblib,
eval.csv and train_row_hashes.txt are gitignored and regenerable.

The script verifies that the plant is real, and that the detector finds it in
the planted fixture and nowhere in the control, before it finishes.
"""
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from fixture_utils import code_commit, library_versions, write_csv_lf, write_text_lf
from mqvg import synthetic as syn
from mqvg.config import load_config
from mqvg.hashing import read_csv_as_text, row_hashes
from mqvg.hidden_groups import detect

PROJECT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT / "fixtures"
SEED, N_TRAIN, N_EVAL = 2026, 6000, 3000
PATTERN = "two_feature"
KEYWORD = "segment = d"          # what a correct rule must mention


def build(name: str, pattern: str | None) -> dict:
    folder = FIXTURES / name
    folder.mkdir(parents=True, exist_ok=True)
    train, evaluation, region = syn.make_pair(N_TRAIN, N_EVAL, SEED, pattern)
    model = syn.fit_logistic(train)

    with tempfile.TemporaryDirectory() as tmp:
        train_csv = Path(tmp) / "train.csv"
        write_csv_lf(train, train_csv)
        train_bytes = train_csv.read_bytes()
        hash_columns = syn.FEATURES + [syn.SENSITIVE]
        hashes = row_hashes(read_csv_as_text(train_csv), hash_columns)
    write_text_lf(folder / "train_row_hashes.txt", "\n".join(hashes) + "\n")
    write_csv_lf(evaluation, folder / "eval.csv")
    joblib.dump(model, folder / "model.joblib")

    train_pred = model.predict(train[syn.FEATURES])
    metadata = {
        "fixture_name": name, "model_type": "LogisticRegression",
        "task_type": "binary_classification", "dataset": "synthetic loan decisions",
        "data_source": "mqvg.synthetic.make_pair (generated)",
        "target_column": syn.TARGET, "positive_label": syn.POSITIVE,
        "sensitive_columns": [syn.SENSITIVE], "excluded_sensitive_features": [syn.SENSITIVE],
        "feature_columns": syn.FEATURES,
        "hyperparameters": {"max_iter": 500}, "random_state": SEED,
        "split": {"method": "generated independently", "seed": SEED},
        "train_rows": N_TRAIN, "eval_rows": N_EVAL,
        "train_data_sha256": hashlib.sha256(train_bytes).hexdigest(),
        "train_row_hashes_file": f"fixtures/{name}/train_row_hashes.txt",
        "row_hash_columns": hash_columns,
        "training_metrics": {
            "accuracy": float(accuracy_score(train[syn.TARGET], train_pred)),
            "f1": float(f1_score(train[syn.TARGET], train_pred, pos_label=syn.POSITIVE))},
        "versions": library_versions(), "code": code_commit(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_text_lf(folder / "metadata.json", json.dumps(metadata, indent=2) + "\n")

    p = model.predict_proba(evaluation[syn.FEATURES])[
        :, list(model.classes_).index(syn.POSITIVE)]
    y = (evaluation[syn.TARGET] == syn.POSITIVE).to_numpy()
    pred = p >= 0.5
    err = y != pred
    truth = {
        "pattern": pattern, "mode": "misspec", "strength": syn.DEFAULT_STRENGTH,
        "description": ("labels are flipped with probability 0.8 inside the region, in "
                        "both training and evaluation data; a linear model cannot "
                        "represent the interaction"
                        if pattern else "no planted region"),
        "region_rule": "segment = d AND tenure_years > 6" if pattern else None,
        "rule_keywords": [KEYWORD] if pattern else [],
        "region_share": float(region.mean()),
        "planted_row_indices": [int(i) for i in np.flatnonzero(region)],
        "measured": {
            "overall_error_rate": float(err.mean()),
            "error_rate_inside": float(err[region].mean()) if region.any() else None,
            "error_rate_outside": float(err[~region].mean()),
        },
    }
    write_text_lf(folder / "ground_truth.json", json.dumps(truth, indent=2) + "\n")
    return {"folder": folder, "truth": truth, "evaluation": evaluation, "region": region,
            "y": y, "pred": pred, "p": p}


def main() -> int:
    config = load_config(PROJECT / "configs" / "hidden_groups.yaml")
    failures = []
    for name, pattern in (("hidden_group", PATTERN), ("hidden_group_null", None)):
        built = build(name, pattern)
        truth = built["truth"]
        d = detect(built["evaluation"][syn.FEATURES], built["y"], built["pred"], built["p"],
                   config.hidden_groups, seed=config.evaluation.random_seed,
                   ece_bins=config.evaluation.ece_bins)
        m = truth["measured"]
        print(f"{name:20s} overall error {m['overall_error_rate']:.1%}"
              + (f"; inside planted region {m['error_rate_inside']:.1%} vs outside "
                 f"{m['error_rate_outside']:.1%}" if pattern else "")
              + f"; detector: available={d.available}, confirmed groups={len(d.confirmed)}")
        for group in d.confirmed[:3]:
            print(f"    [{group.method}] {group.rule}  observed {group.observed_rate:.1%} "
                  f"vs expected {group.expected_rate:.1%}")
        if pattern:
            recall = (d.covered_rows() & built["region"]).sum() / built["region"].sum()
            print(f"    planted rows covered by confirmed groups: {recall:.0%}")
            if m["error_rate_inside"] < m["error_rate_outside"] + 0.15:
                failures.append(f"{name}: the plant is too weak to be a real failure")
            if not d.available or recall < 0.5:
                failures.append(f"{name}: the detector did not find the planted group")
        elif not d.available or d.confirmed:
            failures.append(f"{name}: the detector flagged a group in the null control")
    if failures:
        print("\nFIXTURE VERIFICATION FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nHidden-group fixtures created and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
