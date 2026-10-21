"""Create the fixtures for the extended checks (existing fixtures are not touched).

Run from the C3-Model-Validation folder, after scripts/make_fixtures.py has made
data/eval.csv, data/train_row_hashes.txt and fixtures/good:

    python scripts/make_extended_fixtures.py

Creates

* fixtures/label_conflicts/    the good model's evaluation data plus 3% of its rows repeated
                               with the opposite label
* fixtures/duplicate_training/ the good model's training row hashes plus 15% repeated hashes
* fixtures/unsafe_model/       a pickle whose load runs os.write(1, b"PAYLOAD EXECUTED...")
* fixtures/unreviewed_model/   a pickle that refers to code from a module the allowlist
                               does not know (__main__.UnreviewedModel)

WARNING: fixtures/unsafe_model/model.joblib is DELIBERATELY UNSAFE. Loading it with
joblib.load or pickle.load runs its payload. The payload is harmless (it writes one
line to standard output) but the file is a test for the safety scan: do not load it
outside the gate, and never with the default profile on a machine you care about.
Each folder holds metadata.json, plus model.joblib / eval.csv / train_row_hashes.txt
where it differs from the good model; the binaries and CSVs are gitignored and regenerable.

The script verifies, with the extended profile only, that each fixture produces its
intended finding, and that the unsafe payload did NOT run.
"""
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fixture_utils import write_csv_lf, write_text_lf
from mqvg.config import load_config
from mqvg.pipeline import run_validation
from mqvg.schemas import CandidateBundle, CheckStatus

PROJECT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT / "fixtures"
DATA = PROJECT / "data"
SEED = 7
CONFLICT_SHARE = 0.03
DUPLICATE_SHARE = 0.15


class UnsafePayload:
    """Unpickling this calls os.write(1, ...): harmless, but exactly what the scan must catch."""

    def __reduce__(self):
        return (os.write, (1, b"PAYLOAD EXECUTED: loading this file ran code\n"))


class UnreviewedModel:
    """A stand-in for a user's own model class: not from a library the allowlist knows."""

    classes_ = np.array(["<=50K", ">50K"])

    def predict(self, X):
        return np.array(["<=50K"] * len(X))


def metadata_for(name: str, hashes: str | None = None) -> None:
    meta = json.loads((FIXTURES / "good" / "metadata.json").read_text(encoding="utf-8"))
    meta["fixture_name"] = name
    if hashes:
        meta["train_row_hashes_file"] = hashes
    write_text_lf(FIXTURES / name / "metadata.json", json.dumps(meta, indent=2) + "\n")


def make_label_conflicts() -> None:
    folder = FIXTURES / "label_conflicts"
    folder.mkdir(exist_ok=True)
    evaluation = pd.read_csv(DATA / "eval.csv")
    repeated = evaluation.sample(frac=CONFLICT_SHARE, random_state=SEED).copy()
    labels = sorted(evaluation["target"].unique())
    repeated["target"] = repeated["target"].map({labels[0]: labels[1], labels[1]: labels[0]})
    write_csv_lf(pd.concat([evaluation, repeated], ignore_index=True), folder / "eval.csv")
    metadata_for("label_conflicts")
    print(f"label_conflicts      {len(repeated)} rows repeated with the opposite label")


def make_duplicate_training() -> None:
    folder = FIXTURES / "duplicate_training"
    folder.mkdir(exist_ok=True)
    hashes = [h for h in (DATA / "train_row_hashes.txt").read_text(encoding="utf-8").split() if h]
    extra = hashes[: int(len(hashes) * DUPLICATE_SHARE)]
    write_text_lf(folder / "train_row_hashes.txt", "\n".join(hashes + extra) + "\n")
    metadata_for("duplicate_training", "fixtures/duplicate_training/train_row_hashes.txt")
    print(f"duplicate_training   {len(extra)} of {len(hashes) + len(extra)} hashes are repeats")


def make_unsafe_model() -> None:
    folder = FIXTURES / "unsafe_model"
    folder.mkdir(exist_ok=True)
    joblib.dump(UnsafePayload(), folder / "model.joblib")
    metadata_for("unsafe_model")
    print("unsafe_model         written (DELIBERATELY UNSAFE: do not load it with joblib)")


def make_unreviewed_model() -> None:
    folder = FIXTURES / "unreviewed_model"
    folder.mkdir(exist_ok=True)
    joblib.dump(UnreviewedModel(), folder / "model.joblib")
    metadata_for("unreviewed_model")
    print("unreviewed_model     written (refers to __main__.UnreviewedModel)")


def verify() -> list[str]:
    config = load_config(PROJECT / "configs" / "extended.yaml")
    expectations = {
        "label_conflicts": ("training_validity.eval_label_conflicts", CheckStatus.FAIL),
        "duplicate_training": ("training_validity.train_duplicate_rows", CheckStatus.FAIL),
        "unsafe_model": ("operational.model_file_safety", CheckStatus.FAIL),
        "unreviewed_model": ("operational.model_file_safety", CheckStatus.WARN),
    }
    problems = []
    for name, (check_id, status) in expectations.items():
        folder = FIXTURES / name
        eval_path = folder / "eval.csv"
        hashes = folder / "train_row_hashes.txt"
        model = folder / "model.joblib"
        result = run_validation(CandidateBundle(
            candidate_id=name,
            model_path=model if model.exists() else FIXTURES / "good" / "model.joblib",
            eval_data_path=eval_path if eval_path.exists() else DATA / "eval.csv",
            metadata_path=folder / "metadata.json",
            train_row_hashes_path=hashes if hashes.exists() else DATA / "train_row_hashes.txt"),
            config)
        got = next(c for c in result.checks if c.id == check_id)
        flag = "ok " if got.status == status else "BAD"
        print(f"{flag} {name:20s} {check_id}: {got.status.value} -> {result.decision.decision.value}")
        if got.status != status:
            problems.append(f"{name}: expected {check_id} {status.value}, got {got.status.value}")
    return problems


def main() -> int:
    for needed in (DATA / "eval.csv", DATA / "train_row_hashes.txt", FIXTURES / "good" / "metadata.json"):
        if not needed.exists():
            print(f"missing {needed}; run scripts/make_fixtures.py first")
            return 1
    make_label_conflicts()
    make_duplicate_training()
    make_unsafe_model()
    make_unreviewed_model()
    problems = verify()
    for problem in problems:
        print("PROBLEM:", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
