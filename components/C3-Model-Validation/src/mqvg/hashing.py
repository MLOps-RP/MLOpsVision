"""Row hashing used to compare evaluation rows with training rows.

The convention MUST match the one the training side used when it wrote the
row-hash file (scripts/fixture_utils.py does this for the sample fixtures; a
test keeps the two in step). Rows are rendered through `to_csv` so that an
in-memory frame and the same data re-read from CSV as text hash identically.
Only feature values are hashed, not the label.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def read_csv_as_text(path: Path) -> pd.DataFrame:
    """Read a CSV with no type inference, so hashes do not depend on dtypes."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def row_hashes(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    text = frame.loc[:, columns].astype(str).replace("nan", "").to_csv(
        index=False, header=False, lineterminator="\n")
    return [hashlib.sha256(line.encode("utf-8")).hexdigest()
            for line in text.split("\n")[:-1]]
