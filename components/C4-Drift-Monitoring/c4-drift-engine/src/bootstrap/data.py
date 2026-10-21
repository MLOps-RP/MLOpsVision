"""
Load the Adult Income dataset, clean it, and split it into 4 parts:

    train 40% | validation 10% | test 10% | future_stream 40%

The future stream is NEVER used for training - it becomes "production traffic".
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.config import DATA_RAW, DATA_PROCESSED, LABEL, all_features, feature_lists

SEED = 42

UCI_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
               "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
               "hours-per-week", "native-country", "class"]


def _download_adult() -> pd.DataFrame:
    """Try OpenML first; fall back to UCI files you downloaded by hand into data/raw/."""
    try:
        from sklearn.datasets import fetch_openml
        print("Downloading Adult from OpenML (first time only)...")
        bunch = fetch_openml("adult", version=2, as_frame=True)
        df = bunch.frame
        return df
    except Exception as e:  # no internet, OpenML down, ...
        print(f"OpenML download failed ({e}). Looking for UCI files in data/raw/ ...")

    parts = []
    for name in ["adult.data", "adult.test"]:
        p = DATA_RAW / name
        if p.exists():
            parts.append(pd.read_csv(p, header=None, names=UCI_COLUMNS, skipinitialspace=True,
                                     na_values="?", comment="|"))
    if not parts:
        raise FileNotFoundError(
            "Could not get the Adult dataset. Either connect to the internet, or download "
            "adult.data and adult.test from https://archive.ics.uci.edu/dataset/2/adult "
            "and put them in data/raw/")
    return pd.concat(parts, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Unify column types, turn '?' into missing values and make the 0/1 label."""
    df = df.copy()
    label_col = "class" if "class" in df.columns else "income"
    num, cat = feature_lists()

    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cat:
        s = df[c].astype(object)
        s = s.where(s.notna(), None)
        s = s.map(lambda v: None if v is None else str(v).strip())
        s = s.replace({"?": None, "": None, "nan": None})
        df[c] = s

    label_text = df[label_col].astype(str).str.strip()
    df[LABEL] = label_text.str.startswith(">50K").astype(int)

    df = df[all_features() + [LABEL]]          # drops fnlwgt and the text label
    df = df.dropna(subset=["age"]).reset_index(drop=True)
    return df


def load_adult() -> pd.DataFrame:
    """Return the cleaned Adult dataset, using data/raw/adult.csv as a cache."""
    cache = DATA_RAW / "adult.csv"
    if cache.exists():
        return pd.read_csv(cache)
    df = clean(_download_adult())
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"Saved cleaned dataset to {cache}")
    return pd.read_csv(cache)


def split_and_save(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Stratified 40/10/10/40 split, saved to data/processed/."""
    rest, future = train_test_split(df, test_size=0.40, stratify=df[LABEL], random_state=SEED)
    train, holdout = train_test_split(rest, test_size=1 / 3, stratify=rest[LABEL], random_state=SEED)
    val, test = train_test_split(holdout, test_size=0.5, stratify=holdout[LABEL], random_state=SEED)

    parts = {"train": train, "val": val, "test": test, "future_stream": future}
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part = part.reset_index(drop=True)
        parts[name] = part
        part.to_csv(DATA_PROCESSED / f"{name}.csv", index=False)
    return parts


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / f"{name}.csv")


def describe(df: pd.DataFrame) -> None:
    """Print the facts you will put on a slide."""
    num, cat = feature_lists()
    print(f"Rows: {len(df):,}   Features: {len(num) + len(cat)} "
          f"({len(num)} numeric, {len(cat)} categorical)")
    print(f"Share earning >50K: {df[LABEL].mean():.1%}")
    missing = df[num + cat].isna().mean()
    missing = missing[missing > 0]
    if len(missing):
        print("Missing values: " + ", ".join(f"{k} {v:.1%}" for k, v in missing.items()))
    _ = np  # keeps numpy import for notebooks that use this module
