"""Benchmark registry (RESEARCH R1). OpenML ids; raw frames cached as parquet in data/raw/.
Mix chosen to span missingness x imbalance x size x feature types. `split` = dev (tune rules) / test (held-out)."""
from pathlib import Path

import numpy as np
import openml
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# name: (openml_id, split)
REGISTRY = {
    "credit-g": (31, "dev"), "diabetes": (37, "dev"), "colic": (25, "dev"), "hepatitis": (55, "dev"),
    "credit-approval": (29, "dev"), "kc1": (1067, "dev"), "spambase": (44, "dev"), "sick": (38, "dev"),
    "bank-marketing": (1461, "dev"), "adult": (1590, "dev"),
    "vote": (56, "test"), "cylinder-bands": (6332, "test"), "pc1": (1068, "test"), "phoneme": (1489, "test"),
    "blood-transfusion": (1464, "test"), "ilpd": (1480, "test"), "ozone-level": (1487, "test"),
    "churn": (40701, "test"), "mushroom": (24, "test"), "kr-vs-kp": (3, "test"),
}


def load(name: str, max_rows: int | None = None, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    f = RAW / f"{name}.parquet"
    if not f.exists():
        d = openml.datasets.get_dataset(ALL[name][0], download_qualities=False)
        X, y, _, _ = d.get_data(target=d.default_target_attribute)
        for c in X.columns:                      # categoricals -> str with np.nan, so sklearn imputers behave
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype(object).map(lambda v: np.nan if pd.isna(v) else str(v))
        X["__y__"] = pd.factorize(y.astype(str))[0]
        RAW.mkdir(parents=True, exist_ok=True)
        X.to_parquet(f)
    df = pd.read_parquet(f)
    y = df.pop("__y__").to_numpy()
    if max_rows and len(df) > max_rows:          # stratified cap; ponytail: simple proportional sample
        idx = df.assign(__y__=y).groupby("__y__", group_keys=False).sample(frac=max_rows / len(df), random_state=seed).index
        df, y = df.loc[idx], y[df.index.get_indexer(idx)]
    return df.reset_index(drop=True), y


# Extra OpenML datasets for the Gap 1 v2 study (split alternates so dev/test stay balanced). Loaded via load(); REGISTRY (E1-E5) untouched.
EXTRA = {
    "mammography": (310, "dev"), "oil-spill": (311, "test"), "mc1": (1056, "dev"), "pc2": (1069, "test"),
    "pc3": (1050, "dev"), "pc4": (1049, "test"), "jm1": (1053, "dev"), "kc2": (1063, "test"),
    "hypothyroid": (57, "dev"), "breast-w": (15, "test"), "australian": (40981, "dev"), "heart-c": (49, "test"),
    "amazon-access": (4135, "dev"), "speed-dating": (40536, "test"), "eucalyptus": (188, "dev"), "soybean": (42, "test"),
    "wilt": (40983, "dev"), "mice-protein": (40966, "test"), "car": (40975, "dev"), "steel-plates": (1504, "test"),
}
ALL = {**REGISTRY, **EXTRA}
