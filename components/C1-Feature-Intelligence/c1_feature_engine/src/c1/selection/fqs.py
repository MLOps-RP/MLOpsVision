"""Feature Quality Score = w_P*P + w_S*S - w_M*M - w_R*R, every term in [0, 1] (FR6). Weights configurable."""
import pandas as pd

from c1.schemas import FeatureScore

DEFAULT_WEIGHTS = {"P": 0.4, "S": 0.3, "M": 0.15, "R": 0.15}   # not tuned: E5 reports sensitivity


def missing_ratio(X_raw: pd.DataFrame, lineage: dict) -> pd.Series:
    """Raw missing ratio; a generated feature inherits the max of its parents."""
    base = X_raw.isna().mean()
    return pd.Series({n: max([base[p] for p in l["parents"]] or [base.get(n, 0.0)]) for n, l in lineage.items()})


def fqs(comp: pd.DataFrame, w: dict | None = None) -> pd.Series:
    w = {**DEFAULT_WEIGHTS, **(w or {})}
    return w["P"] * comp["P"] + w["S"] * comp["S"] - w["M"] * comp["M"] - w["R"] * comp["R"]


def feature_scores(comp: pd.DataFrame, lineage: dict, w: dict | None, selected: list[str]) -> list[FeatureScore]:
    q, sel = fqs(comp, w), set(selected)
    rows = [FeatureScore(name=n, P=r.P, S=r.S, M=r.M, R=r.R, fqs=q[n], op=lineage[n]["op"],
                         parents=lineage[n]["parents"], selected=n in sel) for n, r in comp.iterrows()]
    return sorted(rows, key=lambda f: -f.fqs)
