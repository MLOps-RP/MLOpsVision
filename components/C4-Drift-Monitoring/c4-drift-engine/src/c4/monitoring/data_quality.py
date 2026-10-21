"""
Data-quality checks. These catch BROKEN data (nulls, wrong units, unknown codes),
which must not be mistaken for real drift and must never trigger retraining.
"""
import pandas as pd


def check(window_X: pd.DataFrame, baseline: dict, rules: dict) -> dict:
    issues = []
    details = {}

    for f in baseline["numeric_features"]:
        b = baseline["numeric"][f]
        col = pd.to_numeric(window_X[f], errors="coerce")
        null_up = col.isna().mean() - b["null_rate"]
        present = col.dropna()
        out_of_range = float(((present < b["min"]) | (present > b["max"])).mean()) if len(present) else 0.0
        details[f] = {"null_increase": float(null_up), "out_of_range_share": out_of_range}
        if null_up > rules["max_null_increase"]:
            issues.append(f"{f}: null rate +{null_up:.0%}")
        if out_of_range > rules["max_out_of_range_share"]:
            issues.append(f"{f}: {out_of_range:.0%} of values outside baseline range "
                          f"[{b['min']:g}, {b['max']:g}]")

    for f in baseline["categorical_features"]:
        b = baseline["categorical"][f]
        col = window_X[f]
        null_up = col.isna().mean() - b["null_rate"]
        present = col.dropna().astype(str)
        unseen = float((~present.isin(list(b["shares"].keys()))).mean()) if len(present) else 0.0
        details[f] = {"null_increase": float(null_up), "unseen_share": unseen}
        if null_up > rules["max_null_increase"]:
            issues.append(f"{f}: null rate +{null_up:.0%}")
        if unseen > rules["max_unseen_category_share"]:
            issues.append(f"{f}: {unseen:.0%} unseen categories")

    return {"fired": bool(issues), "issues": issues, "details": details}
