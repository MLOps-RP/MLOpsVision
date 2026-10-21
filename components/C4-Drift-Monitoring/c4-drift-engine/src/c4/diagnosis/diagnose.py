"""
DIAGNOSIS (processing step 2)

Decides WHAT kind of drift is happening from the PATTERN of signals,
and WHERE (which features / segments).

Rules are checked in order; the first match wins:
  1. data-quality rule fired                         -> DATA_QUALITY
  2. class mix changed, per-class recall stable      -> LABEL_SHIFT
  3. performance dropped, (almost) no inputs drifted -> CONCEPT_DRIFT
  4. inputs drifted AND performance dropped          -> COVARIATE_DEGRADING
  5. inputs drifted, performance stable              -> COVARIATE_BENIGN
  6. inputs drifted, labels not here yet             -> SUSPECTED (provisional)
  7. otherwise                                       -> NO_DRIFT
"""
import numpy as np
import pandas as pd

CONCEPT_MAX_FIRED_SHARE = 0.10   # "almost no inputs drifted" = under 10% of features fired


def diagnose(signals: dict, onset_window) -> dict:
    perf = signals["performance"]
    dq = signals["data_quality"]
    input_drift = bool(signals["fired_features"]) or signals["prediction"]["fired"]

    # label-based checks (label shift, concept drift) only need labels to exist;
    # covariate checks need labels from AFTER the input drift began
    labels_here = bool(perf.get("available"))
    perf_valid = performance_is_valid(perf, onset_window)

    evidence = []
    if dq["fired"]:
        return _result("DATA_QUALITY", "confirmed", dq["issues"])

    if labels_here and perf["label_share_fired"] and perf["per_class_stable"]:
        evidence.append(f"share of >50K moved by {perf['label_share_delta']:+.2f}, "
                        f"per-class recall stable (TPR {perf['tpr_delta']:+.2f}, TNR {perf['tnr_delta']:+.2f})")
        return _result("LABEL_SHIFT", "confirmed", evidence)

    if labels_here and perf["perf_fired"] and signals["fired_share"] < CONCEPT_MAX_FIRED_SHARE:
        evidence.append(f"performance dropped by {perf['perf_drop']:.3f} "
                        f"(F1 {perf['f1']:.3f}) while inputs stayed within thresholds")
        return _result("CONCEPT_DRIFT", "confirmed", evidence)

    if input_drift and perf_valid and perf["perf_fired"]:
        evidence.append(f"{len(signals['fired_features'])} features drifted and performance "
                        f"dropped by {perf['perf_drop']:.3f}")
        return _result("COVARIATE_DEGRADING", "confirmed", evidence)

    if input_drift and perf_valid:
        evidence.append(f"{len(signals['fired_features'])} features drifted, performance stable "
                        f"(drop {perf['perf_drop']:+.3f})")
        return _result("COVARIATE_BENIGN", "confirmed", evidence)

    if input_drift:
        evidence.append("inputs or predictions drifted; waiting for delayed labels to confirm impact")
        return _result("SUSPECTED", "provisional", evidence)

    return _result("NO_DRIFT", "confirmed", [])


def performance_is_valid(perf: dict, onset_window) -> bool:
    """
    Performance evidence only counts if ALL labelled windows used are from AFTER the
    input drift began - otherwise we would judge the drift with pre-drift labels.
    """
    return bool(perf.get("available")) and (
        onset_window is None or perf["earliest_labelled_window"] >= onset_window)


def _result(diagnosis: str, confidence: str, evidence: list) -> dict:
    return {"diagnosis": diagnosis, "confidence": confidence, "evidence": evidence}


def rank_affected_features(signals: dict, importance: dict, top_n: int = 5) -> list[dict]:
    """Fired features ranked by impact = (score / threshold) x feature importance."""
    rows = []
    for f in signals["fired_features"]:
        r = signals["features"][f]
        rows.append({
            "feature": f, "test": r["test"], "score": round(r["score"], 4),
            "threshold": round(r["threshold"], 4), "ratio": round(r["ratio"], 2),
            "importance": round(importance.get(f, 0.0), 4),
            "impact": round(r["ratio"] * max(importance.get(f, 0.0), 0.01), 4),
        })
    return sorted(rows, key=lambda d: d["impact"], reverse=True)[:top_n]


def affected_segments(X: pd.DataFrame, y, pred, baseline: dict, top_n: int = 3,
                      min_rows: int = 40) -> list[dict]:
    """
    Which groups got worse? Error rate per category in the recent labelled windows,
    compared with the same group's error rate in the baseline.
    """
    err = (np.asarray(y) != np.asarray(pred)).astype(float)
    rows = []
    for f, base_err in baseline.get("segment_error", {}).items():
        groups = pd.Series(err).groupby(X[f].astype(str).to_numpy())
        for cat, e in groups:
            if len(e) < min_rows or cat not in base_err:
                continue
            rows.append({"segment": f"{f} = {cat}", "rows": int(len(e)),
                         "error_now": round(float(e.mean()), 3),
                         "error_baseline": round(base_err[cat], 3),
                         "increase": round(float(e.mean()) - base_err[cat], 3)})
    return sorted(rows, key=lambda d: d["increase"], reverse=True)[:top_n]
