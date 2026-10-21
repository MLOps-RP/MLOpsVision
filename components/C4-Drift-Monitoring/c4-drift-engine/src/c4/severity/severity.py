"""
SEVERITY (processing step 3)

score = w_mag * magnitude + w_per * persistence + w_sco * scope + w_perf * performance
each component scaled 0..1, then mapped to Low / Medium / High / Critical.
"""


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compute(signals: dict, diagnosis: str, streak: int, perf_valid: bool, cfg: dict) -> dict:
    w, caps, lv = cfg["weights"], cfg["caps"], cfg["levels"]

    ratios = [signals["features"][f]["ratio"] for f in signals["fired_features"]]
    ratios.append(signals["prediction"]["ratio"])
    top_ratio = max(ratios) if ratios else 0.0
    magnitude = _clip01((top_ratio - 1) / (caps["magnitude_ratio"] - 1)) if top_ratio > 1 else 0.0

    persistence = _clip01(streak / caps["persistence_windows"])
    scope = _clip01(signals["fired_share"])

    perf = signals["performance"]
    perf_drop = perf["perf_drop"] if perf_valid else 0.0
    performance = _clip01(perf_drop / caps["performance_drop"])

    components = {"magnitude": round(magnitude, 3), "persistence": round(persistence, 3),
                  "scope": round(scope, 3), "performance": round(performance, 3)}
    score = sum(w[k] * components[k] for k in w)

    if diagnosis == "NO_DRIFT":
        level = "None"
    elif score >= lv["critical"] or perf_drop >= cfg["critical_performance_drop"]:
        level = "Critical"
    elif score >= lv["high"]:
        level = "High"
    elif score >= lv["medium"]:
        level = "Medium"
    else:
        level = "Low"

    return {"score": round(score, 3), "level": level, "components": components,
            "perf_drop_used": round(perf_drop, 4)}
