"""
ADAPTIVE THRESHOLD CALIBRATION (basic version)

Instead of one fixed number for every feature (e.g. "PSI > 0.2"), each signal gets
its own threshold, learned from how much it wobbles on CLEAN data for THIS model:

    threshold = max(floor, mean + k * std)   over n clean windows

Clean windows are random resamples of the test split (never used for training
or for the baseline), so they show normal sampling noise only.

Thresholds are saved per model version (artifacts/model_v<N>/thresholds.json)
and attached to that model's MLflow run.

Run from the project root (after generate_scenarios):
    python -m src.c4.monitoring.calibration
"""
import numpy as np
import pandas as pd

from src.common.config import LABEL, all_features, load_config, model_dir, load_json, save_json
from src.common import registry
from src.c4.monitoring import feature_tests, prediction_tests
from src.c4.monitoring.stats import classification_summary
from src.bootstrap.data import load_split


def _threshold(values, k: float, floor: float) -> float:
    v = np.asarray(values, dtype=float)
    return float(max(floor, v.mean() + k * v.std()))


def calibrate_version(version: str, test: pd.DataFrame, cfg: dict) -> dict:
    cal = cfg["calibration"]
    k, floors = cal["k"], cal["floors"]
    folder = model_dir(version)
    baseline = load_json(folder / "baseline.json")
    reference = pd.read_csv(folder / "reference_numeric.csv")
    model = registry.load_model(version)
    rng = np.random.default_rng(7)

    feat_scores = {f: [] for f in all_features()}
    pred_share, conf_mean, conf_jsd = [], [], []
    perf_drop, label_share, tpr_d, tnr_d = [], [], [], []
    n_perf = cfg["window_size"] * cfg["performance_windows"]

    for _ in range(cal["n_clean_windows"]):
        w = test.sample(cfg["window_size"], random_state=int(rng.integers(1e9)))
        conf = model.predict_proba(w[all_features()])[:, 1]
        for f, r in feature_tests.compute(w[all_features()], baseline, reference).items():
            feat_scores[f].append(r["score"])
        p = prediction_tests.compute(conf, baseline, cfg["confidence_bins"])
        pred_share.append(p["prediction_share_delta"])
        conf_mean.append(p["confidence_mean_delta"])
        conf_jsd.append(p["confidence_jsd"])

        wp = test.sample(n_perf, replace=n_perf > len(test), random_state=int(rng.integers(1e9)))
        pred = (model.predict_proba(wp[all_features()])[:, 1] >= 0.5).astype(int)
        m = classification_summary(wp[LABEL], pred)
        b = baseline["performance"]
        perf_drop.append(max(b["accuracy"] - m["accuracy"], b["f1"] - m["f1"]))
        label_share.append(abs(m["positive_share"] - baseline["label"]["positive_share"]))
        tpr_d.append(abs(m["tpr"] - b["tpr"]))
        tnr_d.append(abs(m["tnr"] - b["tnr"]))

    features = {}
    for f, scores in feat_scores.items():
        floor = floors["psi"] if f in baseline["numeric_features"] else floors["jsd"]
        features[f] = _threshold(scores, k, floor)

    return {
        "model_version": version,
        "method": f"max(floor, mean + {k} x std) over {cal['n_clean_windows']} clean windows",
        "features": features,
        "prediction_share": _threshold(pred_share, k, floors["prediction_share"]),
        "confidence_mean": _threshold(conf_mean, k, floors["confidence_mean"]),
        "confidence_jsd": _threshold(conf_jsd, k, floors["confidence_jsd"]),
        "perf_drop": _threshold(perf_drop, k, floors["perf_drop"]),
        "label_share": _threshold(label_share, k, floors["label_share"]),
        "tpr_delta": _threshold(tpr_d, k, floors["tpr_delta"]),
        "tnr_delta": _threshold(tnr_d, k, floors["tnr_delta"]),
    }


def main() -> None:
    cfg = load_config("monitoring")
    test = load_split("test")
    info = registry.registry_info()
    for version in [info["v2_xgboost"], info["v1_logreg"]]:
        thr = calibrate_version(version, test, cfg)
        path = model_dir(version) / "thresholds.json"
        save_json(thr, path)
        registry.log_artifact_to_version(version, path, "c4_baseline")
        print(f"Model version {version}: thresholds saved to {path}")
        for f, t in thr["features"].items():
            print(f"   {f:<16} {t:.3f}")
        print(f"   perf_drop        {thr['perf_drop']:.3f}")
    print("Next step:  python -m src.c4.run_c4 --scenario S3_concept")


if __name__ == "__main__":
    main()
