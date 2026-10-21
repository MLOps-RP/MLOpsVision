"""
Turn the audit log (outputs/audit.db) into flat, time-indexed tables for Grafana:
    outputs/c4_monitoring.db

Grafana plots against time, but C4 works in windows. Each window is mapped to
one simulated hour from GRAFANA_START (window 0 = 2026-01-01 00:00 UTC), so a
39-window scenario fills 2026-01-01 00:00 -> 2026-01-02 14:00.
Only the latest run of each scenario is exported.

run_c4 calls this after every run. To rebuild it by hand:
    python -m src.tools.export_grafana
"""
import json
import sqlite3
from datetime import datetime, timezone

from src.common.config import DATA_SCENARIOS, OUTPUTS, load_json, load_config
from src.c4.audit.audit_log import DB_PATH

GRAFANA_DB = OUTPUTS / "c4_monitoring.db"   # the SQLite plugin blocks paths containing "grafana"
GRAFANA_START = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
WINDOW_SECONDS = 3600

SCHEMA = """
DROP TABLE IF EXISTS scenario_info;
DROP TABLE IF EXISTS window_metrics;
DROP TABLE IF EXISTS feature_drift;
DROP TABLE IF EXISTS decisions;
CREATE TABLE scenario_info (
    scenario TEXT PRIMARY KEY, run_id TEXT, run_started_at TEXT, description TEXT,
    drift_type TEXT, drift_start_window INTEGER, drift_start_time INTEGER,
    expected_diagnosis TEXT, expected_action TEXT,
    first_action_window INTEGER, detection_delay INTEGER
);
CREATE TABLE window_metrics (
    scenario TEXT, window INTEGER, time INTEGER, model_version TEXT, injected INTEGER,
    diagnosis TEXT, severity_level TEXT, severity_score REAL, action TEXT,
    n_fired INTEGER, fired_share REAL,
    prediction_share REAL, confidence_mean REAL, prediction_ratio REAL,
    labels_available INTEGER, accuracy REAL, f1 REAL, perf_drop REAL,
    dq_fired INTEGER, dq_issues INTEGER
);
CREATE TABLE feature_drift (
    scenario TEXT, window INTEGER, time INTEGER, feature TEXT, kind TEXT, test TEXT,
    score REAL, threshold REAL, ratio REAL, fired INTEGER
);
CREATE TABLE decisions (
    scenario TEXT, window INTEGER, time INTEGER,
    model_version_before TEXT, model_version_after TEXT,
    diagnosis TEXT, confidence TEXT, severity_level TEXT, severity_score REAL,
    action TEXT, target TEXT, reason TEXT
);
"""


def window_time(w: int) -> int:
    return GRAFANA_START + int(w) * WINDOW_SECONDS


def latest_runs(src: sqlite3.Connection) -> dict:
    """scenario -> (run_id, started_at) of its most recent run."""
    runs = {}
    for run_id, scenario, started in src.execute("SELECT run_id, scenario, started_at FROM runs ORDER BY rowid"):
        runs[scenario] = (run_id, started)
    return runs


def export() -> None:
    if not DB_PATH.exists():
        print(f"No audit log at {DB_PATH} - run C4 first")
        return
    significant = set(load_config("policy")["significant_actions"]) | {"WARN_ONLY"}
    src = sqlite3.connect(DB_PATH)
    dst = sqlite3.connect(GRAFANA_DB)
    dst.executescript(SCHEMA)

    for scenario, (run_id, started) in sorted(latest_runs(src).items()):
        truth_path = DATA_SCENARIOS / scenario / "ground_truth.json"
        truth = load_json(truth_path) if truth_path.exists() else {}
        start = truth.get("drift_start_window")

        for w, version, diagnosis, level, score, action, raw in src.execute(
                """SELECT window, model_version, diagnosis, severity_level, severity_score, action,
                          signals_json FROM window_signals WHERE run_id = ? ORDER BY window""", (run_id,)):
            s = json.loads(raw)
            t, pred, perf, dq = window_time(w), s["prediction"], s["performance"], s["data_quality"]
            dst.execute("INSERT INTO window_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                scenario, w, t, version, int(start is not None and w >= start),
                diagnosis, level, score, action,
                len(s["fired_features"]), s["fired_share"],
                pred["prediction_share"], pred["confidence_mean"], pred["ratio"],
                int(bool(perf.get("available"))), perf.get("accuracy"), perf.get("f1"), perf.get("perf_drop"),
                int(dq["fired"]), len(dq["issues"])))
            dst.executemany("INSERT INTO feature_drift VALUES (?,?,?,?,?,?,?,?,?,?)", [
                (scenario, w, t, f, r["kind"], r["test"], r["score"], r["threshold"], r["ratio"], int(r["fired"]))
                for f, r in s["features"].items()])

        rows = src.execute(
            """SELECT window, model_version_before, model_version_after, diagnosis, confidence,
                      severity_level, severity_score, action, target, reason
               FROM decisions WHERE run_id = ? ORDER BY window""", (run_id,)).fetchall()
        dst.executemany("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        [(scenario, r[0], window_time(r[0]), *r[1:]) for r in rows])

        # same definition as experiments/run_all: first non-WATCH action after the drift began
        acted = [r[0] for r in rows if r[7] in significant and (start is None or r[0] >= start)]
        first = acted[0] if acted else None
        dst.execute("INSERT INTO scenario_info VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            scenario, run_id, started, truth.get("description"), truth.get("drift_type"),
            start, window_time(start) if start is not None else None,
            truth.get("expected_diagnosis"), truth.get("expected_action"),
            first, first - start if first is not None and start is not None else None))

    dst.commit()
    dst.close()
    src.close()
    print(f"Grafana data written to {GRAFANA_DB}")


if __name__ == "__main__":
    export()
