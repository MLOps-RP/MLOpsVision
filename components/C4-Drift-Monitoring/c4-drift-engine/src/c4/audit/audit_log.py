"""
AUDIT LOG (FR9) - every window's signals and every decision C4 makes,
stored in a SQLite file: outputs/audit.db
"""
import json
import sqlite3
from datetime import datetime, timezone

from src.common.config import OUTPUTS, _to_builtin

DB_PATH = OUTPUTS / "audit.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, scenario TEXT, started_at TEXT
);
CREATE TABLE IF NOT EXISTS window_signals (
    run_id TEXT, scenario TEXT, window INTEGER, model_version TEXT,
    diagnosis TEXT, severity_level TEXT, severity_score REAL, action TEXT,
    signals_json TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT, created_at TEXT, scenario TEXT, window INTEGER,
    model_version_before TEXT, model_version_after TEXT,
    diagnosis TEXT, confidence TEXT, severity_level TEXT, severity_score REAL,
    action TEXT, target TEXT, reason TEXT, report_path TEXT
);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    return con


def start_run(con, run_id: str, scenario: str) -> None:
    con.execute("INSERT INTO runs VALUES (?, ?, ?)",
                (run_id, scenario, datetime.now(timezone.utc).isoformat(timespec="seconds")))
    con.commit()


def log_window(con, run_id, scenario, window, model_version, diagnosis, level, score, action, signals):
    con.execute("INSERT INTO window_signals VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, scenario, int(window), str(model_version), diagnosis, level, float(score),
                 action, json.dumps(signals, default=_to_builtin)))


def log_decision(con, run_id, scenario, window, v_before, v_after, diagnosis, confidence,
                 level, score, action, target, reason, report_path):
    con.execute("""INSERT INTO decisions (run_id, created_at, scenario, window,
                   model_version_before, model_version_after, diagnosis, confidence,
                   severity_level, severity_score, action, target, reason, report_path)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, datetime.now(timezone.utc).isoformat(timespec="seconds"), scenario, int(window),
                 str(v_before), str(v_after), diagnosis, confidence, level, float(score),
                 action, target, reason, str(report_path)))
    con.commit()
