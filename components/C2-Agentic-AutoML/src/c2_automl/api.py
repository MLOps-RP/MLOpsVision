"""C2 REST API (FastAPI) - the backend for the React dashboard.

Run:  uvicorn c2_automl.api:app --reload --port 8000      (docs at http://127.0.0.1:8000/docs)

The API never re-implements ML logic: it starts the same LangGraph workflow as scripts/run_c2.py
and reads the same files the workflow writes (outputs/<dataset>/<run_id>/...).
"""
import ast
import json
import math
import os
import re
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

from . import config
from .agents.agent2_hpo import new_run_id
from .graph import build_graph
from .io.dummy_c1 import build_dummy_c1, load_c1_output, load_split, load_table
from .schemas import TaskType
from .tools.holdout import fit_final_model
from .tools.task_profile import profile_task

JUPYTER_URL = os.environ.get("C2_JUPYTER_URL", "http://localhost:8888")
SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")

NOTEBOOKS = [
    {"file": "01_explore_datasets.ipynb", "title": "Explore input datasets",
     "description": "C1 output: class balance, feature quality scores, feature distributions.",
     "context": "input"},
    {"file": "02_baselines.ipynb", "title": "Baselines B0 / B1 / B2",
     "description": "Default settings vs random search vs static Optuna TPE: best-so-far curves.",
     "context": "baselines"},
    {"file": "03_run_analysis.ipynb", "title": "Analyse one C2 run",
     "description": "Search history by round, overfitting gap, range changes and the reasoning log.",
     "context": "run"},
    {"file": "04_adaptive_comparison.ipynb", "title": "Adaptive vs static comparison",
     "description": "B2 vs adaptive method vs ablations: tables and charts for the thesis.",
     "context": "comparison"},
]

app = FastAPI(title="C2 Agentic AutoML API", version="0.1.0")
app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

_RUNS: dict[str, dict] = {}            # runs started by this server process (live + recent)
_LOCK = threading.Lock()


# ---------- helpers ----------
def _clean(obj: Any) -> Any:
    """Make any value JSON-safe (numpy types, NaN/inf -> None, Path -> str)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _check_id(value: str) -> str:
    if not SAFE_ID.match(value):
        raise HTTPException(400, f"Invalid id: {value!r}")
    return value


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _run_dir(run_id: str) -> Optional[Path]:
    _check_id(run_id)
    for path in config.OUTPUT_DIR.glob(f"*/{run_id}"):
        if path.is_dir():
            return path
    return None


def _read_output(run_id: str) -> Optional[dict]:
    d = _run_dir(run_id)
    if d and (d / "c2_output.json").exists():
        return json.loads((d / "c2_output.json").read_text())
    return None


def _run_summary(output: dict, created: str) -> dict:
    hold = output.get("metrics_holdout", {})
    return {"run_id": output["run_id"], "dataset_id": output["dataset_id"], "status": "completed",
            "created": created, "model_family": output["model_family"], "metric": output["metric"],
            "best_cv": (output.get("score_by_round") or [None])[-1], "holdout": hold,
            "rounds": output["rounds"], "total_trials": output["total_trials"],
            "stop_reason": output.get("stop_reason")}


def _parse_params(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    try:
        return ast.literal_eval(value) if isinstance(value, str) else {}
    except (ValueError, SyntaxError):
        return {}


# ---------- live run tracking ----------
def _summarize(node: str, update: dict) -> dict:
    """A compact, JSON-safe description of what one graph node did."""
    log = update.get("reasoning_log", [])
    data: dict = {}
    if node == "load_input":
        data = {"run_id": update["run_id"], "total_trials": update["total_trials"]}
    elif node == "profile_task":
        data = update["profile"]
    elif node == "agent1_select":
        data = {"families": update["families"], "budget": update["budget"]}
    elif node == "agent2_hpo":
        data = {"round": log[0]["round"], "budget": log[0]["decision"]["budget"],
                "candidates": {f: {k: c.get(k) for k in ("mean_val", "mean_train", "gap", "cv_std",
                                                          "n_complete", "n_pruned", "params")}
                               for f, c in update["round_candidates"].items()}}
    elif node == "evaluate":
        data = {"signals": update["signals"]}
    elif node == "agent1_diagnose":
        data = {"diagnoses": [{"family": d["model_family"], "labels": d["labels"],
                               "reasons": d["reasons"]} for d in update["diagnoses"]],
                "actions": update["plan"]["actions"], "referrals": update["referrals_to_c1"],
                "stop_reason": update["stop_reason"], "best_history": update["best_history"],
                "decision": log[-1]["rationale"] if log else ""}
    elif node == "agent2_adapt":
        data = {"round": update["round"], "changes": log[0]["decision"]["changes"]}
    elif node == "agent3_select":
        data = update["selection"]
    elif node == "package_output":
        o = update["output"]
        data = {k: o[k] for k in ("model_family", "rounds", "total_trials", "stop_reason",
                                  "score_by_round", "metrics_holdout")}
    return _clean({"node": node, "time": _now(), "data": data, "reasoning": log})


def _worker(run_id: str, inputs: dict) -> None:
    rec = _RUNS[run_id]
    try:
        for step in build_graph().stream(inputs, stream_mode="updates"):
            for node, update in step.items():
                rec["events"].append(_summarize(node, update))
                rec["current"] = node
                if node == "package_output":
                    rec["output"] = _clean(update["output"])
        rec["status"] = "completed"
    except Exception as exc:  # report the failure to the UI instead of losing it
        rec["status"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
    rec["finished"] = _now()


class RunRequest(BaseModel):
    dataset_id: str
    total_trials: int = Field(90, ge=3, le=2000)
    folds: int = Field(5, ge=2, le=10)
    max_rounds: int = Field(3, ge=1, le=6)
    adapt_ranges: bool = True
    adapt_families: bool = True


# ---------- endpoints: general ----------
@app.get("/api/health")
def health():
    return {"status": "ok", "time": _now()}


@app.get("/api/config")
def get_config():
    names = ["default.yaml", "model_registry.yaml", "diagnosis_rules.yaml", "search_spaces.yaml",
             "space_adaptation.yaml"]
    return _clean({n.removesuffix(".yaml"): config.load_yaml(n) for n in names})


@app.get("/api/notebooks")
def notebooks():
    nb_dir = config.ROOT / "notebooks"
    items = [{**nb, "exists": (nb_dir / nb["file"]).exists(),
              "url": f"{JUPYTER_URL}/lab/tree/notebooks/{nb['file']}"} for nb in NOTEBOOKS]
    return {"jupyter_url": JUPYTER_URL, "notebooks": items}


# ---------- endpoints: input data (C1 output) ----------
@app.get("/api/datasets")
def list_datasets():
    items = []
    for manifest in sorted((config.DATA_DIR / "c1_output").glob("*/manifest.json")):
        c1 = load_c1_output(manifest.parent.name)
        d = Path(c1.data_dir)
        items.append({"dataset_id": c1.dataset_id, "task_type": c1.task_type.value,
                      "target": c1.target, "positive_class": c1.positive_class,
                      "n_features": len(c1.feature_types),
                      "n_categorical": sum(t == "categorical" for t in c1.feature_types.values()),
                      "n_train": pq.ParquetFile(d / "y_train.parquet").metadata.num_rows,
                      "n_test": pq.ParquetFile(d / "y_test.parquet").metadata.num_rows,
                      "class_distribution": c1.class_distribution})
    return items


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    _check_id(dataset_id)
    try:
        c1 = load_c1_output(dataset_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    X, y = load_split(c1, "train")
    profile = profile_task(c1, X, y, config.load_default_config())
    sample = X.head(12).copy()
    sample[c1.target] = y.head(12).values
    fqs = sorted(c1.fqs.items(), key=lambda kv: -kv[1])
    return _clean({"manifest": c1.model_dump(mode="json"), "profile": profile.model_dump(),
                   "fqs": [{"feature": f, "score": s, "type": c1.feature_types[f]} for f, s in fqs],
                   "sample": {"columns": list(sample.columns), "rows": sample.values.tolist()}})


@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...), target: str = Form(...),
                         task_type: TaskType = Form(TaskType.BINARY), dataset_id: str = Form(...),
                         positive_class: Optional[str] = Form(None)):
    """Upload a CSV/Excel file and turn it into a (dummy) C1 output that C2 can use."""
    _check_id(dataset_id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(400, "Only .csv, .xlsx or .xls files are supported")
    raw_dir = config.DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{dataset_id}{suffix}"
    raw_path.write_bytes(await file.read())
    df = load_table(str(raw_path))
    if target not in df.columns:
        raise HTTPException(400, f"Target column {target!r} not found. Columns: {list(df.columns)}")
    try:
        c1 = build_dummy_c1(df, target, task_type, dataset_id, positive_class=positive_class or None)
    except Exception as exc:
        raise HTTPException(400, f"Could not prepare dataset: {exc}")
    return _clean(c1.model_dump(mode="json"))


# ---------- endpoints: runs ----------
@app.post("/api/runs")
def start_run(req: RunRequest):
    _check_id(req.dataset_id)
    if not (config.DATA_DIR / "c1_output" / req.dataset_id / "manifest.json").exists():
        raise HTTPException(404, f"Dataset {req.dataset_id} not found")
    with _LOCK:
        if any(r["status"] == "running" for r in _RUNS.values()):
            raise HTTPException(409, "Another run is still in progress. Wait for it to finish.")
        run_id = new_run_id(req.dataset_id, "c2")
        _RUNS[run_id] = {"run_id": run_id, "dataset_id": req.dataset_id, "status": "running",
                         "params": req.model_dump(), "events": [], "current": None,
                         "output": None, "error": None, "started": _now(), "finished": None}
    inputs = {"dataset_id": req.dataset_id, "run_id": run_id, "total_trials": req.total_trials,
              "folds": req.folds,
              "overrides": {"budget": {"max_rounds": req.max_rounds},
                            "adaptive": {"adapt_ranges": req.adapt_ranges,
                                         "adapt_families": req.adapt_families}}}
    threading.Thread(target=_worker, args=(run_id, inputs), daemon=True).start()
    return {"run_id": run_id}


@app.get("/api/runs")
def list_runs():
    runs: dict[str, dict] = {}
    for out_file in config.OUTPUT_DIR.glob("*/*/c2_output.json"):
        output = json.loads(out_file.read_text())
        created = datetime.fromtimestamp(out_file.stat().st_mtime).isoformat(timespec="seconds")
        runs[output["run_id"]] = _run_summary(output, created)
    for run_id, rec in _RUNS.items():
        if rec["status"] != "completed" or run_id not in runs:
            runs[run_id] = {"run_id": run_id, "dataset_id": rec["dataset_id"], "status": rec["status"],
                            "created": rec["started"], "error": rec["error"]}
    return _clean(sorted(runs.values(), key=lambda r: r["created"], reverse=True))


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    _check_id(run_id)
    rec = _RUNS.get(run_id)
    output = _read_output(run_id)
    if rec is None and output is None:
        raise HTTPException(404, f"Run {run_id} not found")
    if rec is None:
        rec = {"run_id": run_id, "dataset_id": output["dataset_id"], "status": "completed",
               "params": None, "events": [], "current": "package_output", "error": None}
    return _clean({**rec, "output": output or rec.get("output")})


@app.get("/api/runs/{run_id}/trials")
def get_trials(run_id: str):
    """Every Optuna trial of the run. Live (from optuna.db) while running, from Parquet afterwards."""
    d = _run_dir(run_id)
    if d and (d / "search_history.parquet").exists():
        df = pd.read_parquet(d / "search_history.parquet")
        df["params"] = df["params"].map(_parse_params)
        return _clean(df.drop(columns=["run_id"]).to_dict(orient="records"))

    storage_path = config.OUTPUT_DIR / "optuna.db"
    if not storage_path.exists():
        return []
    rows = []
    try:
        storage = optuna.storages.RDBStorage(f"sqlite:///{storage_path}")
        for summary in optuna.get_all_study_summaries(storage, include_best_trial=False):
            name = summary.study_name
            if not name.startswith(f"{run_id}_r"):
                continue
            round_txt, family = name[len(run_id) + 2:].split("_", 1)
            study = optuna.load_study(study_name=name, storage=storage)
            for t in study.get_trials(deepcopy=False):
                if t.user_attrs.get("warm_start") or t.state.name == "RUNNING":
                    continue
                vals = t.user_attrs.get("val_scores", [])
                mean_val = t.value if t.value is not None else (float(np.mean(vals)) if vals else None)
                mean_train = t.user_attrs.get("mean_train")
                rows.append({"family": family, "round": int(round_txt), "trial": t.number,
                             "state": t.state.name, "params": t.params, "mean_val": mean_val,
                             "mean_train": mean_train,
                             "gap": None if mean_train is None or mean_val is None else mean_train - mean_val})
    except Exception:
        return []   # the database can be briefly locked while a trial is written; the UI polls again
    return _clean(rows)


@app.get("/api/runs/{run_id}/reasoning")
def get_reasoning(run_id: str):
    d = _run_dir(run_id)
    if d and (d / "reasoning_log.json").exists():
        return json.loads((d / "reasoning_log.json").read_text())
    rec = _RUNS.get(run_id)
    if rec is None:
        raise HTTPException(404, f"Run {run_id} not found")
    return [entry for event in rec["events"] for entry in event["reasoning"]]


def _downsample(xs: np.ndarray, ys: np.ndarray, n: int = 150) -> list[list[float]]:
    idx = np.unique(np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int))
    return [[float(xs[i]), float(ys[i])] for i in idx]


def _importance(model, columns: list[str]) -> list[dict]:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        values = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        values = np.abs(np.asarray(est.coef_, dtype=float)).mean(axis=0) if np.ndim(est.coef_) > 1 \
            else np.abs(np.asarray(est.coef_, dtype=float))
    else:
        return []
    total = values.sum() or 1.0
    pairs = sorted(zip(columns, values / total), key=lambda kv: -kv[1])
    return [{"feature": f, "importance": float(v)} for f, v in pairs[:15]]


@app.get("/api/runs/{run_id}/evaluation")
def get_evaluation(run_id: str):
    """Hold-out ROC / PR curves, confusion matrix and feature importance (cached in the run folder).

    The test split is used here only to REPORT, never to choose: the choice was already made on CV.
    """
    d = _run_dir(run_id)
    output = _read_output(run_id)
    if d is None or output is None:
        raise HTTPException(404, "Run not finished yet")
    cache = d / "evaluation.json"
    if cache.exists():
        return json.loads(cache.read_text())

    c1 = load_c1_output(output["dataset_id"])
    if c1.task_type != TaskType.BINARY:
        return {"supported": False, "reason": "ROC/PR curves are available for binary tasks only"}
    X_train, y_train = load_split(c1, "train")
    X_test, y_test = load_split(c1, "test")
    seed = config.load_default_config()["seed"]
    selected = joblib.load(output["selected_model_path"])

    def curves(model) -> dict:
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        return {"auc": float(roc_auc_score(y_test, proba)), "roc": _downsample(fpr, tpr),
                "ap": float(average_precision_score(y_test, proba)),
                "pr": _downsample(rec[::-1], prec[::-1]),
                "confusion": confusion_matrix(y_test, (proba >= 0.5).astype(int)).tolist()}

    result = {"supported": True, "selected_family": output["model_family"],
              "prevalence": float(np.mean(y_test)), "n_test": int(len(y_test)),
              "selected": curves(selected),
              "importance": _importance(selected, list(X_test.columns)), "candidates": {}}
    for cand in output.get("candidates", []):
        family = cand["family"]
        if family == output["model_family"]:
            c = result["selected"]
        else:
            c = curves(fit_final_model(family, cand["params"], X_train, y_train, c1.task_type, seed))
        result["candidates"][family] = {"auc": c["auc"], "roc": c["roc"], "ap": c["ap"]}
    result = _clean(result)
    cache.write_text(json.dumps(result))
    return result


@app.get("/api/runs/{run_id}/files/{name}")
def download(run_id: str, name: str):
    allowed = {"c2_output.json", "reasoning_log.json", "model.joblib", "search_history.csv"}
    if name not in allowed:
        raise HTTPException(400, f"Allowed files: {sorted(allowed)}")
    d = _run_dir(run_id)
    if d is None:
        raise HTTPException(404, "Run not found")
    if name == "search_history.csv":
        csv_path = d / "search_history.csv"
        pd.read_parquet(d / "search_history.parquet").to_csv(csv_path, index=False)
        return FileResponse(csv_path, filename=f"{run_id}_search_history.csv")
    path = d / name
    if not path.exists():
        raise HTTPException(404, f"{name} not found")
    return FileResponse(path, filename=f"{run_id}_{name}")


# ---------- endpoints: comparisons ----------
@app.get("/api/comparisons")
def comparisons():
    items = []
    for path in sorted(config.OUTPUT_DIR.glob("*/comparison_*.csv"), reverse=True):
        df = pd.read_csv(path)
        items.append({"dataset_id": path.parent.name, "file": path.name,
                      "created": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                      "rows": df.to_dict(orient="records")})
    return _clean(items)
