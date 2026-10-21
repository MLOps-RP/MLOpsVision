# C4 – Drift Diagnosis & Remediation Engine

C4 watches a deployed model in production. It detects drift, diagnoses what kind of drift it is, scores how serious it is, and decides what to do about it.

```
INPUT                          PROCESSING (C4)                      OUTPUT
deployed model + baseline  ->  1 monitoring  (PSI, KS, JSD,     ->  Drift Report (JSON)
production stream (windows)      ADWIN, data-quality checks)        action: warn / route C1 / route C2 /
delayed labels                 2 diagnosis   (which drift type,             block / rollback
                                 which features / segments)          audit log (SQLite)
                               3 severity    (Low..Critical)         dashboard
                               4 decision    (policy.yaml)
```

C1, C2 and C3 are not needed to run this. A training script stands in for them before C4 (the "deployment simulator"). After C4, the Drift Report is saved for them to pick up.

---

## 1. One-time setup

Install **Python 3.11** (python.org; on Windows tick *Add Python to PATH*), **VS Code** with the *Python* and *Jupyter* extensions, and **Git**.

Open a terminal **in the project folder** (in VS Code: *View → Terminal*), then run the commands for your system.

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell says *running scripts is disabled*, run this once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Mac / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

When the environment is active, your prompt starts with `(.venv)`. Run every command below from the project folder with `(.venv)` active.

In VS Code, press Ctrl+Shift+P → *Python: Select Interpreter* → pick the one inside `.venv`.

---

## 2. Run everything, in this order

| # | Command | What it does | Time |
|---|---------|--------------|------|
| 1 | `python -m src.bootstrap.train_and_register` | Downloads Adult (first time only), splits it 40/10/10/40, trains **v1 Logistic Regression** and **v2 XGBoost**, registers both in MLflow (production → v2, previous → v1), and saves each model's **versioned baseline** | 1–2 min |
| 2 | `python -m src.simulator.generate_scenarios` | Cuts the future stream into 39 windows of 500 rows and builds 7 drift scenarios plus their ground-truth files | 10 s |
| 3 | `python -m src.c4.monitoring.calibration` | Learns a separate drift threshold for every feature and signal, for each model version | 30 s |
| 4 | `python -m src.c4.run_c4 --scenario S3_concept` | Runs C4 on one scenario, window by window, and prints one line per window | 10 s |
| 5 | `python -m src.experiments.run_all` | Runs all 7 scenarios, scores them against ground truth, and compares with naive retraining. Writes `results/` | 1 min |
| 6 | `streamlit run dashboard/app.py` | Opens the dashboard in your browser | – |
| 7 | `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000` | Opens the MLflow UI at http://localhost:5000 (models, versions, aliases) | – |

Run steps 6 and 7 in **separate terminals**; activate `.venv` in each one. Stop either with Ctrl+C.

### Grafana monitoring view (optional, needs Docker Desktop)

Streamlit is the interactive demo. Grafana OSS is the production-style monitoring view: severity score, F1, PSI / JSD per feature and prediction drift over time. Markers on every chart show where drift was injected (blue), each C4 action such as ROUTE_C2 or ROLLBACK (red) and WATCH / WARN (yellow).

```bash
docker compose up -d        # first start downloads Grafana + the SQLite plugin (~1 min)
```

Open http://localhost:3000 (anyone can view; log in as `admin` / `admin` to edit) and pick a scenario at the top. Stop it with `docker compose down`.

Each run of `run_c4` (or `run_all`) refreshes `outputs/c4_monitoring.db`, which Grafana reads. Press *Refresh* in Grafana afterwards. Grafana needs a time axis, so each window is shown as one simulated hour from 2026-01-01 00:00 UTC. To rebuild the file by hand: `python -m src.tools.export_grafana`.

### What step 4 prints (example)

```
w20 * v2 | fired  0 | F1 0.481 | CONCEPT_DRIFT       conf | Medium   0.32 | NONE
w21 * v2 | fired  0 | F1 0.453 | CONCEPT_DRIFT       conf | Medium   0.41 | ROUTE_C2 <== ACTION
```

Each line shows the window, `*` if drift was injected in it, the model version serving it, how many features fired, F1 from delayed labels, the diagnosis and its confidence, the severity, and the action.

### Scenarios

| Scenario | What is injected (from window 15) | C4 should decide |
|----------|-----------------------------------|------------------|
| `S0_no_drift` | nothing | NONE |
| `S1_covariate_benign` | race mix changes (a feature the model hardly uses) | WARN_ONLY (no retraining) |
| `S2_covariate_degrading` | age +12, education-num +3, hours +10 for 80% of rows | ROUTE_C1 |
| `S3_concept` | inputs unchanged; 40% of labels flip for married people | ROUTE_C2 |
| `S4_label_shift` | share of >50K rises to 50% | ROUTE_C2 |
| `S5_data_quality` | 40% of hours missing, the rest ×60 | BLOCK_AND_FLAG |
| `S6_sudden_drop` | 60% of all labels flip at once | ROLLBACK (then ROUTE_C2) |

All settings are in `configs/scenarios.yaml`.

---

## 3. Demo commands for the presentation

```bash
python -m src.tools.reset_production                         # production -> v2 again
python -m src.c4.run_c4 --scenario S1_covariate_benign       # warn only, no retraining
python -m src.c4.run_c4 --scenario S3_concept                # concept drift -> C2
python -m src.c4.run_c4 --scenario S5_data_quality           # broken data -> blocked
python -m src.c4.run_c4 --scenario S6_sudden_drop            # critical -> ROLLBACK to v1
```

After S6, refresh the MLflow UI. *Models → income-classifier* shows the `production` alias on **version 1**. Run `python -m src.tools.reset_production` before the next demo.

In the dashboard, pick a scenario in the sidebar and press **Run C4 on this scenario**. Then drag the **Replay** slider from window 0 to 38 to show detection happening over time.

---

## 4. Where to find the outputs

| Path | Contents |
|------|----------|
| `artifacts/model_v<N>/baseline.json` | versioned baseline (also attached to the model's MLflow run) |
| `artifacts/model_v<N>/thresholds.json` | calibrated thresholds |
| `data/scenarios/<scenario>/` | production windows + `ground_truth.json` |
| `outputs/drift_reports/<scenario>/*.json` | **Drift Reports** (the contract for C1/C2/C3) |
| `outputs/audit.db` | **audit log** (tables `runs`, `window_signals`, `decisions`) |
| `outputs/c4_monitoring.db` | latest run per scenario, flattened for Grafana |
| `outputs/runs/<scenario>/window_log.csv` | one row per window, used by the dashboard |
| `results/results_table.csv`, `results/unnecessary_retrains.png` | results for the slides |

To read the audit log in Python:

```python
import sqlite3, pandas as pd
pd.read_sql("SELECT * FROM decisions", sqlite3.connect("outputs/audit.db"))
```

---

## 5. What each file does

| File | Role |
|------|------|
| `src/bootstrap/data.py` | load, clean and split the Adult dataset |
| `src/bootstrap/train_and_register.py` | deployment simulator: train v1/v2, register in MLflow, set aliases |
| `src/bootstrap/baseline.py` | builds the versioned baseline |
| `src/simulator/drift_injector.py` | one function per drift type |
| `src/simulator/generate_scenarios.py` | production stream simulator (windows, scenarios, ground truth) |
| `src/c4/monitoring/stats.py` | PSI, JSD, bins, metrics (shared by baseline and monitor) |
| `src/c4/monitoring/feature_tests.py` | PSI and KS for numeric features, JSD for categorical |
| `src/c4/monitoring/prediction_tests.py` | prediction-mix and confidence drift |
| `src/c4/monitoring/performance.py` | delayed labels, accuracy/F1, ADWIN (river) |
| `src/c4/monitoring/data_quality.py` | null, range and unseen-category rules |
| `src/c4/monitoring/calibration.py` | adaptive thresholds: mean + k × std on clean windows |
| `src/c4/monitoring/monitor.py` | combines all signals for one window |
| `src/c4/diagnosis/diagnose.py` | drift type rules, affected features, affected segments |
| `src/c4/severity/severity.py` | severity score and level |
| `src/c4/controller/policy.py` | applies `configs/policy.yaml` |
| `src/c4/controller/rollback.py` | real rollback via MLflow aliases |
| `src/c4/audit/drift_report.py` | Drift Report contract (dataclass → JSON) |
| `src/c4/audit/audit_log.py` | SQLite audit log |
| `src/c4/run_c4.py` | **the C4 engine**: runs everything per window |
| `src/experiments/run_all.py` | all scenarios + naive baseline + results |
| `src/tools/reset_production.py` | set production back to v2 |
| `src/tools/export_grafana.py` | audit log → time-indexed tables for Grafana (`outputs/c4_monitoring.db`) |
| `dashboard/app.py` | Streamlit dashboard (input → processing → output) |
| `docker-compose.yml`, `grafana/` | Grafana OSS service, data source and the provisioned *C4 – Drift Monitoring* dashboard |
| `notebooks/01_learn_step_by_step.ipynb` | learn the ideas interactively (training, PSI, KS, baseline) |

---

## 6. Tuning (only change YAML files)

* **Too many false alarms in S0**: raise `calibration.k` in `configs/monitoring.yaml` (3 → 4), or `min_persistence` in `configs/severity.yaml` (2 → 3).
* **A drift is not detected**: make it stronger in `configs/scenarios.yaml` (`share`, `shifts`, `flip_prob`).
* **The wrong action is chosen**: check the rule order in `configs/policy.yaml`.

After changing `scenarios.yaml`, rerun steps 2 and 5. After changing `monitoring.yaml`, rerun steps 3 and 5.

To start completely fresh, delete `mlflow.db`, `mlruns/`, `artifacts/`, `outputs/` and `data/processed/`, then run steps 1–5 again.

---

## 7. Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'src'` | Run commands from the **project folder** (where this README is), using `python -m ...` |
| `ModuleNotFoundError` for pandas, mlflow, … | `.venv` is not active: activate it, then `pip install -r requirements.txt` |
| Adult download fails (no internet / OpenML down) | Download `adult.data` and `adult.test` from https://archive.ics.uci.edu/dataset/2/adult into `data/raw/`, then rerun step 1 |
| `RESOURCE_DOES_NOT_EXIST` / alias not found | Run step 1 first (it registers the models and aliases) |
| `thresholds.json` not found | Run step 3 (calibration) |
| `windows.csv` not found | Run step 2 |
| Port 5000 already in use (often on Mac) | Use `--port 5001` and open http://localhost:5001 |
| Results differ from the example | Normal: real Adult data differs from the example output. Tune with section 6 |

---

## 8. Save your work to GitHub

```bash
git init
git add .
git commit -m "C4 drift engine: monitoring, diagnosis, severity, decision, rollback"
git branch -M main
git remote add origin https://github.com/<your-username>/c4-drift-engine.git
git push -u origin main
```

Generated files (MLflow store, data, outputs) are excluded by `.gitignore`. `results/` is kept, because it holds your slide material.
