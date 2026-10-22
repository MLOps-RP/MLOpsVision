# C2 – Dashboard Guide (React frontend + FastAPI + Jupyter)

> A light-theme, demo-friendly web dashboard for the C2 component: watch the agents work live, see results and ROC charts, read every decision, and access inputs and outputs. Jupyter notebooks are linked for deeper analysis.

---

## 1. Start it

One command starts all three services (stop them all with **Ctrl+C**):

```bash
cd components/C2-Agentic-AutoML
./scripts/start_dashboard.sh
```

| Service | URL | Started by |
|---|---|---|
| **Dashboard** (React + Vite) | http://localhost:5173 | `cd frontend && npm run dev` |
| **API** (FastAPI) | http://127.0.0.1:8000 · docs at `/docs` | `./scripts/start_api.sh` |
| **JupyterLab** | http://localhost:8888 | `./scripts/start_jupyter.sh` |

First time only:

```bash
pip install -r requirements.txt     # adds fastapi, uvicorn, python-multipart
cd frontend && npm install          # Node 18+ (tested with Node 24)
```

> JupyterLab is started **without a token, bound to 127.0.0.1**, so the dashboard's notebook buttons open directly. This is for a local demo only. Don't expose it on a network.

---

## 2. Pages

| Page | What it shows | Use in the demo |
|---|---|---|
| **Overview** | Novelty statement, C1 → C2 → C3 flow, the three agents, one round of the loop, latest result, implementation progress | Opening slide, live |
| **Input data (C1)** | Dataset cards, profile, class balance, Feature Quality Scores, sample rows, CSV/Excel upload | "This is what C1 gives us" |
| **New run** | Choose dataset, budget, rounds; switch range/family adaptation off for ablations | Start the live run |
| **Runs & results** | All runs (live or finished) | Pick a pre-computed run |
| **Run → Agentic workflow** | **Live LangGraph graph** in agent swimlanes (active node pulses, loop arrow animates), agent activity feed, live trial charts | ⭐ The main demo moment |
| **Run → Results & charts** | KPIs, ROC curves of every family, PR curve, confusion matrix, feature importance, best score per round, overfitting-gap dumbbell, diagnoses per round | Evidence |
| **Run → Search & adaptation** | Trial history per family (coloured by round), every range change with its reason, budget per round | "The ranges really changed" |
| **Run → Reasoning** | Timeline of every agent decision with evidence (filter by agent) | Auditability |
| **Run → Inputs & outputs** | Input dataset, C2Output for C3, referrals to C1, downloads (model, JSON, CSV) | Integration readiness |
| **Baseline comparison** | B2 static vs adaptive vs ablations (grouped bars + table) | Novelty evidence |
| **Jupyter notebooks** | 4 notebooks with "Open in Jupyter" buttons | Research depth |
| **Rules & config** | Diagnosis rules, range-adaptation rules, limits, search spaces | "Knowledge is data, not code" |

Deep links: `/runs/<run_id>?tab=results` (also `workflow`, `search`, `reasoning`, `outputs`).

### Notebooks (`notebooks/`)

| Notebook | Scenario | Linked from |
|---|---|---|
| `01_explore_datasets.ipynb` | Explore C1 output: class balance, FQS, feature distributions | Input data |
| `02_baselines.ipynb` | B0/B1/B2 best-so-far curves | Baseline comparison |
| `03_run_analysis.ipynb` | One run: best per round, gap per round, search history, diagnoses, range changes | Run pages |
| `04_adaptive_comparison.ipynb` | Comparison tables, charts, Wilcoxon test | Baseline comparison |

---

## 3. Suggested demo flow (about 8 minutes)

1. **Overview**: explain the novelty and the 5-step loop.
2. **Input data → diabetes**: point out the small dataset and the class balance.
3. **New run**: diabetes, 90 trials, 3 rounds → **Start run** (about 1 minute).
4. **Agentic workflow tab** while it runs: the pulsing node, the "↺ next round" loop, Agent 1 diagnosing *CatBoost overfitting*, Agent 2's range changes, live trial dots.
5. The run finishes and the page jumps to **Results & charts**: ROC curves, the gap dumbbell (CatBoost's gap shrinks after adaptation), best score per round.
6. **Search & adaptation**: the range-change table (`depth.high 8 → 6`, *shrink depth range*).
7. **Reasoning**: open one decision and show its evidence.
8. **Baseline comparison → adult_imbalanced**: recall and F1 vs B2; the ablation explains the gain.
9. **Inputs & outputs**: C2Output for C3; open `03_run_analysis.ipynb` in Jupyter.

> Pre-run the longer adult_imbalanced comparison before the demo (it takes minutes). Only the short diabetes run should be live.

---

## 4. Architecture

```
React (Vite, :5173) ──/api proxy──▶ FastAPI (:8000) ──▶ LangGraph workflow (same as scripts/run_c2.py)
       │                                   │                 └─ writes outputs/<dataset>/<run_id>/...
       │                                   ├─ reads outputs/ (runs, history, reasoning, model)
       │                                   └─ reads outputs/optuna.db (live trials during a run)
       └── "Open in Jupyter" links ──▶ JupyterLab (:8888) ──▶ notebooks/ import c2_automl
```

- The API **never re-implements ML logic**. It starts the same graph in a background thread and serves the files the graph writes.
- **Live progress:** after each graph node the API stores a compact event. Live trials are read from the Optuna SQLite database while Agent 2 tunes.
- **One run at a time** (HTTP 409 if busy), so a demo laptop isn't overloaded.
- The hold-out ROC/PR curves are computed on demand and cached in `evaluation.json`. The test split is used only to *report*, never to choose.

### API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health`, `/api/config`, `/api/notebooks` | status, YAML rules, notebook links |
| GET | `/api/datasets`, `/api/datasets/{id}` | C1 inputs: list, profile, FQS, sample |
| POST | `/api/datasets` | upload CSV/Excel → dummy C1 output |
| POST | `/api/runs` | start a run `{dataset_id, total_trials, folds, max_rounds, adapt_ranges, adapt_families}` |
| GET | `/api/runs`, `/api/runs/{id}` | runs list; status + live events + C2Output |
| GET | `/api/runs/{id}/trials` | all trials (live during a run) |
| GET | `/api/runs/{id}/reasoning` | reasoning log |
| GET | `/api/runs/{id}/evaluation` | ROC/PR curves, confusion matrix, importance |
| GET | `/api/runs/{id}/files/{name}` | download `c2_output.json`, `model.joblib`, `reasoning_log.json`, `search_history.csv` |
| GET | `/api/comparisons` | saved `compare_adaptive.py` results |

### Files

```
src/c2_automl/api.py            FastAPI backend
tests/test_api.py               5 API tests (incl. a full run through the API)
scripts/start_api.sh            uvicorn on :8000
scripts/start_jupyter.sh        JupyterLab on :8888
scripts/start_dashboard.sh      all three together
notebooks/0*.ipynb              4 notebooks
frontend/
  src/App.tsx                   layout, navigation, routes
  src/api.ts                    typed API client
  src/theme.ts                  colours (validated palette), labels, formatting
  src/components/               UI kit, WorkflowGraph (live swimlane graph)
  src/charts/                   ROC, PR, confusion matrix, dumbbell, trial small multiples, comparison bars
  src/pages/                    Overview, Data, NewRun, Runs, RunDetail (+ run/ tabs), Compare, Notebooks, Config
```

### Chart design rules used

- Colours come from a palette validated for colour-blind safety. A model family **keeps its colour on every chart**.
- Every chart has a **table view** (the switch at the top right of each chart card) and a legend with text labels, so values never depend on colour alone.
- Diagnoses use status colours **with an icon and a label** (✓ healthy, ⚠ overfitting, ⚖ imbalance).
- Thin marks, hairline grid, one y-axis per chart, a tooltip on hover.
- Search history uses **small multiples** (one panel per family) instead of a crowded single scatter.

---

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| Top bar shows **API offline** | Start `./scripts/start_api.sh` (port 8000) |
| Notebook buttons show "can't connect" | Start `./scripts/start_jupyter.sh` (port 8888) |
| "Another run is still in progress" | Wait for the current run, or restart the API |
| Results tab spins for a while | The first visit refits each family once for the ROC curves (seconds on Adult); it's cached afterwards |
| Workflow tab shows activity "rebuilt from the reasoning log" | The run was started from the command line; live node events exist only for runs started from the dashboard |
| Port already in use | `lsof -i :5173` / `:8000` / `:8888` and stop the old process |
