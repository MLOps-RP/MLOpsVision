# C2 – Step 2 Guide: Agent 2 (Hyperparameter Optimization), Levels 1–3

> This guide covers **Roadmap Step 2**: **Agent 2, Levels 1–3** (§11 and §13.3 of [C2_IMPLEMENTATION_GUIDE.md](C2_IMPLEMENTATION_GUIDE.md)).
> It continues from [C2_SETUP_GUIDE_AGENT1.md](C2_SETUP_GUIDE_AGENT1.md). You need Parts A–C of that guide done first.
>
> At the end you will have:
> - Agent 2 that tunes each model family with **Optuna**,
> - your three **baselines**: **B0** (default settings), **B1** (random search), **B2** (Optuna TPE with fixed ranges),
> - pruning (bad trials are stopped early), a SQLite study database, and a fair **trial budget**,
> - a full **search history** (one row per trial) saved as Parquet,
> - 10 new unit tests (23 in total).

---

## Contents

- [1. What Agent 2 does, and why baselines come first](#1-what-agent-2-does-and-why-baselines-come-first)
- [2. Files in this step](#2-files-in-this-step)
- [3. Foundation for Agent 2](#3-foundation-for-agent-2)
  - [3.1 Config: add the `hpo` section](#31-config-add-the-hpo-section)
  - [3.2 Search spaces as data](#32-search-spaces-as-data)
  - [3.3 Search-space tool](#33-search-space-tool)
  - [3.4 Small change to `train_eval.py`](#34-small-change-to-train_evalpy)
  - [3.5 Optuna runner](#35-optuna-runner)
- [4. Agent 2, level by level](#4-agent-2-level-by-level)
  - [4.1 Level 1 – Default parameters (Baseline B0)](#41-level-1--default-parameters-baseline-b0)
  - [4.2 Level 2 – Static Optuna search (Baselines B1, B2)](#42-level-2--static-optuna-search-baselines-b1-b2)
  - [4.3 Level 3 – Pruning, storage, budget](#43-level-3--pruning-storage-budget)
  - [4.4 The full Agent 2 file](#44-the-full-agent-2-file)
- [5. Unit tests](#5-unit-tests)
- [6. Run the baselines](#6-run-the-baselines)
- [7. What the results tell you (important for the thesis)](#7-what-the-results-tell-you-important-for-the-thesis)
- [8. Commit, troubleshooting, next steps](#8-commit-troubleshooting-next-steps)

---

## 1. What Agent 2 does, and why baselines come first

Agent 2 takes the model families chosen by Agent 1 and finds good **hyperparameters** (settings such as tree depth and learning rate) for each one.

```
Agent 1: "try xgboost, lightgbm, catboost"   +   trial budget
                       │
                       ▼
Agent 2: for each family → Optuna study → many trials (each trial = one 5-fold CV run)
                       │
                       ▼
best params + score per family   +   full search history (every trial)
```

In this step, the search ranges are **fixed** (static). That is on purpose:

| Baseline | What it is | Why you need it |
|---|---|---|
| **B0 – defaults** | Library default settings, no tuning | Shows how much tuning helps at all |
| **B1 – random** | Random search inside fixed ranges | A simple tuning baseline |
| **B2 – TPE** | Optuna TPE inside fixed ranges | **The main baseline.** Standard AutoML works like this. Your adaptive method (Step 4) must beat it **with the same trial budget**. |

> Without B2 you cannot prove your novelty. Build it before the adaptive version.

---

## 2. Files in this step

| File | New / changed | Purpose |
|---|---|---|
| `configs/default.yaml` | changed | add `hpo` section |
| `configs/search_spaces.yaml` | new | base ranges per family (data, not code) |
| `src/c2_automl/tools/search_space.py` | new | load ranges, ask Optuna for values |
| `src/c2_automl/tools/train_eval.py` | changed | optional `on_fold` hook (for pruning) |
| `src/c2_automl/tools/hpo.py` | new | Optuna runner + history rows |
| `src/c2_automl/agents/agent2_hpo.py` | new | **Agent 2**, Levels 1–3 |
| `scripts/run_agent2_baseline.py` | new | run B0 / B1 / B2 |
| `scripts/compare_baselines.py` | new | compare saved runs |
| `tests/test_agent2.py` | new | 10 unit tests |

No new libraries are needed: `optuna` is already in `requirements.txt`.
Tested with **Optuna 5.0**, scikit-learn 1.9, XGBoost 3.2, LightGBM 4.7, CatBoost 1.2.10.

---

## 3. Foundation for Agent 2

### 3.1 Config: add the `hpo` section

Replace the end of **`configs/default.yaml`** (from `budget:` down) with:

```yaml
# Trial budget for one C2 run (shared by all families and rounds)
budget:
  total_trials: 150
  max_rounds: 4

# Agent 2 - hyperparameter optimization
hpo:
  method: tpe               # tpe = Baseline B2, random = Baseline B1
  pruning: true             # stop clearly bad trials after a few CV folds
```

Full file for reference:

```yaml
seed: 42
cv_folds: 5
test_size: 0.2

# Main metric per task type (scikit-learn scorer names; higher is always better)
metrics:
  binary: roc_auc
  multiclass: f1_macro
  regression: neg_root_mean_squared_error

# Used by Agent 1 to describe the dataset (task profile)
profile:
  small_rows: 1000          # fewer rows than this = "small" dataset
  large_rows: 200000        # more rows than this = "large" dataset
  many_categorical_ratio: 0.3

# Trial budget for one C2 run (shared by all families and rounds)
budget:
  total_trials: 150
  max_rounds: 4

# Agent 2 - hyperparameter optimization
hpo:
  method: tpe               # tpe = Baseline B2, random = Baseline B1
  pruning: true             # stop clearly bad trials after a few CV folds
```

### 3.2 Search spaces as data

The ranges live in YAML, **not** inside the objective function. In Step 4, Agent 2 Level 4 will **change these ranges** based on Agent 1's diagnosis (for example, lower the maximum `max_depth` after overfitting). Keeping them as data makes that possible.

**`configs/search_spaces.yaml`**

```yaml
# Base (static) hyperparameter search spaces per model family.
# Stored as DATA so Agent 2 Level 4 can shrink / widen / shift them per diagnosis.
#   type: int | float | cat     log: true -> sample on a log scale

random_forest:
  n_estimators:      {type: int,   low: 100,  high: 500}
  max_depth:         {type: int,   low: 3,    high: 20}
  min_samples_leaf:  {type: int,   low: 1,    high: 20}
  max_features:      {type: float, low: 0.3,  high: 1.0}

xgboost:
  n_estimators:      {type: int,   low: 100,  high: 600}
  max_depth:         {type: int,   low: 3,    high: 10}
  learning_rate:     {type: float, low: 0.01, high: 0.3,  log: true}
  min_child_weight:  {type: float, low: 1,    high: 10,   log: true}
  subsample:         {type: float, low: 0.6,  high: 1.0}
  colsample_bytree:  {type: float, low: 0.6,  high: 1.0}
  reg_lambda:        {type: float, low: 0.001, high: 10,  log: true}

lightgbm:
  n_estimators:      {type: int,   low: 100,  high: 600}
  num_leaves:        {type: int,   low: 8,    high: 128,  log: true}
  max_depth:         {type: int,   low: 3,    high: 12}
  learning_rate:     {type: float, low: 0.01, high: 0.3,  log: true}
  min_child_samples: {type: int,   low: 5,    high: 100}
  subsample:         {type: float, low: 0.6,  high: 1.0}
  subsample_freq:    {type: cat,   choices: [1]}
  colsample_bytree:  {type: float, low: 0.6,  high: 1.0}
  reg_lambda:        {type: float, low: 0.001, high: 10,  log: true}

catboost:
  iterations:        {type: int,   low: 100,  high: 500}
  depth:             {type: int,   low: 3,    high: 8}
  learning_rate:     {type: float, low: 0.01, high: 0.3,  log: true}
  l2_leaf_reg:       {type: float, low: 1,    high: 10,   log: true}

logistic_regression:
  C:                 {type: float, low: 0.001, high: 100, log: true}

ridge:
  alpha:             {type: float, low: 0.001, high: 100, log: true}
```

> `subsample_freq: [1]` for LightGBM: LightGBM ignores `subsample` unless `subsample_freq > 0`.

### 3.3 Search-space tool

**`src/c2_automl/tools/search_space.py`**

```python
"""Hyperparameter search spaces (trusted tool). Spaces are plain dicts so they can be adapted later."""
import copy

import optuna

from ..config import load_yaml


def load_base_space(family: str) -> dict:
    """The static search space for one family (from configs/search_spaces.yaml)."""
    return copy.deepcopy(load_yaml("search_spaces.yaml")[family])


def suggest_params(trial: optuna.Trial, space: dict) -> dict:
    """Ask Optuna for one value per hyperparameter inside the given ranges."""
    params = {}
    for name, r in space.items():
        if r["type"] == "int":
            params[name] = trial.suggest_int(name, r["low"], r["high"], log=r.get("log", False))
        elif r["type"] == "float":
            params[name] = trial.suggest_float(name, r["low"], r["high"], log=r.get("log", False))
        elif r["type"] == "cat":
            params[name] = trial.suggest_categorical(name, r["choices"])
        else:
            raise ValueError(f"Unknown range type for {name}: {r['type']}")
    return params
```

`load_base_space` returns a **copy**, so changing a space for one round can never change the YAML defaults for the next run.

### 3.4 Small change to `train_eval.py`

Optuna can stop (**prune**) a trial early if its first CV folds already look bad. To do that it must see the score **after each fold**. Add an optional `on_fold` callback to `cross_validate_family`. Agent 1's code doesn't pass it, so it keeps working unchanged.

Three edits in **`src/c2_automl/tools/train_eval.py`**:

1. Import line:
   ```python
   from typing import Callable, Optional
   ```
2. Function signature + docstring:
   ```python
   def cross_validate_family(family: str, X: pd.DataFrame, y: pd.Series, task_type: TaskType,
                             metric: str, folds: int = 5, seed: int = 42,
                             params: Optional[dict] = None,
                             on_fold: Optional[Callable[[int, list[float]], None]] = None) -> dict:
       """Train one model family with CV. Returns train/validation scores and out-of-fold predictions.

       on_fold(fold_index, val_scores_so_far) is called after each fold. Agent 2 uses it to let
       Optuna stop (prune) a bad trial early by raising optuna.TrialPruned.
       """
   ```
3. Loop: number the folds, and call the hook at the end of each fold:
   ```python
       for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
           ...
           oof_pred[val_idx] = np.ravel(model.predict(X_va))
           if on_fold is not None:
               on_fold(fold, val_scores)
   ```

Full file after the change:

```python
"""Cross-validated training (trusted tool). Agent 1 reads these results; it never trains models itself."""
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from ..schemas import TaskType
from .models import make_model


def make_splitter(task_type: TaskType, folds: int, seed: int):
    if task_type == TaskType.REGRESSION:
        return KFold(n_splits=folds, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def cross_validate_family(family: str, X: pd.DataFrame, y: pd.Series, task_type: TaskType,
                          metric: str, folds: int = 5, seed: int = 42,
                          params: Optional[dict] = None,
                          on_fold: Optional[Callable[[int, list[float]], None]] = None) -> dict:
    """Train one model family with CV. Returns train/validation scores and out-of-fold predictions.

    on_fold(fold_index, val_scores_so_far) is called after each fold. Agent 2 uses it to let
    Optuna stop (prune) a bad trial early by raising optuna.TrialPruned.
    """
    scorer = get_scorer(metric)
    splitter = make_splitter(task_type, folds, seed)
    train_scores, val_scores, fit_times = [], [], []
    oof_pred = np.zeros(len(y), dtype=y.dtype)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        model = make_model(family, task_type, params, seed)
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        start = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_times.append(time.perf_counter() - start)
        train_scores.append(scorer(model, X_tr, y_tr))
        val_scores.append(scorer(model, X_va, y_va))
        oof_pred[val_idx] = np.ravel(model.predict(X_va))
        if on_fold is not None:
            on_fold(fold, val_scores)

    return {
        "family": family,
        "params": params or {},
        "train_scores": [float(s) for s in train_scores],
        "val_scores": [float(s) for s in val_scores],
        "mean_train": float(np.mean(train_scores)),
        "mean_val": float(np.mean(val_scores)),
        "cv_std": float(np.std(val_scores)),
        "fit_time_s": float(np.sum(fit_times)),
        "oof_pred": oof_pred,
    }


def baseline_score(X: pd.DataFrame, y: pd.Series, task_type: TaskType, metric: str,
                   folds: int = 5, seed: int = 42) -> float:
    """Score of a 'no-skill' model. Used to measure lift (are the features useful at all?)."""
    dummy = DummyRegressor() if task_type == TaskType.REGRESSION else DummyClassifier(strategy="prior")
    scores = cross_val_score(dummy, X, y, scoring=metric, cv=make_splitter(task_type, folds, seed))
    return float(np.mean(scores))
```

### 3.5 Optuna runner

This is a **trusted tool**: it runs Optuna for **one** family inside **one** given space. Every trial is a full CV run. The evidence Agent 1 needs later (train score, gap, fold scores) is stored on each trial.

**`src/c2_automl/tools/hpo.py`**

```python
"""Optuna runner (trusted tool). Tunes ONE model family inside a given search space."""
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import RandomSampler, TPESampler

from ..schemas import TaskType
from .search_space import suggest_params
from .train_eval import cross_validate_family

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_sampler(method: str, seed: int) -> optuna.samplers.BaseSampler:
    if method == "tpe":
        return TPESampler(seed=seed, multivariate=True, n_startup_trials=10)
    if method == "random":
        return RandomSampler(seed=seed)
    raise ValueError(f"Unknown sampler: {method}")


def tune_family(family: str, space: dict, X: pd.DataFrame, y: pd.Series, task_type: TaskType,
                metric: str, n_trials: int, folds: int = 5, seed: int = 42,
                method: str = "tpe", pruning: bool = True, study_name: Optional[str] = None,
                storage: Optional[str] = None, enqueue: Optional[list[dict]] = None) -> optuna.Study:
    """Run one Optuna study. Every trial is a full CV run; evidence is stored as trial user attributes."""
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1) if pruning else NopPruner()
    study = optuna.create_study(direction="maximize", sampler=make_sampler(method, seed),
                                pruner=pruner, study_name=study_name, storage=storage)
    for params in enqueue or []:
        study.enqueue_trial(params)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, space)

        def on_fold(fold: int, val_scores: list[float]) -> None:
            trial.set_user_attr("val_scores", [float(s) for s in val_scores])
            trial.report(float(np.mean(val_scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result = cross_validate_family(family, X, y, task_type, metric, folds, seed,
                                       params=params, on_fold=on_fold)
        trial.set_user_attr("mean_train", result["mean_train"])
        trial.set_user_attr("cv_std", result["cv_std"])
        trial.set_user_attr("fit_time_s", result["fit_time_s"])
        return result["mean_val"]

    study.optimize(objective, n_trials=n_trials)
    return study


def study_to_records(study: optuna.Study, family: str, method: str, round_no: int,
                     run_id: str) -> list[dict]:
    """Turn every trial (complete AND pruned) into a flat row: this is the search history."""
    rows = []
    for t in study.trials:
        val_scores = t.user_attrs.get("val_scores", [])
        mean_train = t.user_attrs.get("mean_train")
        mean_val = t.value if t.value is not None else (float(np.mean(val_scores)) if val_scores else None)
        rows.append({
            "run_id": run_id, "method": method, "round": round_no, "family": family,
            "trial": t.number, "state": t.state.name, "params": t.params,
            "mean_val": mean_val, "mean_train": mean_train,
            "gap": None if mean_train is None or mean_val is None else mean_train - mean_val,
            "cv_std": t.user_attrs.get("cv_std"), "folds_done": len(val_scores),
            "fit_time_s": t.user_attrs.get("fit_time_s"),
            "duration_s": t.duration.total_seconds() if t.duration else None,
        })
    return rows
```

Key points:

| Setting | Value | Why |
|---|---|---|
| Direction | `maximize` | All scorers are "higher is better" (regression uses `neg_root_mean_squared_error`) |
| TPE | `seed`, `multivariate=True`, `n_startup_trials=10` | Reproducible; learns how parameters work together; first 10 trials are random to explore |
| Pruner | `MedianPruner(n_startup_trials=5, n_warmup_steps=1)` | After 5 complete trials, stop a trial whose running mean after fold 2+ is below the median |
| `enqueue` | list of params | Tried first. Used in Step 4 to warm-start a round with the previous best |
| Pruned trials | kept in history | They are evidence too (`folds_done` shows where they stopped) |

---

## 4. Agent 2, level by level

### 4.1 Level 1 – Default parameters (Baseline B0)

**Goal:** train every family with library defaults, and return the result in **the same row format** as an Optuna trial. Then every method's history can be compared in one table.

```python
def run_defaults(families: list[str], X: pd.DataFrame, y: pd.Series, c1: C1Output,
                 profile: TaskProfile, cfg: dict, run_id: str, round_no: int = 1) -> list[dict]:
    """One 'trial' per family with library default settings (same row format as Optuna trials)."""
    rows = []
    for family in families:
        r = cross_validate_family(family, X, y, c1.task_type, profile.metric,
                                  cfg["cv_folds"], cfg["seed"])
        rows.append({
            "run_id": run_id, "method": "defaults", "round": round_no, "family": family,
            "trial": 0, "state": "COMPLETE", "params": {},
            "mean_val": r["mean_val"], "mean_train": r["mean_train"],
            "gap": r["mean_train"] - r["mean_val"], "cv_std": r["cv_std"],
            "folds_done": len(r["val_scores"]), "fit_time_s": r["fit_time_s"],
            "duration_s": r["fit_time_s"],
        })
    return rows
```

✅ **Check:** one row per family, `method="defaults"`, `params={}`.

### 4.2 Level 2 – Static Optuna search (Baselines B1, B2)

**Goal:** tune every family with Optuna inside the fixed base ranges. `method="tpe"` gives **B2**, `method="random"` gives **B1**.

```python
def run_hpo(families: list[str], budget: dict[str, int], X: pd.DataFrame, y: pd.Series,
            c1: C1Output, profile: TaskProfile, cfg: dict, run_id: str, method: str = "tpe",
            round_no: int = 1, spaces: Optional[dict[str, dict]] = None,
            use_storage: bool = True) -> tuple[list[dict], dict[str, dict], ReasoningEntry]:
    """Tune every family with its share of the budget.

    Returns (trial rows for the search history, best result per family, reasoning entry).
    `spaces` lets Level 4 pass adapted ranges; by default the static base spaces are used.
    """
    hpo_cfg = cfg["hpo"]
    storage = f"sqlite:///{OUTPUT_DIR / 'optuna.db'}" if use_storage else None
    if use_storage:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, best = [], {}
    for family in families:
        space = (spaces or {}).get(family) or load_base_space(family)
        study = tune_family(
            family, space, X, y, c1.task_type, profile.metric, n_trials=budget[family],
            folds=cfg["cv_folds"], seed=cfg["seed"], method=method, pruning=hpo_cfg["pruning"],
            study_name=f"{run_id}_r{round_no}_{family}", storage=storage)
        family_rows = study_to_records(study, family, method, round_no, run_id)
        rows += family_rows
        complete = [r for r in family_rows if r["state"] == "COMPLETE"]
        if complete:
            top = max(complete, key=lambda r: r["mean_val"])
            best[family] = {**top, "space": space,
                            "n_complete": len(complete),
                            "n_pruned": sum(r["state"] == "PRUNED" for r in family_rows)}

    entry = ReasoningEntry(
        round=round_no, agent="agent2", step="run_hpo",
        decision={"method": method, "budget": budget},
        evidence={f: {"best_val": b["mean_val"], "gap": b["gap"], "params": b["params"],
                      "n_complete": b["n_complete"], "n_pruned": b["n_pruned"]}
                  for f, b in best.items()},
        rationale=f"{method} search with static base spaces; pruning={hpo_cfg['pruning']}")
    return rows, best, entry
```

Notice the `spaces` argument. It is unused now (static ranges), but in Step 4 Agent 2 Level 4 will pass **adapted** ranges here. Nothing else needs to change.

✅ **Check:** each family gets the number of trials in its budget; `best` holds the best complete trial per family; the reasoning entry records the method and budget.

### 4.3 Level 3 – Pruning, storage, budget

**Pruning** is in `tools/hpo.py` (section 3.5) and is switched on/off by `hpo.pruning` in the config.

**Storage:** every study is saved in `outputs/optuna.db` (SQLite) with the name `<run_id>_r<round>_<family>`. You can inspect it later (see section 6.4).

**Budget:** for a **fair** comparison every method must use the **same total number of trials**. `split_budget` shares the total between families. It is equal for now; in Step 4, Agent 1 will pass weights (more trials for good families, none for dropped ones).

```python
def split_budget(total_trials: int, families: list[str],
                 weights: Optional[dict[str, float]] = None) -> dict[str, int]:
    """Share the trial budget between families. Equal by default; Agent 1 can pass weights later.

    Always sums exactly to total_trials, and every family gets at least 1 trial.
    """
    if total_trials < len(families):
        raise ValueError(f"budget {total_trials} is smaller than the number of families {len(families)}")
    weights = weights or {f: 1.0 for f in families}
    total_w = sum(weights[f] for f in families)
    raw = {f: total_trials * weights[f] / total_w for f in families}
    budget = {f: max(1, int(raw[f])) for f in families}
    # hand out the remaining trials to the largest fractional parts
    leftover = total_trials - sum(budget.values())
    for f in sorted(families, key=lambda f: raw[f] - int(raw[f]), reverse=True)[:max(leftover, 0)]:
        budget[f] += 1
    while sum(budget.values()) > total_trials:  # only if the "at least 1" rule overshot
        biggest = max(budget, key=budget.get)
        budget[biggest] -= 1
    return budget
```

| Call | Result |
|---|---|
| `split_budget(60, 5 families)` | 12 each |
| `split_budget(10, ["a","b","c"])` | 4 / 3 / 3 (always sums to 10) |
| `split_budget(30, ["a","b"], weights={"a": 2, "b": 1})` | a=20, b=10 |

### 4.4 The full Agent 2 file

**`src/c2_automl/agents/agent2_hpo.py`**

```python
"""Agent 2 - Hyperparameter Optimization.

Level 1: evaluate default parameters            -> Baseline B0
Level 2: static Optuna search (TPE or random)    -> Baselines B2 (TPE) and B1 (random)
Level 3: pruning, SQLite storage, trial budget from Agent 1

Level 4 (next step) will adapt the search spaces using Agent 1's actions.
"""
from datetime import datetime
from typing import Optional

import pandas as pd

from ..config import OUTPUT_DIR
from ..schemas import C1Output, ReasoningEntry, TaskProfile
from ..tools.hpo import study_to_records, tune_family
from ..tools.search_space import load_base_space
from ..tools.train_eval import cross_validate_family


def new_run_id(dataset_id: str, method: str) -> str:
    return f"{dataset_id}_{method}_{datetime.now():%Y%m%d_%H%M%S}"


# ---------- Level 1 ----------
def run_defaults(families: list[str], X: pd.DataFrame, y: pd.Series, c1: C1Output,
                 profile: TaskProfile, cfg: dict, run_id: str, round_no: int = 1) -> list[dict]:
    """One 'trial' per family with library default settings (same row format as Optuna trials)."""
    rows = []
    for family in families:
        r = cross_validate_family(family, X, y, c1.task_type, profile.metric,
                                  cfg["cv_folds"], cfg["seed"])
        rows.append({
            "run_id": run_id, "method": "defaults", "round": round_no, "family": family,
            "trial": 0, "state": "COMPLETE", "params": {},
            "mean_val": r["mean_val"], "mean_train": r["mean_train"],
            "gap": r["mean_train"] - r["mean_val"], "cv_std": r["cv_std"],
            "folds_done": len(r["val_scores"]), "fit_time_s": r["fit_time_s"],
            "duration_s": r["fit_time_s"],
        })
    return rows


# ---------- Level 3 (budget) ----------
def split_budget(total_trials: int, families: list[str],
                 weights: Optional[dict[str, float]] = None) -> dict[str, int]:
    """Share the trial budget between families. Equal by default; Agent 1 can pass weights later.

    Always sums exactly to total_trials, and every family gets at least 1 trial.
    """
    if total_trials < len(families):
        raise ValueError(f"budget {total_trials} is smaller than the number of families {len(families)}")
    weights = weights or {f: 1.0 for f in families}
    total_w = sum(weights[f] for f in families)
    raw = {f: total_trials * weights[f] / total_w for f in families}
    budget = {f: max(1, int(raw[f])) for f in families}
    # hand out the remaining trials to the largest fractional parts
    leftover = total_trials - sum(budget.values())
    for f in sorted(families, key=lambda f: raw[f] - int(raw[f]), reverse=True)[:max(leftover, 0)]:
        budget[f] += 1
    while sum(budget.values()) > total_trials:  # only if the "at least 1" rule overshot
        biggest = max(budget, key=budget.get)
        budget[biggest] -= 1
    return budget


# ---------- Level 2 + 3 ----------
def run_hpo(families: list[str], budget: dict[str, int], X: pd.DataFrame, y: pd.Series,
            c1: C1Output, profile: TaskProfile, cfg: dict, run_id: str, method: str = "tpe",
            round_no: int = 1, spaces: Optional[dict[str, dict]] = None,
            use_storage: bool = True) -> tuple[list[dict], dict[str, dict], ReasoningEntry]:
    """Tune every family with its share of the budget.

    Returns (trial rows for the search history, best result per family, reasoning entry).
    `spaces` lets Level 4 pass adapted ranges; by default the static base spaces are used.
    """
    hpo_cfg = cfg["hpo"]
    storage = f"sqlite:///{OUTPUT_DIR / 'optuna.db'}" if use_storage else None
    if use_storage:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, best = [], {}
    for family in families:
        space = (spaces or {}).get(family) or load_base_space(family)
        study = tune_family(
            family, space, X, y, c1.task_type, profile.metric, n_trials=budget[family],
            folds=cfg["cv_folds"], seed=cfg["seed"], method=method, pruning=hpo_cfg["pruning"],
            study_name=f"{run_id}_r{round_no}_{family}", storage=storage)
        family_rows = study_to_records(study, family, method, round_no, run_id)
        rows += family_rows
        complete = [r for r in family_rows if r["state"] == "COMPLETE"]
        if complete:
            top = max(complete, key=lambda r: r["mean_val"])
            best[family] = {**top, "space": space,
                            "n_complete": len(complete),
                            "n_pruned": sum(r["state"] == "PRUNED" for r in family_rows)}

    entry = ReasoningEntry(
        round=round_no, agent="agent2", step="run_hpo",
        decision={"method": method, "budget": budget},
        evidence={f: {"best_val": b["mean_val"], "gap": b["gap"], "params": b["params"],
                      "n_complete": b["n_complete"], "n_pruned": b["n_pruned"]}
                  for f, b in best.items()},
        rationale=f"{method} search with static base spaces; pruning={hpo_cfg['pruning']}")
    return rows, best, entry


def save_history(rows: list[dict], path) -> pd.DataFrame:
    """Save the full search history (one row per trial) as Parquet."""
    df = pd.DataFrame(rows)
    df["params"] = df["params"].astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df
```

---

## 5. Unit tests

These use a small synthetic dataset (300 rows), so all 10 run in about 3 seconds.

**`tests/test_agent2.py`**

```python
import optuna
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from c2_automl.agents.agent2_hpo import run_defaults, split_budget
from c2_automl.config import load_default_config, load_yaml
from c2_automl.schemas import C1Output, TaskProfile, TaskType
from c2_automl.tools.hpo import study_to_records, tune_family
from c2_automl.tools.search_space import load_base_space, suggest_params

REGISTRY = load_yaml("model_registry.yaml")


@pytest.fixture(scope="module")
def small_data():
    X, y = make_classification(n_samples=300, n_features=6, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(6)]), pd.Series(y)


# ---------- search spaces ----------
def test_every_registry_family_has_a_search_space():
    for family in REGISTRY["families"]:
        assert load_base_space(family), family


def test_suggested_params_stay_inside_ranges():
    space = load_base_space("xgboost")
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    study.optimize(lambda t: len(suggest_params(t, space)), n_trials=20)
    for t in study.trials:
        for name, r in space.items():
            assert r["low"] <= t.params[name] <= r["high"]
        assert isinstance(t.params["max_depth"], int)


def test_load_base_space_returns_a_copy():
    space = load_base_space("xgboost")
    space["max_depth"]["high"] = 999
    assert load_base_space("xgboost")["max_depth"]["high"] != 999


# ---------- budget (Level 3) ----------
def test_split_budget_equal_and_exact():
    b = split_budget(10, ["a", "b", "c"])
    assert sum(b.values()) == 10 and max(b.values()) - min(b.values()) <= 1


def test_split_budget_with_weights():
    b = split_budget(30, ["a", "b"], weights={"a": 2, "b": 1})
    assert b == {"a": 20, "b": 10}


def test_split_budget_gives_every_family_at_least_one_trial():
    b = split_budget(3, ["a", "b", "c"], weights={"a": 100, "b": 1, "c": 1})
    assert sum(b.values()) == 3 and min(b.values()) >= 1


def test_split_budget_too_small_raises():
    with pytest.raises(ValueError):
        split_budget(2, ["a", "b", "c"])


# ---------- tuning (Level 2) ----------
def test_tune_family_records_every_trial(small_data, tmp_path):
    X, y = small_data
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    study = tune_family("logistic_regression", load_base_space("logistic_regression"), X, y,
                        TaskType.BINARY, "roc_auc", n_trials=4, folds=3, method="tpe",
                        study_name="test", storage=storage)
    rows = study_to_records(study, "logistic_regression", "tpe", 1, "run1")
    assert len(rows) == 4
    assert all(r["state"] == "COMPLETE" for r in rows)
    assert all(0.5 < r["mean_val"] <= 1.0 for r in rows)
    # the study was saved and can be loaded again from SQLite
    assert len(optuna.load_study(study_name="test", storage=storage).trials) == 4


def test_enqueued_params_are_tried_first(small_data):
    X, y = small_data
    study = tune_family("logistic_regression", load_base_space("logistic_regression"), X, y,
                        TaskType.BINARY, "roc_auc", n_trials=2, folds=3,
                        enqueue=[{"C": 0.5}])
    assert study.trials[0].params["C"] == 0.5


# ---------- defaults (Level 1) ----------
def test_defaults_use_same_row_format_as_hpo(small_data):
    X, y = small_data
    cfg = {**load_default_config(), "cv_folds": 3}
    c1 = C1Output(dataset_id="t", task_type=TaskType.BINARY, target="y", data_dir=".",
                  feature_types={}, fqs={})
    profile = TaskProfile(n_rows=300, n_features=6, n_categorical=0, categorical_ratio=0,
                          size_class="small", metric="roc_auc")
    rows = run_defaults(["logistic_regression"], X, y, c1, profile, cfg, run_id="run1")
    study = tune_family("logistic_regression", load_base_space("logistic_regression"), X, y,
                        TaskType.BINARY, "roc_auc", n_trials=1, folds=3)
    hpo_rows = study_to_records(study, "logistic_regression", "tpe", 1, "run1")
    assert set(rows[0]) == set(hpo_rows[0])
```

Run all tests (Agent 1 + Agent 2):

```bash
pytest -q
```

Expected:

```text
.......................                                                  [100%]
23 passed
```

| Test group | What it checks |
|---|---|
| search spaces | every registry family has a space; values stay inside ranges; `int` params are ints; loading returns a copy |
| budget | sums exactly to the total; weights work; every family gets ≥ 1 trial; too-small budget raises an error |
| tuning | every trial is recorded; study is saved to and re-loaded from SQLite; enqueued params are tried first |
| defaults | default rows have exactly the same columns as Optuna rows |

---

## 6. Run the baselines

### 6.1 The scripts

**`scripts/run_agent2_baseline.py`** – baselines try **all eligible families**, because a static AutoML has no Agent 1 to choose for it:

```python
"""Run Agent 2 alone to produce the baselines:

  B0 = defaults   (library default settings, no tuning)
  B1 = random     (random search, static ranges)
  B2 = tpe        (Optuna TPE, static ranges)  <- the main baseline your adaptive method must beat

Usage:
  python scripts/run_agent2_baseline.py diabetes --method defaults
  python scripts/run_agent2_baseline.py diabetes --method tpe --trials 60
  python scripts/run_agent2_baseline.py adult --method tpe --trials 40 --folds 3
"""
import argparse
import json

from c2_automl.agents.agent1_generation import eligible_families
from c2_automl.agents.agent2_hpo import new_run_id, run_defaults, run_hpo, save_history, split_budget
from c2_automl.config import OUTPUT_DIR, load_default_config, load_yaml
from c2_automl.io.dummy_c1 import load_c1_output, load_split
from c2_automl.tools.task_profile import profile_task


def main(dataset_id: str, method: str, trials: int, folds: int):
    cfg = load_default_config()
    cfg["cv_folds"] = folds
    registry = load_yaml("model_registry.yaml")

    c1 = load_c1_output(dataset_id)
    X, y = load_split(c1, "train")              # tuning NEVER sees the test split
    profile = profile_task(c1, X, y, cfg)
    # Baselines are "static AutoML": they try ALL eligible families, no Agent 1 intelligence
    families = eligible_families(c1.task_type.value, registry)
    run_id = new_run_id(dataset_id, method)
    print(f"run_id={run_id}  metric={profile.metric}  folds={folds}  families={families}")

    if method == "defaults":
        rows = run_defaults(families, X, y, c1, profile, cfg, run_id)
        best = {r["family"]: {**r, "n_complete": 1, "n_pruned": 0} for r in rows}
        budget = {f: 1 for f in families}
    else:
        budget = split_budget(trials, families)
        print(f"budget per family: {budget}")
        rows, best, _ = run_hpo(families, budget, X, y, c1, profile, cfg, run_id, method=method)

    print(f"\n{'family':<22}{'trials':>7}{'pruned':>7}{'best val':>10}{'train':>8}{'gap':>8}")
    for f, b in sorted(best.items(), key=lambda kv: -kv[1]["mean_val"]):
        print(f"{f:<22}{budget[f]:>7}{b['n_pruned']:>7}{b['mean_val']:>10.4f}"
              f"{b['mean_train']:>8.3f}{b['gap']:>8.3f}")
    winner = max(best.values(), key=lambda b: b["mean_val"])
    print(f"\nBEST: {winner['family']}  {profile.metric}={winner['mean_val']:.4f}  params={winner['params']}")

    out = OUTPUT_DIR / dataset_id
    save_history(rows, out / f"{run_id}_trials.parquet")
    summary = {"run_id": run_id, "method": method, "metric": profile.metric, "folds": folds,
               "total_trials": sum(budget.values()), "best_family": winner["family"],
               "best_val": winner["mean_val"], "best_params": winner["params"],
               "per_family": {f: {k: b[k] for k in ("mean_val", "gap", "params", "n_complete", "n_pruned")}
                              for f, b in best.items()}}
    (out / f"{run_id}_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved {out / (run_id + '_trials.parquet')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id")
    parser.add_argument("--method", choices=["defaults", "random", "tpe"], default=None,
                        help="default: hpo.method in configs/default.yaml")
    parser.add_argument("--trials", type=int, default=None, help="total trials (default: budget.total_trials)")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    cfg = load_default_config()
    method = args.method or cfg["hpo"]["method"]
    trials = args.trials or cfg["budget"]["total_trials"]
    main(args.dataset_id, method, trials, args.folds)
```

**`scripts/compare_baselines.py`**

```python
"""Compare all saved runs for one dataset (best score per method and family).

Usage:  python scripts/compare_baselines.py diabetes
"""
import json
import sys

import pandas as pd

from c2_automl.config import OUTPUT_DIR

dataset_id = sys.argv[1]
rows = []
for path in sorted((OUTPUT_DIR / dataset_id).glob("*_summary.json")):
    s = json.loads(path.read_text())
    rows.append({"run_id": s["run_id"], "method": s["method"], "trials": s["total_trials"],
                 "best_family": s["best_family"], "best_val": round(s["best_val"], 4)})

if not rows:
    sys.exit(f"No runs found in {OUTPUT_DIR / dataset_id}")
print(pd.DataFrame(rows).sort_values("best_val", ascending=False).to_string(index=False))
```

### 6.2 Pima Diabetes – same budget (60 trials) for B1 and B2

```bash
python scripts/run_agent2_baseline.py diabetes --method defaults
python scripts/run_agent2_baseline.py diabetes --method random --trials 60
python scripts/run_agent2_baseline.py diabetes --method tpe --trials 60
python scripts/compare_baselines.py diabetes
```

Actual output from testing this guide (about 4 s for defaults, 20–30 s for each tuned run):

```text
##### B0 defaults
family                 trials pruned  best val   train     gap
logistic_regression         1      0    0.8334   0.842   0.008
catboost                    1      0    0.8245   0.994   0.169
random_forest               1      0    0.8110   1.000   0.189
lightgbm                    1      0    0.7902   1.000   0.210
xgboost                     1      0    0.7732   1.000   0.227

##### B1 random (60 trials)
family                 trials pruned  best val   train     gap
logistic_regression        12      3    0.8348   0.841   0.007
random_forest              12      1    0.8315   0.931   0.100
catboost                   12      1    0.8312   0.966   0.135
lightgbm                   12      4    0.8299   0.904   0.074
xgboost                    12      2    0.8285   0.913   0.084

##### B2 tpe (60 trials)
family                 trials pruned  best val   train     gap
logistic_regression        12      2    0.8353   0.841   0.006
catboost                   12      1    0.8325   0.973   0.140
random_forest              12      0    0.8315   0.931   0.100
lightgbm                   12      3    0.8299   0.904   0.074
xgboost                    12      3    0.8273   0.928   0.101

##### compare_baselines.py diabetes  (run_id column removed here)
   method  trials         best_family  best_val
      tpe      60 logistic_regression    0.8353
   random      60 logistic_regression    0.8348
 defaults       5 logistic_regression    0.8334
```

### 6.3 Adult – bigger dataset (use `--folds 3`)

```bash
python scripts/run_agent2_baseline.py adult --method defaults --folds 3
python scripts/run_agent2_baseline.py adult --method tpe --trials 40 --folds 3
python scripts/run_agent2_baseline.py adult --method tpe --trials 150 --folds 3
python scripts/compare_baselines.py adult
```

Actual output (about 9 s, 46 s and 154 s on an Apple M4):

```text
##### B2 tpe (150 trials = 30 per family)
family                 trials pruned  best val   train     gap
xgboost                    30      2    0.9282   0.948   0.020
lightgbm                   30     12    0.9275   0.950   0.023
catboost                   30      3    0.9271   0.949   0.022
random_forest              30      3    0.9162   0.971   0.055
logistic_regression        30     14    0.8536   0.854   0.000

##### compare_baselines.py adult  (run_id column removed here)
   method  trials best_family  best_val
      tpe     150     xgboost    0.9282
 defaults       5    catboost    0.9272
      tpe      40    lightgbm    0.9266
```

### 6.4 Look at the search history

Every run saves `outputs/<dataset>/<run_id>_trials.parquet` (one row per trial) and `<run_id>_summary.json`.

```python
import pandas as pd, glob
df = pd.read_parquet(sorted(glob.glob("outputs/adult/*tpe*_trials.parquet"))[-1])
df.columns.tolist()
# ['run_id', 'method', 'round', 'family', 'trial', 'state', 'params', 'mean_val',
#  'mean_train', 'gap', 'cv_std', 'folds_done', 'fit_time_s', 'duration_s']
df.groupby(["family", "state"]).size()
```

Optional – browse the studies in a web page:

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///outputs/optuna.db      # open http://127.0.0.1:8080
```

> Notebook idea for the thesis (`notebooks/02_baselines.ipynb`): plot **best-so-far score vs. trial number** for B1 and B2 from the history files. Later you will add your adaptive method (A1) to the same plot.

---

## 7. What the results tell you (important for the thesis)

These are real observations from the runs above. Each one points to why your **diagnosis-conditioned** method (Step 4) is needed.

| # | Observation | Evidence | What it means for Step 4 |
|---|---|---|---|
| 1 | **Tuning reduces overfitting**, but only partly | Pima random forest gap: 0.189 (defaults) → 0.100 (tuned) | Fixed ranges still allow deep trees. Agent 1's `shrink_depth_range` should close the gap further |
| 2 | **TPE ≈ random** at small budgets | Pima: 0.8353 vs 0.8348 | With 12 trials per family, TPE has only 2 trials after its 10 random start-up trials. More trials per family are needed |
| 3 | **A small budget can lose to defaults** | Adult: TPE-40 = 0.9266 < defaults = 0.9272 | 8 trials per family < 10 start-up trials, so TPE never started learning. Spreading the budget over all families is wasteful |
| 4 | **Budget is wasted on hopeless families** | Adult: logistic regression used 30 of 150 trials (20%) and stayed at 0.8536, far below 0.928 | Agent 1 diagnoses it as `unsuitable_model_family` and drops it, giving those trials to strong families |
| 5 | Gains on Adult are small (+0.001) | 0.9282 vs 0.9272 | Report this honestly. Also measure **trials needed to reach a target score** and **cost**, not only the final score (§14 E2) |

> **Rule for a fair comparison:** B0, B1, B2 and your adaptive method must use the **same dataset split, folds, seed and total trial budget**. Only the *search strategy* may differ.

---

## 8. Commit, troubleshooting, next steps

### Commit

```bash
git add components/C2-Agentic-AutoML
git commit -m "C2: Agent 2 levels 1-3 (defaults, static Optuna, pruning, budget) + baselines B0/B1/B2"
```

### Troubleshooting

| Problem | Fix |
|---|---|
| `KeyError: 'hpo'` | Add the `hpo` section to `configs/default.yaml` (section 3.1) |
| `KeyError: '<family>'` in `load_base_space` | The family is in `model_registry.yaml` but not in `search_spaces.yaml` – add it |
| `TypeError: ... unexpected keyword argument 'on_fold'` | Apply the `train_eval.py` change (section 3.4) |
| `DuplicatedStudyError` | Two runs got the same `run_id` within one second – just run again |
| Everything is very slow | Use `--folds 3` and fewer `--trials` while developing; CatBoost is the slowest family |
| Many trials `PRUNED` | Normal: pruning saves time. Switch it off with `hpo.pruning: false` to compare |
| `outputs/optuna.db` gets big | It is git-ignored; delete it any time (history Parquet files are separate) |

### Next steps

| Next | What | Main guide |
|---|---|---|
| **Step 3 – LangGraph** | Connect Agent 1 → Agent 2 → evaluate → Agent 3 (simple) as one graph | §8 |
| **Step 4 – Agent 2 Level 4** ⭐ | `adapt_space()`: turn Agent 1's actions (`shrink_depth_range`, `increase_regularization`, `enable_class_weights`, …) into new ranges | §11.2, §13.3 |
| **Step 4 – Agent 1 Level 4** ⭐ | The multi-round loop, with budget weights from the diagnosis (`split_budget(..., weights=...)`) | §13.2 |
