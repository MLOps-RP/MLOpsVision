"""Agent 2 - Hyperparameter Optimization.

Level 1: evaluate default parameters            -> Baseline B0
Level 2: static Optuna search (TPE or random)    -> Baselines B2 (TPE) and B1 (random)
Level 3: pruning, SQLite storage, trial budget from Agent 1
Level 4: adapt the search spaces using Agent 1's actions + warm start from earlier rounds
"""
from datetime import datetime
from typing import Optional

import pandas as pd

from ..config import OUTPUT_DIR
from ..schemas import C1Output, ReasoningEntry, TaskProfile
from ..tools.hpo import study_to_records, tune_family
from ..tools.search_space import adapt_space, clip_to_space, in_space, load_base_space
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
            use_storage: bool = True, warm_start: Optional[dict[str, dict]] = None
            ) -> tuple[list[dict], dict[str, dict], ReasoningEntry]:
    """Tune every family with its share of the budget.

    Returns (trial rows for the search history, best result per family, reasoning entry).
    `spaces` lets Level 4 pass adapted ranges; by default the static base spaces are used.
    `warm_start` (Level 4): family -> {"prior_rows": [...], "best_params": {...}} from earlier rounds.
    """
    hpo_cfg = cfg["hpo"]
    storage = f"sqlite:///{OUTPUT_DIR / 'optuna.db'}" if use_storage else None
    if use_storage:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, best = [], {}
    for family in families:
        space = (spaces or {}).get(family) or load_base_space(family)
        ws = (warm_start or {}).get(family, {})
        # If the old best still fits the new space it is already in the warm-start history.
        # Otherwise (range moved, parameter added) try a clipped copy of it first.
        best_params = ws.get("best_params")
        enqueue = ([clip_to_space(best_params, space)]
                   if best_params and not in_space(best_params, space) else None)
        study = tune_family(
            family, space, X, y, c1.task_type, profile.metric, n_trials=budget[family],
            folds=cfg["cv_folds"], seed=cfg["seed"], method=method,
            pruning=hpo_cfg["pruning"], study_name=f"{run_id}_r{round_no}_{family}",
            storage=storage, enqueue=enqueue, prior_trials=ws.get("prior_rows"))
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
        rationale=(f"{method} search, round {round_no}, "
                   f"{'adapted' if spaces else 'static base'} spaces; pruning={hpo_cfg['pruning']}"))
    return rows, best, entry


# ---------- Level 4 ----------
def adapt_spaces(families: list[str], plan_actions: dict[str, list[str]], current: dict[str, dict],
                 round_best: dict[str, dict], rules: dict, context: dict, round_no: int,
                 enabled: bool = True) -> tuple[dict[str, dict], ReasoningEntry]:
    """Turn Agent 1's actions into new ranges for the next round (one call per round).

    enabled=False keeps the ranges unchanged (ablation: "no range adaptation").
    """
    spaces, changes = {}, {}
    for family in families:
        space = current.get(family) or load_base_space(family)
        if enabled:
            best_params = round_best.get(family, {}).get("params")
            space, family_changes = adapt_space(space, plan_actions.get(family, []), family, rules,
                                                best_params=best_params, context=context)
            changes[family] = family_changes
        spaces[family] = space
    n = sum(len(c) for c in changes.values())
    entry = ReasoningEntry(
        round=round_no, agent="agent2", step="adapt_spaces",
        decision={"changes": changes, "enabled": enabled},
        evidence={f: plan_actions.get(f, []) for f in families},
        rationale=(f"{n} range changes for round {round_no}" if enabled
                   else "range adaptation disabled (ablation)"))
    return spaces, entry


def save_history(rows: list[dict], path) -> pd.DataFrame:
    """Save the full search history (one row per trial) as Parquet."""
    df = pd.DataFrame(rows)
    df["params"] = df["params"].astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df
