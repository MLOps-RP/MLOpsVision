"""LangGraph nodes. Each node: read what it needs from the state, call an agent or tool,
return ONLY the fields it changes. Nodes contain wiring, not ML logic."""
import json
from functools import lru_cache

import pandas as pd

from . import config
from .agents.agent1_generation import diagnose_family, plan_next_round, select_initial_families
from .agents.agent2_hpo import new_run_id, run_hpo, save_history, split_budget
from .agents.agent3_selection import select_best
from .io.dummy_c1 import load_c1_output, load_split
from .schemas import C1Output, C2Output, ReasoningEntry, TaskProfile
from .tools.holdout import fit_final_model, holdout_metrics, save_model
from .tools.signals import compute_signals
from .tools.task_profile import profile_task as compute_profile
from .tools.train_eval import baseline_score, cross_validate_family


# ---------- helpers ----------
def _c1(state) -> C1Output:
    return C1Output(**state["c1"])


@lru_cache(maxsize=8)
def _cached_split(c1_json: str, split: str) -> tuple[pd.DataFrame, pd.Series]:
    return load_split(C1Output.model_validate_json(c1_json), split)


def _data(state, split: str = "train"):
    return _cached_split(_c1(state).model_dump_json(), split)


def _run_dir(state):
    return config.OUTPUT_DIR / state["dataset_id"] / state["run_id"]


# ---------- nodes ----------
def load_input(state):
    cfg = config.load_default_config()
    cfg["cv_folds"] = state.get("folds") or cfg["cv_folds"]
    c1 = load_c1_output(state["dataset_id"])
    return {"cfg": cfg, "c1": c1.model_dump(mode="json"), "round": 1,
            "run_id": new_run_id(state["dataset_id"], "c2"),
            "total_trials": state.get("total_trials") or cfg["budget"]["total_trials"]}


def profile_task(state):
    X, y = _data(state)
    profile = compute_profile(_c1(state), X, y, state["cfg"])
    return {"profile": profile.model_dump()}


def agent1_select(state):
    registry = config.load_yaml("model_registry.yaml")
    families, entry = select_initial_families(TaskProfile(**state["profile"]), _c1(state),
                                              registry, state["cfg"])
    budget = split_budget(state["total_trials"], families)
    entry.decision["budget"] = budget
    return {"families": families, "budget": budget, "reasoning_log": [entry.model_dump()]}


def agent2_hpo(state):
    X, y = _data(state)
    cfg = state["cfg"]
    rows, best, entry = run_hpo(state["families"], state["budget"], X, y, _c1(state),
                                TaskProfile(**state["profile"]), cfg, state["run_id"],
                                method=cfg["hpo"]["method"], round_no=state["round"])
    return {"trial_rows": rows, "candidates": best, "reasoning_log": [entry.model_dump()]}


def evaluate(state):
    """Re-run CV once with each family's best params to collect diagnosis signals."""
    X, y = _data(state)
    c1, cfg, metric = _c1(state), state["cfg"], state["profile"]["metric"]
    base = baseline_score(X, y, c1.task_type, metric, cfg["cv_folds"], cfg["seed"])
    results = {f: cross_validate_family(f, X, y, c1.task_type, metric, cfg["cv_folds"],
                                        cfg["seed"], params=cand["params"])
               for f, cand in state["candidates"].items()}
    best_val = max(r["mean_val"] for r in results.values())
    signals = {f: compute_signals(r, y, c1.task_type, base, best_val, c1.fqs)
               for f, r in results.items()}
    return {"signals": signals}


def agent1_diagnose(state):
    rules = config.load_yaml("diagnosis_rules.yaml")
    diagnoses = [diagnose_family(f, s, rules) for f, s in state["signals"].items()]
    plan, plan_entry = plan_next_round(diagnoses, rules, state["round"])

    log = [ReasoningEntry(round=state["round"], agent="agent1", step="diagnose",
                          decision={"family": d.model_family, "labels": [l.value for l in d.labels]},
                          evidence=d.evidence, rationale="; ".join(d.reasons)).model_dump()
           for d in diagnoses]
    log.append(plan_entry.model_dump())

    # Step 3 = one round only. Step 4 replaces this with a real stop/continue decision.
    stop_reason = "weak_features" if plan.stop else "single_round (adaptive loop comes in Step 4)"
    return {"diagnoses": [d.model_dump(mode="json") for d in diagnoses],
            "plan": plan.model_dump(), "referrals_to_c1": plan.referrals_to_c1,
            "stop_reason": stop_reason, "reasoning_log": log}


def agent3_select(state):
    c1, cfg = _c1(state), state["cfg"]
    winner, entry = select_best(state["candidates"], state["round"])
    best = state["candidates"][winner]

    X, y = _data(state, "train")
    model = fit_final_model(winner, best["params"], X, y, c1.task_type, cfg["seed"])
    X_test, y_test = _data(state, "test")               # the ONLY place the test split is used
    metrics_holdout = holdout_metrics(model, X_test, y_test, c1.task_type)
    model_path = _run_dir(state) / "model.joblib"
    size_kb = save_model(model, model_path)

    selection = {
        "family": winner, "params": best["params"], "model_path": str(model_path),
        "model_size_kb": size_kb, "metrics_holdout": metrics_holdout,
        "metrics_cv": {state["profile"]["metric"]: best["mean_val"], "mean_train": best["mean_train"],
                       "gap": best["gap"], "cv_std": best["cv_std"]},
    }
    entry.evidence["holdout"] = metrics_holdout
    return {"selection": selection, "reasoning_log": [entry.model_dump()]}


def package_output(state):
    run_dir = _run_dir(state)
    history_path = run_dir / "search_history.parquet"
    log_path = run_dir / "reasoning_log.json"
    save_history(state["trial_rows"], history_path)
    log_path.write_text(json.dumps(state["reasoning_log"], indent=2, default=str))

    sel, c1 = state["selection"], _c1(state)
    output = C2Output(
        dataset_id=state["dataset_id"], run_id=state["run_id"], task_type=c1.task_type,
        metric=state["profile"]["metric"], selected_model_path=sel["model_path"],
        model_family=sel["family"], hyperparameters=sel["params"],
        metrics_cv=sel["metrics_cv"], metrics_holdout=sel["metrics_holdout"],
        model_size_kb=sel["model_size_kb"],
        candidates=[{"family": f, "mean_val": c["mean_val"], "gap": c["gap"], "params": c["params"],
                     "n_complete": c["n_complete"], "n_pruned": c["n_pruned"]}
                    for f, c in state["candidates"].items()],
        diagnoses=state["diagnoses"], search_history_path=str(history_path),
        reasoning_log_path=str(log_path), total_trials=len(state["trial_rows"]),
        rounds=state["round"], referrals_to_c1=state.get("referrals_to_c1", []),
    )
    (run_dir / "c2_output.json").write_text(output.model_dump_json(indent=2))
    return {"output": output.model_dump(mode="json")}
