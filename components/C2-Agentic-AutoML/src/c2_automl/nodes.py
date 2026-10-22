"""LangGraph nodes. Each node: read what it needs from the state, call an agent or tool,
return ONLY the fields it changes. Nodes contain wiring, not ML logic."""
import copy
import json
from functools import lru_cache

import pandas as pd

from . import config
from .agents.agent1_generation import (decide_next_round, diagnose_family, plan_next_round,
                                       round_budget, select_initial_families)
from .agents.agent2_hpo import adapt_spaces, new_run_id, run_hpo, save_history, split_budget
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


def _merge(base: dict, overrides: dict) -> dict:
    """Nested dict merge: overrides win. Used for ablations, e.g. {"adaptive": {"adapt_ranges": False}}."""
    out = copy.deepcopy(base)
    for k, v in (overrides or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _best_score(candidates: dict[str, dict]) -> float:
    return max(c["mean_val"] for c in candidates.values())


# ---------- nodes ----------
def load_input(state):
    cfg = _merge(config.load_default_config(), state.get("overrides"))
    cfg["cv_folds"] = state.get("folds") or cfg["cv_folds"]
    c1 = load_c1_output(state["dataset_id"])
    return {"cfg": cfg, "c1": c1.model_dump(mode="json"), "round": 1, "spaces": {},
            "candidates": {}, "run_id": state.get("run_id") or new_run_id(state["dataset_id"], "c2"),
            "total_trials": state.get("total_trials") or cfg["budget"]["total_trials"]}


def profile_task(state):
    X, y = _data(state)
    profile = compute_profile(_c1(state), X, y, state["cfg"])
    return {"profile": profile.model_dump()}


def agent1_select(state):
    cfg = state["cfg"]
    registry = config.load_yaml("model_registry.yaml")
    families, entry = select_initial_families(TaskProfile(**state["profile"]), _c1(state),
                                              registry, cfg)
    n_trials = round_budget(state["total_trials"], 0, 1, cfg["budget"]["max_rounds"], len(families))
    budget = split_budget(n_trials, families)
    entry.decision["budget"] = budget
    return {"families": families, "budget": budget, "reasoning_log": [entry.model_dump()]}


def agent2_hpo(state):
    X, y = _data(state)
    cfg = state["cfg"]
    # Level 4 warm start: earlier trials of the same family + the best params found so far
    warm_start = {f: {"prior_rows": [r for r in state.get("trial_rows", []) if r["family"] == f],
                      "best_params": state["candidates"].get(f, {}).get("params")}
                  for f in state["families"]}
    rows, best, entry = run_hpo(state["families"], state["budget"], X, y, _c1(state),
                                TaskProfile(**state["profile"]), cfg, state["run_id"],
                                method=cfg["hpo"]["method"], round_no=state["round"],
                                spaces=state.get("spaces") or None, warm_start=warm_start)
    merged = dict(state["candidates"])
    for f, b in best.items():
        if f not in merged or b["mean_val"] > merged[f]["mean_val"]:
            merged[f] = b
    return {"trial_rows": rows, "round_candidates": best, "candidates": merged,
            "reasoning_log": [entry.model_dump()]}


def evaluate(state):
    """Re-run CV once with each family's best params (this round) to collect diagnosis signals."""
    X, y = _data(state)
    c1, cfg, metric = _c1(state), state["cfg"], state["profile"]["metric"]
    base = baseline_score(X, y, c1.task_type, metric, cfg["cv_folds"], cfg["seed"])
    results = {f: cross_validate_family(f, X, y, c1.task_type, metric, cfg["cv_folds"],
                                        cfg["seed"], params=cand["params"])
               for f, cand in state["round_candidates"].items()}
    best_val = max(r["mean_val"] for r in results.values())
    signals = {f: compute_signals(r, y, c1.task_type, base, best_val, c1.fqs)
               for f, r in results.items()}
    return {"signals": signals}


def agent1_diagnose(state):
    cfg, round_no = state["cfg"], state["round"]
    rules = config.load_yaml("diagnosis_rules.yaml")
    diagnoses = [diagnose_family(f, s, rules) for f, s in state["signals"].items()]
    plan, plan_entry = plan_next_round(diagnoses, rules, round_no)
    if not cfg["adaptive"]["adapt_families"]:              # ablation: never drop a family
        plan.families_next_round = list(state["families"])

    log = [ReasoningEntry(round=round_no, agent="agent1", step="diagnose",
                          decision={"family": d.model_family, "labels": [l.value for l in d.labels]},
                          evidence=d.evidence, rationale="; ".join(d.reasons)).model_dump()
           for d in diagnoses]
    log.append(plan_entry.model_dump())

    # Level 4: continue or stop?
    best_history = state.get("best_history", []) + [_best_score(state["candidates"])]
    trials_used = len(state["trial_rows"])
    stop_reason, decision_entry = decide_next_round(
        plan, round_no, best_history, trials_used, state["total_trials"],
        cfg["budget"]["max_rounds"], cfg["adaptive"]["min_improvement"])
    log.append(decision_entry.model_dump())

    update = {"diagnoses": [d.model_dump(mode="json") for d in diagnoses],
              "plan": plan.model_dump(), "referrals_to_c1": plan.referrals_to_c1,
              "best_history": [best_history[-1]], "stop_reason": stop_reason,
              "reasoning_log": log}
    if stop_reason is None:
        families = plan.families_next_round
        n_trials = round_budget(state["total_trials"], trials_used, round_no + 1,
                                cfg["budget"]["max_rounds"], len(families))
        update.update({"families": families, "budget": split_budget(n_trials, families)})
    return update


def route_after_diagnosis(state) -> str:
    """The loop: back to Agent 2 with adapted ranges, or on to Agent 3."""
    return "agent3_select" if state.get("stop_reason") else "agent2_adapt"


def agent2_adapt(state):
    """Agent 2 Level 4: turn Agent 1's actions into the search ranges for the next round."""
    cfg, profile = state["cfg"], state["profile"]
    rules = config.load_yaml("space_adaptation.yaml")
    context = {"task_type": _c1(state).task_type.value, "imbalance_ratio": profile.get("imbalance_ratio")}
    spaces, entry = adapt_spaces(state["families"], state["plan"]["actions"], state.get("spaces", {}),
                                 state["round_candidates"], rules, context, state["round"] + 1,
                                 enabled=cfg["adaptive"]["adapt_ranges"])
    return {"spaces": spaces, "round": state["round"] + 1, "reasoning_log": [entry.model_dump()]}


def agent3_select(state):
    c1, cfg = _c1(state), state["cfg"]
    winner, entry = select_best(state["candidates"], state["round"])   # best over ALL rounds
    best = state["candidates"][winner]

    X, y = _data(state, "train")
    model = fit_final_model(winner, best["params"], X, y, c1.task_type, cfg["seed"])
    X_test, y_test = _data(state, "test")               # the ONLY place the test split is used
    metrics_holdout = holdout_metrics(model, X_test, y_test, c1.task_type)
    model_path = _run_dir(state) / "model.joblib"
    size_kb = save_model(model, model_path)

    selection = {
        "family": winner, "params": best["params"], "round": best["round"],
        "model_path": str(model_path), "model_size_kb": size_kb, "metrics_holdout": metrics_holdout,
        "metrics_cv": {state["profile"]["metric"]: best["mean_val"], "mean_train": best["mean_train"],
                       "gap": best["gap"], "cv_std": best["cv_std"]},
    }
    entry.evidence["holdout"] = metrics_holdout
    entry.evidence["found_in_round"] = best["round"]
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
                     "round": c["round"], "n_complete": c["n_complete"], "n_pruned": c["n_pruned"]}
                    for f, c in state["candidates"].items()],
        diagnoses=state["diagnoses"], search_history_path=str(history_path),
        reasoning_log_path=str(log_path), total_trials=len(state["trial_rows"]),
        rounds=state["round"], score_by_round=state.get("best_history", []),
        stop_reason=state.get("stop_reason"), referrals_to_c1=sorted(set(state.get("referrals_to_c1", []))),
    )
    (run_dir / "c2_output.json").write_text(output.model_dump_json(indent=2))
    return {"output": output.model_dump(mode="json")}
