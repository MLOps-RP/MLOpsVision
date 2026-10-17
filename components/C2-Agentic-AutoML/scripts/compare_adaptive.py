"""Compare the static baseline B2 with the adaptive method and its ablations - SAME trial budget.

  B2_static     Optuna TPE, all eligible families, fixed ranges, one round (Step 2)
  A_full        Agent 1 + Agent 2 adaptive loop (this step)
  A_no_ranges   ablation: loop + family selection, but ranges never change
  A_no_families ablation: loop + range adaptation, but no family is ever dropped

Usage:
  python scripts/compare_adaptive.py diabetes --trials 90
  python scripts/compare_adaptive.py adult_imbalanced --trials 90 --folds 3
  python scripts/compare_adaptive.py diabetes --trials 90 --seeds 3     # repeat with 3 seeds
"""
import argparse
import time
from datetime import datetime

import pandas as pd

from c2_automl.agents.agent1_generation import eligible_families
from c2_automl.agents.agent2_hpo import new_run_id, run_hpo, split_budget
from c2_automl.config import OUTPUT_DIR, load_default_config, load_yaml
from c2_automl.graph import build_graph
from c2_automl.io.dummy_c1 import load_c1_output, load_split
from c2_automl.tools.holdout import fit_final_model, holdout_metrics
from c2_automl.tools.task_profile import profile_task

ADAPTIVE = {
    "A_full": {},
    "A_no_ranges": {"adaptive": {"adapt_ranges": False}},
    "A_no_families": {"adaptive": {"adapt_families": False}},
}


def run_b2(dataset_id: str, trials: int, folds: int, seed: int) -> dict:
    cfg = load_default_config()
    cfg.update({"cv_folds": folds, "seed": seed})
    c1 = load_c1_output(dataset_id)
    X, y = load_split(c1, "train")
    profile = profile_task(c1, X, y, cfg)
    families = eligible_families(c1.task_type.value, load_yaml("model_registry.yaml"))
    rows, best, _ = run_hpo(families, split_budget(trials, families), X, y, c1, profile, cfg,
                            new_run_id(dataset_id, "b2"), method="tpe", use_storage=False)
    winner = max(best.values(), key=lambda b: b["mean_val"])
    model = fit_final_model(winner["family"], winner["params"], X, y, c1.task_type, seed)
    X_test, y_test = load_split(c1, "test")
    return {"family": winner["family"], "rounds": 1, "trials": len(rows),
            "best_cv": winner["mean_val"], "stop": "-",
            **holdout_metrics(model, X_test, y_test, c1.task_type)}


def run_adaptive(dataset_id: str, trials: int, folds: int, seed: int, overrides: dict) -> dict:
    inputs = {"dataset_id": dataset_id, "total_trials": trials, "folds": folds,
              "overrides": {**overrides, "seed": seed}}
    out = build_graph().invoke(inputs)["output"]
    return {"family": out["model_family"], "rounds": out["rounds"], "trials": out["total_trials"],
            "best_cv": out["score_by_round"][-1], "stop": out["stop_reason"], **out["metrics_holdout"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id")
    parser.add_argument("--trials", type=int, default=90)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=1, help="repeat with seeds 42, 43, ...")
    args = parser.parse_args()

    results = []
    for seed in range(42, 42 + args.seeds):
        jobs = [("B2_static", lambda: run_b2(args.dataset_id, args.trials, args.folds, seed))]
        jobs += [(name, lambda o=o: run_adaptive(args.dataset_id, args.trials, args.folds, seed, o))
                 for name, o in ADAPTIVE.items()]
        for name, job in jobs:
            print(f"seed {seed}: running {name} ...", flush=True)
            start = time.perf_counter()
            res = job()
            results.append({"method": name, "seed": seed, **res,
                            "time_s": round(time.perf_counter() - start, 1)})

    df = pd.DataFrame(results)
    out = OUTPUT_DIR / args.dataset_id / f"comparison_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    print("\n" + df.round(4).to_string(index=False))
    if args.seeds > 1:
        print("\nMean over seeds:")
        print(df.drop(columns=["seed", "family", "stop"]).groupby("method").mean().round(4).to_string())
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
