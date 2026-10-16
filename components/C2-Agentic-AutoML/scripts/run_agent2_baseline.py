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
