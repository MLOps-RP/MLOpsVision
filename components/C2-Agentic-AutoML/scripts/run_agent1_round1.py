"""Run Agent 1 on one dataset: profile -> select families -> train (defaults) -> diagnose -> plan.

Usage:  python scripts/run_agent1_round1.py diabetes [--folds 3]
"""
import argparse
import json

from c2_automl.agents.agent1_generation import (diagnose_family, plan_next_round,
                                                select_initial_families)
from c2_automl.config import OUTPUT_DIR, load_default_config, load_yaml
from c2_automl.io.dummy_c1 import load_c1_output, load_split
from c2_automl.tools.signals import compute_signals
from c2_automl.tools.task_profile import profile_task
from c2_automl.tools.train_eval import baseline_score, cross_validate_family


def main(dataset_id: str, folds: int):
    cfg = load_default_config()
    registry = load_yaml("model_registry.yaml")
    rules = load_yaml("diagnosis_rules.yaml")
    seed = cfg["seed"]
    log = []

    c1 = load_c1_output(dataset_id)
    X, y = load_split(c1, "train")           # never the test split here
    profile = profile_task(c1, X, y, cfg)
    print("Task profile:", profile.model_dump())

    families, entry = select_initial_families(profile, c1, registry, cfg)
    log.append(entry)
    print("\nAgent 1 selected:", families, "\nReason:", entry.rationale)

    results = {}
    for family in families:
        print(f"  training {family} ...")
        results[family] = cross_validate_family(family, X, y, c1.task_type, profile.metric, folds, seed)
    base = baseline_score(X, y, c1.task_type, profile.metric, folds, seed)
    best_val = max(r["mean_val"] for r in results.values())

    diagnoses = []
    print(f"\n{'family':<22}{'train':>8}{'val':>8}{'gap':>8}  diagnosis")
    for family, r in results.items():
        signals = compute_signals(r, y, c1.task_type, base, best_val, c1.fqs)
        d = diagnose_family(family, signals, rules)
        diagnoses.append(d)
        print(f"{family:<22}{signals['mean_train']:>8.3f}{signals['mean_val']:>8.3f}"
              f"{signals['gap']:>8.3f}  {[l.value for l in d.labels]}")
        for reason in d.reasons:
            print(f"{'':<22}- {reason}")

    plan, entry = plan_next_round(diagnoses, rules, round_no=1)
    log.append(entry)
    print("\nPlan for round 2:", json.dumps(plan.model_dump(), indent=2))

    out = OUTPUT_DIR / dataset_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "agent1_reasoning_log.json").write_text(
        json.dumps([e.model_dump() for e in log], indent=2, default=str))
    (out / "agent1_diagnoses.json").write_text(
        json.dumps([d.model_dump() for d in diagnoses], indent=2, default=str))
    print(f"\nSaved logs to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    main(args.dataset_id, args.folds)
