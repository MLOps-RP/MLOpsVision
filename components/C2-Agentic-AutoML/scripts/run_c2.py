"""Run the whole C2 workflow (LangGraph) on one dataset and show what each node did.

Usage:
  python scripts/run_c2.py diabetes --trials 90
  python scripts/run_c2.py adult_imbalanced --trials 90 --folds 3
  python scripts/run_c2.py diabetes --trials 90 --no-adapt-ranges     # ablation
  python scripts/run_c2.py diabetes --draw                            # print the graph as Mermaid
"""
import argparse

from c2_automl.graph import build_graph


def show(node: str, update: dict) -> None:
    """Print a short summary of each node's update."""
    if node == "load_input":
        print(f"[load_input]      run_id={update['run_id']}  total_trials={update['total_trials']}")
    elif node == "profile_task":
        p = update["profile"]
        print(f"[profile_task]    {p['n_rows']} rows, {p['n_features']} features, "
              f"size={p['size_class']}, metric={p['metric']}, imbalance={p['imbalance_ratio']}")
    elif node == "agent1_select":
        print(f"[agent1_select]   families={update['families']}")
        print(f"                  reason: {update['reasoning_log'][0]['rationale']}")
    elif node == "agent2_hpo":
        rnd = update["reasoning_log"][0]["round"]
        budget = update["reasoning_log"][0]["decision"]["budget"]
        print(f"\n===== ROUND {rnd} =====  budget={budget}")
        for f, c in update["round_candidates"].items():
            print(f"[agent2_hpo]      {f:<20} best={c['mean_val']:.4f} gap={c['gap']:.3f} "
                  f"(complete={c['n_complete']}, pruned={c['n_pruned']})")
    elif node == "agent1_diagnose":
        for d in update["diagnoses"]:
            print(f"[agent1_diagnose] {d['model_family']:<20} {d['labels']}")
        actions = {f: a for f, a in update["plan"]["actions"].items() if a}
        print(f"                  actions: {actions or 'none'}")
        if update["referrals_to_c1"]:
            print(f"                  referrals to C1: {update['referrals_to_c1']}")
        print(f"                  decision: {update['reasoning_log'][-1]['rationale']}")
    elif node == "agent2_adapt":
        changes = update["reasoning_log"][0]["decision"]["changes"]
        print(f"[agent2_adapt]    {update['reasoning_log'][0]['rationale']}")
        for f, ch in changes.items():
            for c in ch:
                print(f"                  {f}: {c}")
    elif node == "agent3_select":
        s = update["selection"]
        hold = ", ".join(f"{k}={v:.4f}" for k, v in s["metrics_holdout"].items())
        print(f"\n[agent3_select]   selected {s['family']} (found in round {s['round']})")
        print(f"                  hold-out: {hold}")
    elif node == "package_output":
        o = update["output"]
        print(f"[package_output]  rounds={o['rounds']}, trials={o['total_trials']}, "
              f"stop={o['stop_reason']}, best by round={[round(v, 4) for v in o['score_by_round']]}")
        print(f"                  saved to {o['selected_model_path'].rsplit('/', 1)[0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id")
    parser.add_argument("--trials", type=int, default=None, help="total trial budget")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--no-adapt-ranges", action="store_true", help="ablation: keep static ranges")
    parser.add_argument("--no-adapt-families", action="store_true", help="ablation: never drop a family")
    parser.add_argument("--draw", action="store_true", help="print the graph as Mermaid")
    args = parser.parse_args()

    graph = build_graph()
    if args.draw:
        print(graph.get_graph().draw_mermaid())
        return

    overrides = {"adaptive": {}}
    if args.no_adapt_ranges:
        overrides["adaptive"]["adapt_ranges"] = False
    if args.no_adapt_families:
        overrides["adaptive"]["adapt_families"] = False
    if args.max_rounds:
        overrides["budget"] = {"max_rounds": args.max_rounds}

    inputs = {"dataset_id": args.dataset_id, "total_trials": args.trials, "folds": args.folds,
              "overrides": overrides}
    # stream_mode="updates" yields {node_name: update} after every node finishes
    for step in graph.stream(inputs, stream_mode="updates"):
        for node, update in step.items():
            show(node, update)


if __name__ == "__main__":
    main()
