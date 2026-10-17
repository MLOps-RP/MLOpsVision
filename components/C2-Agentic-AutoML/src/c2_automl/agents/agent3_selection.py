"""Agent 3 - Multi-Objective Selection.

Level 1 (this step): choose the family with the best CV score.
Level 2-3 (Step 6) will add latency, model size, constraints and the Pareto front.
"""
from ..schemas import ReasoningEntry


def select_best(candidates: dict[str, dict], round_no: int) -> tuple[str, ReasoningEntry]:
    """candidates: family -> best trial row (needs 'mean_val'). Higher mean_val is better."""
    if not candidates:
        raise ValueError("Agent 3 received no candidates (every trial failed or was pruned)")
    ranking = sorted(candidates.items(), key=lambda kv: kv[1]["mean_val"], reverse=True)
    winner, best = ranking[0]
    runner_up = ranking[1] if len(ranking) > 1 else None

    rationale = f"{winner} has the best CV score ({best['mean_val']:.4f})"
    if runner_up:
        margin = best["mean_val"] - runner_up[1]["mean_val"]
        rationale += f", {margin:.4f} ahead of {runner_up[0]}"

    entry = ReasoningEntry(
        round=round_no, agent="agent3", step="select_best",
        decision={"selected_family": winner, "params": best["params"]},
        evidence={f: round(c["mean_val"], 4) for f, c in ranking},
        rationale=rationale + " (Level 1: score only; latency/size come in Step 6)")
    return winner, entry
