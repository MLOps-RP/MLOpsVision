"""
Run every scenario through C4, score it against the ground truth, and compare
with a NAIVE baseline policy:  "retrain whenever any feature drift alarm fires"
(fixed thresholds: PSI > 0.2 for numeric, JSD > 0.1 for categorical).

Outputs (in results/):
    results_table.csv          one row per scenario
    unnecessary_retrains.png   naive vs C4 bar chart
    detection_summary.txt      short text summary

Run from the project root:
    python -m src.experiments.run_all
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common.config import DATA_SCENARIOS, OUTPUTS, RESULTS, load_config, load_json
from src.c4.run_c4 import run_scenario

RETRAIN_ACTIONS = {"ROUTE_C1", "ROUTE_C2"}
NO_RETRAIN_EXPECTED = {"NONE", "WARN_ONLY", "BLOCK_AND_FLAG"}
NAIVE_PSI, NAIVE_JSD = 0.2, 0.1


def naive_retrains(scenario: str, cooldown: int) -> tuple[int, int | None]:
    """Count how often the naive policy would retrain (same cooldown as C4)."""
    scores = pd.read_csv(OUTPUTS / "runs" / scenario / "feature_scores.csv")
    num, cat = load_config("monitoring")["numeric_features"], load_config("monitoring")["categorical_features"]
    retrains, first, next_allowed = 0, None, -1
    for _, r in scores.iterrows():
        alarm = any(r[f] > NAIVE_PSI for f in num) or any(r[f] > NAIVE_JSD for f in cat)
        if alarm and r["window"] > next_allowed:
            retrains += 1
            first = int(r["window"]) if first is None else first
            next_allowed = r["window"] + cooldown
    return retrains, first


def score(scenario: str, log: pd.DataFrame, truth: dict, cooldown: int) -> dict:
    start = truth["drift_start_window"]
    acted = log[log["executed"]]
    significant = acted[acted["action"] != "WATCH"]

    before = significant if start is None else significant[significant["window"] < start]
    after = significant if start is None else significant[significant["window"] >= start]
    first = after.iloc[0] if len(after) else None

    # first confirmed diagnosis after the drift started (ignoring SUSPECTED / NO_DRIFT)
    confirmed = log[(log["active"]) & (~log["diagnosis"].isin(["SUSPECTED", "NO_DRIFT"]))]
    if start is not None:
        confirmed = confirmed[confirmed["window"] >= start]
    first_diag = confirmed.iloc[0]["diagnosis"] if len(confirmed) else "NO_DRIFT"

    c4_retrains = int(acted["action"].isin(RETRAIN_ACTIONS).sum())
    naive_n, naive_first = naive_retrains(scenario, cooldown)
    expected_action = truth["expected_action"]

    return {
        "scenario": scenario,
        "expected_diagnosis": truth["expected_diagnosis"],
        "c4_diagnosis": first_diag,
        "diagnosis_correct": first_diag == truth["expected_diagnosis"],
        "expected_action": expected_action,
        "c4_first_action": first["action"] if first is not None else "NONE",
        "action_correct": (first["action"] if first is not None else "NONE") == expected_action,
        "detection_window": int(first["window"]) if first is not None else None,
        "detection_delay": (int(first["window"]) - start) if (first is not None and start is not None) else None,
        "false_alarms_before_drift": int(len(before)),
        "c4_retrains": c4_retrains,
        "c4_unnecessary_retrains": c4_retrains if expected_action in NO_RETRAIN_EXPECTED else 0,
        "naive_retrains": naive_n,
        "naive_unnecessary_retrains": naive_n if expected_action in NO_RETRAIN_EXPECTED else 0,
        "naive_missed": naive_n == 0 and expected_action not in NO_RETRAIN_EXPECTED,
        "rollback_done": bool((acted["action"] == "ROLLBACK").any()),
    }


def chart(df: pd.DataFrame) -> None:
    x = range(len(df))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([i - 0.2 for i in x], df["naive_unnecessary_retrains"], width=0.4, label="Naive: retrain on any alarm",
           color="#b0b0b0")
    ax.bar([i + 0.2 for i in x], df["c4_unnecessary_retrains"], width=0.4, label="C4", color="#2a6fdb")
    for i, (n, c) in enumerate(zip(df["naive_unnecessary_retrains"], df["c4_unnecessary_retrains"])):
        ax.text(i - 0.2, n + 0.05, str(n), ha="center", fontsize=9)
        ax.text(i + 0.2, c + 0.05, str(c), ha="center", fontsize=9, color="#2a6fdb")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_xticks(list(x))
    ax.set_xticklabels([s.split("_", 1)[0] for s in df["scenario"]])
    ax.set_ylabel("Unnecessary retrains")
    ax.set_title(f"Unnecessary retrains: naive {df['naive_unnecessary_retrains'].sum()} vs "
                 f"C4 {df['c4_unnecessary_retrains'].sum()}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "unnecessary_retrains.png", dpi=160)
    plt.close(fig)


def main() -> None:
    sc = load_config("scenarios")
    cooldown = load_config("policy")["cooldown_windows"]
    rows = []
    for scenario in sc["scenarios"]:
        log = run_scenario(scenario, reset=True, verbose=False)
        truth = load_json(DATA_SCENARIOS / scenario / "ground_truth.json")
        r = score(scenario, log, truth, cooldown)
        rows.append(r)
        print(f"{scenario:<24} diagnosis {r['c4_diagnosis']:<20} "
              f"{'OK ' if r['diagnosis_correct'] else 'X  '} action {r['c4_first_action']:<15} "
              f"{'OK ' if r['action_correct'] else 'X  '} delay {r['detection_delay']}")

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "results_table.csv", index=False)
    chart(df)

    summary = (f"Diagnosis correct: {df['diagnosis_correct'].sum()}/{len(df)}\n"
               f"First action correct: {df['action_correct'].sum()}/{len(df)}\n"
               f"False alarms before drift: {df['false_alarms_before_drift'].sum()}\n"
               f"Mean detection delay (windows): {df['detection_delay'].dropna().mean():.1f}\n"
               f"Unnecessary retrains - naive: {df['naive_unnecessary_retrains'].sum()}, "
               f"C4: {df['c4_unnecessary_retrains'].sum()}\n"
               f"Scenarios naive policy missed: {int(df['naive_missed'].sum())}\n")
    (RESULTS / "detection_summary.txt").write_text(summary)
    from src.common import registry
    registry.reset_production()          # leave the registry in its starting state
    print("\n" + summary)
    print(f"Saved results to {RESULTS}")


if __name__ == "__main__":
    main()
