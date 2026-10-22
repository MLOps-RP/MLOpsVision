"""The shared state that flows through the LangGraph workflow.

Rules:
- Only JSON-friendly values (str, int, float, list, dict). DataFrames and models are NOT stored
  here; nodes load data from disk when needed. This keeps the state easy to log and checkpoint.
- Fields marked `operator.add` are APPENDED to by each node instead of replaced.
"""
import operator
from typing import Annotated, Optional, TypedDict


class C2State(TypedDict, total=False):
    # input
    dataset_id: str
    total_trials: int
    folds: int
    overrides: dict                           # optional config overrides (used for ablations)
    run_id: str                               # optional: the API sets it so it can track the run

    # set by load_input / profile_task
    cfg: dict
    c1: dict                                  # C1Output as dict
    profile: dict                             # TaskProfile as dict
    round: int

    # Agent 1 -> Agent 2
    families: list[str]
    budget: dict[str, int]
    spaces: dict[str, dict]                   # current search space per family (Agent 2 Level 4)

    # Agent 2 -> evaluate / Agent 3
    trial_rows: Annotated[list[dict], operator.add]   # full search history (all rounds)
    round_candidates: dict[str, dict]                  # best trial per family, THIS round
    candidates: dict[str, dict]                        # best trial per family, ALL rounds
    best_history: Annotated[list[float], operator.add] # best score so far, after each round

    # evaluate -> Agent 1 diagnose
    signals: dict[str, dict]

    # Agent 1 diagnose -> router / Agent 3
    diagnoses: list[dict]
    plan: dict
    referrals_to_c1: Annotated[list[str], operator.add]
    stop_reason: Optional[str]

    # Agent 3 -> package_output
    selection: dict

    # every node adds its decisions here (auditable reasoning)
    reasoning_log: Annotated[list[dict], operator.add]

    # final result
    output: dict
