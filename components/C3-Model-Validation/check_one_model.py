
from pathlib import Path

from mqvg.config import load_config
from mqvg.pipeline import run_validation
from mqvg.schemas import CandidateBundle, Dimension

config = load_config("configs/default.yaml")

def validate_model(name):
    folder = Path("fixtures") / name

    eval_file = (
        folder / "eval.csv"
        if (folder / "eval.csv").exists()
        else Path("data/eval.csv")
    )

    bundle = CandidateBundle(
        candidate_id=name,
        model_path=folder / "model.joblib",
        eval_data_path=eval_file,
        metadata_path=folder / "metadata.json",
        train_row_hashes_path=Path("data/train_row_hashes.txt"),
    )

    result = run_validation(bundle, config)

    print("\n========== MODEL QUALITY RESULT ==========")
    print("Model:", name)
    print("Decision:", result.decision.decision.value)
    print("Summary:", result.decision.summary)
    print("MQS:", result.scorecard.mqs)

    print("\n--- Individual Checks ---")
    for check in result.checks:
        print(f"{check.id}: {check.status.value}")
        print(f"  {check.message}")

    print("\n--- Triggered Rules ---")
    for rule in result.decision.triggered_rules:
        print(f"{rule.rule_id}: {rule.reason}")

if __name__ == "__main__":
    name = input(
        "Enter model (good/leaky/biased/miscalibrated/overfit/slow): "
    ).strip().lower()

    allowed = {
        "good", "leaky", "biased",
        "miscalibrated", "overfit", "slow"
    }

    if name in allowed:
        validate_model(name)
    else:
        print("Invalid model name.")
