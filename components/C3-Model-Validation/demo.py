
from pathlib import Path

from mqvg.config import load_config
from mqvg.pipeline import run_validation
from mqvg.schemas import CandidateBundle


def main():
    # Step 1: Load validation rules
    config = load_config("configs/default.yaml")

    # Step 2: Create the candidate model bundle
    bundle = CandidateBundle(
        candidate_id="demo-good",
        model_path=Path("fixtures/good/model.joblib"),
        eval_data_path=Path("data/eval.csv"),
        metadata_path=Path("fixtures/good/metadata.json"),
        train_row_hashes_path=Path("data/train_row_hashes.txt"),
    )

    # Step 3: Run the complete validation pipeline
    result = run_validation(bundle, config)

    # Step 4: Display the final verdict and MQS
    print("\n===== C3 MODEL VALIDATION RESULT =====")

    print(
        f"Decision: {result.decision.decision.value}"
    )

    if result.scorecard.mqs is not None:
        print(f"MQS Score: {result.scorecard.mqs:.1f}")
    else:
        print("MQS Score: Not available")

    print(f"\nSummary: {result.decision.summary}")


if __name__ == "__main__":
    main()
