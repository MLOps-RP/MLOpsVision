"""Validate each sample model separately and show its score and result.

Run from the C3-Model-Validation folder:

    python scripts/validate_models.py                 # all six trained models
    python scripts/validate_models.py leaky biased    # only some
    python scripts/validate_models.py --config configs/strict.yaml
    python scripts/validate_models.py hidden_group hidden_group_null         --config configs/hidden_groups.yaml          # Hidden Failure Group Detection

For every model it prints the decision, the three dimension scores, the MQS,
every check with its evidence, and each rule that fired. The full result is
saved to reports/<model>/result.json, with a readable reports/<model>/report.md
(the reports/ folder is gitignored).

This is a demonstration helper. It is not part of the gate (src/mqvg).
"""
import argparse
from pathlib import Path

from mqvg.config import load_config
from mqvg.pipeline import run_validation
from mqvg.reporting import write_report
from mqvg.schemas import CandidateBundle, Dimension

PROJECT = Path(__file__).resolve().parents[1]
MODELS = ["good", "leaky", "biased", "miscalibrated", "overfit", "slow"]
LABELS = {Dimension.TRAINING_VALIDITY: "TVS  training validity",
          Dimension.MODEL_BEHAVIOUR: "MBS  model behaviour",
          Dimension.OPERATIONAL_READINESS: "ORS  operational readiness"}


def bundle_for(name: str) -> CandidateBundle:
    folder = PROJECT / "fixtures" / name
    own_eval = folder / "eval.csv"
    own_hashes = folder / "train_row_hashes.txt"
    return CandidateBundle(
        candidate_id=name,
        model_path=folder / "model.joblib",
        eval_data_path=own_eval if own_eval.exists() else PROJECT / "data" / "eval.csv",
        metadata_path=folder / "metadata.json",
        train_row_hashes_path=(own_hashes if own_hashes.exists()
                               else PROJECT / "data" / "train_row_hashes.txt"))


def number(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def show(name: str, result) -> None:
    card, decision = result.scorecard, result.decision
    print("\n" + "=" * 100)
    print(f"MODEL: {name}")
    print("=" * 100)
    print(f"DECISION: {decision.decision.value}   "
          f"(recommendation only; nothing is deployed or rolled back)")
    print(decision.summary)

    print("\nSCORES (0-100)")
    for dimension, label in LABELS.items():
        d = card.dimensions[dimension]
        score = "n/a" if d.score is None else f"{d.score:6.1f}"
        print(f"  {label:28s} {score}   weight {card.weights[dimension]:.2f}   "
              f"evidence {d.coverage:.0%}  ({d.available_checks}/{d.total_checks} checks)")
    mqs = "n/a" if card.mqs is None else f"{card.mqs:.1f}"
    print(f"  {'MQS  model quality score':28s} {mqs:>6s}   "
          f"worst case {card.mqs_lower_bound:.1f}   overall evidence "
          f"{card.overall_coverage:.0%}   provisional={card.provisional}")
    if not card.mqs_sufficient_evidence:
        print(f"  !! MQS NOT RELIABLE: {card.insufficient_evidence_reason}")

    print("\nCHECKS")
    print(f"  {'check':42s} {'status':12s} {'value':>9s} {'score':>6s}  evidence")
    for c in result.checks:
        score = "" if c.score is None else f"{c.score:.0f}"
        print(f"  {c.id:42s} {c.status.value:12s} {number(c.raw_value):>9s} "
              f"{score:>6s}  {c.evidence_source.value}")

    print("\nRULES THAT FIRED")
    for rule in decision.triggered_rules:
        print(f"  [{rule.kind.value:8s} -> {rule.effect.value:8s}] {rule.reason}")
    if decision.overridden_by_critical:
        print("  >> A critical rule overrode the score: the MQS alone "
              "would not have caused a rollback.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("models", nargs="*", default=MODELS,
                        help=f"models to validate (default: {' '.join(MODELS)})")
    parser.add_argument("--config", default=str(PROJECT / "configs" / "default.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    rows = []
    for name in args.models:
        result = run_validation(bundle_for(name), config)
        show(name, result)
        write_report(result, PROJECT / "reports" / name)
        rows.append((name, result))

    print("\n" + "=" * 100)
    print(f"SUMMARY  (profile: {config.profile})")
    print("=" * 100)
    print(f"{'model':14s} {'decision':9s} {'MQS':>6s} {'TVS':>6s} {'MBS':>6s} {'ORS':>6s}")
    for name, r in rows:
        d = r.scorecard.dimensions
        cell = lambda dim: ("   n/a" if d[dim].score is None  # noqa: E731
                            else f"{d[dim].score:6.1f}")
        reliable = r.scorecard.mqs is not None and r.scorecard.mqs_sufficient_evidence
        mqs = f"{r.scorecard.mqs:6.1f}" if reliable else "   n/a"
        print(f"{name:14s} {r.decision.decision.value:9s} {mqs} "
              f"{cell(Dimension.TRAINING_VALIDITY)} {cell(Dimension.MODEL_BEHAVIOUR)} "
              f"{cell(Dimension.OPERATIONAL_READINESS)}")
    print(f"\nFull results saved under {PROJECT / 'reports'}")


if __name__ == "__main__":
    main()
