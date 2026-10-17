"""The catalogue of every check the gate knows about.

This is the single source of truth for check ids, the dimension each belongs
to, and which direction is "better". The configuration is validated against
it (so a typo in YAML is an error), and the check modules must implement
exactly these ids.

Direction is intrinsic to a metric, so it lives here and not in the config.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mqvg.schemas import Dimension

Direction = Literal["higher_is_better", "lower_is_better", "binary"]

TV = Dimension.TRAINING_VALIDITY
MB = Dimension.MODEL_BEHAVIOUR
OR = Dimension.OPERATIONAL_READINESS


@dataclass(frozen=True)
class CheckSpec:
    id: str
    dimension: Dimension
    direction: Direction
    unit: str
    description: str


CATALOGUE: tuple[CheckSpec, ...] = (
    # --- Training validity ------------------------------------------------
    CheckSpec("training_validity.metadata_completeness", TV, "higher_is_better",
              "fraction", "Share of required training-metadata fields present"),
    CheckSpec("training_validity.reproducibility_info", TV, "higher_is_better",
              "fraction", "Share of reproducibility fields present (seed, versions, hashes, commit)"),
    CheckSpec("training_validity.overfit_gap", TV, "lower_is_better",
              "accuracy gap", "Train accuracy minus eval accuracy (needs train metrics)"),
    CheckSpec("training_validity.train_eval_overlap", TV, "lower_is_better",
              "fraction of eval rows", "Eval rows that also appear in training data (needs row hashes)"),
    CheckSpec("leakage.single_feature_predictiveness", TV, "lower_is_better",
              "ROC AUC", "Highest ROC AUC reached by any single feature on eval data"),
    # --- Model behaviour --------------------------------------------------
    CheckSpec("behaviour.f1", MB, "higher_is_better", "score", "F1 on the evaluation data"),
    CheckSpec("behaviour.roc_auc", MB, "higher_is_better", "score", "ROC AUC on the evaluation data"),
    CheckSpec("behaviour.precision", MB, "higher_is_better", "score", "Precision on the evaluation data"),
    CheckSpec("behaviour.recall", MB, "higher_is_better", "score", "Recall on the evaluation data"),
    CheckSpec("behaviour.subgroup_f1_gap", MB, "lower_is_better",
              "F1 gap", "Largest F1 difference between sufficiently large subgroups"),
    CheckSpec("fairness.selection_rate_ratio", MB, "higher_is_better",
              "ratio", "Lowest/highest group selection rate (worst over sensitive attributes)"),
    CheckSpec("fairness.equalized_odds", MB, "lower_is_better",
              "difference", "Equalized-odds difference (worst over sensitive attributes)"),
    CheckSpec("behaviour.hidden_failure_groups", MB, "lower_is_better",
              "excess error rate",
              "Largest excess error rate (observed minus the model's own expected "
              "error) among feature-defined groups found on held-out rows; 0 if none"),
    CheckSpec("calibration.ece", MB, "lower_is_better", "ECE", "Expected calibration error"),
    CheckSpec("robustness.noise_degradation", MB, "lower_is_better",
              "relative F1 drop", "F1 drop under input perturbation"),
    # --- Operational readiness -------------------------------------------
    CheckSpec("operational.model_loadable", OR, "binary", "pass/fail", "Artifact exists and loads"),
    CheckSpec("operational.input_schema", OR, "binary", "pass/fail", "Model predicts on the evaluation data schema"),
    CheckSpec("operational.prediction_determinism", OR, "binary",
              "pass/fail", "Repeated prediction on the same data gives identical output"),
    CheckSpec("operational.malformed_input_handling", OR, "binary",
              "pass/fail", "Malformed input is rejected cleanly instead of crashing"),
    CheckSpec("operational.artifact_size", OR, "lower_is_better", "MB", "Model file size"),
    CheckSpec("operational.latency", OR, "lower_is_better", "ms", "95th-percentile single-row prediction latency"),
    CheckSpec("operational.memory", OR, "lower_is_better", "MB", "Memory increase when loading and predicting"),
    CheckSpec("operational.dependency_match", OR, "higher_is_better",
              "fraction", "Share of declared library versions matching the current environment"),
)

CHECKS_BY_ID: dict[str, CheckSpec] = {spec.id: spec for spec in CATALOGUE}
assert len(CHECKS_BY_ID) == len(CATALOGUE), "duplicate check id in catalogue"
