"""Typed contracts shared by every module of the validation gate.

These models are the integration surface of the component: later phases
(AutoML hand-off, deployment, monitoring) must exchange data in these shapes.
Several validators encode design rules directly in the types, most
importantly that a check without evidence can never carry a score or a
passing status.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "0.1.0"
WEIGHT_TOLERANCE = 1e-9


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------
class Dimension(StrEnum):
    TRAINING_VALIDITY = "training_validity"          # TVS
    MODEL_BEHAVIOUR = "model_behaviour"              # MBS
    OPERATIONAL_READINESS = "operational_readiness"  # ORS


class CheckStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    UNAVAILABLE = "UNAVAILABLE"        # evidence needed was not provided
    NOT_APPLICABLE = "NOT_APPLICABLE"  # check does not apply to this model/task
    ERROR = "ERROR"                    # the check itself crashed


class EvidenceSource(StrEnum):
    MEASURED = "MEASURED"        # computed by the gate from the artifact/data
    DECLARED = "DECLARED"        # taken from metadata, not independently verified
    SIMULATED = "SIMULATED"      # injected by a test scenario
    UNAVAILABLE = "UNAVAILABLE"  # no evidence


class Decision(StrEnum):
    PROMOTE = "PROMOTE"
    HOLD = "HOLD"
    ROLLBACK = "ROLLBACK"


class ActionMode(StrEnum):
    # Standalone mode never executes anything; it only recommends.
    RECOMMENDATION = "RECOMMENDATION"


class RuleKind(StrEnum):
    CRITICAL = "critical"  # overrides the MQS
    EVIDENCE = "evidence"  # missing or insufficient evidence
    BLOCKING = "blocking"  # a must-pass check did not pass
    SCORE = "score"        # MQS threshold bands


class TaskType(StrEnum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class MissingPolicy(StrEnum):
    EXCLUDE_AND_CAP = "exclude_and_cap"  # drop from averages, flag as provisional
    PENALIZE = "penalize"                # score missing checks as 0


class ScenarioMode(StrEnum):
    REAL = "real"            # every result comes from real inputs
    SIMULATED = "simulated"  # at least one input was injected for a demo/test


class _Strict(BaseModel):
    """Base for gate-produced objects: unknown fields are an error."""
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
class ScenarioInfo(_Strict):
    """Declares whether a run uses simulated inputs, so reports can say so."""
    mode: ScenarioMode = ScenarioMode.REAL
    name: str | None = None
    description: str | None = None
    simulated_inputs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _simulated_must_be_explained(self) -> ScenarioInfo:
        if self.mode == ScenarioMode.SIMULATED:
            if not self.simulated_inputs:
                raise ValueError(
                    "A simulated scenario must list which inputs are simulated")
        elif self.simulated_inputs:
            raise ValueError("simulated_inputs given but mode is 'real'")
        return self


class CandidateBundle(_Strict):
    """Everything needed to validate one candidate model.

    Paths are not required to exist: a missing or unreadable artifact is a
    validation finding, not a malformed request.
    """
    candidate_id: str | None = None
    model_path: Path
    eval_data_path: Path
    metadata_path: Path | None = None
    train_row_hashes_path: Path | None = None
    config_path: Path | None = None
    # Optional overrides of what the metadata declares, so a candidate can be
    # validated even when its metadata is incomplete.
    target_column: str | None = None
    positive_label: str | int | bool | None = None
    sensitive_columns: list[str] | None = None
    scenario: ScenarioInfo = Field(default_factory=ScenarioInfo)


class SplitInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    method: str | None = None
    stratify_on: str | None = None
    seed: int | None = None
    ratios: dict[str, float] | None = None


class CodeInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    commit: str | None = None
    uncommitted_source_changes: bool | None = None


class TrainingMetadata(BaseModel):
    """Producer-declared facts about how the model was trained.

    Every field is optional because incomplete metadata is a normal case the
    gate must handle. Extra fields are kept. NOTE: anything in here is
    DECLARED evidence. Declared metrics (for example `evaluation_metrics`,
    if present as an extra field) must never be used as if the gate measured
    them.
    """
    model_config = ConfigDict(extra="allow")

    model_name: str | None = None
    model_version: str | None = None
    model_type: str | None = None
    task_type: TaskType | None = None
    dataset: str | None = None
    data_source: str | None = None
    target_column: str | None = None
    positive_label: str | int | bool | None = None
    sensitive_columns: list[str] | None = None
    excluded_sensitive_features: list[str] | None = None
    feature_columns: list[str] | None = None
    hyperparameters: dict[str, Any] | None = None
    random_state: int | None = None
    split: SplitInfo | None = None
    train_rows: int | None = Field(default=None, ge=0)
    eval_rows: int | None = Field(default=None, ge=0)
    train_data_sha256: str | None = None
    train_row_hashes_file: str | None = None
    row_hash_columns: list[str] | None = None
    training_metrics: dict[str, float] | None = None
    versions: dict[str, str] | None = None
    code: CodeInfo | None = None
    trained_at_utc: str | None = None

    def missing_fields(self, required: Iterable[str]) -> list[str]:
        """Names from `required` that are absent or empty.

        Dotted names such as "split.seed" look inside nested objects.
        """
        dumped = self.model_dump()
        missing = []
        for name in required:
            value: Any = dumped
            for part in name.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            if value is None or value == [] or value == {} or value == "":
                missing.append(name)
        return missing


# --------------------------------------------------------------------------
# Check results
# --------------------------------------------------------------------------
_NO_SCORE = {CheckStatus.UNAVAILABLE, CheckStatus.NOT_APPLICABLE,
             CheckStatus.ERROR}


class CheckResult(_Strict):
    """Outcome of one validation check.

    Invariants (enforced below):
    * PASS/WARN/FAIL have a 0-100 score and real evidence.
    * UNAVAILABLE/NOT_APPLICABLE/ERROR have no score and no evidence, so a
      check that could not run can never be mistaken for a pass.
    """
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    dimension: Dimension
    name: str
    status: CheckStatus
    evidence_source: EvidenceSource
    score: float | None = Field(default=None, ge=0, le=100)
    raw_value: float | int | bool | str | None = None
    direction: Literal["higher_is_better", "lower_is_better", "binary"] | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)
    required: bool = False
    message: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _status_matches_evidence(self) -> CheckResult:
        if self.status in _NO_SCORE:
            if self.score is not None:
                raise ValueError(f"{self.status} check must not have a score")
            if self.evidence_source != EvidenceSource.UNAVAILABLE:
                raise ValueError(
                    f"{self.status} check must have evidence_source UNAVAILABLE")
        else:
            if self.score is None:
                raise ValueError(f"{self.status} check requires a score")
            if self.evidence_source == EvidenceSource.UNAVAILABLE:
                raise ValueError(
                    f"{self.status} check cannot have evidence_source UNAVAILABLE")
        return self

    @property
    def has_evidence(self) -> bool:
        return self.status not in _NO_SCORE


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
class CheckTreatment(StrEnum):
    COUNTED = "counted"                              # had evidence; scored
    EXCLUDED_UNAVAILABLE = "excluded_unavailable"    # left out of the average
    PENALIZED_AS_ZERO = "penalized_as_zero"          # missing, scored as 0
    NOT_APPLICABLE = "not_applicable"                # never applied; ignored


class CheckContribution(_Strict):
    """How one check fed into its dimension score.

    `points` are the check's share of the dimension score, so the points of a
    dimension add up to that dimension's score.
    """
    check_id: str
    weight: float = Field(gt=0)
    status: CheckStatus | None   # None: no result was supplied for the check
    score: float | None = Field(default=None, ge=0, le=100)
    treatment: CheckTreatment
    points: float = Field(ge=0, le=100)


class DimensionScore(_Strict):
    dimension: Dimension
    score: float | None = Field(default=None, ge=0, le=100)
    coverage: float = Field(ge=0, le=1)   # weight of available / applicable checks
    provisional: bool
    available_checks: int = Field(ge=0)
    total_checks: int = Field(ge=0)
    note: str | None = None
    contributions: list[CheckContribution] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> DimensionScore:
        if self.score is not None and self.contributions:
            total = sum(c.points for c in self.contributions)
            if abs(total - self.score) > 1e-6:
                raise ValueError(
                    f"contribution points ({total}) must add up to the "
                    f"score ({self.score})")
        if self.available_checks > self.total_checks:
            raise ValueError("available_checks cannot exceed total_checks")
        if self.score is not None and self.coverage == 0:
            raise ValueError("a score cannot exist with zero coverage")
        if self.score is not None and self.coverage < 1 and not self.provisional:
            raise ValueError("partial coverage must be marked provisional")
        return self


class ScoreCard(_Strict):
    """Dimension scores, weights, and the Model Quality Score."""
    dimensions: dict[Dimension, DimensionScore]
    weights: dict[Dimension, float]
    effective_weights: dict[Dimension, float] | None = None
    mqs: float | None = Field(default=None, ge=0, le=100)
    # What the MQS would be if every missing check (and every dimension left
    # without a score) had scored 0. Never above `mqs`; equal under `penalize`.
    mqs_lower_bound: float | None = Field(default=None, ge=0, le=100)
    overall_coverage: float = Field(ge=0, le=1)
    provisional: bool
    missing_policy: MissingPolicy

    @staticmethod
    def _check_weights(weights: dict[Dimension, float], label: str) -> None:
        if set(weights) != set(Dimension):
            raise ValueError(f"{label} must define all three dimensions")
        if any(w < 0 for w in weights.values()):
            raise ValueError(f"{label} must not be negative")
        if abs(sum(weights.values()) - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(f"{label} must sum to 1")

    @model_validator(mode="after")
    def _consistent(self) -> ScoreCard:
        if set(self.dimensions) != set(Dimension):
            raise ValueError("dimensions must contain all three dimensions")
        self._check_weights(self.weights, "weights")
        if self.effective_weights is not None:
            self._check_weights(self.effective_weights, "effective_weights")
        if self.mqs is not None and not self.provisional:
            if any(d.provisional or d.score is None
                   for d in self.dimensions.values()):
                raise ValueError(
                    "MQS built from incomplete dimensions must be provisional")
        if (self.mqs is not None and self.mqs_lower_bound is not None
                and self.mqs_lower_bound > self.mqs + 1e-9):
            raise ValueError("mqs_lower_bound cannot exceed mqs")
        return self


# --------------------------------------------------------------------------
# Decision
# --------------------------------------------------------------------------
class TriggeredRule(_Strict):
    rule_id: str
    name: str
    kind: RuleKind
    effect: Decision
    reason: str = Field(min_length=1)
    observed: Any = None
    threshold: Any = None
    check_ids: list[str] = Field(default_factory=list)


class DecisionRecord(_Strict):
    decision: Decision
    action_mode: ActionMode = ActionMode.RECOMMENDATION
    mqs: float | None = Field(default=None, ge=0, le=100)
    overridden_by_critical: bool = False
    triggered_rules: list[TriggeredRule] = Field(default_factory=list)
    summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> DecisionRecord:
        effects = {r.effect for r in self.triggered_rules}
        if self.decision == Decision.PROMOTE and effects - {Decision.PROMOTE}:
            raise ValueError("PROMOTE cannot coexist with HOLD/ROLLBACK rules")
        if self.decision == Decision.ROLLBACK and Decision.ROLLBACK not in effects:
            raise ValueError("ROLLBACK requires a triggered ROLLBACK rule")
        if self.overridden_by_critical:
            if self.decision != Decision.ROLLBACK:
                raise ValueError("only a ROLLBACK can be a critical override")
            if not any(r.kind == RuleKind.CRITICAL for r in self.triggered_rules):
                raise ValueError("critical override needs a triggered critical rule")
        return self


# --------------------------------------------------------------------------
# Full result
# --------------------------------------------------------------------------
class CandidateSummary(_Strict):
    candidate_id: str | None = None
    model_path: str
    model_sha256: str | None = None
    eval_data_sha256: str | None = None
    metadata_sha256: str | None = None


class ValidationResult(_Strict):
    """The complete, auditable output of one validation run."""
    schema_version: str = SCHEMA_VERSION
    gate_version: str
    run_id: str
    created_at_utc: datetime
    candidate: CandidateSummary
    scenario: ScenarioInfo
    config_hash: str
    config_snapshot: dict[str, Any]
    environment: dict[str, Any] = Field(default_factory=dict)
    checks: list[CheckResult]
    scorecard: ScoreCard
    decision: DecisionRecord
    explanations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> ValidationResult:
        ids = [c.id for c in self.checks]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate check ids: {sorted(duplicates)}")
        simulated = any(c.evidence_source == EvidenceSource.SIMULATED
                        for c in self.checks)
        if simulated and self.scenario.mode != ScenarioMode.SIMULATED:
            raise ValueError(
                "SIMULATED evidence present but the scenario is marked real")
        unknown = {cid for r in self.decision.triggered_rules
                   for cid in r.check_ids} - set(ids)
        if unknown:
            raise ValueError(f"rules reference unknown checks: {sorted(unknown)}")
        return self
