"""Request and response bodies. The full result reuses `mqvg.schemas.ValidationResult`."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from mqvg.schemas import ActionMode, Decision, RuleKind

NOTICE = ("This is a recommendation only. Nothing was deployed, promoted or rolled back.")

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9._-]{1,80}$")]
ProfileName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]{1,60}$")]
RelativePath = Annotated[str, StringConstraints(min_length=1, max_length=400)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ValidationRequest(_Model):
    """One candidate to validate. Paths are relative to the server's artifact root."""
    candidate_id: Identifier | None = None
    model_path: RelativePath = Field(description="joblib/pickle model file")
    eval_data_path: RelativePath = Field(description="CSV with the features and the true labels")
    metadata_path: RelativePath | None = Field(default=None, description="training metadata (JSON/YAML)")
    train_row_hashes_path: RelativePath | None = Field(
        default=None, description="one SHA-256 row hash per training row")
    target_column: str | None = Field(default=None, max_length=200)
    positive_label: str | int | bool | None = None
    sensitive_columns: list[Annotated[str, StringConstraints(max_length=200)]] | None = Field(
        default=None, max_length=50)
    profile: ProfileName | None = Field(
        default=None, description="validation profile name; the server default if omitted")


class RuleSummary(_Model):
    rule_id: str
    kind: RuleKind
    effect: Decision
    reason: str


class ValidationSummary(_Model):
    """The answer to a validation: the recommendation, its score and why."""
    run_id: str
    candidate_id: str | None
    profile: str
    decision: Decision
    action_mode: ActionMode = ActionMode.RECOMMENDATION
    summary: str
    mqs: float | None = Field(description="Model Quality Score 0-100; null if it could not be computed")
    mqs_lower_bound: float | None = Field(description="the MQS if every missing check had scored 0")
    mqs_sufficient_evidence: bool = Field(description="false: quote the MQS as n/a")
    overall_coverage: float
    provisional: bool
    dimension_scores: dict[str, float | None]
    triggered_rules: list[RuleSummary]
    check_counts: dict[str, int]
    created_at_utc: datetime
    result_url: str
    report_url: str
    notice: str = NOTICE


class RunEntry(_Model):
    run_id: str
    candidate_id: str | None
    profile: str
    decision: Decision
    mqs: float | None
    created_at_utc: datetime


class RunList(_Model):
    runs: list[RunEntry]


class ProfileInfo(_Model):
    name: str
    description: str | None
    enabled_checks: int
    scans_model_files: bool = Field(
        description="whether the model file is scanned before it is loaded")
    allowed: bool = Field(description="whether this server will run the profile")
    weights: dict[str, float]


class ProfileList(_Model):
    default_profile: str
    profiles: list[ProfileInfo]


class CheckInfo(_Model):
    id: str
    dimension: str
    direction: str
    unit: str
    description: str


class Health(_Model):
    status: str
    gate_version: str
    authentication: str
