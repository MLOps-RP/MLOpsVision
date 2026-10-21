"""Request and response bodies. The full result reuses `mqvg.schemas.ValidationResult`."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

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
    # Plain-language guide, generated from the profile's own configuration (never from its name).
    summary: str = Field(description="what the profile does, in simple words")
    highlights: list[str] = Field(description="what sets it apart: scanning, extra checks, subgroup detection")
    strictness: Literal["lenient", "standard", "strict"] = Field(
        description="how demanding it is: derived from its promotion threshold and how it treats missing evidence")
    strictness_text: str = Field(description="the thresholds and rules behind the strictness, in simple words")
    unavailable_reason: str | None = Field(default=None, description="why this server will not run it (when allowed is false)")


class ProfileList(_Model):
    default_profile: str
    profiles: list[ProfileInfo]


class CheckInfo(_Model):
    id: str
    dimension: str
    direction: str
    unit: str
    description: str


UploadKind = Literal["model", "eval", "metadata", "hashes"]


class UploadInfo(_Model):
    """A stored upload. `path` is relative to the artifact root, ready for a validation request."""
    upload_id: str
    kind: UploadKind
    path: str
    bytes: int
    sha256: str


class UploadLimits(_Model):
    max_bytes: dict[str, int] = Field(description="largest accepted file per kind")
    extensions: dict[str, list[str]] = Field(description="accepted file extensions per kind")


class Capabilities(_Model):
    """What this server will do, so a client can show unavailable features instead of failing."""
    gate_version: str
    uploads_enabled: bool
    upload_limits: UploadLimits
    unscanned_profiles_allowed: bool
    default_profile: str


class StudyInfo(_Model):
    id: str
    title: str
    description: str
    has_data: bool = Field(description="a machine-readable result exists")
    has_report: bool = Field(description="a Markdown report exists")
    generated_at_utc: datetime | None = None
    script: str = Field(description="the helper script that generates it (the API never runs it)")


class StudyList(_Model):
    studies: list[StudyInfo]


class StudyData(_Model):
    id: str
    title: str
    generated_at_utc: datetime | None = None
    data: dict[str, Any]


class CandidateInfo(_Model):
    """A model the server offers, with the files that belong to it, exactly as declared in the catalogue."""
    id: str
    description: str
    model_path: str = Field(description="relative to the artifact root")
    eval_data_path: str
    metadata_path: str | None
    train_row_hashes_path: str | None
    suggested_profile: str | None = Field(default=None, description="a profile the catalogue recommends for it")
    available: bool = Field(description="false if a declared file is missing or outside the artifact root")
    unavailable_reason: str | None = None


class CandidateList(_Model):
    catalogue_found: bool = Field(description="false if the server has no candidates.yaml")
    error: str | None = Field(default=None, description="why the catalogue could not be used, if it is malformed")
    candidates: list[CandidateInfo]


class Health(_Model):
    status: str
    gate_version: str
    authentication: str
