from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class C1Output(BaseModel):
    """What C1 gives to C2. Simulated by io/dummy_c1.py until C1 is ready."""
    dataset_id: str
    task_type: TaskType
    target: str
    data_dir: str
    feature_types: dict[str, str]                  # feature -> "numeric" | "categorical"
    fqs: dict[str, float]                          # feature -> quality score 0..1
    class_distribution: Optional[dict[str, float]] = None
    positive_class: Optional[str] = None
    decision_log: list[str] = []


class TaskProfile(BaseModel):
    n_rows: int
    n_features: int
    n_categorical: int
    categorical_ratio: float
    size_class: str                                # "small" | "medium" | "large"
    metric: str
    n_classes: Optional[int] = None
    imbalance_ratio: Optional[float] = None        # minority / majority


class DiagnosisLabel(str, Enum):
    UNDERFITTING = "underfitting"
    OVERFITTING = "overfitting"
    IMBALANCE = "class_imbalance"
    WEAK_FEATURES = "weak_features"
    UNSUITABLE_FAMILY = "unsuitable_model_family"
    HEALTHY = "healthy"


class Diagnosis(BaseModel):
    model_family: str
    labels: list[DiagnosisLabel]
    evidence: dict[str, float]
    reasons: list[str]                             # human-readable, one per label
    confidence: float = Field(default=1.0, ge=0, le=1)


class PlanDecision(BaseModel):
    """Agent 1's plan for the next round."""
    families_next_round: list[str]
    actions: dict[str, list[str]]                  # family -> actions for Agent 2
    referrals_to_c1: list[str] = []
    stop: bool = False
    decided_by: str = "rules"                      # "rules" | "rules+llm" | "fallback"
    rationale: str = ""


class ReasoningEntry(BaseModel):
    round: int
    agent: str
    step: str
    decision: dict
    evidence: dict = {}
    decided_by: str = "rules"
    rationale: str = ""
