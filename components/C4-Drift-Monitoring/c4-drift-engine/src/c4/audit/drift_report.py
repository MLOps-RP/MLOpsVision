"""
DRIFT REPORT - the output contract C4 hands to C1, C2 and C3.
Agree these fields with your teammates; integration later only changes where
the report is SENT, not what is in it.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
import uuid

from src.common.config import OUTPUTS, save_json


@dataclass
class DriftReport:
    model_name: str
    model_version: str
    baseline_version: str
    preprocessing_version: str
    scenario: str
    window: int
    drift_type: str
    diagnosis_confidence: str
    severity_score: float
    severity_level: str
    severity_breakdown: dict
    recommended_action: str
    target_component: str
    reason: str
    affected_features: list = field(default_factory=list)
    affected_segments: list = field(default_factory=list)
    performance: dict = field(default_factory=dict)
    data_quality_issues: list = field(default_factory=list)
    action_result: dict = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    schema_version: str = "1.0"

    def save(self) -> Path:
        path = OUTPUTS / "drift_reports" / self.scenario / f"w{self.window:02d}_{self.recommended_action}_{self.report_id}.json"
        save_json(asdict(self), path)
        return path

    def to_dict(self) -> dict:
        return asdict(self)
