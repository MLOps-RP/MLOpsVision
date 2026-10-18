"""The complete validation workflow in one call.

    supported model + separate evaluation data (with the true labels)
      -> Layer 1  Training Validity
      -> Layer 2  Model Behaviour, including Hidden Failure Group Detection
      -> Layer 3  Operational Readiness
      -> MQS (configured dimension weights)
      -> PROMOTE / HOLD / ROLLBACK recommendation (configured thresholds and rules)
      -> one unified report (Markdown) and the full result (JSON)

`validate_and_report` adds nothing to the validation itself: it calls
`run_validation` and `write_report`. The outcome is a RECOMMENDATION. Nothing
is deployed, promoted or rolled back; real deployment actions are outside this
component.

SECURITY: loading the model executes code stored in the model file; validate
only models from trusted sources.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from mqvg.config import GateConfig
from mqvg.pipeline import run_validation
from mqvg.reporting import render_markdown, write_report
from mqvg.schemas import CandidateBundle, ValidationResult


@dataclass
class WorkflowOutput:
    result: ValidationResult
    report_markdown: str
    paths: dict[str, Path] = field(default_factory=dict)   # empty if nothing was written

    @property
    def decision(self) -> str:
        return self.result.decision.decision.value


def validate_and_report(bundle: CandidateBundle, config: GateConfig | None = None,
                        report_dir: str | Path | None = None, *,
                        run_id: str | None = None,
                        created_at: datetime | None = None) -> WorkflowOutput:
    """Validate one candidate and produce the unified report.

    `report_dir`, if given, receives `report.md` and `result.json`.
    A bad candidate never raises (it becomes findings and a decision); only an
    unusable configuration raises `ConfigError`.
    """
    result = run_validation(bundle, config, run_id=run_id, created_at=created_at)
    paths = write_report(result, report_dir) if report_dir is not None else {}
    return WorkflowOutput(result=result, report_markdown=render_markdown(result),
                          paths=paths)
