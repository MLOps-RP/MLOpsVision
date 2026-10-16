"""Readable validation report and result files.

`render_markdown` turns a ValidationResult into a self-contained Markdown
report; `write_report` writes it next to the full `result.json`. The report
only presents what is in the result: no new judgement is made here.

Every rule that fired and every check appears in the report, so a reader never
has to open the JSON to see why a model was held, promoted or rejected.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from mqvg.schemas import (
    CheckStatus, CheckTreatment, Decision, Dimension, ScenarioMode,
    ValidationResult,
)

DIMENSION_LABELS = {
    Dimension.TRAINING_VALIDITY: "Training Validity (TVS)",
    Dimension.MODEL_BEHAVIOUR: "Model Behaviour (MBS)",
    Dimension.OPERATIONAL_READINESS: "Operational Readiness (ORS)",
}
NO_EVIDENCE = (CheckStatus.UNAVAILABLE, CheckStatus.ERROR)
MISSING = "—"


# --------------------------------------------------------------- helpers
def _cell(text: Any) -> str:
    """One table cell: single line, pipes escaped."""
    return " ".join(str(text).split()).replace("|", "\\|")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def _value(value: Any) -> str:
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _score(value: float | None, digits: int = 1) -> str:
    return MISSING if value is None else f"{value:.{digits}f}"


def _title(result: ValidationResult) -> str:
    return (result.candidate.candidate_id
            or Path(result.candidate.model_path).name)


# ------------------------------------------------------------- sections
def _banner(result: ValidationResult) -> list[str]:
    decision, card = result.decision, result.scorecard
    lines = [f"## Decision: {decision.decision.value}", "",
             f"> {decision.summary}", "",
             "*This is a recommendation. Nothing was deployed, promoted or "
             "rolled back.*", ""]
    if result.scenario.mode == ScenarioMode.SIMULATED:
        lines += ["**SIMULATED RUN.** Simulated inputs: "
                  + ", ".join(result.scenario.simulated_inputs)
                  + ". Results are not real measurements.", ""]
    if not card.mqs_sufficient_evidence:
        lines += [f"**The MQS is not reliable:** {card.insufficient_evidence_reason}.", ""]
    return lines


def _scores(result: ValidationResult) -> list[str]:
    card = result.scorecard
    reliable = card.mqs is not None and card.mqs_sufficient_evidence
    mqs = _score(card.mqs) if reliable else (
        f"n/a (insufficient evidence; recorded value {_score(card.mqs)})"
        if card.mqs is not None else "n/a")
    lines = ["## Scores (0–100)", "",
             f"**Model Quality Score (MQS): {mqs}**  ",
             f"Worst case if every missing check had failed: "
             f"{_score(card.mqs_lower_bound)}  ",
             f"Overall evidence coverage: {card.overall_coverage:.0%}  ",
             f"Missing-evidence policy: `{card.missing_policy.value}`", ""]
    rows = []
    for dimension, dim in card.dimensions.items():
        rows.append([DIMENSION_LABELS[dimension], _score(dim.score),
                     f"{card.weights[dimension]:.2f}", f"{dim.coverage:.0%}",
                     f"{dim.available_checks}/{dim.total_checks}",
                     "yes" if dim.provisional else "no"])
    lines += _table(["Dimension", "Score", "Weight", "Evidence", "Checks with evidence",
                     "Provisional"], rows)
    lines += ["", "`MQS = TVS × W1 + MBS × W2 + ORS × W3`"
              + (" (weights renormalized over the scored dimensions: "
                 + ", ".join(f"{DIMENSION_LABELS[d].split(' (')[0]} "
                             f"{w:.2f}" for d, w in card.effective_weights.items())
                 + ")" if card.effective_weights
                 and any(d.score is None for d in card.dimensions.values()) else ""), ""]
    for dimension, dim in card.dimensions.items():
        if dim.note:
            lines.append(f"* {DIMENSION_LABELS[dimension]}: {dim.note}")
    return lines + [""]


def _why(result: ValidationResult) -> list[str]:
    rules = result.decision.triggered_rules
    lines = ["## Why this decision", ""]
    if result.decision.overridden_by_critical:
        lines += ["**A critical rule overrode the score.**", ""]
    lines += _table(["Rule", "Kind", "Effect", "Reason"],
                    [[r.name, r.kind.value, r.effect.value, r.reason] for r in rules])
    # The decision line and the rule reasons are already in the table above.
    explanation = [line for line in result.explanations
                   if not line.startswith(("Decision:", "Why ("))]
    if explanation:
        lines += ["", "### Explanation", ""] + [f"* {line}" for line in explanation]
    return lines + [""]


def _breakdown(result: ValidationResult) -> list[str]:
    checks = {c.id: c for c in result.checks}
    lines = ["## Score breakdown", "",
             "Each check's points add up to its dimension's score.", ""]
    for dimension, dim in result.scorecard.dimensions.items():
        shown = (f"{dim.score:.1f}" if dim.score is not None else "no score")
        lines += [f"### {DIMENSION_LABELS[dimension]}: {shown}", ""]
        rows = []
        for c in dim.contributions:
            check = checks.get(c.check_id)
            rows.append([c.check_id, c.status.value if c.status else "no result",
                         _score(c.score, 0), f"{c.weight:g}", f"{c.points:.2f}",
                         c.treatment.value.replace("_", " ")])
        lines += _table(["Check", "Status", "Score", "Weight", "Points", "Treatment"], rows)
        lines.append("")
    return lines


def _checks(result: ValidationResult) -> list[str]:
    rows = [[c.id, c.status.value, _value(c.raw_value), _score(c.score, 0),
             c.evidence_source.value, "yes" if c.required else ""]
            for c in result.checks]
    lines = ["## All checks", ""]
    lines += _table(["Check", "Status", "Value", "Score", "Evidence", "Required"], rows)
    lines += ["", "### Check details", ""]
    for c in result.checks:
        lines.append(f"* **`{c.id}`** — {c.status.value}: {_cell(c.message)}")
    lines += ["", "*Evidence: MEASURED = computed by the gate from the model and "
                  "data; DECLARED = taken from producer metadata, not verified; "
                  "SIMULATED = injected for a demonstration; UNAVAILABLE = none.*", ""]
    return lines


def _gaps(result: ValidationResult) -> list[str]:
    gaps = [c for c in result.checks if c.status in NO_EVIDENCE]
    na = [c for c in result.checks if c.status == CheckStatus.NOT_APPLICABLE]
    lines = ["## Evidence gaps", ""]
    if not gaps and not na:
        return lines + ["No gaps: every check had evidence.", ""]
    lines.append("Checks without evidence are never counted as passing.")
    lines.append("")
    rows = [[c.id, c.status.value, c.message] for c in gaps]
    rows += [[c.id, "NOT_APPLICABLE", c.message] for c in na]
    return lines + _table(["Check", "Status", "Reason"], rows) + [""]


def _audit(result: ValidationResult) -> list[str]:
    cand, env = result.candidate, result.environment
    machine = env.get("machine", {})
    versions = ", ".join(f"{k} {v}" for k, v in machine.get("versions", {}).items())
    rows = [
        ["Run id", result.run_id], ["Created (UTC)", result.created_at_utc.isoformat()],
        ["Gate version", result.gate_version], ["Schema version", result.schema_version],
        ["Candidate", cand.candidate_id or MISSING],
        ["Model file", cand.model_path], ["Model SHA-256", cand.model_sha256 or MISSING],
        ["Evaluation data SHA-256", cand.eval_data_sha256 or MISSING],
        ["Metadata SHA-256", cand.metadata_sha256 or MISSING],
        ["Configuration profile", result.config_snapshot.get("profile", MISSING)],
        ["Configuration hash", result.config_hash],
        ["Machine", f"{machine.get('platform', MISSING)}; "
                    f"{machine.get('cpu_count', MISSING)} CPUs; "
                    f"{machine.get('ram_gb', MISSING)} GB RAM"],
        ["Libraries", versions or MISSING],
    ]
    timings = env.get("timings_seconds")
    if timings:
        rows.append(["Stage timings (s)", ", ".join(f"{k} {v:.2f}" for k, v in timings.items())])
    return ["## Inputs and audit trail", ""] + _table(["Item", "Value"], rows) + [""]


def _configuration(result: ValidationResult) -> list[str]:
    cfg = result.config_snapshot
    decision = cfg.get("decision", {})
    rules = cfg.get("critical_rules", [])
    lines = ["## Configuration used", "",
             f"* Profile `{cfg.get('profile')}`: {cfg.get('description') or ''}",
             f"* Dimension weights: {cfg.get('weights')}",
             f"* Promote at MQS ≥ {decision.get('promote_at')}; roll back below "
             f"{decision.get('reject_below')}",
             f"* Must-pass checks: {', '.join(decision.get('must_pass_checks', [])) or 'none'}",
             f"* Minimum evidence coverage: {cfg.get('coverage')}",
             f"* Critical rules: " + (", ".join(r.get("name", r.get("id", "?"))
                                              for r in rules) or "none"),
             "", "The thresholds and weights are configuration choices. They need "
                 "justification for the domain; they are not universal standards.", ""]
    return lines


def _limits() -> list[str]:
    return ["## Limitations", "",
            "* This report is a recommendation; no action was taken.",
            "* DECLARED evidence comes from producer metadata and is not verified.",
            "* The gate looks for signs of leakage, overfitting and bias; it "
            "cannot prove their absence.",
            "* Only binary classification on tabular data is evaluated "
            "(see `docs/supported_models.md`).",
            "* Loading a model file executes code stored in it; validate only "
            "models from trusted sources.", ""]


# ---------------------------------------------------------------- public
def render_markdown(result: ValidationResult) -> str:
    """The complete report as Markdown (LF line endings)."""
    parts = ([f"# Model Quality Validation Report: {_title(result)}", ""]
             + _banner(result) + _scores(result) + _why(result)
             + _breakdown(result) + _gaps(result) + _checks(result)
             + _audit(result) + _configuration(result) + _limits())
    return "\n".join(parts).rstrip("\n") + "\n"


def write_report(result: ValidationResult, directory: str | Path) -> dict[str, Path]:
    """Write `result.json` and `report.md` into `directory` (created if needed)."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {"json": folder / "result.json", "markdown": folder / "report.md"}
    with open(paths["json"], "w", encoding="utf-8", newline="\n") as handle:
        handle.write(result.model_dump_json(indent=2) + "\n")
    with open(paths["markdown"], "w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_markdown(result))
    return paths
