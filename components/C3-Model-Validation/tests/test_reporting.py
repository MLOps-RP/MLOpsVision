import re
from datetime import datetime, timezone

import pytest

from mqvg.explain import explain_failures
from mqvg.pipeline import run_validation
from mqvg.reporting import _cell, render_markdown, write_report
from mqvg.rules import evaluate_rules
from mqvg.schemas import (
    CandidateBundle, CandidateSummary, CheckResult, CheckStatus, Dimension,
    EvidenceSource, ScenarioInfo, ScenarioMode, ValidationResult,
)
from mqvg.scoring import compute_scorecard

from builders import UNAVAILABLE, make_results
from conftest import DATA, FIXTURES, ROOT

TV, MB, OR = (Dimension.TRAINING_VALIDITY, Dimension.MODEL_BEHAVIOUR,
              Dimension.OPERATIONAL_READINESS)
LEAKAGE = "leakage.single_feature_predictiveness"
WHEN = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def build(config, scenario=None, results=None, **kwargs) -> ValidationResult:
    """A complete ValidationResult from synthetic check results."""
    results = results if results is not None else make_results(config, **kwargs)
    scenario = scenario or ScenarioInfo()
    card = compute_scorecard(results, config)
    decision = evaluate_rules(results, card, config)
    return ValidationResult(
        gate_version="0.1.0", run_id="run-123", created_at_utc=WHEN,
        candidate=CandidateSummary(candidate_id="demo", model_path="m/model.joblib",
                                   model_sha256="a" * 64, eval_data_sha256="b" * 64),
        scenario=scenario, config_hash="c" * 64,
        config_snapshot=config.model_dump(mode="json"),
        environment={"machine": {"platform": "TestOS", "cpu_count": 4, "ram_gb": 16.0,
                                 "versions": {"python": "3.12.1"}},
                     "timings_seconds": {"load": 0.1}},
        checks=results, scorecard=card, decision=decision,
        explanations=explain_failures(
            results, card, decision, config,
            scenario.simulated_inputs if scenario.mode == ScenarioMode.SIMULATED else ()))


def unescaped_pipes(line):
    return len(re.findall(r"(?<!\\)\|", line))


# ------------------------------------------------------------- structure
def test_report_has_every_section_in_order(default_config):
    text = render_markdown(build(default_config))
    headings = re.findall(r"^#{1,2} (.+)$", text, re.M)
    assert headings == [
        "Model Quality Validation Report: demo", "Decision: PROMOTE", "Scores (0–100)",
        "Why this decision", "Score breakdown", "Evidence gaps", "All checks",
        "Inputs and audit trail", "Configuration used", "Limitations"]


def test_report_is_plain_lf_text_ending_in_one_newline(default_config):
    text = render_markdown(build(default_config))
    assert "\r" not in text and text.endswith("\n") and not text.endswith("\n\n")


def test_report_never_leaks_python_placeholders(default_config):
    for kwargs in ({}, {"per_dimension": {MB: UNAVAILABLE}},
                   {"scores": {LEAKAGE: ("FAIL", 5)}}):
        text = render_markdown(build(default_config, **kwargs))
        assert "None" not in text and "nan" not in text.lower().split()


def test_rendering_is_deterministic(default_config):
    result = build(default_config, scores={LEAKAGE: ("FAIL", 5)})
    assert render_markdown(result) == render_markdown(result)


def test_tables_are_well_formed_even_with_awkward_messages(default_config):
    results = make_results(default_config)
    results[0] = results[0].model_copy(update={
        "message": "has a pipe | and\na newline | here", "raw_value": "a|b"})
    text = render_markdown(build(default_config, results=results))
    block = []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            block.append(line)
            continue
        if block:
            counts = {unescaped_pipes(row) for row in block}
            assert len(counts) == 1, block[0]
            block = []
    assert "has a pipe \\| and a newline \\| here" in text


def test_cell_escaping():
    assert _cell("a|b\nc") == "a\\|b c"


# ------------------------------------------------------ nothing is omitted
def test_every_rule_that_fired_appears_with_its_reason(default_config):
    result = build(default_config, scores={
        LEAKAGE: ("FAIL", 5), "calibration.ece": ("WARN", 50),
        "fairness.selection_rate_ratio": UNAVAILABLE})
    text = render_markdown(result)
    assert len(result.decision.triggered_rules) >= 3
    for rule in result.decision.triggered_rules:
        assert rule.name in text
        assert _cell(rule.reason) in text


def test_every_check_appears_in_the_checks_breakdown_and_details(default_config):
    result = build(default_config)
    text = render_markdown(result)
    for check in result.checks:
        assert text.count(f"`{check.id}`") >= 1                  # details list
        assert f"| {check.id} |" in text                           # tables
        assert _cell(check.message) in text


def test_breakdown_has_one_row_per_contribution_and_its_points(default_config):
    result = build(default_config, scores={"behaviour.f1": ("WARN", 70)})
    text = render_markdown(result)
    for dimension in result.scorecard.dimensions.values():
        for c in dimension.contributions:
            assert re.search(rf"\| {re.escape(c.check_id)} \| .* \| {c.points:.2f} \|", text)


def test_the_rules_table_matches_the_decision(default_config):
    result = build(default_config, scores={LEAKAGE: ("FAIL", 5)})
    text = render_markdown(result)
    assert "## Decision: ROLLBACK" in text
    assert "A critical rule overrode the score." in text
    assert "| Target leakage detected | critical | ROLLBACK |" in text


def test_the_explanation_does_not_repeat_the_rules_table(default_config):
    text = render_markdown(build(default_config, scores={LEAKAGE: ("FAIL", 5)}))
    explanation = text.split("### Explanation")[1].split("## Score breakdown")[0]
    assert "* Decision:" not in explanation and "* Why (" not in explanation
    assert "Override:" in explanation and "Suggested action" in explanation


# --------------------------------------------------- honesty in the report
def test_evidence_gaps_are_listed(default_config):
    text = render_markdown(build(default_config, scores={
        "fairness.selection_rate_ratio": UNAVAILABLE}))
    gaps = text.split("## Evidence gaps")[1].split("## All checks")[0]
    assert "never counted as passing" in gaps and "fairness.selection_rate_ratio" in gaps


def test_no_gaps_section_says_so(default_config):
    gaps = render_markdown(build(default_config)).split("## Evidence gaps")[1]
    assert "No gaps: every check had evidence." in gaps.split("## All checks")[0]


def test_an_unreliable_mqs_is_shown_as_not_available(default_config):
    result = build(default_config, per_dimension={MB: UNAVAILABLE})
    text = render_markdown(result)
    assert not result.scorecard.mqs_sufficient_evidence
    assert "**The MQS is not reliable:**" in text
    assert "n/a (insufficient evidence; recorded value" in text
    assert "weights renormalized over the scored dimensions" in text


def test_renormalization_is_only_mentioned_when_a_dimension_was_dropped(default_config):
    provisional_only = build(default_config, scores={"behaviour.precision": UNAVAILABLE})
    assert provisional_only.scorecard.provisional
    assert "renormalized" not in render_markdown(provisional_only)
    dropped = build(default_config, per_dimension={MB: UNAVAILABLE})
    assert "renormalized" in render_markdown(dropped)


def test_a_reliable_mqs_is_shown_plainly(default_config):
    result = build(default_config)
    text = render_markdown(result)
    assert f"**Model Quality Score (MQS): {result.scorecard.mqs:.1f}**" in text
    assert "not reliable" not in text


def test_a_simulated_run_is_labelled(default_config):
    scenario = ScenarioInfo(mode=ScenarioMode.SIMULATED, name="demo",
                            simulated_inputs=["metadata"])
    simulated = build(default_config, scenario=scenario)
    # a simulated scenario requires the result to say so (schema rule)
    text = render_markdown(simulated)
    assert "**SIMULATED RUN.**" in text and "metadata" in text
    assert "SIMULATED RUN" not in render_markdown(build(default_config))


def test_declared_evidence_is_labelled_in_the_checks_table(default_config):
    results = make_results(default_config)
    results = [r.model_copy(update={"evidence_source": EvidenceSource.DECLARED})
               if r.id == "training_validity.overfit_gap" else r for r in results]
    text = render_markdown(build(default_config, results=results))
    assert re.search(r"\| training_validity\.overfit_gap \| PASS \| .* \| DECLARED \|", text)


def test_a_missing_dimension_score_is_shown_as_a_dash(default_config):
    text = render_markdown(build(default_config, per_dimension={MB: UNAVAILABLE}))
    assert "| Model Behaviour (MBS) | — |" in text
    assert "### Model Behaviour (MBS): no score" in text


# ------------------------------------------------------------ audit trail
def test_audit_section_records_the_run(default_config):
    text = render_markdown(build(default_config))
    audit = text.split("## Inputs and audit trail")[1].split("## Configuration used")[0]
    for expected in ("run-123", WHEN.isoformat(), "a" * 64, "b" * 64, "c" * 64,
                     "TestOS", "python 3.12.1", "load 0.10", "default"):
        assert expected in audit


def test_configuration_section_summarizes_the_rules(default_config):
    text = render_markdown(build(default_config))
    section = text.split("## Configuration used")[1].split("## Limitations")[0]
    assert "Promote at MQS ≥ 75.0" in section and "Target leakage detected" in section
    assert "calibration.ece" in section                       # a must-pass check
    assert "not universal standards" in section


def test_limitations_are_always_stated(default_config):
    limits = render_markdown(build(default_config)).split("## Limitations")[1]
    for phrase in ("recommendation", "DECLARED", "cannot prove", "executes code"):
        assert phrase in limits


# -------------------------------------------------------------- writing
def test_write_report_creates_both_files(default_config, tmp_path):
    result = build(default_config)
    paths = write_report(result, tmp_path / "nested" / "run")
    assert paths["json"].name == "result.json" and paths["markdown"].name == "report.md"
    assert paths["markdown"].read_text(encoding="utf-8") == render_markdown(result)
    assert ValidationResult.model_validate_json(
        paths["json"].read_text(encoding="utf-8")) == result
    for path in paths.values():
        raw = path.read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n")


def test_writing_twice_overwrites_cleanly(default_config, tmp_path):
    write_report(build(default_config), tmp_path)
    paths = write_report(build(default_config, scores={LEAKAGE: ("FAIL", 5)}), tmp_path)
    assert "ROLLBACK" in paths["markdown"].read_text(encoding="utf-8")


# ------------------------------------------- end to end on a real run
def test_a_real_pipeline_result_renders(tiny_bundle, default_config, tmp_path):
    result = run_validation(tiny_bundle, default_config)
    text = render_markdown(result)
    assert f"## Decision: {result.decision.decision.value}" in text
    assert result.candidate.model_sha256 in text
    for rule in result.decision.triggered_rules:
        assert _cell(rule.reason) in text
    paths = write_report(result, tmp_path)
    assert paths["markdown"].exists()


def _fixture_result(name, eval_path=None, config=None):
    model = FIXTURES / ("good" if name.startswith("good") else name) / "model.joblib"
    eval_path = eval_path or DATA / "eval.csv"
    if not model.exists() or not eval_path.exists():
        pytest.skip("fixtures not generated (run scripts/make_fixtures.py)")
    return run_validation(CandidateBundle(
        candidate_id=name, model_path=model, eval_data_path=eval_path,
        metadata_path=FIXTURES / name / "metadata.json",
        train_row_hashes_path=ROOT / "data" / "train_row_hashes.txt"), config)


def test_leaky_fixture_report_names_the_leaking_feature(default_config):
    text = render_markdown(_fixture_result(
        "leaky", FIXTURES / "leaky" / "eval.csv", default_config))
    assert "## Decision: ROLLBACK" in text
    assert "A critical rule overrode the score." in text
    assert "prior_high_income_flag" in text


def test_corrupt_fixture_report_explains_the_load_failure(default_config):
    text = render_markdown(_fixture_result("corrupt", config=default_config))
    assert "## Decision: ROLLBACK" in text
    assert "cannot load model" in text and "not reliable" in text


def test_good_fixture_report_is_a_clean_promotion(default_config):
    result = _fixture_result("good", config=default_config)
    text = render_markdown(result)
    assert "## Decision: PROMOTE" in text and "No gaps: every check had evidence." in text
    assert "Suggested action" not in text
