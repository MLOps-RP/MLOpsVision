"""The sensitivity-analysis helper must reproduce the gate before it perturbs it."""
import importlib.util
import sys

import pytest

from mqvg.config import Thresholds, load_config
from mqvg.pipeline import run_validation
from mqvg.schemas import CheckStatus

from conftest import CONFIGS, ROOT


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location(
        "sensitivity_analysis", ROOT / "scripts" / "sensitivity_analysis.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sensitivity_analysis"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def good(script):
    config = load_config(CONFIGS / "default.yaml")
    return config, run_validation(script.bundle_for("good"), config)


def test_rescoring_with_the_unchanged_config_reproduces_the_gate(script, good):
    config, result = good
    again = script.rescore(result.checks, config)
    for before, after in zip(result.checks, again):
        assert after.status == before.status, before.id
        if before.score is not None:
            assert after.score == pytest.approx(before.score, abs=1e-9), before.id
    decision, mqs = script.judge(result.checks, config)
    assert decision == result.decision.decision.value
    assert mqs == pytest.approx(result.scorecard.mqs, abs=1e-9)


def test_scaling_keeps_the_threshold_order_in_both_directions(script):
    higher = Thresholds(target=0.80, pass_at=0.60, fail_at=0.45, floor=0.0)
    lower = Thresholds(target=0.01, pass_at=0.05, fail_at=0.10, floor=0.25)
    for th, ordered in ((higher, lambda t: t.floor <= t.fail_at <= t.pass_at <= t.target),
                        (lower, lambda t: t.floor >= t.fail_at >= t.pass_at >= t.target)):
        for factor in (0.8, 0.9, 1.1, 1.2, 5.0):
            moved = script.scaled(th, factor)
            assert ordered(moved), (th, factor, moved)
            assert (moved.target, moved.floor) == (th.target, th.floor)
    assert script.scaled(higher, 0.5).pass_at == pytest.approx(0.70)    # halfway to target


def test_a_stricter_threshold_can_only_make_a_check_worse(script, good):
    config, result = good
    strict = script.rescore(result.checks, script.with_thresholds(config, 0.8))
    order = {CheckStatus.PASS: 0, CheckStatus.WARN: 1, CheckStatus.FAIL: 2}
    for before, after in zip(result.checks, strict):
        if before.status in order:
            assert order[after.status] >= order[before.status], before.id


def test_the_perturbation_grid_is_valid_and_complete(script, good):
    config, _ = good
    items = list(script.perturbations(config))
    families = {family for family, _, _ in items}
    assert families == {"weights", "thresholds", "per-check", "bands", "missing"}
    weights = [c.weights for f, _, c in items if f == "weights"]
    assert len(weights) == 36 and all(
        abs(w.training_validity + w.model_behaviour + w.operational_readiness - 1) < 1e-9
        for w in weights)
    assert all(c.decision.reject_below < c.decision.promote_at for f, _, c in items if f == "bands")
    # the original configuration was not mutated
    assert config.weights.model_behaviour == 0.50 and config.decision.promote_at == 75


def test_binary_and_unavailable_results_are_left_alone(script, good):
    config, result = good
    harsh = script.rescore(result.checks, script.with_thresholds(config, 0.5))
    for before, after in zip(result.checks, harsh):
        if before.direction == "binary":
            assert (after.status, after.score) == (before.status, before.score)


def test_the_report_is_written_and_covers_every_case(script, tmp_path):
    out = tmp_path / "sensitivity.md"
    assert script.main(["--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    for name, _ in script.CASES:
        assert f"| {name} |" in text
    assert "does not show that the thresholds are correct" in text
