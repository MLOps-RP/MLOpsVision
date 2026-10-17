"""The unified report's MQS arithmetic, layer overview and policy wording."""
import re

import pytest

from mqvg.config import MissingPolicy
from mqvg.reporting import render_markdown
from mqvg.schemas import Dimension

from builders import UNAVAILABLE, make_results
from test_reporting import build

CHECK = "behaviour.hidden_failure_groups"
MB = Dimension.MODEL_BEHAVIOUR


def enabled(make_config, **hidden):
    def tweak(d):
        d["checks"][CHECK]["enabled"] = True
        d["hidden_groups"].update(**hidden)
    return make_config(tweak)


def calc(text):
    return re.search(r"\*\*Calculation:\*\* (.+)", text).group(1)


def test_the_calculation_lists_every_layer_with_score_and_weight(default_config):
    result = build(default_config)
    line = calc(render_markdown(result))
    assert re.fullmatch(
        r"MQS = TVS [\d.]+ × 0\.30 \+ MBS [\d.]+ × 0\.50 \+ ORS [\d.]+ × 0\.20 = [\d.]+", line)
    assert line.endswith(f"= {result.scorecard.mqs:.1f}")


def ors_unavailable(config):
    return {cid: UNAVAILABLE for cid, c in config.checks.items()
            if c.enabled and cid.startswith("operational.")}


def test_a_dimension_without_a_score_is_left_out_and_said_so(default_config):
    result = build(default_config, scores=ors_unavailable(default_config))
    assert result.scorecard.dimensions[Dimension.OPERATIONAL_READINESS].score is None
    line = calc(render_markdown(result))
    assert "ORS" not in line.split("(")[0] and "Operational Readiness: no score, left out" in line


def test_no_mqs_is_stated_plainly(default_config):
    every = {cid: UNAVAILABLE for cid, c in default_config.checks.items() if c.enabled}
    result = build(default_config, scores=every)
    assert result.scorecard.mqs is None
    assert "no dimension had enough evidence" in render_markdown(result)


def test_penalize_shows_the_missing_dimension_as_zero(make_config):
    config = make_config(lambda d: d.update(missing_policy="penalize"))
    assert config.missing_policy == MissingPolicy.PENALIZE
    result = build(config, scores=ors_unavailable(config))
    assert "ORS 0 (no score)" in calc(render_markdown(result))


def test_the_overview_marks_subgroup_detection_off_when_the_check_is_absent(default_config):
    text = render_markdown(build(default_config))
    assert "The three layers at a glance" in text
    assert "switched off in this profile" in text
    assert "## Hidden failure groups" not in text


def test_the_overview_names_the_check_that_needs_attention(make_config):
    config = enabled(make_config, consequence="hold")
    text = render_markdown(build(config, scores={CHECK: UNAVAILABLE}))
    assert "Subgroup detection: **not assessed**" in text
    overview = text.split("### The three layers at a glance")[1].split("Subgroup detection")[0]
    assert CHECK in overview


@pytest.mark.parametrize("consequence", ["score_only", "hold", "critical"])
def test_the_report_states_the_configured_consequence(make_config, consequence):
    config = enabled(make_config, consequence=consequence)
    text = render_markdown(build(config, scores={CHECK: UNAVAILABLE}))
    assert f"`{consequence}`" in text and "never counted as a pass" in text


def test_require_assessment_is_stated_in_the_report(make_config):
    config = enabled(make_config, consequence="hold", require_assessment=True)
    assert "blocks promotion" in render_markdown(build(config, scores={CHECK: UNAVAILABLE}))


def test_the_report_never_calls_a_recommendation_a_deployment(make_config):
    config = enabled(make_config, consequence="critical")
    text = render_markdown(build(config, scores={CHECK: UNAVAILABLE}))
    assert "Nothing was deployed, promoted or rolled back" in text
