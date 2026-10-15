import numpy as np
import pytest
from pydantic import ValidationError

from mqvg.catalogue import CHECKS_BY_ID
from mqvg.checks.behaviour import run_behaviour_checks
from mqvg.checks.operational import run_operational_checks
from mqvg.checks.training_validity import run_training_validity_checks
from mqvg.loader import load_candidate
from mqvg.schemas import (
    CandidateBundle, CheckResult, CheckStatus, CheckTreatment, Dimension,
    EvidenceSource, MissingPolicy, ScoreCard,
)
from mqvg.scoring import compute_scorecard

from conftest import CONFIGS, DATA, FIXTURES

from builders import (  # noqa: E402
    ERROR, NA, UNAVAILABLE, enabled_ids, make_results,
)

TV, MB, OR = (Dimension.TRAINING_VALIDITY, Dimension.MODEL_BEHAVIOUR,
              Dimension.OPERATIONAL_READINESS)


def score(config, **kwargs):
    return compute_scorecard(make_results(config, **kwargs), config)


@pytest.fixture
def penalize_config(make_config):
    return make_config(lambda d: d.update(missing_policy="penalize"))


# Behaviour checks (default profile) and their weights; robustness is disabled.
MB_WEIGHTS = {"behaviour.f1": 2, "behaviour.roc_auc": 2, "behaviour.precision": 1,
              "behaviour.recall": 1, "behaviour.subgroup_f1_gap": 1,
              "fairness.selection_rate_ratio": 2, "fairness.equalized_odds": 1,
              "calibration.ece": 1.5}
MB_SCORES = {"behaviour.f1": 80, "behaviour.roc_auc": 90, "behaviour.precision": 70,
             "behaviour.recall": 60, "behaviour.subgroup_f1_gap": 50,
             "fairness.selection_rate_ratio": 40, "fairness.equalized_odds": 30,
             "calibration.ece": 100}
# weighted sum = 160+180+70+60+50+80+30+150 = 780 over total weight 11.5


# ------------------------------------------------------- the plan's example
def test_plan_example_mqs_is_78(default_config):
    card = score(default_config, per_dimension={TV: 80, MB: 72, OR: 90})
    assert card.dimensions[TV].score == pytest.approx(80)
    assert card.dimensions[MB].score == pytest.approx(72)
    assert card.dimensions[OR].score == pytest.approx(90)
    # 80*0.30 + 72*0.50 + 90*0.20
    assert card.mqs == pytest.approx(78.0)
    assert card.overall_coverage == 1.0 and not card.provisional
    assert card.mqs_lower_bound == pytest.approx(78.0)
    assert card.effective_weights == {TV: 0.3, MB: 0.5, OR: 0.2}


def test_extremes(default_config):
    assert score(default_config, base=100).mqs == pytest.approx(100)
    assert score(default_config, base=0).mqs == pytest.approx(0)


# ------------------------------------------------ dimension score (hand-computed)
def test_dimension_is_the_weighted_average_of_its_checks(default_config):
    card = score(default_config, scores=MB_SCORES)
    assert card.dimensions[MB].score == pytest.approx(780 / 11.5)
    assert card.dimensions[MB].coverage == 1.0


def test_contributions_explain_the_score(default_config):
    dim = score(default_config, scores=MB_SCORES).dimensions[MB]
    assert {c.check_id for c in dim.contributions} == set(MB_WEIGHTS)
    assert sum(c.points for c in dim.contributions) == pytest.approx(dim.score)
    f1 = next(c for c in dim.contributions if c.check_id == "behaviour.f1")
    assert f1.points == pytest.approx(2 * 80 / 11.5)
    assert f1.treatment == CheckTreatment.COUNTED and f1.status == CheckStatus.PASS


def test_check_weights_matter(default_config, make_config):
    heavier = make_config(lambda d: d["checks"]["behaviour.f1"].update(weight=10))
    base = score(default_config, scores=MB_SCORES).dimensions[MB].score
    changed = score(heavier, scores=MB_SCORES).dimensions[MB].score
    # f1 (score 80) is above the 67.8 average, so weighting it more raises MBS
    assert changed > base
    assert changed == pytest.approx((780 + 8 * 80) / 19.5)


# ------------------------------------------------------- missing evidence
UNAVAIL_PAIR = {"fairness.selection_rate_ratio": UNAVAILABLE,
                "fairness.equalized_odds": UNAVAILABLE}


@pytest.mark.parametrize("missing", [UNAVAILABLE, ERROR])
def test_exclude_policy_leaves_missing_checks_out_and_lowers_coverage(
        default_config, missing):
    scores = dict(MB_SCORES, **{k: missing for k in UNAVAIL_PAIR})
    dim = score(default_config, scores=scores).dimensions[MB]
    assert dim.score == pytest.approx(670 / 8.5)             # (780-80-30) / (11.5-3)
    assert dim.coverage == pytest.approx(8.5 / 11.5)
    assert dim.provisional and "Provisional" in dim.note
    assert (dim.available_checks, dim.total_checks) == (6, 8)
    treatments = {c.check_id: c.treatment for c in dim.contributions}
    assert treatments["fairness.equalized_odds"] == CheckTreatment.EXCLUDED_UNAVAILABLE
    assert sum(c.points for c in dim.contributions) == pytest.approx(dim.score)


def test_missing_checks_are_never_counted_as_passes(default_config):
    """Unavailable checks must not behave like a score of 100."""
    scores = dict(MB_SCORES, **UNAVAIL_PAIR)
    as_unavailable = score(default_config, scores=scores).dimensions[MB].score
    as_pass = score(default_config, scores=dict(
        MB_SCORES, **{k: 100 for k in UNAVAIL_PAIR})).dimensions[MB].score
    assert as_unavailable != pytest.approx(as_pass)


def test_penalize_policy_scores_missing_checks_as_zero(penalize_config):
    dim = score(penalize_config, scores=dict(MB_SCORES, **UNAVAIL_PAIR)).dimensions[MB]
    assert dim.score == pytest.approx(670 / 11.5)
    assert dim.coverage == pytest.approx(8.5 / 11.5)         # coverage is still honest
    treatments = {c.check_id: c.treatment for c in dim.contributions}
    assert treatments["fairness.equalized_odds"] == CheckTreatment.PENALIZED_AS_ZERO
    assert sum(c.points for c in dim.contributions) == pytest.approx(dim.score)


def test_not_applicable_checks_cost_nothing(default_config):
    dim = score(default_config,
                scores=dict(MB_SCORES, **{"calibration.ece": NA})).dimensions[MB]
    assert dim.score == pytest.approx(630 / 10)              # ece (weight 1.5) removed
    assert dim.coverage == 1.0 and not dim.provisional
    assert dim.total_checks == 7
    ece = next(c for c in dim.contributions if c.check_id == "calibration.ece")
    assert ece.treatment == CheckTreatment.NOT_APPLICABLE and ece.points == 0


def test_not_applicable_does_not_help_under_penalize_either(penalize_config):
    dim = score(penalize_config,
                scores=dict(MB_SCORES, **{"calibration.ece": NA})).dimensions[MB]
    assert dim.score == pytest.approx(630 / 10)


def test_a_check_with_no_result_counts_as_unavailable(default_config):
    results = [r for r in make_results(default_config, scores=MB_SCORES)
               if r.id != "behaviour.recall"]
    dim = compute_scorecard(results, default_config).dimensions[MB]
    assert dim.coverage == pytest.approx(10.5 / 11.5)
    recall = next(c for c in dim.contributions if c.check_id == "behaviour.recall")
    assert recall.status is None
    assert recall.treatment == CheckTreatment.EXCLUDED_UNAVAILABLE


def test_low_coverage_dimension_gets_no_score(default_config):
    scores = {k: UNAVAILABLE for k in MB_SCORES}
    scores["behaviour.f1"] = 80                              # coverage 2/11.5 = 17%
    dim = score(default_config, scores=scores).dimensions[MB]
    assert dim.score is None and dim.provisional
    assert "below the minimum" in dim.note


def test_a_dimension_with_nothing_applicable_has_no_score(default_config):
    dim = score(default_config,
                scores={k: NA for k in MB_WEIGHTS}).dimensions[MB]
    assert dim.score is None and "No applicable check" in dim.note
    assert dim.coverage == 0.0


# -------------------------------------------------------------------- MQS
def test_exclude_policy_renormalizes_over_scored_dimensions(default_config):
    card = score(default_config, per_dimension={TV: 80, MB: UNAVAILABLE, OR: 90})
    assert card.dimensions[MB].score is None
    assert card.effective_weights == {TV: pytest.approx(0.6), MB: 0.0,
                                      OR: pytest.approx(0.4)}
    assert card.mqs == pytest.approx(0.6 * 80 + 0.4 * 90)    # 84
    assert card.provisional
    assert card.mqs_lower_bound == pytest.approx(0.3 * 80 + 0.2 * 90)   # 42


def test_penalize_policy_counts_an_unscored_dimension_as_zero(penalize_config):
    card = score(penalize_config, per_dimension={TV: 80, MB: UNAVAILABLE, OR: 90})
    assert card.mqs == pytest.approx(42.0)
    assert card.mqs == card.mqs_lower_bound
    assert card.effective_weights is None
    assert card.missing_policy == MissingPolicy.PENALIZE


def test_lower_bound_shows_the_optimism_of_excluding_missing_checks(default_config):
    card = score(default_config, scores=dict(MB_SCORES, **UNAVAIL_PAIR))
    assert card.mqs_lower_bound < card.mqs
    full = score(default_config, scores=MB_SCORES)
    # omitting two weak checks RAISES the excluded-policy MBS and hence the MQS
    assert card.dimensions[MB].score > full.dimensions[MB].score
    assert card.mqs > full.mqs


def test_nothing_available_gives_no_mqs_under_exclude(default_config):
    card = score(default_config, base=UNAVAILABLE)
    assert card.mqs is None and card.mqs_lower_bound == 0.0
    assert all(d.score is None for d in card.dimensions.values())
    assert card.overall_coverage == 0.0 and card.provisional


def test_nothing_available_scores_zero_under_penalize(penalize_config):
    card = score(penalize_config, base=UNAVAILABLE)
    assert card.mqs == 0.0 and card.provisional


def test_overall_coverage_is_weighted_by_dimension_weight(default_config):
    scores = dict(MB_SCORES, **UNAVAIL_PAIR)
    card = score(default_config, scores=scores)
    mb_coverage = 8.5 / 11.5
    assert card.overall_coverage == pytest.approx(0.3 + 0.5 * mb_coverage + 0.2)


def test_dimension_weights_drive_the_mqs(make_config):
    def tweak(d):
        d["weights"] = {"training_validity": 0.6, "model_behaviour": 0.3,
                        "operational_readiness": 0.1}
    card = score(make_config(tweak), per_dimension={TV: 100, MB: 50, OR: 0})
    assert card.mqs == pytest.approx(0.6 * 100 + 0.3 * 50)


def test_zero_weight_dimension_does_not_affect_the_mqs(make_config):
    def tweak(d):
        d["weights"] = {"training_validity": 0.5, "model_behaviour": 0.5,
                        "operational_readiness": 0.0}
    card = score(make_config(tweak), per_dimension={TV: 80, MB: 60, OR: 0})
    assert card.mqs == pytest.approx(70)


def test_zero_weight_on_every_scored_dimension_gives_no_mqs(make_config):
    def tweak(d):
        d["weights"] = {"training_validity": 0.0, "model_behaviour": 1.0,
                        "operational_readiness": 0.0}
    card = score(make_config(tweak),
                 per_dimension={TV: 80, MB: UNAVAILABLE, OR: 90})
    assert card.mqs is None


# ----------------------------------------------------------- input checks
def test_duplicate_results_are_rejected(default_config):
    results = make_results(default_config)
    with pytest.raises(ValueError, match="duplicate result"):
        compute_scorecard(results + results[:1], default_config)


def test_results_for_disabled_checks_are_rejected(default_config):
    extra = CheckResult(
        id="robustness.noise_degradation", dimension=MB, name="x",
        status=CheckStatus.PASS, score=50, evidence_source=EvidenceSource.MEASURED,
        message="m")
    with pytest.raises(ValueError, match="unknown or disabled"):
        compute_scorecard(make_results(default_config) + [extra], default_config)


def test_results_for_unknown_checks_are_rejected(default_config):
    extra = CheckResult(
        id="made.up", dimension=MB, name="x", status=CheckStatus.PASS, score=50,
        evidence_source=EvidenceSource.MEASURED, message="m")
    with pytest.raises(ValueError, match="unknown or disabled"):
        compute_scorecard(make_results(default_config) + [extra], default_config)


# -------------------------------------------------------------- properties
def test_scorecard_is_always_valid_and_internally_consistent(default_config,
                                                             penalize_config):
    rng = np.random.default_rng(0)
    for config in (default_config, penalize_config):
        ids = enabled_ids(config)
        for _ in range(150):
            scores = {}
            for check_id in ids:
                roll = rng.random()
                scores[check_id] = (UNAVAILABLE if roll < 0.2 else ERROR if roll < 0.3
                                    else NA if roll < 0.4 else float(rng.uniform(0, 100)))
            card = compute_scorecard(make_results(config, scores=scores), config)
            ScoreCard.model_validate(card.model_dump())           # re-runs validators
            for dim in card.dimensions.values():
                if dim.score is not None:
                    assert sum(c.points for c in dim.contributions) == pytest.approx(
                        dim.score)
            if card.mqs is not None:
                assert 0 <= card.mqs_lower_bound <= card.mqs + 1e-9 <= 100 + 1e-9


def test_better_check_scores_never_lower_the_mqs(default_config):
    rng = np.random.default_rng(1)
    ids = enabled_ids(default_config)
    for _ in range(60):
        scores = {i: float(rng.uniform(0, 90)) for i in ids}
        before = compute_scorecard(make_results(default_config, scores=scores),
                                   default_config)
        target = ids[rng.integers(len(ids))]
        better = dict(scores, **{target: scores[target] + 10})
        after = compute_scorecard(make_results(default_config, scores=better),
                                  default_config)
        assert after.mqs >= before.mqs - 1e-9


def test_scorecard_validators_reject_inconsistent_values(default_config):
    good = score(default_config)
    data = good.model_dump()
    data["mqs"], data["mqs_lower_bound"] = 50.0, 60.0
    with pytest.raises(ValidationError, match="mqs_lower_bound"):
        ScoreCard.model_validate(data)


def test_strict_profile_scores_too():
    from mqvg.config import load_config
    strict = load_config(CONFIGS / "strict.yaml")
    card = compute_scorecard(make_results(strict), strict)
    assert card.mqs == pytest.approx(100) and card.missing_policy == MissingPolicy.PENALIZE


# ------------------------------------------------ generated fixture models
FIXTURE_EVAL = {"good_no_sensitive": FIXTURES / "good_no_sensitive" / "eval.csv",
                "leaky": FIXTURES / "leaky" / "eval.csv"}


@pytest.fixture(scope="module")
def cards(default_config_module):
    config = default_config_module
    out = {}
    for name in ("good", "leaky", "biased", "corrupt", "good_no_sensitive"):
        model = FIXTURES / ("good" if name.startswith("good") else name) / "model.joblib"
        eval_path = FIXTURE_EVAL.get(name, DATA / "eval.csv")
        if not model.exists() or not eval_path.exists():
            pytest.skip("fixtures not generated (run scripts/make_fixtures.py)")
        c = load_candidate(CandidateBundle(
            model_path=model, eval_data_path=eval_path,
            metadata_path=FIXTURES / name / "metadata.json"))
        results = (run_training_validity_checks(c, config)
                   + run_behaviour_checks(c, config) + run_operational_checks(c, config))
        out[name] = compute_scorecard(results, config)
    return out


@pytest.fixture(scope="module")
def default_config_module():
    from mqvg.config import load_config
    return load_config(CONFIGS / "default.yaml")


def test_good_fixture_has_full_coverage_and_a_high_score(cards):
    good = cards["good"]
    assert good.overall_coverage == 1.0 and not good.provisional
    assert good.mqs > 80 and good.mqs_lower_bound == pytest.approx(good.mqs)


def test_leaky_model_looks_great_on_behaviour_but_loses_training_validity(cards):
    assert cards["leaky"].dimensions[MB].score > cards["good"].dimensions[MB].score
    assert (cards["leaky"].dimensions[TV].score
            < cards["good"].dimensions[TV].score - 10)


def test_biased_model_loses_behaviour_score_only(cards):
    assert cards["biased"].dimensions[MB].score < cards["good"].dimensions[MB].score - 10
    assert cards["biased"].dimensions[TV].score == pytest.approx(
        cards["good"].dimensions[TV].score, abs=5)


def test_corrupt_model_has_a_wide_mqs_range_and_is_provisional(cards):
    corrupt = cards["corrupt"]
    assert corrupt.provisional and corrupt.overall_coverage < 0.5
    assert corrupt.dimensions[OR].score is None
    # the optimistic MQS hides the failure; the lower bound exposes it
    assert corrupt.mqs - corrupt.mqs_lower_bound > 50


def test_omitting_fairness_evidence_raises_the_excluded_score(cards):
    """Documents the optimism of 'exclude missing': a model can score higher by
    omitting an unfavourable check. The lower bound and the PROMOTE block on
    missing required evidence are the safeguards."""
    missing, full = cards["good_no_sensitive"], cards["good"]
    assert missing.provisional and missing.overall_coverage < 1
    assert missing.mqs > full.mqs
    assert missing.mqs_lower_bound < full.mqs
