"""The validation rules engine: PROMOTE, HOLD or ROLLBACK.

Rules, in the order they are recorded:

1. CRITICAL (config `critical_rules`): a matching condition forces ROLLBACK
   whatever the MQS is.
2. SCORE: MQS below `decision.reject_below` -> ROLLBACK.
3. EVIDENCE (-> HOLD): a required check has no evidence; overall coverage is
   below `coverage.min_overall`; no MQS could be computed.
4. BLOCKING (-> HOLD): a `decision.must_pass_checks` entry is not PASS. A check
   that is UNAVAILABLE, ERROR or missing is not PASS, so absent evidence
   cannot slip through. NOT_APPLICABLE never blocks.
5. SCORE (-> HOLD): MQS below `decision.promote_at`.
6. If nothing above fired, PROMOTE.

Every rule that fires is recorded (a full audit, not just the first). The
final decision is the strongest effect: ROLLBACK > HOLD > PROMOTE.

The decision uses the policy MQS, not `mqs_lower_bound`: missing evidence
should lead to HOLD, not ROLLBACK. (Under the `penalize` policy missing
evidence already lowers the MQS itself, which is that policy's intent.)

The outcome is a recommendation. Standalone, nothing is executed.
"""
from __future__ import annotations

from typing import Sequence

from mqvg.config import CriticalRule, GateConfig
from mqvg.schemas import (
    ActionMode, CheckResult, CheckStatus, Decision, DecisionRecord, RuleKind,
    ScoreCard, TriggeredRule,
)

NO_EVIDENCE = {CheckStatus.UNAVAILABLE, CheckStatus.ERROR}


def _numeric(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _critical_rules(rules: Sequence[CriticalRule],
                    results: dict[str, CheckResult]) -> list[TriggeredRule]:
    triggered = []
    for rule in rules:
        cond = rule.condition
        result = results.get(cond.check)
        # a check with no result is treated as UNAVAILABLE, as in scoring
        status = CheckStatus.UNAVAILABLE if result is None else result.status
        if cond.status_in:
            if status not in cond.status_in:
                continue
            reason = (f"{rule.name}: check '{cond.check}' is {status.value}"
                      + (f" ({result.message})" if result else "") + ".")
            observed, threshold = status.value, [s.value for s in cond.status_in]
        else:
            value = None if result is None or not result.has_evidence \
                else _numeric(result.raw_value)
            if value is None:
                continue                   # no measurable value: cannot match
            if cond.value_above is not None and value > cond.value_above:
                threshold = {"value_above": cond.value_above}
                relation = f"above {cond.value_above:g}"
            elif cond.value_below is not None and value < cond.value_below:
                threshold = {"value_below": cond.value_below}
                relation = f"below {cond.value_below:g}"
            else:
                continue
            reason = (f"{rule.name}: '{cond.check}' is {value:.4g}, "
                      f"{relation}.")
            observed = value
        triggered.append(TriggeredRule(
            rule_id=rule.id, name=rule.name, kind=RuleKind.CRITICAL,
            effect=Decision.ROLLBACK, reason=reason, observed=observed,
            threshold=threshold, check_ids=[cond.check]))
    return triggered


def _mqs_text(card: ScoreCard) -> str:
    if card.mqs is None:
        return "no MQS could be computed"
    text = f"MQS {card.mqs:.1f}"
    if card.mqs_lower_bound is not None and card.mqs - card.mqs_lower_bound > 0.05:
        text += f" (at worst {card.mqs_lower_bound:.1f} if missing checks had failed)"
    return text


def _summary(decision: Decision, rules: list[TriggeredRule], card: ScoreCard,
             config: GateConfig, overridden: bool) -> str:
    mqs = _mqs_text(card)
    if decision == Decision.PROMOTE:
        note = " The score is provisional." if card.provisional else ""
        return (f"PROMOTE recommended: {mqs} meets the promote threshold of "
                f"{config.decision.promote_at:g}; no critical rule fired, "
                f"required evidence is present and every must-pass check "
                f"passed.{note}")
    effect_rules = [r for r in rules if r.effect == decision]
    names = "; ".join(r.name for r in effect_rules)
    if decision == Decision.ROLLBACK:
        why = ("a critical rule overrides the score" if overridden
               else "the score is below the rejection threshold"
               if any(r.kind == RuleKind.SCORE for r in effect_rules)
               else "a critical rule fired")
        return f"ROLLBACK recommended ({mqs}; {why}): {names}."
    return (f"HOLD: the model needs review before promotion ({mqs}). "
            f"{len(effect_rules)} issue(s): {names}.")


def evaluate_rules(checks: Sequence[CheckResult], scorecard: ScoreCard,
                   config: GateConfig) -> DecisionRecord:
    """Decide PROMOTE, HOLD or ROLLBACK from the checks and the scorecard."""
    results = {c.id: c for c in checks}
    if len(results) != len(checks):
        raise ValueError("duplicate check results")
    rules: list[TriggeredRule] = []
    mqs = scorecard.mqs
    decision_cfg = config.decision

    # 1. critical rules
    critical = _critical_rules(config.critical_rules, results)
    rules += critical

    # 2. score below the rejection threshold
    score_rollback = mqs is not None and mqs < decision_cfg.reject_below
    if score_rollback:
        rules.append(TriggeredRule(
            rule_id="mqs_below_reject_threshold",
            name="MQS below the rejection threshold", kind=RuleKind.SCORE,
            effect=Decision.ROLLBACK,
            reason=f"MQS {mqs:.1f} is below the rejection threshold of "
                   f"{decision_cfg.reject_below:g}.",
            observed=mqs, threshold=decision_cfg.reject_below))

    # 3. evidence
    required_missing = sorted(
        i for i, c in config.checks.items()
        if c.enabled and c.required
        and (i not in results or results[i].status in NO_EVIDENCE))
    if required_missing:
        rules.append(TriggeredRule(
            rule_id="required_evidence_missing",
            name="Required evidence is missing", kind=RuleKind.EVIDENCE,
            effect=Decision.HOLD,
            reason="Required checks have no evidence: "
                   + ", ".join(required_missing) + ".",
            observed=required_missing,
            check_ids=[i for i in required_missing if i in results]))
    if scorecard.overall_coverage < config.coverage.min_overall:
        rules.append(TriggeredRule(
            rule_id="overall_coverage_below_minimum",
            name="Evidence coverage is too low", kind=RuleKind.EVIDENCE,
            effect=Decision.HOLD,
            reason=f"Overall evidence coverage {scorecard.overall_coverage:.0%} "
                   f"is below the required {config.coverage.min_overall:.0%}.",
            observed=scorecard.overall_coverage,
            threshold=config.coverage.min_overall))
    if mqs is None:
        rules.append(TriggeredRule(
            rule_id="mqs_unavailable", name="No MQS could be computed",
            kind=RuleKind.EVIDENCE, effect=Decision.HOLD,
            reason="No dimension had enough evidence to be scored."))

    # 4. must-pass checks
    for check_id in decision_cfg.must_pass_checks:
        result = results.get(check_id)
        status = None if result is None else result.status
        if status in (CheckStatus.PASS, CheckStatus.NOT_APPLICABLE):
            continue
        shown = "has no result" if status is None else f"is {status.value}"
        rules.append(TriggeredRule(
            rule_id=f"must_pass:{check_id}",
            name=f"Must-pass check '{check_id}' did not pass",
            kind=RuleKind.BLOCKING, effect=Decision.HOLD,
            reason=f"Check '{check_id}' {shown}; it must PASS before promotion"
                   + (f" ({result.message})" if result else "") + ".",
            observed=None if status is None else status.value,
            threshold="PASS", check_ids=[check_id] if result else []))

    # 5. score below the promotion threshold
    in_hold_band = (mqs is not None and not score_rollback
                    and mqs < decision_cfg.promote_at)
    if in_hold_band:
        rules.append(TriggeredRule(
            rule_id="mqs_below_promote_threshold",
            name="MQS below the promotion threshold", kind=RuleKind.SCORE,
            effect=Decision.HOLD,
            reason=f"MQS {mqs:.1f} is between the rejection threshold "
                   f"({decision_cfg.reject_below:g}) and the promotion "
                   f"threshold ({decision_cfg.promote_at:g}).",
            observed=mqs, threshold=decision_cfg.promote_at))

    # decision: the strongest effect wins
    effects = {r.effect for r in rules}
    if Decision.ROLLBACK in effects:
        decision = Decision.ROLLBACK
    elif Decision.HOLD in effects:
        decision = Decision.HOLD
    else:
        decision = Decision.PROMOTE
        rules.append(TriggeredRule(
            rule_id="mqs_meets_promote_threshold",
            name="MQS meets the promotion threshold", kind=RuleKind.SCORE,
            effect=Decision.PROMOTE,
            reason=f"MQS {mqs:.1f} is at or above the promotion threshold of "
                   f"{decision_cfg.promote_at:g}.",
            observed=mqs, threshold=decision_cfg.promote_at))

    # a critical rule only "overrides" the MQS if the score alone would not
    # have produced a ROLLBACK
    overridden = (decision == Decision.ROLLBACK and bool(critical)
                  and not score_rollback)
    return DecisionRecord(
        decision=decision, action_mode=ActionMode.RECOMMENDATION, mqs=mqs,
        overridden_by_critical=overridden, triggered_rules=rules,
        summary=_summary(decision, rules, scorecard, config, overridden))
