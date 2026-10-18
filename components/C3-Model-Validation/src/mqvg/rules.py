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

Hidden Failure Group Detection has its own explicit consequence setting
(`hidden_groups.consequence`), independent of the lists above:

* `score_only` (default): the check only affects the score; no rule is added;
* `hold`: a confirmed group (WARN or FAIL) adds a BLOCKING rule -> HOLD;
* `critical`: as `hold`, and a FAIL (excess above the check's `fail_at`) also adds
  a CRITICAL rule -> ROLLBACK recommendation.

`hidden_groups.require_assessment` additionally blocks PROMOTE when the check
is UNAVAILABLE or ERROR. A subgroup alone never forces a ROLLBACK unless the
policy is explicitly `critical`.
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


HIDDEN_CHECK = "behaviour.hidden_failure_groups"


def _hidden_group_rules(results: dict[str, CheckResult],
                        config: GateConfig) -> list[TriggeredRule]:
    """Rules for the configured consequence of Hidden Failure Group Detection."""
    policy = config.hidden_groups
    check = results.get(HIDDEN_CHECK)
    if check is None or not config.checks[HIDDEN_CHECK].enabled:
        return []
    rules: list[TriggeredRule] = []
    if check.status in NO_EVIDENCE:
        if policy.require_assessment:
            rules.append(TriggeredRule(
                rule_id="hidden_failure_groups_not_assessed",
                name="Hidden failure groups could not be assessed",
                kind=RuleKind.EVIDENCE, effect=Decision.HOLD,
                reason=("hidden_groups.require_assessment is set and the check is "
                        f"{check.status.value}, which is not a pass: {check.message}"),
                observed=check.status.value, threshold="assessed",
                check_ids=[HIDDEN_CHECK]))
        return rules
    if policy.consequence == "score_only" or check.status not in (
            CheckStatus.WARN, CheckStatus.FAIL):
        return rules
    rules.append(TriggeredRule(
        rule_id="hidden_failure_groups_hold", name="Hidden failure group found",
        kind=RuleKind.BLOCKING, effect=Decision.HOLD,
        reason=(f"{check.message} Policy hidden_groups.consequence is "
                f"'{policy.consequence}': a confirmed group blocks promotion."),
        observed=check.raw_value, threshold="no confirmed group",
        check_ids=[HIDDEN_CHECK]))
    if policy.consequence == "critical" and check.status == CheckStatus.FAIL:
        fail_at = config.checks[HIDDEN_CHECK].thresholds.fail_at
        rules.append(TriggeredRule(
            rule_id="hidden_failure_groups_critical",
            name="Severe hidden failure group", kind=RuleKind.CRITICAL,
            effect=Decision.ROLLBACK,
            reason=(f"The largest excess error rate of a confirmed group is "
                    f"{check.raw_value:.1%}, above the check's fail threshold of "
                    f"{fail_at:.1%}. Policy hidden_groups.consequence is 'critical', "
                    "so this is a critical failure. This is an association found in "
                    "the evaluation data, not a proven cause."),
            observed=check.raw_value, threshold={"fail_at": fail_at},
            check_ids=[HIDDEN_CHECK]))
    return rules


def _mqs_text(card: ScoreCard) -> str:
    if card.mqs is None:
        return "no MQS could be computed"
    text = f"MQS {card.mqs:.1f}"
    notes = []
    if not card.mqs_sufficient_evidence:
        notes.append(f"not reliable: {card.insufficient_evidence_reason}")
    if card.mqs_lower_bound is not None and card.mqs - card.mqs_lower_bound > 0.05:
        notes.append(f"at worst {card.mqs_lower_bound:.1f} if missing checks had failed")
    return text + (f" ({'; '.join(notes)})" if notes else "")


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
        return f"ROLLBACK recommended: {why}. {mqs}. Reason(s): {names}."
    return (f"HOLD: the model needs review before promotion. {mqs}. "
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
    hidden_rules = _hidden_group_rules(results, config)
    critical += [r for r in hidden_rules if r.kind == RuleKind.CRITICAL]
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
    unscored = [d.value for d, dim in scorecard.dimensions.items()
                if dim.score is None]
    if (mqs is not None and unscored
            and scorecard.overall_coverage >= config.coverage.min_overall):
        rules.append(TriggeredRule(
            rule_id="mqs_insufficient_evidence",
            name="MQS rests on insufficient evidence", kind=RuleKind.EVIDENCE,
            effect=Decision.HOLD,
            reason="These dimensions have too little evidence to be scored, so "
                   "the MQS is not reliable: " + ", ".join(unscored) + ".",
            observed=unscored))
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

    # 4b. the configured consequence of hidden failure groups (non-critical part)
    rules += [r for r in hidden_rules if r.kind != RuleKind.CRITICAL]

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
