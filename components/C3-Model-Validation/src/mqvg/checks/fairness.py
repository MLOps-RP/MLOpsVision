"""Group fairness and subgroup performance checks.

Groups are the values of the declared sensitive columns. Metrics are computed
here (not with Fairlearn at run time) so that small groups can be excluded
and undefined rates handled explicitly; the tests compare every number with
Fairlearn as an independent reference.

Rules applied throughout:
* a group smaller than `evaluation.min_group_size` is excluded and listed,
  never silently included (tiny groups give wild rates);
* rows with a missing group value are left out of that column's analysis;
* a rate that is undefined for a group (for example the true-positive rate of
  a group with no positive examples) leaves that group out of that metric;
* if nothing can be evaluated the check is UNAVAILABLE, never a pass.

Limitation: only the declared sensitive columns are examined, one at a time.
Intersections and non-sensitive slices belong to hidden-failure discovery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from mqvg.checks.base import (
    CheckContext, CheckFunction, guarded, scored_result, unavailable_result,
)
from mqvg.checks.predictions import EvalPredictions, get_predictions
from mqvg.schemas import CheckResult

FAIRNESS_CHECKS: dict[str, CheckFunction] = {}


def _register(check_id: str) -> Callable[[CheckFunction], CheckFunction]:
    def decorator(func: CheckFunction) -> CheckFunction:
        FAIRNESS_CHECKS[check_id] = guarded(check_id)(func)
        return FAIRNESS_CHECKS[check_id]
    return decorator


# --------------------------------------------------------------------------
# Group statistics
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GroupStats:
    n: int
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def selection_rate(self) -> float:
        return (self.tp + self.fp) / self.n

    @property
    def tpr(self) -> float | None:
        positives = self.tp + self.fn
        return self.tp / positives if positives else None

    @property
    def fpr(self) -> float | None:
        negatives = self.fp + self.tn
        return self.fp / negatives if negatives else None

    @property
    def f1(self) -> float | None:
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else None

    def as_dict(self) -> dict:
        return {"n": self.n, "selection_rate": self.selection_rate,
                "tpr": self.tpr, "fpr": self.fpr, "f1": self.f1,
                "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}


@dataclass
class ColumnAnalysis:
    column: str
    groups: dict[str, GroupStats] = field(default_factory=dict)
    excluded_small: dict[str, int] = field(default_factory=dict)
    rows_missing_group: int = 0

    @property
    def evaluable(self) -> bool:
        return len(self.groups) >= 2


@dataclass
class GroupAnalysis:
    columns: dict[str, ColumnAnalysis]
    skipped: dict[str, str]   # column -> why it could not be examined
    min_group_size: int

    @property
    def evaluable(self) -> dict[str, ColumnAnalysis]:
        return {c: a for c, a in self.columns.items() if a.evaluable}


def analyse_column(p: EvalPredictions, values, column: str,
                   min_group_size: int) -> ColumnAnalysis:
    """Per-group confusion counts for one sensitive column."""
    present = values.notna().to_numpy()
    labels = values.astype(str).to_numpy()
    analysis = ColumnAnalysis(column, rows_missing_group=int((~present).sum()))
    for group in sorted(set(labels[present])):
        mask = present & (labels == group)
        n = int(mask.sum())
        if n < min_group_size:
            analysis.excluded_small[group] = n
            continue
        t, y = p.y_true[mask], p.y_pred[mask]
        analysis.groups[group] = GroupStats(
            n=n, tp=int(np.sum(t & y)), fp=int(np.sum(~t & y)),
            fn=int(np.sum(t & ~y)), tn=int(np.sum(~t & ~y)))
    return analysis


def get_group_analysis(ctx: CheckContext) -> tuple[GroupAnalysis | None, str | None]:
    """Group statistics for every declared sensitive column, computed once."""
    cache = ctx.candidate.cache
    if "group_analysis" in cache:
        return cache["group_analysis"]
    cache["group_analysis"] = _build_analysis(ctx)
    return cache["group_analysis"]


def _build_analysis(ctx: CheckContext) -> tuple[GroupAnalysis | None, str | None]:
    predictions, reason = get_predictions(ctx.candidate)
    if predictions is None:
        return None, reason
    declared = ctx.candidate.sensitive_columns
    if not declared:
        return None, ("No sensitive columns are declared; set sensitive_columns "
                      "in the metadata or bundle to evaluate fairness.")
    frame = ctx.candidate.eval_df
    min_size = ctx.config.evaluation.min_group_size
    columns: dict[str, ColumnAnalysis] = {}
    skipped: dict[str, str] = {}
    for column in declared:
        if column not in frame.columns:
            skipped[column] = "column is not in the evaluation data"
            continue
        values = frame.loc[predictions.row_mask, column].reset_index(drop=True)
        analysis = analyse_column(predictions, values, column, min_size)
        columns[column] = analysis
        if not analysis.evaluable:
            skipped[column] = (
                f"fewer than two groups have at least {min_size} rows "
                f"(excluded: {analysis.excluded_small or 'none'}; "
                f"usable: {sorted(analysis.groups)})")
    if not any(a.evaluable for a in columns.values()):
        return None, ("No sensitive column could be evaluated: "
                      + "; ".join(f"'{c}': {why}" for c, why in skipped.items())
                      + ".")
    return GroupAnalysis(columns, skipped, min_size), None


def _common_details(analysis: GroupAnalysis) -> dict:
    return {
        "min_group_size": analysis.min_group_size,
        "evaluated_columns": sorted(analysis.evaluable),
        "skipped_columns": analysis.skipped,
        "columns": {
            c: {"groups": {g: s.as_dict() for g, s in a.groups.items()},
                "excluded_small_groups": a.excluded_small,
                "rows_missing_group": a.rows_missing_group}
            for c, a in analysis.columns.items()},
    }


def _coverage_note(analysis: GroupAnalysis) -> str:
    if not analysis.skipped:
        return ""
    return " Not evaluated: " + ", ".join(
        f"'{c}'" for c in analysis.skipped) + "."


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
@_register("fairness.selection_rate_ratio")
def selection_rate_ratio(ctx: CheckContext) -> CheckResult:
    check_id = "fairness.selection_rate_ratio"
    analysis, reason = get_group_analysis(ctx)
    if analysis is None:
        return unavailable_result(ctx, check_id, reason)

    per_column = {}
    for column, col in analysis.evaluable.items():
        rates = {g: s.selection_rate for g, s in col.groups.items()}
        low, high = min(rates, key=rates.get), max(rates, key=rates.get)
        vacuous = rates[high] == 0
        per_column[column] = {
            "ratio": 1.0 if vacuous else rates[low] / rates[high],
            "parity_difference": rates[high] - rates[low],
            "lowest_group": low, "highest_group": high,
            "lowest_rate": rates[low], "highest_rate": rates[high],
            "vacuous_no_positive_predictions": vacuous,
        }
    worst = min(per_column, key=lambda c: per_column[c]["ratio"])
    w = per_column[worst]
    if w["vacuous_no_positive_predictions"]:
        message = (f"No group received a positive prediction for '{worst}', so "
                   "the selection-rate ratio is vacuous (reported as 1.0).")
    else:
        message = (f"Lowest/highest group selection rate is {w['ratio']:.3f} "
                   f"for '{worst}': {w['lowest_group']} {w['lowest_rate']:.1%} "
                   f"vs {w['highest_group']} {w['highest_rate']:.1%}.")
    return scored_result(ctx, check_id, w["ratio"],
                         message + _coverage_note(analysis),
                         details={**_common_details(analysis),
                                  "worst_column": worst,
                                  "per_column": per_column})


def _spread(values: list[float | None]) -> float | None:
    usable = [v for v in values if v is not None]
    return max(usable) - min(usable) if len(usable) >= 2 else None


@_register("fairness.equalized_odds")
def equalized_odds(ctx: CheckContext) -> CheckResult:
    check_id = "fairness.equalized_odds"
    analysis, reason = get_group_analysis(ctx)
    if analysis is None:
        return unavailable_result(ctx, check_id, reason)

    per_column, undefined = {}, {}
    for column, col in analysis.evaluable.items():
        tpr_gap = _spread([s.tpr for s in col.groups.values()])
        fpr_gap = _spread([s.fpr for s in col.groups.values()])
        gaps = [g for g in (tpr_gap, fpr_gap) if g is not None]
        if not gaps:
            undefined[column] = ("true/false-positive rates are undefined for "
                                 "all but one group")
            continue
        per_column[column] = {"equalized_odds_difference": max(gaps),
                              "tpr_difference": tpr_gap,
                              "fpr_difference": fpr_gap}
    if not per_column:
        return unavailable_result(
            ctx, check_id,
            "Equalized odds is undefined: " + "; ".join(
                f"'{c}': {w}" for c, w in undefined.items()) + ".")
    worst = max(per_column,
                key=lambda c: per_column[c]["equalized_odds_difference"])
    w = per_column[worst]
    return scored_result(
        ctx, check_id, w["equalized_odds_difference"],
        f"Equalized-odds difference is {w['equalized_odds_difference']:.3f} "
        f"for '{worst}' (true-positive-rate gap "
        f"{_fmt(w['tpr_difference'])}, false-positive-rate gap "
        f"{_fmt(w['fpr_difference'])})." + _coverage_note(analysis),
        details={**_common_details(analysis), "worst_column": worst,
                 "per_column": per_column, "undefined_columns": undefined})


def _fmt(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.3f}"


@_register("behaviour.subgroup_f1_gap")
def subgroup_f1_gap(ctx: CheckContext) -> CheckResult:
    check_id = "behaviour.subgroup_f1_gap"
    analysis, reason = get_group_analysis(ctx)
    if analysis is None:
        return unavailable_result(ctx, check_id, reason)

    per_column, undefined = {}, {}
    for column, col in analysis.evaluable.items():
        f1s = {g: s.f1 for g, s in col.groups.items() if s.f1 is not None}
        if len(f1s) < 2:
            undefined[column] = ("F1 is undefined (no positive examples or "
                                 "predictions) in all but one group")
            continue
        low, high = min(f1s, key=f1s.get), max(f1s, key=f1s.get)
        per_column[column] = {"gap": f1s[high] - f1s[low],
                              "lowest_group": low, "lowest_f1": f1s[low],
                              "highest_group": high, "highest_f1": f1s[high]}
    if not per_column:
        return unavailable_result(
            ctx, check_id,
            "Subgroup F1 gap is undefined: " + "; ".join(
                f"'{c}': {w}" for c, w in undefined.items()) + ".")
    worst = max(per_column, key=lambda c: per_column[c]["gap"])
    w = per_column[worst]
    return scored_result(
        ctx, check_id, w["gap"],
        f"Largest subgroup F1 gap is {w['gap']:.3f} for '{worst}': "
        f"{w['lowest_group']} {w['lowest_f1']:.3f} vs "
        f"{w['highest_group']} {w['highest_f1']:.3f}." + _coverage_note(analysis),
        details={**_common_details(analysis), "worst_column": worst,
                 "per_column": per_column, "undefined_columns": undefined})
