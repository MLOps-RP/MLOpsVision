"""How much do the gate's decisions depend on its arbitrary numbers?

The weights, thresholds and decision bands in the configuration are
research-defined placeholders. This script asks whether reasonable changes to
them would change the PROMOTE / HOLD / ROLLBACK recommendation for the sample
models.

Method: every model is validated ONCE (the checks are the slow part). The
stored check results are then re-scored and re-judged under each perturbed
configuration, using the gate's own functions (`normalize`, `compute_scorecard`,
`evaluate_rules`). Nothing is re-measured, so the measured values are identical
across perturbations and only the policy changes.

Perturbation families
  weights     every (TVS, MBS, ORS) triple on a 0.1 grid with each weight >= 0.1
  thresholds  every pass_at / fail_at moved toward or away from its target by
              10% and 20% of its distance, for all checks at once
  per-check   the same +/-10% applied to one check at a time (finds the
              single threshold a decision hinges on)
  bands       promote_at 65..85 and reject_below 40..60, in steps of 5
  missing     exclude_and_cap against penalize

Run from the C3-Model-Validation folder:

    python scripts/sensitivity_analysis.py
    python scripts/sensitivity_analysis.py --out docs/sensitivity_analysis.md

Standalone demonstration helper; it is not part of the gate (src/mqvg).
The result describes the sample models only; it is not a validation of the
thresholds.
"""
import argparse
import itertools
import json
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path

from mqvg.config import DimensionWeights, MissingPolicy, Thresholds, load_config
from mqvg.normalize import normalize
from mqvg.pipeline import run_validation
from mqvg.rules import evaluate_rules
from mqvg.schemas import CandidateBundle, CheckStatus
from mqvg.scoring import compute_scorecard

PROJECT = Path(__file__).resolve().parents[1]
# (fixture, profile it is judged under)
CASES = [("good", "default"), ("leaky", "default"), ("overfit", "default"),
         ("biased", "default"), ("miscalibrated", "default"), ("slow", "default"),
         ("corrupt", "default"), ("hidden_group", "hidden_groups"),
         ("hidden_group_null", "hidden_groups")]
SCORED = (CheckStatus.PASS, CheckStatus.WARN, CheckStatus.FAIL)


def bundle_for(name):
    folder = PROJECT / "fixtures" / name
    eval_path = folder / "eval.csv"
    hashes = folder / "train_row_hashes.txt"
    return CandidateBundle(
        candidate_id=name, model_path=folder / "model.joblib",
        eval_data_path=eval_path if eval_path.exists() else PROJECT / "data" / "eval.csv",
        metadata_path=folder / "metadata.json",
        train_row_hashes_path=hashes if hashes.exists() else PROJECT / "data" / "train_row_hashes.txt")


# ------------------------------------------------------------ perturbations
def scaled(th: Thresholds, factor: float) -> Thresholds:
    """Move pass_at and fail_at toward (<1) or away from (>1) the target."""
    lo, hi = sorted((th.floor, th.target))
    clamp = lambda x: min(hi, max(lo, x))   # noqa: E731
    return Thresholds(target=th.target, floor=th.floor,
                      pass_at=clamp(th.target + (th.pass_at - th.target) * factor),
                      fail_at=clamp(th.target + (th.fail_at - th.target) * factor))


def with_thresholds(config, factor, only=None):
    checks = {}
    for cid, check in config.checks.items():
        if check.thresholds is not None and (only is None or cid == only):
            check = check.model_copy(update={"thresholds": scaled(check.thresholds, factor)})
        checks[cid] = check
    return config.model_copy(update={"checks": checks})


def weight_grid():
    for a, b in itertools.product(range(1, 9), repeat=2):
        c = 10 - a - b
        if c >= 1:
            yield a / 10, b / 10, c / 10


def perturbations(config):
    """Yield (family, label, perturbed configuration)."""
    for t, m, o in weight_grid():
        weights = DimensionWeights(training_validity=t, model_behaviour=m,
                                   operational_readiness=o)
        yield "weights", f"TVS {t:.1f} / MBS {m:.1f} / ORS {o:.1f}", \
            config.model_copy(update={"weights": weights})
    for factor in (0.8, 0.9, 1.1, 1.2):
        yield "thresholds", f"all pass_at/fail_at x{factor}", with_thresholds(config, factor)
    for cid, check in config.checks.items():
        if check.enabled and check.thresholds is not None:
            for factor in (0.9, 1.1):
                yield "per-check", f"{cid} x{factor}", with_thresholds(config, factor, only=cid)
    for promote, reject in itertools.product(range(65, 90, 5), range(40, 65, 5)):
        if reject < promote:
            decision = config.decision.model_copy(
                update={"promote_at": promote, "reject_below": reject})
            yield "bands", f"promote_at {promote} / reject_below {reject}", \
                config.model_copy(update={"decision": decision})
    for policy in MissingPolicy:
        yield "missing", f"missing_policy {policy.value}", \
            config.model_copy(update={"missing_policy": policy})


# ------------------------------------------------------------------ judging
def rescore(checks, config):
    """The stored measurements, scored again against the config's thresholds."""
    out = []
    for check in checks:
        spec = config.checks.get(check.id)
        if (check.status in SCORED and check.direction in ("higher_is_better", "lower_is_better")
                and isinstance(check.raw_value, (int, float)) and not isinstance(check.raw_value, bool)
                and spec is not None and spec.thresholds is not None):
            score, status = normalize(float(check.raw_value), spec.thresholds, check.direction)
            check = check.model_copy(update={"score": score, "status": status})
        out.append(check)
    return out


def judge(checks, config):
    results = rescore(checks, config)
    card = compute_scorecard(results, config)
    return evaluate_rules(results, card, config).decision.value, card.mqs


# ------------------------------------------------------------------- report
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=PROJECT / "docs" / "sensitivity_analysis.md")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the results as JSON (default: next to --out)")
    args = parser.parse_args(argv)

    families = ["weights", "thresholds", "per-check", "bands", "missing"]
    table = {}        # fixture -> family -> Counter(decision)
    baseline = {}
    flips = defaultdict(list)       # fixture -> [(family, label, decision)]
    mqs_range = {}
    for name, profile in CASES:
        config = load_config(PROJECT / "configs" / f"{profile}.yaml")
        checks = run_validation(bundle_for(name), config).checks
        base_decision, base_mqs = judge(checks, config)
        baseline[name] = (profile, base_decision, base_mqs)
        counts = {f: Counter() for f in families}
        scores = [base_mqs] if base_mqs is not None else []
        for family, label, perturbed in perturbations(config):
            decision, mqs = judge(checks, perturbed)
            counts[family][decision] += 1
            if mqs is not None and family in ("weights", "thresholds", "per-check", "missing"):
                scores.append(mqs)
            if decision != base_decision:
                flips[name].append((family, label, decision))
        table[name] = counts
        mqs_range[name] = (min(scores), max(scores)) if scores else (None, None)
        print(f"{name:<18} baseline {base_decision:<8} flips {len(flips[name])}")

    lines = ["# Sensitivity of the decisions to weights, thresholds and bands", "",
             "Generated by `scripts/sensitivity_analysis.py`. Each sample model was validated once; "
             "its stored measurements were then re-scored and re-judged under each perturbed "
             "configuration with the gate's own functions. Nothing was re-measured.", "",
             "**This analyses the sample models only. It shows how stable the recommendation is "
             "under these perturbations; it does not show that the thresholds are correct.**", "",
             "## Share of perturbations that keep the baseline decision", "",
             "| Model | Profile | Baseline | MQS range | " + " | ".join(families) + " |",
             "|---|---|---|---|" + "---|" * len(families)]
    for name, (profile, decision, mqs) in baseline.items():
        cells = []
        for family in families:
            c = table[name][family]
            total = sum(c.values())
            cells.append(f"{c[decision] / total:.0%} ({c[decision]}/{total})" if total else "n/a")
        lo, hi = mqs_range[name]
        rng = "n/a" if lo is None else f"{lo:.1f}-{hi:.1f}"
        lines.append(f"| {name} | {profile} | {decision} | {rng} | " + " | ".join(cells) + " |")

    lines += ["", "## Perturbations that change the decision", ""]
    for name, items in flips.items():
        base = baseline[name][1]
        if not items:
            lines.append(f"- **{name}** ({base}): no perturbation changed the decision.")
            continue
        by_family = defaultdict(list)
        for family, label, decision in items:
            by_family[family].append((label, decision))
        lines.append(f"- **{name}** (baseline {base}): {len(items)} of "
                     f"{sum(sum(c.values()) for c in table[name].values())} perturbations flip it.")
        for family, entries in by_family.items():
            shown = "; ".join(f"{label} -> {d}" for label, d in entries[:6])
            more = f" (+{len(entries) - 6} more)" if len(entries) > 6 else ""
            lines.append(f"  - {family}: {shown}{more}")

    lines += ["", "## How to read this", "",
              "- A model whose decision survives nearly every perturbation is robust to the "
              "arbitrary choices. A model that flips under small changes sits near a boundary, and "
              "its decision should be treated as uncertain.",
              "- Critical rules (leakage, selection-rate disparity, F1 floor, unloadable model) "
              "depend on thresholds only through the FAIL status, not on the weights or the "
              "bands. That is why weight and band changes cannot rescue a model such as `leaky`.",
              "- `MQS range` is taken over the weight, threshold and missing-policy "
              "perturbations; the band perturbations do not change the score.",
              "- The perturbations are a design of this study (a 0.1 weight grid, +/-10% and "
              "+/-20% threshold distances, bands in steps of 5). Other designs would give other "
              "numbers.", ""]
    args.out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"\nwritten {args.out}")

    cases = []
    for name, (profile, decision, mqs) in baseline.items():
        lo, hi = mqs_range[name]
        cases.append({
            "model": name, "profile": profile, "baseline_decision": decision, "baseline_mqs": mqs,
            "mqs_range": None if lo is None else [lo, hi],
            "families": {f: {"total": sum(table[name][f].values()), "kept": table[name][f][decision],
                             "decisions": dict(table[name][f])} for f in families},
            "flips": [{"family": fam, "label": label, "decision": d} for fam, label, d in flips[name]]})
    study = {
        "study": "sensitivity_analysis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "families": families, "cases": cases,
        "design": ("weights on a 0.1 grid; pass_at/fail_at moved 10% and 20% toward or away from "
                   "the target (all checks, and one check at a time); promote_at 65-85 and "
                   "reject_below 40-60 in steps of 5; both missing-evidence policies"),
        "caveats": [
            "Each sample model was validated once; its stored measurements were re-scored under each "
            "perturbed configuration. Nothing was re-measured.",
            "This analyses the sample models only. It shows how stable the recommendation is under these "
            "perturbations; it does not show that the thresholds are correct.",
            "The perturbation design is a choice of this study; other designs would give other numbers.",
        ]}
    json_path = args.json or args.out.with_suffix(".json")
    json_path.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"written {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
