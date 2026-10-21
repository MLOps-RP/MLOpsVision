"""Does the full gate catch defects that a plain evaluation-metrics gate misses?

Two baselines are compared with the full gate on the same sample models, using
the SAME measurements (every model is validated once per profile; the baselines
read the F1, precision, recall and ROC-AUC checks from that run and apply the
configuration's own pass_at limits).

  baseline 1  "F1 only"       promote iff behaviour.f1 passes
  baseline 2  "metrics only"  promote iff F1, precision, recall and ROC-AUC all pass
                              (what a typical evaluation script does)
  full gate   default profile
  full gate   hidden_groups profile (hidden failure group check on, policy hold)

Ground truth comes from the fixture manifests, which record the defect that was
injected into each model: a model is a "defect" if one was injected, "clean" if
the manifest says `none`. `slow` is excluded from the rates because its
expected decision depends on the validating machine's speed.

A baseline that cannot compute its metrics (for example the model will not
load) is counted as "not promoted"; a real script would crash.

Run from the C3-Model-Validation folder:

    python scripts/baseline_comparison.py
    python scripts/baseline_comparison.py --out docs/baseline_comparison.md

Standalone demonstration helper; it is not part of the gate (src/mqvg). The
sample models were built by the project author, so detection rates measure the
gate against defects it was designed for, not against defects found in the wild.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from mqvg.config import load_config
from mqvg.pipeline import run_validation
from mqvg.schemas import CandidateBundle, CheckStatus

PROJECT = Path(__file__).resolve().parents[1]
MANIFESTS = ("manifest.yaml", "hidden_groups_manifest.yaml")
EXCLUDED = {"slow"}      # expected decision depends on the validating machine
F1 = "behaviour.f1"
METRICS = (F1, "behaviour.precision", "behaviour.recall", "behaviour.roc_auc")
GATES = ("F1 only", "metrics only", "full gate (default)", "full gate (hidden_groups)")


# ------------------------------------------------------------------ inputs
def load_cases():
    """name -> manifest entry, for every fixture with a recorded expectation."""
    cases = {}
    for filename in MANIFESTS:
        document = yaml.safe_load((PROJECT / "fixtures" / filename).read_text(encoding="utf-8"))
        cases.update(document["fixtures"])
    return cases


def bundle_for(name, entry):
    hashes = entry.get("train_row_hashes")
    own = PROJECT / "fixtures" / name / "train_row_hashes.txt"
    hashes_path = (PROJECT / hashes if hashes else own if own.exists()
                   else PROJECT / "data" / "train_row_hashes.txt")
    return CandidateBundle(
        candidate_id=name, model_path=PROJECT / entry["model"],
        eval_data_path=PROJECT / entry["eval_data"],
        metadata_path=PROJECT / entry["metadata"], train_row_hashes_path=hashes_path)


def is_defect(entry):
    """True when a defect was injected into the model; the manifest records 'none' otherwise.

    The expected decision is not used: for `hidden_group` the default profile
    expects PROMOTE, because that is exactly the defect it cannot see.
    """
    return not " ".join(str(entry["injected_defect"]).split()).lower().startswith("none")


# --------------------------------------------------------------- baselines
def promotes_f1_only(checks):
    by_id = {c.id: c for c in checks}
    return F1 in by_id and by_id[F1].status == CheckStatus.PASS


def promotes_metrics_only(checks):
    by_id = {c.id: c for c in checks}
    return all(m in by_id and by_id[m].status == CheckStatus.PASS for m in METRICS)


# ----------------------------------------------------------------- summary
def rates(outcomes):
    """Detection of defects and false alarms on clean models, per gate.

    `outcomes` maps name -> {"defect": bool, "promoted": {gate: bool}}.
    """
    summary = {}
    defects = [o for o in outcomes.values() if o["defect"] is True]
    clean = [o for o in outcomes.values() if o["defect"] is False]
    for gate in GATES:
        caught = sum(not o["promoted"][gate] for o in defects)
        alarms = sum(not o["promoted"][gate] for o in clean)
        summary[gate] = {"caught": caught, "defects": len(defects),
                         "false_alarms": alarms, "clean": len(clean)}
    return summary


def collect(cases):
    default = load_config(PROJECT / "configs" / "default.yaml")
    hidden = load_config(PROJECT / "configs" / "hidden_groups.yaml")
    outcomes = {}
    for name, entry in cases.items():
        bundle = bundle_for(name, entry)
        base = run_validation(bundle, default)
        extended = run_validation(bundle, hidden)
        decisions = {"full gate (default)": base.decision.decision.value,
                     "full gate (hidden_groups)": extended.decision.decision.value}
        outcomes[name] = {
            "defect": None if name in EXCLUDED else is_defect(entry),
            "injected": " ".join(str(entry["injected_defect"]).split()),
            "decisions": decisions,
            "promoted": {
                "F1 only": promotes_f1_only(base.checks),
                "metrics only": promotes_metrics_only(base.checks),
                "full gate (default)": decisions["full gate (default)"] == "PROMOTE",
                "full gate (hidden_groups)": decisions["full gate (hidden_groups)"] == "PROMOTE"},
            "f1": next((c.raw_value for c in base.checks if c.id == F1), None),
            "weak_metrics": [m.split(".")[1] for m in METRICS
                             if next((c.status for c in base.checks if c.id == m), None)
                             != CheckStatus.PASS]}
        print(f"{name:<26} " + " ".join(
            f"{g}={'promote' if outcomes[name]['promoted'][g] else 'stop'}" for g in GATES))
    return outcomes


def render(outcomes):
    summary = rates(outcomes)
    lines = ["# Baseline comparison: plain metrics gate vs the full gate", "",
             "Generated by `scripts/baseline_comparison.py`. Every sample model was validated once "
             "per profile; the baselines read the F1, precision, recall and ROC-AUC checks from "
             "the same run, using the configuration's own `pass_at` limits.", "",
             "- **F1 only**: promote iff F1 passes.",
             "- **metrics only**: promote iff F1, precision, recall and ROC-AUC all pass "
             "(a typical evaluation script).",
             "- **full gate (default)**: the default profile.",
             "- **full gate (hidden_groups)**: hidden failure group check on, policy `hold`.", "",
             "Ground truth is the fixture manifests: a model is a *defect* if a defect was injected "
             "into it (`injected_defect` is not `none`) and *clean* otherwise. `slow` is excluded "
             "because its expected decision depends on machine speed. A baseline that cannot compute its "
             "metrics counts as not promoted.", "",
             "**The sample models were built by the project author, so this measures the gate "
             "against defects it was designed for, not defects found in practice. There are only "
             "two clean models, so the false-alarm figures are weak evidence.**", "",
             "## Summary", "",
             "| Gate | Defects stopped | False alarms on clean models |", "|---|---|---|"]
    for gate, s in summary.items():
        lines.append(f"| {gate} | {s['caught']}/{s['defects']} | {s['false_alarms']}/{s['clean']} |")

    lines += ["", "## Per model", "",
              "| Model | Truth | F1 | Metrics not passing | " + " | ".join(GATES) + " |",
              "|---|---|---|---|" + "---|" * len(GATES)]
    for name, o in outcomes.items():
        truth = {True: "defect", False: "clean", None: "excluded"}[o["defect"]]
        f1 = "" if o["f1"] is None else f"{o['f1']:.3f}"
        cells = []
        for gate in GATES:
            promoted = o["promoted"][gate]
            decision = o["decisions"].get(gate)
            cells.append(("PROMOTE" if promoted else "stop") if decision is None
                         else decision)
        weak = ", ".join(o["weak_metrics"]) or "none"
        lines.append(f"| {name} | {truth} | {f1} | {weak} | " + " | ".join(cells) + " |")

    missed = [(n, o) for n, o in outcomes.items()
              if o["defect"] and o["promoted"]["metrics only"]]
    lines += ["", "## Defects a plain metrics gate promotes", ""]
    for name, o in missed:
        caught = [g for g in GATES[2:] if not o["promoted"][g]]
        lines.append(f"- **{name}**: {o['injected']} "
                     + (f"Stopped by: {', '.join(caught)}." if caught
                        else "**Not stopped by any gate.**"))
    lines += ["", "## Reading this", "",
              "- A metrics-only gate promotes models whose F1, precision, recall and ROC-AUC look "
              "fine but that leak the label, lack metadata or lack evaluation columns (listed "
              "above). Those need the training-validity and evidence checks.",
              "- It also stops `overfit`, `biased` and `miscalibrated`, but only because a metric "
              "happens to be a WARN on this dataset (see `Metrics not passing`), not because it "
              "sees the overfitting, the unfairness or the miscalibration. The same weak metric "
              "makes it stop the clean `hidden_group_null` control (a false alarm). Its 'stopped' "
              "count therefore overstates what it detects.",
              "- The F1-only gate stops only the model it cannot score.",
              "- The default full gate does not stop `hidden_group`: every standard check passes. "
              "Only the profile with the hidden failure group check does.",
              "- A model that a baseline cannot evaluate (for example a corrupt file) is stopped "
              "only because it cannot be scored; this is not a fair detection.", ""]
    return "\n".join(lines)


def to_json(outcomes):
    """The same results as the Markdown, for the dashboard (docs/baseline_comparison.json)."""
    truth = {True: "defect", False: "clean", None: "excluded"}
    return {
        "study": "baseline_comparison",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gates": list(GATES),
        "summary": rates(outcomes),
        "models": [{
            "name": name, "truth": truth[o["defect"]], "f1": o["f1"],
            "metrics_not_passing": o["weak_metrics"], "injected_defect": o["injected"],
            "decisions": o["decisions"], "promoted": o["promoted"]} for name, o in outcomes.items()],
        "caveats": [
            "The sample models were built by the project author, so this measures the gate against "
            "defects it was designed for, not defects found in practice.",
            "There are only two clean models, so the false-alarm figures are weak evidence.",
            "The metrics-only gate also stops models whose precision, recall or ROC-AUC merely warns on "
            "this dataset; its stopped count overstates what it detects.",
            "A baseline that cannot compute its metrics counts as not promoted.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=PROJECT / "docs" / "baseline_comparison.md")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the results as JSON (default: next to --out)")
    args = parser.parse_args(argv)
    outcomes = collect(load_cases())
    args.out.write_text(render(outcomes), encoding="utf-8", newline="\n")
    print(f"\nwritten {args.out}")
    json_path = args.json or args.out.with_suffix(".json")
    json_path.write_text(json.dumps(to_json(outcomes), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"written {json_path}")
    for gate, s in rates(outcomes).items():
        print(f"{gate:<28} stopped {s['caught']}/{s['defects']} defects, "
              f"{s['false_alarms']}/{s['clean']} false alarms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
