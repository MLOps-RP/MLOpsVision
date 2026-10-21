"""Plain-language descriptions of a validation profile, generated from its real configuration.

Nothing here is keyed on a profile's name: every sentence is computed from the loaded configuration,
so a profile an operator adds is described correctly with no change to this module, and a profile
whose settings change is described as it now behaves. The words are for people who are not
validation engineers; the numbers in them are the configuration's own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mqvg.config import GateConfig
from mqvg.config import MissingPolicy

Strictness = Literal["lenient", "standard", "strict"]

# What an extra check adds, in words. Keyed by CHECK id (the catalogue), never by profile.
EXTRA_CHECKS = {
    "training_validity.eval_label_conflicts": "conflicting labels in the evaluation data",
    "training_validity.train_duplicate_rows": "repeated training rows",
    "operational.cpu_time": "how much CPU the model uses",
    "robustness.noise_degradation": "how well it copes with noisy input",
}
CONSEQUENCE = {
    "score_only": "a failing group only lowers the score",
    "hold": "a failing group puts the model on hold",
    "critical": "a severe failing group can trigger a rollback recommendation",
}
# A profile is "strict" when it asks for a high score or counts missing evidence against the model,
# and "lenient" when promotion needs a low score. Anything else is "standard".
STRICT_PROMOTE_AT = 85.0
LENIENT_BELOW = 70.0


@dataclass(frozen=True)
class Guide:
    summary: str
    highlights: list[str]
    strictness: Strictness
    strictness_text: str


def strictness_of(config: GateConfig) -> Strictness:
    if config.decision.promote_at >= STRICT_PROMOTE_AT or config.missing_policy == MissingPolicy.PENALIZE:
        return "strict"
    if config.decision.promote_at < LENIENT_BELOW:
        return "lenient"
    return "standard"


def describe(config: GateConfig, scans_files: bool) -> Guide:
    enabled = {check_id for check_id, c in config.checks.items() if c.enabled}
    decision = config.decision
    critical = len(config.critical_rules)
    must_pass = len(decision.must_pass_checks)

    evidence = ("A check that could not run counts as zero."
                if config.missing_policy == MissingPolicy.PENALIZE
                else "A check that could not run is flagged and never counted as a pass.")
    strictness_text = (
        f"Recommends promotion at a score of {decision.promote_at:g} or more and rollback below "
        f"{decision.reject_below:g}. {critical} critical {'rule forces' if critical == 1 else 'rules force'} "
        f"a rollback whatever the score, and {must_pass} {'check' if must_pass == 1 else 'checks'} must "
        f"pass before promotion. {evidence}")

    highlights = []
    if scans_files:
        highlights.append("Scans the model file before loading it")
    if "behaviour.hidden_failure_groups" in enabled:
        highlights.append(f"Looks for hidden failing groups of rows ({CONSEQUENCE[config.hidden_groups.consequence]})")
    extras = [phrase for check_id, phrase in EXTRA_CHECKS.items() if check_id in enabled]
    if extras:
        highlights.append("Also checks " + ", ".join(extras[:-1]) + (" and " if len(extras) > 1 else "") + extras[-1])

    summary = (f"Runs {len(enabled)} checks of training quality, model behaviour and operational readiness, "
               "then recommends promote, hold or rollback.")
    return Guide(summary=summary, highlights=highlights, strictness=strictness_of(config), strictness_text=strictness_text)


def unavailable_reason(scans_files: bool) -> str:
    """Why a profile this server will not run is restricted."""
    if not scans_files:
        return ("It would load the model file without scanning it first. Loading a model runs the code inside it, "
                "so this server only runs profiles that scan the file before loading it.")
    return "This server does not allow it."
