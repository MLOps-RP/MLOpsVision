"""Agent 1 - Adaptive Model Generation.

Level 1: eligible families from the registry
Level 2: profile-aware initial selection
Level 3: rule-based diagnosis + plan for the next round

All functions are pure (no training here), so they are easy to unit test.
"""
from ..schemas import (C1Output, Diagnosis, DiagnosisLabel, PlanDecision, ReasoningEntry,
                       TaskProfile)


# ---------- Level 1 ----------
def eligible_families(task_type: str, registry: dict, blocked: list[str] = ()) -> list[str]:
    """All model families that support this task type, minus families blocked by C3."""
    return [name for name, spec in registry["families"].items()
            if task_type in spec["tasks"] and name not in blocked]


# ---------- Level 2 ----------
def select_initial_families(profile: TaskProfile, c1: C1Output, registry: dict, cfg: dict,
                            blocked: list[str] = ()) -> tuple[list[str], ReasoningEntry]:
    """Choose round-1 families from the dataset profile (guide §10.3)."""
    eligible = eligible_families(c1.task_type.value, registry, blocked)
    baseline_family = "ridge" if c1.task_type.value == "regression" else "logistic_regression"
    reasons = []

    if profile.size_class == "small":
        wanted = ["random_forest", "catboost", baseline_family]
        reasons.append(f"small dataset ({profile.n_rows} rows): boosting overfits easily, "
                       "prefer random forest, catboost and a simple baseline")
    elif profile.size_class == "large":
        wanted = ["lightgbm", "xgboost"]
        reasons.append(f"large dataset ({profile.n_rows} rows): start with the fastest boosters")
    else:
        wanted = ["random_forest", "xgboost", "lightgbm"]
        reasons.append(f"medium dataset ({profile.n_rows} rows): default tree-based families")

    if profile.categorical_ratio > cfg["profile"]["many_categorical_ratio"] and "catboost" not in wanted:
        wanted.append("catboost")
        reasons.append(f"{profile.categorical_ratio:.0%} categorical features: add catboost")

    families = [f for f in wanted if f in eligible]
    if blocked:
        reasons.append(f"blocked by C3: {list(blocked)}")
    if not families:
        families = eligible
        reasons.append("no rule matched an eligible family: fall back to all eligible families")

    entry = ReasoningEntry(round=1, agent="agent1", step="select_initial_families",
                           decision={"families": families}, evidence=profile.model_dump(),
                           rationale="; ".join(reasons))
    return families, entry


# ---------- Level 3 ----------
def diagnose_family(family: str, signals: dict[str, float], rules: dict) -> Diagnosis:
    """Turn evidence signals into named failure modes using the YAML rules."""
    labels, reasons = [], []

    r = rules["overfitting"]
    if signals["gap"] > r["min_gap"] and signals["mean_train"] > r["min_train_score"]:
        labels.append(DiagnosisLabel.OVERFITTING)
        reasons.append(f"train-validation gap {signals['gap']:.3f} > {r['min_gap']} "
                       f"with train score {signals['mean_train']:.3f}")

    r = rules["underfitting"]
    if signals["gap"] < r["max_gap"] and signals["mean_train"] < r["max_train_score"]:
        labels.append(DiagnosisLabel.UNDERFITTING)
        reasons.append(f"low train score {signals['mean_train']:.3f} and small gap {signals['gap']:.3f}")

    r = rules["class_imbalance"]
    if ("imbalance_ratio" in signals and signals["imbalance_ratio"] < r["max_imbalance_ratio"]
            and signals["minority_recall"] < r["max_minority_recall"]):
        labels.append(DiagnosisLabel.IMBALANCE)
        reasons.append(f"imbalance ratio {signals['imbalance_ratio']:.3f} and "
                       f"minority recall {signals['minority_recall']:.3f}")

    r = rules["weak_features"]
    if signals["lift_over_baseline"] < r["max_lift"]:
        labels.append(DiagnosisLabel.WEAK_FEATURES)
        reasons.append(f"only {signals['lift_over_baseline']:.3f} better than a dummy model "
                       f"(mean FQS {signals['mean_fqs']:.2f})")

    r = rules["unsuitable_model_family"]
    if signals["gap_to_best_family"] >= r["min_gap_to_best"]:
        labels.append(DiagnosisLabel.UNSUITABLE_FAMILY)
        reasons.append(f"{signals['gap_to_best_family']:.3f} below the best family")

    if not labels:
        labels.append(DiagnosisLabel.HEALTHY)
        reasons.append("no rule triggered")

    return Diagnosis(model_family=family, labels=labels, evidence=signals, reasons=reasons)


def plan_next_round(diagnoses: list[Diagnosis], rules: dict, round_no: int
                    ) -> tuple[PlanDecision, ReasoningEntry]:
    """Decide which families to keep and which search changes Agent 2 should make."""
    action_map = rules["actions"]
    keep, actions, referrals, notes = [], {}, [], []

    for d in diagnoses:
        labels = {lbl.value for lbl in d.labels}
        if DiagnosisLabel.UNSUITABLE_FAMILY.value in labels:
            notes.append(f"drop {d.model_family}")
            continue
        keep.append(d.model_family)
        actions[d.model_family] = sorted({a for lbl in labels for a in action_map[lbl]
                                          if a not in ("drop_family", "refer_features_to_c1")})

    if not keep:  # never drop everything: keep the best family
        best = min(diagnoses, key=lambda d: d.evidence["gap_to_best_family"])
        keep, actions[best.model_family] = [best.model_family], []
        notes.append(f"all families looked unsuitable; keep best: {best.model_family}")

    all_weak = all(DiagnosisLabel.WEAK_FEATURES in d.labels for d in diagnoses)
    if all_weak:
        referrals.append("weak_features: no model family beats the dummy baseline; "
                         "feature engineering needed in C1")
    if any(DiagnosisLabel.IMBALANCE in d.labels for d in diagnoses):
        referrals.append("class_imbalance: consider resampling (e.g. SMOTE/ADASYN) in C1")

    plan = PlanDecision(families_next_round=keep, actions=actions, referrals_to_c1=referrals,
                        stop=all_weak, rationale="; ".join(notes) or "keep all families")
    entry = ReasoningEntry(round=round_no, agent="agent1", step="plan_next_round",
                           decision=plan.model_dump(),
                           evidence={d.model_family: [l.value for l in d.labels] for d in diagnoses},
                           rationale=plan.rationale)
    return plan, entry
