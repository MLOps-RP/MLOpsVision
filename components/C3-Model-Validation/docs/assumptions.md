# Assumptions, decisions and open research justifications

Record each threshold or design choice here with its reason. Items marked
**OPEN** still need literature, experiment or supervisor input.

## Fixtures

- Dataset: UCI Adult via Fairlearn, split 60/20/20 stratified, seed 42.
  Not representative of every domain.
- Sensitive attributes: `sex`, `race`. The good model excludes them from its
  features but is still not fair (parity difference 0.171 by sex, selection-rate
  ratio 0.31). **OPEN:** fairness thresholds decide whether this is WARN or PASS.
- Adult has naturally duplicated records: 12 of 9,768 eval rows (0.12%) exactly
  match a training row. The overlap check must compare against this baseline
  and must not fail on any non-zero overlap. **OPEN:** choose the tolerance.
- Fairness metric choice: the parity *difference* is capped by the majority
  group's selection rate (0.171 for good vs 0.249 for the clearly biased
  fixture), while the selection-rate *ratio* separates them clearly
  (0.31 vs 0.006). Use a ratio or equalized-odds metric as well.
- Latency depends on hardware (about 7 ms vs 78 ms single-row on the
  development machine). Always record a machine profile with latency results.
- `manifest.yaml` expected decisions are provisional until thresholds exist.
  Do not tune thresholds only to make them pass.

## Schemas

- No per-check severity; criticality comes from configured rules.
- Declared metrics in metadata are never to be used as measured results.
- Model loading executes pickle code, so only trusted local paths are
  accepted in this prototype. **Limitation** to state in the thesis.

## Open questions (see the implementation plan)

- Meaning of ROLLBACK (reject candidate vs revert deployed model).
- Whether to compare against an incumbent model (champion/challenger).
- Default missing-evidence policy (exclude-and-cap vs penalize).
- How the MQS itself will be validated (injected defects, expert agreement).

## Operational readiness checks (Step 5)

- **Memory check is weak evidence.** It measures resident-memory growth in the
  validating process. It includes library imports on the first load and reads
  about 0 MB when those libraries are already imported (seen for the `leaky`
  fixture, 68 MB for `good`). **OPEN / Phase B:** measure in a fresh subprocess.
  That would also isolate model loading (a security gain) and let the gate
  survive a model that hard-crashes.
- **Malformed-input check is in-process.** It can tell that a model rejects bad
  input but cannot detect a hard crash (such as a segmentation fault), which
  would end the whole run.
- **Latency is single-row, one call at a time, on the validating machine.**
  It says nothing about batch throughput or a production server. Always read
  it next to the recorded machine profile.
- **Dependency match is DECLARED evidence.** It compares the versions the
  producer declared against the current environment; the declaration itself is
  unverified. Python patch versions are treated as compatible.
- **Loading executes code.** `joblib.load` runs pickle code, so the loader
  accepts only trusted local paths. State this as a limitation in the thesis.

## Performance metrics (Step 6)

- Binary classification only. Multiclass and regression targets, or a declared
  non-binary task type, make the behaviour checks UNAVAILABLE (not wrong
  numbers). **OPEN:** extend in Phase B.
- All metrics are computed by the gate from the model's own predictions.
  Metrics recorded in metadata are never used; for the good fixture the two
  agree to 1e-9, which cross-checks both code paths.
- Positive label: taken from the bundle or metadata. If undeclared it is
  inferred as `model.classes_[1]` and the result says so. This is a
  scikit-learn convention, not a fact; declare it where it matters.
- Performance alone cannot reveal leakage or overfitting: the `leaky` fixture
  scores F1 0.954 / AUC 0.994 and passes every behaviour check, and `overfit`
  passes F1. This is the argument for the separate training-validity dimension
  and for critical rules (use it in the thesis).
- Reported baselines (always-positive F1/precision, AUC 0.5) are context only
  and are not part of the score. **OPEN:** decide whether a lift-over-baseline
  check should be scored.
- Precision is undefined when nothing is predicted positive; it is reported as
  0.0 with `precision_undefined` set.
- Behaviour checks that are not built yet (fairness, calibration, subgroup,
  robustness) report ERROR with "not implemented", never PASS, until their step.
