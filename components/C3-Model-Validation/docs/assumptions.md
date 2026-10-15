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
