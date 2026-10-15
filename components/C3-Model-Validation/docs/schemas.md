# Schemas (`src/mqvg/schemas.py`)

The schemas are the contract between the gate's modules and, later, between
the gate and the other platform components. Changing them is a deliberate act.
Current `SCHEMA_VERSION`: 0.1.0.

## Data flow

```
CandidateBundle ─► checks ─► CheckResult[] ─► ScoreCard ─► DecisionRecord
                                  └──────────────┴──────────────┴──► ValidationResult
```

| Type | Role |
|---|---|
| `CandidateBundle` | Input: paths to model, eval data, metadata, config, plus `ScenarioInfo`. Paths need not exist; a missing artifact is a finding. |
| `TrainingMetadata` | Producer-declared training facts. All fields optional. Everything here is DECLARED evidence. |
| `CheckResult` | One check's outcome: status, 0-100 score, evidence source, thresholds used, message. |
| `DimensionScore` | TVS / MBS / ORS with `coverage` and a `provisional` flag. |
| `ScoreCard` | Dimension scores, configured weights, effective weights, MQS, missing-evidence policy. |
| `TriggeredRule` / `DecisionRecord` | Which rules fired and the PROMOTE / HOLD / ROLLBACK recommendation. |
| `ValidationResult` | The complete auditable record of one run. |

## Rules enforced by the types

These make the central design promises impossible to violate by accident.

1. **Missing evidence is never a pass.** `UNAVAILABLE`, `NOT_APPLICABLE` and
   `ERROR` checks must have no score and evidence source `UNAVAILABLE`.
   `PASS`/`WARN`/`FAIL` must have a score and real evidence.
2. **Partial evidence is visible.** A dimension with coverage below 1 must be
   `provisional`; an MQS built from incomplete dimensions must be `provisional`.
3. **Weights are valid.** All three dimensions present, non-negative, sum to 1.
4. **Decisions are consistent.** PROMOTE cannot coexist with HOLD/ROLLBACK
   rules; ROLLBACK needs a ROLLBACK rule; a critical override must be a
   ROLLBACK caused by a critical rule.
5. **Simulation is declared.** SIMULATED evidence requires a scenario marked
   `simulated` that lists the simulated inputs.
6. **Results are internally consistent.** Check ids are unique and every rule
   refers to a check that exists.
7. **Standalone mode only recommends.** `action_mode` is always
   `RECOMMENDATION`.

## Design decisions

- Gate-produced objects use `extra="forbid"` so typos fail loudly.
  `TrainingMetadata` uses `extra="allow"` because producers add fields.
- There is no per-check `severity` field. Whether a failure is critical is
  decided by the rules engine from configuration, so one check can be critical
  under a strict profile and not under a lenient one.
- Check ids look like `behaviour.f1` or `operational.latency` (lower-case,
  one dot) so rules and reports can refer to them reliably.
