# Extended checks: conflicting labels, training duplicates, model-file safety, CPU time

Four additional checks, off in `configs/default.yaml` (enabling them would change every
model's TVS, ORS and MQS and break the hand-calculated scoring tests) and on in
`configs/extended.yaml`:

```
python scripts\validate_models.py good --config configs\extended.yaml
python scripts\make_extended_fixtures.py          # sample defective inputs (see below)
```

All thresholds are **placeholders**, like every other threshold in the gate.

| Check | Layer | Value | Evidence | Placeholder thresholds (target / pass / fail / floor) |
|---|---|---|---|---|
| `training_validity.eval_label_conflicts` | TVS | share of evaluation rows whose features equal another row's but whose label differs | MEASURED | 0 / 0.5% / 3% / 15% |
| `training_validity.train_duplicate_rows` | TVS | share of training rows that repeat another training row (from the supplied row hashes) | DECLARED | 0 / 1% / 10% / 30% |
| `operational.model_file_safety` | ORS | severity 0 recognised only, 1 unrecognised code, 2 dangerous code | MEASURED | 0 / 0 / 1 / 2 |
| `operational.cpu_time` | ORS | process CPU milliseconds per predicted row | MEASURED | 0.05 / 0.5 / 3 / 10 ms |

Natural values on the sample models (default data): label conflicts 0.000; training
duplicates 0.001 (0.06%); CPU 0.009 to 0.018 ms/row, 0.297 for the 1000-tree `slow` model
(on this machine, so hardware dependent). The CPU thresholds were not tuned to make any
sample model warn; the tests use a synthetic CPU-burning model instead.

## Boundaries

- **Conflicting labels** are searched in the **evaluation** data only. The gate never receives
  training labels (only row hashes of the features), so conflicting training labels cannot be
  found here. Only exact repeats of the feature values are found, so data with continuous
  features (such as `fnlwgt`) has little power: the sample data has no repeated feature
  rows at all, and the check says how many repeated groups it examined.
- **Training duplicates** come from the producer's hash file, so the evidence is DECLARED.
  The hashes cover the feature columns, so rows with the same features and different labels
  also count as duplicates.
- Data collection and data-integrity validation belong to another component. These two
  checks examine the data a candidate model was evaluated and trained on, so they overlap
  with it; they are kept in a separate profile and are scoring/blocking choices of the
  profile, not claims of ownership.

## Model-file safety

**Why.** Loading a joblib/pickle file executes whatever code the file asks for. Without this
check the gate itself runs untrusted code the moment it loads a candidate.

**What it does.** Before the model is loaded, the file is read **without being executed**
(`src/mqvg/model_safety.py`). The scan uses joblib's own unpickler, so joblib's arrays,
compression and nested object-array streams are parsed as `joblib.load` would parse them,
but it replaces `find_class`, the single place where a pickle obtains a callable. Every
global (module.name) is recorded and answered with an inert stub; nothing the file names is
ever called. Only a short fixed list of numpy/joblib classes needed to keep the stream
readable is resolved for real. Each reference is classified:

| Level | Meaning | Default consequence |
|---|---|---|
| 0 recognised | module is on `model_safety.allowed_module_prefixes` (sklearn, numpy, scipy, pandas, joblib, xgboost, lightgbm, collections, ...) | loaded |
| 1 unrecognised | not on the allowlist (for example a user's own model class) | not loaded; WARN; HOLD for review |
| 2 dangerous | on the blocklist (os, subprocess, socket, `builtins.eval`/`exec`/`getattr`/`open`, `numpy.load`, `joblib.load`, ...) | not loaded; FAIL; critical rule `unsafe_model_file` gives a ROLLBACK recommendation |

`model_safety.refuse_at` sets the level at which the file is not loaded: `dangerous`,
`unrecognised` (default in the extended profile) or `never` (report only; the file is then
loaded and its code runs). A file the scanner cannot parse is never handed to the real loader
(a file one parser rejects and another accepts is what an attacker looks for): a corrupt
file is a failed load (ROLLBACK, as before), and a file the scanner cannot handle (for
example an unsupported joblib version) is UNAVAILABLE and goes to review.

**Verified** (`tests/test_model_safety.py`, 45 tests): dangerous references are found in joblib,
compressed-joblib and plain-pickle files and inside nested object arrays; the payload
(creating a marker directory) never runs during the scan or the gate, and a control test
shows the same file really does run it when loaded normally; every sample model scans clean
(including string-category pipelines, which contain nested pickles).

**What a clean result does and does not mean.** It means every reference is on the allowlist.
It does **not** mean the model is safe. This is a static allowlist, not a sandbox:

- a trusted library can contain a harmful function, and the allowlist is by module prefix;
- the block and allow lists are heuristics and will sometimes block a legitimate model (for
  example one that pickles a bound method through `builtins.getattr`) or miss a gadget;
- the scanner depends on joblib internals, so a different joblib version may be refused;
- only pickle-based formats (`.joblib`, `.pkl`, `.pickle`) are scanned; other formats are not
  loaded at all;
- only trusted files should be validated, as before.

Observed in `scripts/make_extended_fixtures.py`: with the default profile the unsafe fixture's
payload runs (it prints one line) and the model is a ROLLBACK only because the loaded object
is not a model; with the extended profile it is never loaded.

## CPU time

Predicts the first `evaluation.cpu_sample_rows` (1000) evaluation rows repeatedly until
`evaluation.cpu_min_seconds` (0.3 s) of wall time has passed, then divides the process CPU
time (`time.process_time`, user plus system, all threads) by the rows predicted. The OS
reports process CPU time in coarse steps (about 15.6 ms on Windows), so the figure is only
meaningful over a window of this length; the check reports the CPU/wall ratio (above 1 means
several cores were busy). It is batch prediction, so it differs from the single-row p95
latency check, and it depends on the machine and on other load. It is not a measure of CPU
percentage or of production capacity.

## Profile policy (`configs/extended.yaml`)

- `must_pass_checks`: the default four plus `eval_label_conflicts` (unreliable labels make
  every metric unreliable) and `model_file_safety` (unrecognised code needs a person).
- `critical_rules`: the default four plus `unsafe_model_file` (file safety FAIL).
- `operational.model_file_safety` is `required`: a security check must not be skipped silently.
- Training duplicates and CPU time only change the score.
- Lists replace the default's lists in this configuration format, so the profile restates the
  default rules in full; a test checks they are still a superset.

## Fixtures (`scripts/make_extended_fixtures.py`, `fixtures/extended_manifest.yaml`)

| Fixture | Defect | Default | Extended |
|---|---|---|---|
| `label_conflicts` | 3% of evaluation rows repeated with the opposite label | PROMOTE (blind spot) | HOLD |
| `duplicate_training` | 15% of training hashes repeated | PROMOTE | PROMOTE (FAIL lowers the score only) |
| `unsafe_model` | pickle runs `os.write` when loaded | ROLLBACK, after the payload ran | ROLLBACK, never loaded |
| `unreviewed_model` | refers to `__main__.UnreviewedModel` | HOLD (cannot be loaded here) | HOLD (not loaded; review) |

`unsafe_model/model.joblib` is **deliberately unsafe**: the payload is harmless (one line on
standard output) but it is real code execution on load. The test suite does not load it under
the default profile; that outcome was observed once by hand.

## Not covered

Rollback capability (needs deployment metadata the gate does not receive), conflicting labels
in the training data, a sandboxed or out-of-process load, and models that are not pickles.
