"""HTTP interface to the validation gate.

A thin wrapper: every validation is `mqvg.workflow.validate_and_report`, and the
answer is a RECOMMENDATION (PROMOTE / HOLD / ROLLBACK). Nothing here deploys,
promotes or rolls back a model, and there is no endpoint that could.

SECURITY. Validating a model loads it, and loading a pickle runs the code inside
it. An endpoint that loads a model from a path is therefore a code-execution
surface, so this API is restricted by construction (see `docs/api.md`):

* model and data paths are relative to one configured artifact root and may not
  leave it (including through symlinks);
* a client chooses a profile by name from a fixed folder, never a config file path;
* by default only profiles that scan the model file before loading it are allowed;
* every endpoint but /health needs the API key, and without a configured key the
  API refuses to work (fail closed) unless explicitly told it is for local use;
* the server binds to the loopback interface unless an API key is set.

None of this makes it safe to validate untrusted models. The scan is a heuristic,
not a sandbox, and the API should run as a low-privilege user, preferably in a
container, on a machine that holds nothing the models should not reach.
"""
