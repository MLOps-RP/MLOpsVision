# C3 Validation Gate: dashboard

A React + Vite (TypeScript) dashboard for the C3 Model Quality & Validation Gate API (`../docs/api.md`), styled
with **Tailwind CSS v4**, charted with **Recharts**, icons from **Lucide**. Clean white surfaces with a blue brand;
green, amber and red are used only for decisions and check statuses, always together with an icon and a label.

It adds no validation logic. Every number comes from the API, and a feature the API or the data cannot provide is
shown as **unavailable**, never as a made-up value.

## Pages

| Page | What it shows | API it uses |
|---|---|---|
| **Dashboard** | the latest recommendation, MQS gauge, three layer scores, MQS-by-run and decision charts, recent validations | `/v1/validations`, `/v1/validations/{id}` |
| **Validate Model** | choose one of the server's registered models ("Model 1, 2, …", with file name and description) and one of the profiles the server will run, each explained in plain language; run. No files are uploaded and no technical fields are entered | `/v1/candidates`, `/v1/profiles`, `/v1/validations` |
| **Validation Results** | decision, score arithmetic, key metrics, every check with status/value/thresholds, rules that fired, explanations, evidence, report, raw JSON | `/v1/validations/{id}` (+ `/report`), `/v1/checks` |
| **Validation History** | searchable, filterable, sortable, paged records; CSV export | `/v1/validations` |
| **Hidden Failure Analysis** | detected subgroups, observed against expected error, excess error with its interval | `/v1/validations/{id}` |
| **Reports & Catalogue** | view validation reports and download each as a **PDF** (`Model-1-Validation-Report.pdf`), built in the browser from the report the API returns; the check catalogue and profiles | `/v1/validations/{id}/report`, `/v1/studies`, `/v1/checks`, `/v1/profiles` |

## Run it

```
# 1. the API (from the C3-Model-Validation folder)
.\.venv\Scripts\Activate.ps1                          # use the project's virtual environment (where mqvg is installed)
$env:MQVG_ARTIFACT_ROOT = (Get-Location).Path       # a real folder: the one the API may read files from
$env:MQVG_API_KEY = "change-me"
python -m mqvg.api                    # http://127.0.0.1:8000

# 2. the dashboard (from this folder)
npm install
npm run dev                           # http://127.0.0.1:5173
```

Open the dashboard, press **API key needed** in the top bar and enter the key. The dev and preview servers proxy
`/api` to the API (default `http://127.0.0.1:8000`, override with `MQVG_API_URL`), so the browser needs no CORS and
the API stays on loopback. **CORS is not enabled** on the API because it is not needed.

| Command | Purpose |
|---|---|
| `npm run dev` | development server |
| `npm run build` | type-check, then a production build in `dist/` (route-split: charts and the Markdown renderer load only where used) |
| `npm run preview` | serve the build (also proxies `/api`) |
| `npm test` | the test suite (Vitest + Testing Library) |
| `npm run typecheck` | TypeScript only |
| `npm run types` | regenerate `src/api/schema.d.ts` from `../docs/api_openapi.json` |

## What is unavailable, and how the dashboard says so

| Situation | What you see |
|---|---|
| The server has no `candidates.yaml`, or it is broken or empty | "No models are registered on this server" / "The list of models could not be used" / "is empty", saying what to fix |
| A listed model's files are missing | the model is listed under "Not available right now" with the reason and cannot be selected |
| A profile is restricted (it does not scan model files) | not offered as a choice; listed under "Not available on this server" with the reason |
| A study (baseline, sensitivity, evaluation) has not been generated | "… is not available", naming the script that generates it; nothing is charted |
| A study file has an unexpected structure | "… is in an unexpected format"; nothing is charted |
| A validation used a profile without hidden-group detection | "Hidden failure detection was not run for this validation"; no groups are implied |
| Hidden-group detection could not run (too few rows or errors, miscalibrated model, ...) | "could not be assessed", and that this is not a pass |
| A run has too little evidence for a quotable MQS | `n/a` on the gauge and in lists, the recorded arithmetic marked "not reliable" with the worst case beside it, layers left out marked "counted as 0%" |
| A check had no evidence | "no evidence" and "no score", grouped by shared reason; never a pass |
| The profile's per-check thresholds | not exposed by the API; shown per run on its Checks tab |

## Security notes

- **The API key is typed in, never built in.** It lives in `sessionStorage` (this tab only; cleared when the tab
  closes) and is sent as the `X-API-Key` header. It is not in the bundle, a URL or `localStorage`. Anything running in
  the page (an XSS flaw, a malicious extension) could read it, so use a key meant for this purpose.
- **The dashboard never chooses files or bypasses restrictions.** It sends exactly the paths the server's own model list
  declares for the selected model, and offers only profiles the server reports as allowed. The server still scans a model
  before loading it, runs only scanning profiles and confines every path to its artifact root; a restricted profile sent
  directly is still refused (422). The dashboard does not use `POST /v1/uploads` (it remains an opt-in server feature).
- **Run it only against an API you trust**, behind the same reverse proxy as the API when it is not on your own machine
  (the static build has no server of its own: serve `dist/` and proxy `/api` from one origin).
- The reports are rendered with `react-markdown`, which does not render raw HTML. CSV exports neutralise cells a
  spreadsheet would run as formulas (candidate names are user-supplied).

## How it is built

```
src/
  api/        client.ts (typed fetch, classified errors), types.ts, studies.ts (validated parsers), schema.d.ts (generated)
  state/      connection.tsx (key, connection status, dialog)
  lib/        format.ts, useAsync.ts, runs.ts, catalogue.ts, csv.ts
  components/ ui.tsx (Tailwind primitives), charts.tsx (Recharts), ScoreGauge, LayerCard, DecisionBanner, ChecksTable,
              RulesPanel, HiddenGroups, ReportView, Layout, Gate, badges
  pages/      Dashboard, ValidateModel, Results, History, Comparison, HiddenFailures, Reports
  test/       mockApi.ts, render.tsx, setup.ts, fixtures/ (REAL results, see below)
```

Types come from the API's own OpenAPI file, so a contract change shows up as a TypeScript error. Every chart is
drawn and also offered as a table ("View data as a table"), which is its accessible description and what the tests
assert against. `Gate` is the single place that decides what to show before data exists (skeleton, a connect notice,
the error, or the data), so a page never shows an empty state too early.

## Tests

`npm test` runs 230 tests: helpers, the API client (error classification, candidates, studies), the study parsers,
components, charts (the SVG is actually rendered), and every page including the model and profile selection (each of the 15 models is checked to send exactly its declared files). The fixtures in
`src/test/fixtures/` are **real**: the sample models were run through the real API and the responses saved (the
machine's absolute path replaced with `C:\artifacts`), and the study files are the ones the real helper scripts wrote.
Checks that tie the UI to the backend: the MQS arithmetic must equal the line the gate writes into its own report for
all five runs, and the chart tables must equal the study files.

The tests mock `fetch`; they do not start the API. Separately, the real UI was driven in headless Chromium against the
real API (desktop and 390 px mobile), with no console errors.

## Limits

- Not tested: browsers other than Chromium, very long histories (the list loads the latest 200), slow networks.
  Keyboard and screen-reader support was designed for (labels, roles, focus, tables behind charts, non-colour status
  cues) but has not been audited with assistive technology.
- A validation is synchronous (up to about a minute); "Stop waiting" ends only the wait, and the server may still
  finish the run.
- No sign-in beyond the shared API key, no per-user history.
- Uploaded files are not deleted by the API; the operator removes them.
