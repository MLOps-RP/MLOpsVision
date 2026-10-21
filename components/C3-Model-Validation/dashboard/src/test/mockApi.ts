import { vi } from "vitest";

import type { CandidateList, CheckInfo, ProfileList, RunList, StudyInfo, ValidationResult } from "../api/types";
import candidates from "./fixtures/candidates.json";
import checks from "./fixtures/checks.json";
import corrupt from "./fixtures/corrupt.result.json";
import corruptReport from "./fixtures/corrupt.report.md?raw";
import good from "./fixtures/good.result.json";
import goodReport from "./fixtures/good.report.md?raw";
import hiddenGroup from "./fixtures/hidden_group.result.json";
import hiddenReport from "./fixtures/hidden_group.report.md?raw";
import leaky from "./fixtures/leaky.result.json";
import leakyReport from "./fixtures/leaky.report.md?raw";
import profiles from "./fixtures/profiles.json";
import runs from "./fixtures/runs.json";
import baselineJson from "./fixtures/study_baseline_comparison.json";
import baselineMd from "./fixtures/study_baseline_comparison.md?raw";
import hiddenResultsMd from "./fixtures/study_hidden_groups_results.md?raw";
import sensitivityJson from "./fixtures/study_sensitivity_analysis.json";
import sensitivityMd from "./fixtures/study_sensitivity_analysis.md?raw";
import unsafeModel from "./fixtures/unsafe_model.result.json";
import unsafeReport from "./fixtures/unsafe_model.report.md?raw";

// Every fixture here is a REAL result: the sample models run through the real API, and the study
// files written by the real helper scripts.
export const RESULTS = {
  good: good as unknown as ValidationResult,
  leaky: leaky as unknown as ValidationResult,
  hidden_group: hiddenGroup as unknown as ValidationResult,
  corrupt: corrupt as unknown as ValidationResult,
  unsafe_model: unsafeModel as unknown as ValidationResult,
};
export const REPORTS: Record<string, string> = {
  good: goodReport, leaky: leakyReport, hidden_group: hiddenReport, corrupt: corruptReport, unsafe_model: unsafeReport,
};
export const RUNS = runs as unknown as RunList;
export const PROFILES = profiles as unknown as ProfileList;
export const CHECKS = checks as unknown as CheckInfo[];
export const CANDIDATES = candidates as unknown as CandidateList;
/** The friendly name every page shows for a candidate id: its position in the server's list. */
export const modelName = (id: string) => `Model ${CANDIDATES.candidates.findIndex((c) => c.id === id) + 1}`;
export const BASELINE = baselineJson as unknown as Record<string, unknown>;
export const SENSITIVITY = sensitivityJson as unknown as Record<string, unknown>;
export const KEY = "test-key";

export const STUDY_INFOS: StudyInfo[] = [
  { id: "baseline_comparison", title: "Baseline comparison", description: "A plain metrics gate against the full gate.", has_data: true, has_report: true, generated_at_utc: (baselineJson as { generated_at_utc: string }).generated_at_utc, script: "scripts/baseline_comparison.py" },
  { id: "sensitivity_analysis", title: "Sensitivity analysis", description: "How stable each recommendation is.", has_data: true, has_report: true, generated_at_utc: (sensitivityJson as { generated_at_utc: string }).generated_at_utc, script: "scripts/sensitivity_analysis.py" },
  { id: "hidden_groups_results", title: "Hidden failure group evaluation", description: "Detection power on synthetic data.", has_data: false, has_report: true, generated_at_utc: null, script: "scripts/evaluate_hidden_groups.py" },
];
const STUDY_REPORTS: Record<string, string> = {
  baseline_comparison: baselineMd, sensitivity_analysis: sensitivityMd, hidden_groups_results: hiddenResultsMd,
};

export interface Call { method: string; path: string; headers: Headers; body: unknown }

interface Options {
  /** If set, requests without this key get a 401. */
  key?: string;
  /** Every request fails like an unreachable server. */
  offline?: boolean;
  /** Answer POST /v1/validations. */
  validate?: (body: unknown) => { status: number; json: unknown; headers?: Record<string, string> };
  runs?: RunList;
  profiles?: ProfileList;
  candidates?: CandidateList;
  /** Study ids to answer 404 for (the study has not been generated). */
  missingStudies?: string[];
  /** Replace a study's JSON payload (to test an unexpected format). */
  studyData?: Record<string, Record<string, unknown>>;
  studies?: StudyInfo[];
  /** Make the POST /v1/validations answer wait for this promise (to test the in-progress state). */
  hold?: Promise<unknown>;
  /** Answer this path with an error instead. */
  fail?: { path: string; status: number; detail: string };
  /** Serve these results (by run_id) instead of the fixtures. */
  results?: Record<string, ValidationResult>;
}

const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });

/** Replace fetch with a fake API serving the real fixtures. Returns the recorded calls. */
export function installApi(options: Options = {}): Call[] {
  const calls: Call[] = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    if (options.offline) throw new TypeError("Failed to fetch");
    const url = new URL(String(input), "http://localhost");
    const path = url.pathname.replace(/^\/api/, "") + url.search;
    const headers = new Headers(init.headers);
    const method = (init.method ?? "GET").toUpperCase();
    const body: unknown = init.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ method, path, headers, body });

    if (options.fail && path === options.fail.path) return json(options.fail.status, { detail: options.fail.detail });
    if (path === "/health") return json(200, { status: "ok", gate_version: "0.1.0", authentication: "api key" });
    if (options.key && headers.get("X-API-Key") !== options.key) return json(401, { detail: "missing or wrong API key" });
    if (path === "/v1/candidates") return json(200, options.candidates ?? CANDIDATES);
    if (path === "/v1/profiles") return json(200, options.profiles ?? PROFILES);
    if (path === "/v1/checks") return json(200, CHECKS);
    if (path.startsWith("/v1/validations?")) return json(200, options.runs ?? RUNS);
    if (path === "/v1/validations" && method === "POST") {
      if (options.hold) {
        await new Promise<void>((resolve, reject) => {
          void options.hold!.then(() => resolve());
          init.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
        });
      }
      const answer = options.validate?.(body);
      return answer ? json(answer.status, answer.json, answer.headers) : json(500, { detail: "no validate handler" });
    }
    if (path === "/v1/studies") return json(200, { studies: options.studies ?? STUDY_INFOS });
    const study = path.match(/^\/v1\/studies\/([a-z_]+)(\/report)?$/);
    if (study) {
      const [, id, report] = study;
      const info = STUDY_INFOS.find((s) => s.id === id);
      if (!info) return json(404, { detail: "no such study" });
      if (options.missingStudies?.includes(id)) {
        return json(404, { detail: report ? `the report has not been generated; run ${info.script}` : `this study has no machine-readable result; generate it with ${info.script}` });
      }
      if (report) return new Response(STUDY_REPORTS[id], { status: 200, headers: { "Content-Type": "text/markdown" } });
      const data = options.studyData?.[id] ?? (id === "baseline_comparison" ? BASELINE : id === "sensitivity_analysis" ? SENSITIVITY : null);
      if (!data) return json(404, { detail: "this study has no machine-readable result, only a report" });
      return json(200, { id, title: info.title, generated_at_utc: info.generated_at_utc, data });
    }
    const match = path.match(/^\/v1\/validations\/([0-9a-f]{32})(\/report)?$/);
    if (match) {
      const name = (Object.keys(RESULTS) as (keyof typeof RESULTS)[]).find((n) => RESULTS[n].run_id === match[1]);
      if (!name) return json(404, { detail: "no such validation run" });
      return match[2]
        ? new Response(REPORTS[name], { status: 200, headers: { "Content-Type": "text/markdown" } })
        : json(200, options.results?.[match[1]] ?? RESULTS[name]);
    }
    return json(404, { detail: "not found" });
  }));
  return calls;
}

/** The stored runs with the named candidates first (the dashboard treats the first as the latest). */
export function runsWith(...names: string[]): RunList {
  const first = names.flatMap((n) => RUNS.runs.filter((r) => r.candidate_id === n));
  return { runs: [...first, ...RUNS.runs.filter((r) => !first.includes(r))] };
}
