import type {
  CandidateList, CheckInfo, Health, ProfileList, RunList, StudyData, StudyList, ValidationRequest, ValidationResult,
  ValidationSummary,
} from "./types";

export type ErrorKind =
  | "offline" | "unauthorized" | "unconfigured" | "invalid" | "busy" | "notfound" | "forbidden" | "toolarge" | "server";

/** An API failure, classified so the UI can say something useful about it. */
export class ApiError extends Error {
  constructor(public kind: ErrorKind, public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

function detailText(body: unknown): string | null {
  if (!body || typeof body !== "object" || !("detail" in body)) return null;
  const detail = (body as { detail: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        const item = d as { loc?: unknown[]; msg?: string };
        const where = (item.loc ?? []).filter((p) => p !== "body").join(".");
        return where ? `${where}: ${item.msg}` : String(item.msg);
      })
      .join("; ");
  }
  return null;
}

function classify(status: number): ErrorKind {
  if (status === 401) return "unauthorized";
  if (status === 403) return "forbidden";
  if (status === 404) return "notfound";
  if (status === 413) return "toolarge";
  if (status === 422) return "invalid";
  if (status === 503) return "unconfigured";
  return "server";
}

export interface Client {
  health(signal?: AbortSignal): Promise<Health>;
  candidates(): Promise<CandidateList>;
  profiles(): Promise<ProfileList>;
  checks(): Promise<CheckInfo[]>;
  runs(limit?: number): Promise<RunList>;
  run(id: string): Promise<ValidationResult>;
  report(id: string): Promise<string>;
  validate(body: ValidationRequest, signal?: AbortSignal): Promise<ValidationSummary>;
  studies(): Promise<StudyList>;
  study(id: string): Promise<StudyData>;
  studyReport(id: string): Promise<string>;
}

/** `getKey` is read on every request so a key typed later takes effect immediately. */
export function createClient(getKey: () => string | null, base = "/api"): Client {
  async function request(path: string, init: RequestInit = {}): Promise<Response> {
    const headers = new Headers(init.headers);
    const key = getKey();
    if (key) headers.set("X-API-Key", key);
    let response: Response;
    try {
      response = await fetch(`${base}${path}`, { ...init, headers });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") throw error;
      throw new ApiError("offline", 0, "Cannot reach the validation API. Is it running?");
    }
    if (response.ok) return response;
    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      /* not JSON: a proxy error page, for example */
    }
    // 503 is the API's "busy" and "no key configured" answer; the Retry-After header tells them apart.
    const busy = response.status === 503 && response.headers.has("Retry-After");
    const kind = busy ? "busy" : response.status === 502 || response.status === 504 ? "offline" : classify(response.status);
    const fallback = kind === "offline" ? "Cannot reach the validation API. Is it running?" : `The API answered ${response.status}.`;
    throw new ApiError(kind, response.status, detailText(body) ?? fallback);
  }

  const json = async <T,>(path: string, init?: RequestInit) => (await request(path, init)).json() as Promise<T>;
  const text = async (path: string) => (await request(path)).text();

  return {
    health: (signal) => json<Health>("/health", { signal }),
    candidates: () => json<CandidateList>("/v1/candidates"),
    profiles: () => json<ProfileList>("/v1/profiles"),
    checks: () => json<CheckInfo[]>("/v1/checks"),
    runs: (limit = 200) => json<RunList>(`/v1/validations?limit=${limit}`),
    run: (id) => json<ValidationResult>(`/v1/validations/${id}`),
    report: (id) => text(`/v1/validations/${id}/report`),
    validate: (body, signal) =>
      json<ValidationSummary>("/v1/validations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal,
      }),
    studies: () => json<StudyList>("/v1/studies"),
    study: (id) => json<StudyData>(`/v1/studies/${id}`),
    studyReport: (id) => text(`/v1/studies/${id}/report`),
  };
}
