import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, createClient } from "./client";
import { installApi, RESULTS } from "../test/mockApi";

afterEach(() => vi.unstubAllGlobals());

const reply = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(typeof body === "string" ? body : JSON.stringify(body), { status, headers });

async function failure(promise: Promise<unknown>): Promise<ApiError> {
  try {
    await promise;
  } catch (error) {
    return error as ApiError;
  }
  throw new Error("expected a failure");
}

describe("client requests", () => {
  it("sends the API key on every request and reads it at request time", async () => {
    const calls = installApi({ key: "k1" });
    let key: string | null = null;
    const client = createClient(() => key, "/api");
    expect((await failure(client.profiles())).kind).toBe("unauthorized");
    key = "k1";
    await client.profiles();
    expect(calls.map((c) => c.headers.get("X-API-Key"))).toEqual([null, "k1"]);
  });

  it("calls the documented endpoints", async () => {
    const calls = installApi();
    const client = createClient(() => "k", "/api");
    await client.health();
    await client.checks();
    await client.runs(5);
    await client.run(RESULTS.good.run_id);
    expect(await client.report(RESULTS.good.run_id)).toContain("# Model Quality Validation Report");
    expect(calls.map((c) => `${c.method} ${c.path}`)).toEqual([
      "GET /health", "GET /v1/checks", "GET /v1/validations?limit=5",
      `GET /v1/validations/${RESULTS.good.run_id}`, `GET /v1/validations/${RESULTS.good.run_id}/report`,
    ]);
  });

  it("posts validation requests as JSON", async () => {
    const calls = installApi({ validate: () => ({ status: 201, json: { run_id: "x" } }) });
    await createClient(() => "k", "/api").validate({ model_path: "m", eval_data_path: "e" });
    expect(calls[0].method).toBe("POST");
    expect(calls[0].headers.get("Content-Type")).toBe("application/json");
    expect(calls[0].body).toEqual({ model_path: "m", eval_data_path: "e" });
  });
});

describe("candidates and studies", () => {
  it("reads the candidates and the studies", async () => {
    const calls = installApi();
    const client = createClient(() => "k", "/api");
    expect((await client.candidates()).candidates.length).toBe(15);
    expect((await client.studies()).studies.map((s) => s.id)).toContain("baseline_comparison");
    expect((await client.study("baseline_comparison")).data).toHaveProperty("summary");
    expect(await client.studyReport("baseline_comparison")).toContain("# ");
    expect(calls.map((c) => c.path)).toEqual(["/v1/candidates", "/v1/studies", "/v1/studies/baseline_comparison", "/v1/studies/baseline_comparison/report"]);
  });

  it("reports a study that has not been generated as not found, with the script to run", async () => {
    installApi({ missingStudies: ["sensitivity_analysis"] });
    const error = await failure(createClient(() => "k", "/api").study("sensitivity_analysis"));
    expect(error.kind).toBe("notfound");
    expect(error.message).toContain("scripts/sensitivity_analysis.py");
  });
});

describe("error handling", () => {
  const cases: [string, Response, ApiError["kind"], RegExp][] = [
    ["a wrong key", reply(401, { detail: "missing or wrong API key" }), "unauthorized", /wrong API key/],
    ["a request the server refuses", reply(403, { detail: "this is not allowed on this server" }), "forbidden", /not allowed/],
    ["a file over the limit", reply(413, { detail: "the file is larger than the 1 MB limit" }), "toolarge", /larger than/],
    ["a rejected request (string detail)", reply(422, { detail: "model_path: the path is outside the artifact root" }), "invalid", /outside the artifact root/],
    ["a rejected body (field list)", reply(422, { detail: [{ loc: ["body", "profile"], msg: "String should match pattern" }] }), "invalid", /profile: String should match pattern/],
    ["an unconfigured API", reply(503, { detail: "the API key is not configured" }), "unconfigured", /not configured/],
    ["a busy gate", reply(503, { detail: "the gate is busy" }, { "Retry-After": "30" }), "busy", /busy/],
    ["an unknown run", reply(404, { detail: "no such validation run" }), "notfound", /no such validation run/],
    ["a proxy error", reply(502, "Bad Gateway"), "offline", /Cannot reach/],
    ["a server error without a body", reply(500, "oops"), "server", /500/],
  ];
  it.each(cases)("classifies %s", async (_name, response, kind, text) => {
    vi.stubGlobal("fetch", vi.fn(async () => response));
    const error = await failure(createClient(() => "k", "/api").profiles());
    expect(error).toBeInstanceOf(ApiError);
    expect(error.kind).toBe(kind);
    expect(error.message).toMatch(text);
  });

  it("reports an unreachable server as offline", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    const error = await failure(createClient(() => "k", "/api").health());
    expect(error.kind).toBe("offline");
    expect(error.status).toBe(0);
  });

  it("lets an abort through unchanged so a cancelled wait is not shown as an error", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new DOMException("aborted", "AbortError"); }));
    const error = await failure(createClient(() => "k", "/api").health());
    expect(error).toBeInstanceOf(DOMException);
    expect((error as unknown as DOMException).name).toBe("AbortError");
  });
});
