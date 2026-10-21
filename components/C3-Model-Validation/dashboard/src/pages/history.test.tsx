import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { RunList } from "../api/types";
import { downloads } from "../test/setup";
import { installApi, KEY, modelName, RUNS } from "../test/mockApi";
import { renderApp } from "../test/render";

afterEach(() => vi.unstubAllGlobals());

const blobText = (blob: Blob) => new Promise<string>((resolve) => { const r = new FileReader(); r.onload = () => resolve(String(r.result)); r.readAsText(blob); });
const bodyRows = () => within(screen.getByRole("table")).getAllByRole("row").slice(1);
const names = () => bodyRows().map((r) => within(r).getAllByRole("cell")[0].textContent);
const many = (n: number): RunList => ({
  runs: Array.from({ length: n }, (_, i) => ({ ...RUNS.runs[i % RUNS.runs.length], run_id: i.toString(16).padStart(32, "0"), candidate_id: `model-${i}` })),
});

describe("Validation History", () => {
  it("lists every stored run", async () => {
    installApi({ key: KEY });
    renderApp("/history");
    await screen.findByRole("table");
    expect(bodyRows()).toHaveLength(RUNS.runs.length);
    expect(screen.getByText(`${RUNS.runs.length} of ${RUNS.runs.length}`)).toBeInTheDocument();
  });

  it("searches by candidate, profile and decision", async () => {
    installApi({ key: KEY });
    const user = userEvent.setup();
    renderApp("/history");
    const search = await screen.findByRole("searchbox", { name: "Search runs" });
    await user.type(search, "hidden");
    expect(names()).toEqual(RUNS.runs.filter((r) => `${r.candidate_id} ${r.profile}`.includes("hidden")).map((r) => modelName(r.candidate_id as string)));
    await user.clear(search);
    await user.type(search, "rollback");
    expect(bodyRows()).toHaveLength(RUNS.runs.filter((r) => r.decision === "ROLLBACK").length);
    await user.clear(search);
    await user.type(search, "zzz-nothing");
    expect(screen.getByText("No runs match these filters")).toBeInTheDocument();
  });

  it("filters by decision and by profile", async () => {
    installApi({ key: KEY });
    const user = userEvent.setup();
    renderApp("/history");
    await screen.findByRole("table");
    await user.click(screen.getByRole("button", { name: "HOLD" }));
    expect(bodyRows()).toHaveLength(RUNS.runs.filter((r) => r.decision === "HOLD").length);
    await user.click(screen.getByRole("button", { name: "All" }));
    await user.selectOptions(screen.getByLabelText("Profile"), "extended_hidden_groups");
    expect(bodyRows()).toHaveLength(RUNS.runs.filter((r) => r.profile === "extended_hidden_groups").length);
  });

  it("sorts by score, always putting runs without a score last", async () => {
    installApi({ key: KEY });
    const user = userEvent.setup();
    renderApp("/history");
    await screen.findByRole("table");
    const scored = RUNS.runs.filter((r) => r.mqs !== null).map((r) => r.mqs as number);
    const mqsColumn = () => bodyRows().map((r) => within(r).getAllByRole("cell")[2].textContent);

    await user.selectOptions(screen.getByLabelText("Sort"), "mqs-high");
    expect(mqsColumn().slice(0, scored.length)).toEqual([...scored].sort((a, b) => b - a).map((v) => v.toFixed(1)));
    expect(mqsColumn().slice(scored.length).every((c) => c === "n/a")).toBe(true);

    await user.selectOptions(screen.getByLabelText("Sort"), "mqs-low");
    expect(mqsColumn().slice(0, scored.length)).toEqual([...scored].sort((a, b) => a - b).map((v) => v.toFixed(1)));
    expect(mqsColumn().slice(scored.length).every((c) => c === "n/a")).toBe(true);

    await user.selectOptions(screen.getByLabelText("Sort"), "oldest");
    expect(names()[0]).toBe([...RUNS.runs].sort((a, b) => +new Date(a.created_at_utc) - +new Date(b.created_at_utc))[0].candidate_id as string ? modelName([...RUNS.runs].sort((a, b) => +new Date(a.created_at_utc) - +new Date(b.created_at_utc))[0].candidate_id as string) : "");
  });

  it("pages through a long history and returns to page one when the filter changes", async () => {
    installApi({ key: KEY, runs: many(40) });
    const user = userEvent.setup();
    renderApp("/history");
    await screen.findByRole("table");
    expect(bodyRows()).toHaveLength(15);
    expect(screen.getByText("Page 1 of 3")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Previous/ })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: /Next/ }));
    expect(screen.getByText("Page 2 of 3")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Next/ }));
    expect(bodyRows()).toHaveLength(10);
    expect(screen.getByRole("button", { name: /Next/ })).toBeDisabled();
    await user.type(screen.getByRole("searchbox", { name: "Search runs" }), "model-1");
    expect(screen.queryByText(/Page 3 of/)).not.toBeInTheDocument();
    expect(names().every((n) => n!.startsWith("model-1"))).toBe(true);
  });

  it("exports the filtered list as CSV and neutralises spreadsheet formulas in names", async () => {
    const hostile = { runs: [{ ...RUNS.runs[0], candidate_id: "=HYPERLINK(\"http://evil\")" }, ...RUNS.runs.slice(1, 3)] };
    installApi({ key: KEY, runs: hostile });
    const user = userEvent.setup();
    renderApp("/history");
    await screen.findByRole("table");
    await user.click(screen.getByRole("button", { name: /Export CSV/ }));
    expect(downloads[0].name).toBe("validation-history.csv");
    const lines = (await blobText(downloads[0].blob)).trim().split("\r\n");
    expect(lines[0]).toBe("run_id,model,candidate,profile,decision,mqs,created_at_utc");
    expect(lines).toHaveLength(4);
    expect(lines[1]).toContain(`"'=HYPERLINK(""http://evil"")"`);              // quoted, with a leading apostrophe
  });

  it("only exports what the filters show", async () => {
    installApi({ key: KEY });
    const user = userEvent.setup();
    renderApp("/history");
    await screen.findByRole("table");
    await user.click(screen.getByRole("button", { name: "PROMOTE" }));
    await user.click(screen.getByRole("button", { name: /Export CSV/ }));
    const lines = (await blobText(downloads[0].blob)).trim().split("\r\n");
    expect(lines).toHaveLength(1 + RUNS.runs.filter((r) => r.decision === "PROMOTE").length);
  });

  it("shows an empty state, and cannot export nothing", async () => {
    installApi({ key: KEY, runs: { runs: [] } });
    renderApp("/history");
    expect(await screen.findByText("No validations yet")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Export CSV/ })).toBeDisabled();
  });

  it("opens a run when its row is clicked", async () => {
    installApi({ key: KEY });
    const user = userEvent.setup();
    renderApp("/history");
    await screen.findByRole("table");
    await user.click(screen.getByRole("link", { name: modelName("good") }));
    expect(await screen.findByRole("tab", { name: "Overview" })).toBeInTheDocument();
  });
});
