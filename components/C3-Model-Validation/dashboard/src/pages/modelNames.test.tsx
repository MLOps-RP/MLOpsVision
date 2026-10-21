import { screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CANDIDATES, installApi, KEY, modelName, RESULTS, RUNS, runsWith } from "../test/mockApi";
import { renderApp } from "../test/render";

afterEach(() => vi.unstubAllGlobals());

const TECHNICAL = ["hidden_group", "hidden_group_null", "good_incomplete_metadata", "good_no_sensitive", "label_conflicts"];

describe("friendly model names: one name for one model, on every page", () => {
  it("derives the name from the model's position in the server's list, and matches the files it maps to", () => {
    CANDIDATES.candidates.forEach((c, i) => expect(modelName(c.id)).toBe(`Model ${i + 1}`));
    expect(new Set(CANDIDATES.candidates.map((c) => modelName(c.id))).size).toBe(CANDIDATES.candidates.length);   // unique
    // Every stored run's candidate id is a catalogue id, so each run's name maps to that catalogue entry's own files.
    for (const run of RUNS.runs) expect(CANDIDATES.candidates.some((c) => c.id === run.candidate_id)).toBe(true);
    const good = CANDIDATES.candidates.find((c) => c.id === "good")!;
    expect(RESULTS.good.candidate.model_path.replaceAll("\\", "/")).toMatch(new RegExp(`${good.model_path.replace(".", "\.")}$`));
  });

  it("Dashboard: header, recent table and chart data", async () => {
    installApi({ key: KEY, runs: runsWith("hidden_group") });
    renderApp("/");
    expect(await screen.findByText(/Latest validation:/)).toHaveTextContent(modelName("hidden_group"));
    const recent = (await screen.findByText("Recent validations")).closest("section")!;
    expect(within(recent).getByRole("link", { name: modelName("hidden_group_null") })).toBeInTheDocument();
    for (const raw of TECHNICAL) expect(within(recent).queryByText(raw)).not.toBeInTheDocument();
  });

  it("Validate Model, Validation Results, Validation History, Hidden Failure Analysis and Reports", async () => {
    const run = RUNS.runs.find((r) => r.candidate_id === "hidden_group")!;
    const name = modelName("hidden_group");

    installApi({ key: KEY });
    const validate = renderApp("/validate");
    expect((await screen.findByRole("radio", { name }))).toHaveAttribute("value", "hidden_group");     // the id is what is sent
    validate.unmount();

    installApi({ key: KEY });
    const results = renderApp(`/results/${run.run_id}`);
    expect(await screen.findByRole("heading", { level: 1, name })).toBeInTheDocument();
    results.unmount();

    installApi({ key: KEY });
    const history = renderApp("/history");
    expect(await screen.findByRole("link", { name })).toHaveAttribute("href", `/results/${run.run_id}`);
    expect(screen.queryByText("hidden_group")).not.toBeInTheDocument();
    history.unmount();

    installApi({ key: KEY });
    const hidden = renderApp(`/hidden-failures?run=${run.run_id}`);
    await waitFor(() => expect(screen.getByRole("combobox")).toBeInTheDocument());
    expect(within(screen.getByRole("combobox")).getByRole("option", { name: new RegExp(`^${name} ·`) })).toBeInTheDocument();
    hidden.unmount();

    installApi({ key: KEY });
    renderApp(`/reports?run=${run.run_id}`);
    const picker = await screen.findByRole("combobox", { name: "Validation" });
    expect(within(picker).getByRole("option", { name: new RegExp(`^${name} ·`) })).toBeInTheDocument();
  });

  it("falls back to the stored id, never an invented name, when the server's list is unavailable or lacks the model", async () => {
    installApi({ key: KEY, candidates: { catalogue_found: false, error: null, candidates: [] } });
    renderApp("/history");
    expect(await screen.findByRole("link", { name: "good" })).toBeInTheDocument();
  });
});
