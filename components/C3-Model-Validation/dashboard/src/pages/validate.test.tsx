import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { CandidateInfo, ProfileList } from "../api/types";
import { CANDIDATES, installApi, KEY, PROFILES, RESULTS, type Call } from "../test/mockApi";
import { renderApp } from "../test/render";

afterEach(() => vi.unstubAllGlobals());

const goodSummary = { status: 201, json: { run_id: RESULTS.good.run_id } };
const posts = (calls: Call[]) => calls.filter((c) => c.method === "POST");
const radio = (name: RegExp | string) => screen.getByRole("radio", { name });
const ready = async () => { await screen.findByRole("radiogroup", { name: "Models" }); };
const ids = CANDIDATES.candidates.map((c) => c.id);
const modelNumber = (id: string) => ids.indexOf(id) + 1;
const profilesWith = (patch: (p: ProfileList) => void): ProfileList => { const copy = structuredClone(PROFILES); patch(copy); return copy; };
const candidate = (over: Partial<CandidateInfo> & { id: string }): CandidateInfo => ({
  description: "A test model.", model_path: `m/${over.id}/model.joblib`, eval_data_path: `d/${over.id}.csv`, metadata_path: `m/${over.id}/meta.json`,
  train_row_hashes_path: null, suggested_profile: null, available: true, unavailable_reason: null, ...over });

describe("Validate Model: what the page offers", () => {
  it("lists every model the server returns, with friendly names and the real file name as secondary text", async () => {
    installApi({ key: KEY });
    renderApp("/validate");
    await ready();
    const radios = screen.getAllByRole("radio").filter((r) => (r as HTMLInputElement).name === "candidate");
    expect(radios).toHaveLength(CANDIDATES.candidates.length);
    expect(radios.map((r) => r.getAttribute("aria-labelledby") && document.getElementById(r.getAttribute("aria-labelledby")!)!.textContent))
      .toEqual(ids.map((_, i) => `Model ${i + 1}`));
    const first = radio("Model 1").closest("label")!;
    expect(first).toHaveTextContent("A well-behaved baseline classifier on the Adult income data.");
    expect(within(first).getByText("…/good/model.joblib")).toBeInTheDocument();          // the real file, as secondary information
    expect(within(first).getByTitle("Model file: fixtures/good/model.joblib")).toBeInTheDocument();
  });

  it("uses friendly names instead of technical ids or file names as the model's title", async () => {
    installApi({ key: KEY });
    renderApp("/validate");
    await ready();
    const titles = screen.getAllByText(/^Model \d+$/).map((e) => e.textContent);
    expect(titles).toHaveLength(15);
    expect(screen.queryByText("good", { selector: "span" })).not.toBeInTheDocument();
    expect(screen.queryByText("hidden_group", { selector: "span" })).not.toBeInTheDocument();
  });

  it("numbers models by their position in the server's list", async () => {
    const list = { catalogue_found: true, error: null, candidates: [candidate({ id: "zeta" }), candidate({ id: "alpha" }), candidate({ id: "mid" })] };
    installApi({ key: KEY, candidates: list });
    renderApp("/validate");
    await ready();
    expect(radio("Model 1")).toHaveAttribute("value", "zeta");
    expect(radio("Model 2")).toHaveAttribute("value", "alpha");
    expect(radio("Model 3")).toHaveAttribute("value", "mid");
  });

  it("has no upload controls and none of the technical inputs", async () => {
    installApi({ key: KEY });
    renderApp("/validate");
    await ready();
    expect(document.querySelector('input[type="file"]')).toBeNull();
    expect(screen.queryByRole("tab")).not.toBeInTheDocument();
    expect(screen.queryByText(/upload/i)).not.toBeInTheDocument();
    for (const gone of [/Target column/i, /Positive label/i, /Sensitive columns/i, /Candidate name/i]) {
      expect(screen.queryByLabelText(gone)).not.toBeInTheDocument();
      expect(screen.queryByText(gone)).not.toBeInTheDocument();
    }
    expect(document.querySelector('input[type="text"]')).toBeNull();                 // nothing to type
  });

  it("does not show the long security warning", async () => {
    installApi({ key: KEY });
    renderApp("/validate");
    await ready();
    expect(screen.queryByText(/Validating a model loads it/)).not.toBeInTheDocument();
    expect(screen.queryByText(/heuristic, not a sandbox/)).not.toBeInTheDocument();
    expect(screen.queryByRole("status", { name: "" })).toBeNull();
  });
});

describe("Validate Model: choosing a model", () => {
  it("starts with nothing chosen and cannot be submitted", async () => {
    installApi({ key: KEY });
    renderApp("/validate");
    await ready();
    expect(screen.getAllByRole("radio").filter((r) => (r as HTMLInputElement).name === "candidate").every((r) => !(r as HTMLInputElement).checked)).toBe(true);
    expect(screen.getByRole("button", { name: /Validate model/ })).toBeDisabled();
    expect(screen.getByText("Choose a model to continue.")).toBeInTheDocument();
  });

  it("shows an obvious selected state and says what will run", async () => {
    installApi({ key: KEY });
    const user = userEvent.setup();
    renderApp("/validate");
    await ready();
    await user.click(radio("Model 3"));
    expect(radio("Model 3")).toBeChecked();
    expect(radio("Model 3").closest("label")).toHaveClass("border-blue-600");
    expect(radio("Model 2").closest("label")).not.toHaveClass("border-blue-600");
    expect(screen.getByText(/Ready:/).parentElement).toHaveTextContent("Model 3 (…/overfit/model.joblib) with the Extended profile.");
    expect(screen.getByRole("button", { name: /Validate model/ })).toBeEnabled();
    await user.click(radio("Model 5"));
    expect(radio("Model 3")).not.toBeChecked();
    expect(radio("Model 5")).toBeChecked();
  });

  it.each(CANDIDATES.candidates.map((c) => [c.id]))("sends exactly the files the server declares for %s", async (id) => {
    const calls = installApi({ key: KEY, validate: () => goodSummary });
    const declared = CANDIDATES.candidates.find((c) => c.id === id)!;
    const user = userEvent.setup();
    renderApp("/validate");
    await ready();
    await user.click(radio(`Model ${modelNumber(id)}`));
    await user.click(screen.getByRole("button", { name: /Validate model/ }));
    await screen.findByRole("region", { name: "Decision" });
    const body = posts(calls)[0].body as Record<string, unknown>;
    expect(body).toEqual({
      candidate_id: declared.id, model_path: declared.model_path, eval_data_path: declared.eval_data_path,
      metadata_path: declared.metadata_path, train_row_hashes_path: declared.train_row_hashes_path,
      profile: declared.suggested_profile ?? "extended",
    });
    expect(Object.keys(body)).not.toContain("target_column");                      // nothing the backend defaults are overridden by
    expect(Object.keys(body)).not.toContain("positive_label");
    expect(Object.keys(body)).not.toContain("sensitive_columns");
  });

  it("sends an explicit null, not a guess, for a model that has no training-hash file", async () => {
    const calls = installApi({ key: KEY, validate: () => goodSummary });
    const user = userEvent.setup();
    renderApp("/validate");
    await ready();
    await user.click(radio(`Model ${modelNumber("good_incomplete_metadata")}`));
    await user.click(screen.getByRole("button", { name: /Validate model/ }));
    await screen.findByRole("region", { name: "Decision" });
    expect(posts(calls)[0].body).toMatchObject({ train_row_hashes_path: null, metadata_path: "fixtures/good_incomplete_metadata/metadata.json" });
  });

  it("keeps models that share one model file separate: each gets its own evaluation data", async () => {
    const calls = installApi({ key: KEY, validate: () => goodSummary });
    const user = userEvent.setup();
    renderApp("/validate");
    await ready();
    await user.click(radio(`Model ${modelNumber("label_conflicts")}`));
    await user.click(screen.getByRole("button", { name: /Validate model/ }));
    await screen.findByRole("region", { name: "Decision" });
    expect(posts(calls)[0].body).toMatchObject({
      model_path: "fixtures/good/model.joblib", eval_data_path: "fixtures/label_conflicts/eval.csv", candidate_id: "label_conflicts" });
  });

  it("lists a model whose files are missing as unavailable, with the reason, and cannot select it", async () => {
    const list = { catalogue_found: true, error: null, candidates: [
      candidate({ id: "fine" }),
      candidate({ id: "broken", available: false, unavailable_reason: "the evaluation data was not found" })] };
    installApi({ key: KEY, candidates: list });
    const user = userEvent.setup();
    renderApp("/validate");
    await ready();
    expect(screen.getByText("Not available right now (1)")).toBeInTheDocument();
    const broken = radio("Model 2");
    expect(broken).toBeDisabled();
    expect(broken.closest("label")).toHaveTextContent("Unavailable");
    expect(broken.closest("label")).toHaveTextContent("Not available: the evaluation data was not found.");
    await user.click(broken.closest("label")!);
    expect(broken).not.toBeChecked();
    expect(screen.getByRole("button", { name: /Validate model/ })).toBeDisabled();
  });

  it("says so when none of the listed models can be used", async () => {
    installApi({ key: KEY, candidates: { catalogue_found: true, error: null, candidates: [candidate({ id: "x", available: false, unavailable_reason: "the model file was not found" })] } });
    renderApp("/validate");
    expect(await screen.findByText("None of the listed models can be used right now")).toBeInTheDocument();
  });

  it("explains a server that has no list of models", async () => {
    installApi({ key: KEY, candidates: { catalogue_found: false, error: null, candidates: [] } });
    renderApp("/validate");
    expect(await screen.findByText("No models are registered on this server")).toBeInTheDocument();
    expect(screen.getByText(/candidates\.yaml/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Validate model/ })).toBeDisabled();
  });

  it("explains a list of models that could not be read", async () => {
    installApi({ key: KEY, candidates: { catalogue_found: true, error: "the candidate list repeats these ids: good", candidates: [] } });
    renderApp("/validate");
    expect(await screen.findByText("The list of models could not be used")).toBeInTheDocument();
    expect(screen.getByText(/repeats these ids: good/)).toBeInTheDocument();
  });

  it("explains an empty list of models", async () => {
    installApi({ key: KEY, candidates: { catalogue_found: true, error: null, candidates: [] } });
    renderApp("/validate");
    expect(await screen.findByText("The list of models is empty")).toBeInTheDocument();
  });

  it("shows a loading state first", async () => {
    installApi({ key: KEY });
    renderApp("/validate");
    expect(screen.getByLabelText("Loading")).toBeInTheDocument();
    await ready();
    expect(screen.queryByLabelText("Loading")).not.toBeInTheDocument();
  });

  it("shows an error, with a way to retry, when the server cannot list its models", async () => {
    installApi({ key: KEY, fail: { path: "/v1/candidates", status: 500, detail: "the candidate list is broken" } });
    renderApp("/validate");
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("The API reported an error");
    expect(alert).toHaveTextContent("the candidate list is broken");
    expect(within(alert).getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });
});

describe("Validate Model: choosing a profile", () => {
  it("offers only the profiles this server will run, and preselects the server's default", async () => {
    installApi({ key: KEY });
    renderApp("/validate");
    await ready();
    const profileRadios = screen.getAllByRole("radio").filter((r) => (r as HTMLInputElement).name === "profile");
    expect(profileRadios.map((r) => r.getAttribute("value"))).toEqual(PROFILES.profiles.filter((p) => p.allowed).map((p) => p.name));
    expect(radio("Extended")).toBeChecked();
    expect(radio("Extended").closest("label")).toHaveTextContent("Server default");
    expect(radio("Extended hidden groups")).not.toBeChecked();
  });

  it("explains each profile in simple language, with its strictness", async () => {
    installApi({ key: KEY });
    renderApp("/validate");
    await ready();
    const card = radio("Extended hidden groups").closest("label")!;
    expect(within(card).getByText("Standard")).toBeInTheDocument();
    expect(card).toHaveTextContent(/Runs 26 checks of training quality, model behaviour and operational readiness/);
    expect(card).toHaveTextContent(/Recommends promotion at a score of 75 or more and rollback below 50/);
    expect(card).toHaveTextContent("Scans the model file before loading it");
    expect(card).toHaveTextContent(/Looks for hidden failing groups of rows \(a failing group puts the model on hold\)/);
    expect(within(card).getByText("extended_hidden_groups")).toBeInTheDocument();         // the technical name, secondary
  });

  it("does not offer restricted profiles as choices, and says why they are unavailable", async () => {
    installApi({ key: KEY });
    renderApp("/validate");
    await ready();
    for (const name of ["default", "hidden_groups", "hidden_groups_critical", "hidden_groups_score_only", "strict"]) {
      expect(document.querySelector(`input[name="profile"][value="${name}"]`)).toBeNull();
    }
    const details = screen.getByText("Not available on this server (5)").closest("details")!;
    expect(details).not.toHaveAttribute("open");                                       // tucked away, not shown as options
    for (const name of ["default", "hidden_groups", "hidden_groups_critical", "hidden_groups_score_only", "strict"]) {
      expect(within(details).getByText(name)).toBeInTheDocument();
    }
    expect(within(details).getAllByText(/without scanning it first/)).toHaveLength(1);       // one reason, shared
    expect(within(details).getByText(/runs the code inside it/)).toBeInTheDocument();
  });

  it("describes a strict or lenient profile the server provides", async () => {
    const profiles = profilesWith((p) => {
      const ext = p.profiles.find((x) => x.name === "extended")!;
      ext.strictness = "strict";
      p.profiles.find((x) => x.name === "extended_hidden_groups")!.strictness = "lenient";
      expect(ext.allowed).toBe(true);
    });
    installApi({ key: KEY, profiles });
    renderApp("/validate");
    await ready();
    expect(within(radio("Extended").closest("label")!).getByText("Strict")).toBeInTheDocument();
    expect(within(radio("Extended hidden groups").closest("label")!).getByText("Lenient")).toBeInTheDocument();
  });

  it("falls back to the first available profile when the server's default is restricted", async () => {
    installApi({ key: KEY, profiles: profilesWith((p) => { p.default_profile = "default"; }) });
    renderApp("/validate");
    await ready();
    expect(radio("Extended")).toBeChecked();
    expect(screen.queryByText("Server default")).not.toBeInTheDocument();
  });

  it("sends the profile the user picked", async () => {
    const calls = installApi({ key: KEY, validate: () => goodSummary });
    const user = userEvent.setup();
    renderApp("/validate");
    await ready();
    await user.click(radio("Model 1"));
    await user.click(radio("Extended hidden groups"));
    expect(radio("Extended hidden groups")).toBeChecked();
    await user.click(screen.getByRole("button", { name: /Validate model/ }));
    await screen.findByRole("region", { name: "Decision" });
    expect(posts(calls)[0].body).toMatchObject({ candidate_id: "good", profile: "extended_hidden_groups" });
  });

  it("preselects the profile the server suggests for a model, and resets it for another", async () => {
    installApi({ key: KEY });
    const user = userEvent.setup();
    renderApp("/validate");
    await ready();
    await user.click(radio(`Model ${modelNumber("hidden_group")}`));
    expect(radio("Extended hidden groups")).toBeChecked();
    expect(screen.getByText(/Suggested for Model 12:/).parentElement).toHaveTextContent("Extended hidden groups");
    await user.click(radio("Extended"));                                              // the user overrides it
    expect(radio("Extended")).toBeChecked();
    await user.click(radio("Model 1"));
    expect(radio("Extended")).toBeChecked();                                          // back to the default for a model with no suggestion
    expect(screen.queryByText(/Suggested for/)).not.toBeInTheDocument();
  });

  it("does not use a suggested profile the server does not allow, and says so", async () => {
    const profiles = profilesWith((p) => { p.profiles.find((x) => x.name === "extended_hidden_groups")!.allowed = false; p.profiles.find((x) => x.name === "extended_hidden_groups")!.unavailable_reason = "Restricted for the test."; });
    const calls = installApi({ key: KEY, profiles, validate: () => goodSummary });
    const user = userEvent.setup();
    renderApp("/validate");
    await ready();
    await user.click(radio(`Model ${modelNumber("hidden_group")}`));
    expect(radio("Extended")).toBeChecked();
    expect(screen.getByText(/The profile suggested for Model 12 \(extended_hidden_groups\) is not available on this server/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Validate model/ }));
    await screen.findByRole("region", { name: "Decision" });
    expect(posts(calls)[0].body).toMatchObject({ profile: "extended" });             // the restricted profile is never sent
  });

  it("cannot start a validation when no profile is available", async () => {
    const profiles = profilesWith((p) => { for (const x of p.profiles) { x.allowed = false; x.unavailable_reason = "Restricted for the test."; } });
    installApi({ key: KEY, profiles });
    const user = userEvent.setup();
    renderApp("/validate");
    await ready();
    expect(screen.getByText("No validation profile is available on this server")).toBeInTheDocument();
    await user.click(radio("Model 1"));
    expect(screen.getByRole("button", { name: /Validate model/ })).toBeDisabled();
    expect(screen.getByText("Not available on this server (7)")).toBeInTheDocument();
  });
});

describe("Validate Model: running a validation", () => {
  it("opens the new result when the validation finishes", async () => {
    installApi({ key: KEY, validate: () => goodSummary });
    const user = userEvent.setup();
    renderApp("/validate");
    await ready();
    await user.click(radio("Model 1"));
    await user.click(screen.getByRole("button", { name: /Validate model/ }));
    expect(await screen.findByRole("region", { name: "Decision" })).toHaveTextContent("PROMOTE");
  });

  it("shows progress and locks the choices while it runs, then lets the user stop waiting", async () => {
    let release!: () => void;
    const hold = new Promise<void>((resolve) => { release = resolve; });
    const calls = installApi({ key: KEY, hold, validate: () => goodSummary });
    const user = userEvent.setup();
    renderApp("/validate");
    await ready();
    await user.click(radio("Model 2"));
    await user.click(screen.getByRole("button", { name: /Validate model/ }));
    expect(await screen.findByText(/Validating Model 2…/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Validate model/ })).toBeDisabled();
    expect(radio("Model 1")).toBeDisabled();                                          // the choices cannot change mid-run
    expect(radio("Extended")).toBeDisabled();
    expect(screen.getByText(/server may still finish the run/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Stop waiting" }));
    await waitFor(() => expect(screen.getByRole("button", { name: /Validate model/ })).toBeEnabled());
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();                     // stopping is not an error
    expect(radio("Model 2")).toBeChecked();                                           // the choice is kept
    release();
    expect(posts(calls)).toHaveLength(1);
  });

  it("shows the API's reason when a request is rejected, keeps the choices and allows another try", async () => {
    installApi({ key: KEY, validate: () => ({ status: 422, json: { detail: "profile 'x' does not scan model files" } }) });
    const user = userEvent.setup();
    renderApp("/validate");
    await ready();
    await user.click(radio("Model 4"));
    await user.click(screen.getByRole("button", { name: /Validate model/ }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("The request was rejected");
    expect(alert).toHaveTextContent("does not scan model files");
    expect(radio("Model 4")).toBeChecked();
    expect(screen.getByRole("button", { name: /Validate model/ })).toBeEnabled();
  });

  it("tells the user when the gate is busy", async () => {
    installApi({ key: KEY, validate: () => ({ status: 503, json: { detail: "the gate is busy with another validation" }, headers: { "Retry-After": "30" } }) });
    const user = userEvent.setup();
    renderApp("/validate");
    await ready();
    await user.click(radio("Model 1"));
    await user.click(screen.getByRole("button", { name: /Validate model/ }));
    expect(await screen.findByText("The gate is busy")).toBeInTheDocument();
  });

  it("asks for the key instead of showing models when the key is wrong", async () => {
    installApi({ key: KEY });
    renderApp("/validate", "wrong-key");
    expect(await screen.findByText("Enter the API key to continue")).toBeInTheDocument();
    expect(screen.queryByRole("radio")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Validate model/ })).not.toBeInTheDocument();
  });
});
