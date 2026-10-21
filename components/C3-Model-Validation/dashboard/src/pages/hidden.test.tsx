import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ValidationResult } from "../api/types";
import { installApi, KEY, RESULTS, runsWith } from "../test/mockApi";
import { renderApp } from "../test/render";

afterEach(() => vi.unstubAllGlobals());

const hiddenRun = RESULTS.hidden_group.run_id;
const withHiddenCheck = (patch: Record<string, unknown>): ValidationResult => ({
  ...RESULTS.hidden_group,
  checks: RESULTS.hidden_group.checks.map((c) => (c.id === "behaviour.hidden_failure_groups" ? { ...c, ...patch } : c)),
}) as ValidationResult;

describe("Hidden Failure Analysis", () => {
  it("shows the confirmed subgroups of the selected validation, with charts and a table", async () => {
    installApi({ key: KEY, runs: runsWith("hidden_group") });
    renderApp(`/hidden-failures?run=${hiddenRun}`);
    expect(await screen.findByText("6 failing subgroups confirmed")).toBeInTheDocument();
    const stat = (label: string) => screen.getByText(label, { selector: "[data-stat-label]" }).closest("section")!;
    expect(stat("Groups tested")).toHaveTextContent("15");
    expect(stat("Confirmed")).toHaveTextContent("6");
    expect(stat("Rows in confirmed groups")).toHaveTextContent("11.9%");
    expect(screen.getByRole("figure", { name: "Error rate inside each group" })).toBeInTheDocument();
    expect(screen.getByRole("figure", { name: "Excess error with its 95% interval" })).toBeInTheDocument();
    const table = screen.getByText("6 confirmed of 15 groups tested").closest("section")!;
    expect(within(table).getByText(/segment = d AND tenure_years > 6\.57 AND income_k > 57/)).toBeInTheDocument();
    expect(within(table).getAllByText("Confirmed", { selector: "span" })).toHaveLength(6);
    expect(screen.getByText(/Configured consequence/)).toHaveTextContent("hold");
  });

  it("states the limits of what a group means", async () => {
    installApi({ key: KEY, runs: runsWith("hidden_group") });
    renderApp(`/hidden-failures?run=${hiddenRun}`);
    expect(await screen.findByText(/association found in the evaluation data, not a causal explanation/)).toBeInTheDocument();
    expect(screen.getByText(/proxies for sensitive characteristics/)).toBeInTheDocument();
    expect(screen.getByText(/Detection power depends on the sample size/)).toBeInTheDocument();
  });

  it("opens on a validation that was run with hidden-group detection when none is chosen", async () => {
    installApi({ key: KEY, runs: runsWith("hidden_group") });
    renderApp("/hidden-failures");
    expect(await screen.findByText("6 failing subgroups confirmed")).toBeInTheDocument();
    expect(screen.getByLabelText("Validation")).toHaveValue(hiddenRun);
  });

  it("says detection was not run, instead of showing nothing or inventing groups, for a profile without it", async () => {
    installApi({ key: KEY, runs: runsWith("good", "hidden_group") });
    renderApp(`/hidden-failures?run=${RESULTS.good.run_id}`);
    expect(await screen.findByText("Hidden failure detection was not run for this validation")).toBeInTheDocument();
    expect(screen.getByText(/does not include the check/)).toBeInTheDocument();
    const notice = screen.getByText("Hidden failure detection was not run for this validation").closest("[role=status]") as HTMLElement;
    expect(within(notice).getByRole("link", { name: "Validate Model" })).toHaveAttribute("href", "/validate");
    expect(screen.queryByRole("figure")).not.toBeInTheDocument();
  });

  it("says it could not be assessed, and that this is not a pass, when the check had no evidence", async () => {
    const result = withHiddenCheck({ status: "UNAVAILABLE", score: null, raw_value: null, evidence_source: "UNAVAILABLE", details: {}, message: "only 700 evaluation rows (at least 1000 are needed)" });
    installApi({ key: KEY, runs: runsWith("hidden_group"), results: { [hiddenRun]: result } });
    renderApp(`/hidden-failures?run=${hiddenRun}`);
    expect(await screen.findByText("Hidden failure detection could not be assessed")).toBeInTheDocument();
    expect(screen.getByText(/only 700 evaluation rows/)).toBeInTheDocument();
    expect(screen.getByText(/not a pass: the model was not shown to be free of failing subgroups/)).toBeInTheDocument();
    expect(screen.queryByRole("figure")).not.toBeInTheDocument();
  });

  it("does not claim a clean bill of health when no group was confirmed", async () => {
    const check = RESULTS.hidden_group.checks.find((c) => c.id === "behaviour.hidden_failure_groups")!;
    const result = withHiddenCheck({ status: "PASS", raw_value: 0, details: { ...check.details, groups: [], groups_confirmed: 0, share_of_rows_in_confirmed_groups: null } });
    installApi({ key: KEY, runs: runsWith("hidden_group"), results: { [hiddenRun]: result } });
    renderApp(`/hidden-failures?run=${hiddenRun}`);
    expect(await screen.findByText("No failing subgroup was confirmed")).toBeInTheDocument();
    expect(screen.getByText(/does not prove none exists/)).toBeInTheDocument();
    expect(screen.queryByRole("figure")).not.toBeInTheDocument();                  // nothing to chart
  });

  it("switches to another validation", async () => {
    installApi({ key: KEY, runs: runsWith("hidden_group", "good") });
    const user = userEvent.setup();
    renderApp(`/hidden-failures?run=${hiddenRun}`);
    await screen.findByText("6 failing subgroups confirmed");
    await user.selectOptions(screen.getByLabelText("Validation"), RESULTS.good.run_id);
    expect(await screen.findByText("Hidden failure detection was not run for this validation")).toBeInTheDocument();
  });

  it("invites a validation when none exists", async () => {
    installApi({ key: KEY, runs: { runs: [] } });
    renderApp("/hidden-failures");
    expect(await screen.findByText("No validations yet")).toBeInTheDocument();
    expect(screen.getByText(/extended_hidden_groups profile/)).toBeInTheDocument();
  });
});
