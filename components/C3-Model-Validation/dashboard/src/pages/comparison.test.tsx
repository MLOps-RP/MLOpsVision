import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { BASELINE, installApi, KEY, modelName, SENSITIVITY, runsWith } from "../test/mockApi";
import { renderPage } from "../test/render";
import { Comparison } from "./Comparison";

afterEach(() => vi.unstubAllGlobals());

describe("Model Comparison: run against run", () => {
  it("compares the two newest runs by default, with a table and a chart", async () => {
    installApi({ key: KEY, runs: runsWith("good", "leaky") });
    renderPage(<Comparison />, "/comparison");
    const table = (await screen.findByText("Side by side")).closest("section")!;
    const rows = within(table).getAllByRole("row").slice(1);
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveTextContent(modelName("good"));
    expect(rows[0]).toHaveTextContent("PROMOTE");
    expect(rows[0]).toHaveTextContent("86.9");
    expect(rows[1]).toHaveTextContent(modelName("leaky"));
    expect(rows[1]).toHaveTextContent("ROLLBACK");
    expect(rows[1]).toHaveTextContent("85.7");
    expect(screen.getByRole("figure", { name: "Scores side by side" })).toBeInTheDocument();
    expect(screen.getByText(/compare scores only when the profiles match/)).toBeInTheDocument();
  });

  it("lists only the checks whose status differs between the runs", async () => {
    installApi({ key: KEY, runs: runsWith("good", "leaky") });
    renderPage(<Comparison />, "/comparison");
    const differing = (await screen.findByText("Checks that differ")).closest("section")!;
    expect(within(differing).getByText("leakage.single_feature_predictiveness")).toBeInTheDocument();   // Pass vs Fail
    expect(within(differing).queryByText("operational.model_loadable")).not.toBeInTheDocument();        // Pass in both
    const row = within(differing).getByText("leakage.single_feature_predictiveness").closest("tr")!;
    expect(row).toHaveTextContent("Pass");
    expect(row).toHaveTextContent("Fail");
  });

  it("shows a score that is not available as n/a and a check that was not run as a dash", async () => {
    installApi({ key: KEY, runs: runsWith("good", "corrupt") });
    renderPage(<Comparison />, "/comparison");
    const table = (await screen.findByText("Side by side")).closest("section")!;
    const corrupt = within(table).getAllByRole("row")[2];
    expect(corrupt).toHaveTextContent("n/a");
    expect(corrupt).toHaveTextContent("ROLLBACK");
  });

  it("lets the user change the selection, up to four runs", async () => {
    installApi({ key: KEY, runs: runsWith("good", "leaky") });
    const user = userEvent.setup();
    renderPage(<Comparison />, "/comparison");
    await screen.findByText("Side by side");
    const box = (name: string) => screen.getByRole("checkbox", { name: new RegExp(`^${name}\\b`) });
    await user.click(box(modelName("hidden_group")));
    await user.click(box(modelName("corrupt")));
    expect(screen.getByText(/4 selected/)).toBeInTheDocument();
    expect(box(modelName("overfit"))).toBeDisabled();                                       // the limit
    await user.click(box(modelName("leaky")));
    expect(screen.getByText(/3 selected/)).toBeInTheDocument();
    expect(box(modelName("overfit"))).toBeEnabled();
  });

  it("asks for at least two runs", async () => {
    installApi({ key: KEY, runs: runsWith("good", "leaky") });
    const user = userEvent.setup();
    renderPage(<Comparison />, "/comparison");
    await screen.findByText("Side by side");
    await user.click(screen.getByRole("checkbox", { name: new RegExp(`^${modelName("leaky")}\\b`) }));
    expect(await screen.findByText("Select at least two runs")).toBeInTheDocument();
  });

  it("needs at least two stored runs to compare", async () => {
    installApi({ key: KEY, runs: { runs: runsWith("good").runs.slice(0, 1) } });
    renderPage(<Comparison />, "/comparison");
    expect(await screen.findByText("Validate at least two models to compare them")).toBeInTheDocument();
  });
});

describe("Model Comparison: the baseline gate (real study file)", () => {
  const open = async (options = {}) => {
    installApi({ key: KEY, ...options });
    const user = userEvent.setup();
    renderPage(<Comparison />, "/comparison");
    await user.click(await screen.findByRole("tab", { name: "Baseline gate" }));
    return user;
  };

  it("charts defects stopped and false alarms per gate, from the study", async () => {
    await open();
    expect(await screen.findByRole("figure", { name: "Defects stopped and false alarms, by gate" })).toBeInTheDocument();
    expect(screen.getByText("Real study results")).toBeInTheDocument();
    expect(screen.getByText(/Out of 8 defective and 2 clean sample models/)).toBeInTheDocument();
    const gates = (BASELINE as { gates: string[] }).gates;
    const perModel = screen.getByText("Per model").closest("section")!;
    for (const gate of gates) expect(within(perModel).getByText(gate)).toBeInTheDocument();
    const leaky = within(perModel).getByText(modelName("leaky")).closest("tr")!;
    expect(leaky).toHaveTextContent("defect");
    expect(leaky).toHaveTextContent("promote");                                   // a metrics-only gate promotes it
    expect(leaky).toHaveTextContent("ROLLBACK");                                  // the full gate does not
  });

  it("shows the study's caveats and links to its report", async () => {
    await open();
    expect(await screen.findByText(/Only two clean models|only two clean models/i)).toBeInTheDocument();
    expect(screen.getByText(/built by the project author/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Read the full report/ })).toHaveAttribute("href", "/reports?tab=studies&study=baseline_comparison");
  });

  it("says the study is unavailable, and which script generates it, when it has not been generated", async () => {
    await open({ missingStudies: ["baseline_comparison"] });
    expect(await screen.findByText("The baseline comparison is not available")).toBeInTheDocument();
    expect(screen.getByText(/scripts\/baseline_comparison\.py/)).toBeInTheDocument();
    expect(screen.queryByRole("figure")).not.toBeInTheDocument();                 // nothing is drawn or invented
  });

  it("refuses to chart a file it does not understand", async () => {
    await open({ studyData: { baseline_comparison: { gates: "oops" } } });
    expect(await screen.findByText("The baseline comparison is in an unexpected format")).toBeInTheDocument();
    expect(screen.queryByRole("figure")).not.toBeInTheDocument();
  });
});

describe("Model Comparison: sensitivity analysis (real study file)", () => {
  const open = async (options = {}) => {
    installApi({ key: KEY, ...options });
    const user = userEvent.setup();
    renderPage(<Comparison />, "/comparison");
    await user.click(await screen.findByRole("tab", { name: "Sensitivity analysis" }));
  };

  it("shows how stable each decision is, per family of changes", async () => {
    await open();
    const heat = (await screen.findByText("Stability by family of changes")).closest("section")!;
    const slow = within(heat).getByText(modelName("slow")).closest("tr")!;
    expect(slow).toHaveTextContent("HOLD");
    expect(slow).toHaveTextContent("50%");                                        // the thresholds family flips it
    expect(slow).toHaveTextContent("(2/4)");
    const familyCells = within(within(heat).getByText(modelName("overfit")).closest("tr")!).getAllByRole("cell").slice(3);
    expect(familyCells).toHaveLength(5);                                          // one per family
    expect(familyCells.every((c) => c.textContent!.startsWith("100%"))).toBe(true);   // nothing flips this decision
    for (const family of (SENSITIVITY as { families: string[] }).families) expect(within(heat).getByText(family)).toBeInTheDocument();
    expect(screen.getByRole("figure", { name: "Share of perturbations that keep the decision" })).toBeInTheDocument();
  });

  it("lists the perturbations that change a decision", async () => {
    await open();
    const flips = (await screen.findByText("Perturbations that change a decision")).closest("section")!;
    const slow = within(flips).getByText(modelName("slow")).closest("details")!;
    expect(slow).toHaveTextContent("3 flips");
    expect(within(flips).queryByText(modelName("overfit"))).not.toBeInTheDocument();         // it never flipped
  });

  it("reports the design and the caveats, and is unavailable when not generated", async () => {
    await open();
    expect(await screen.findByText(/weights on a 0\.1 grid/)).toBeInTheDocument();
    expect(screen.getByText(/does not show that the thresholds are correct/)).toBeInTheDocument();
  });

  it("is unavailable, with the script to run, when the study has not been generated", async () => {
    await open({ missingStudies: ["sensitivity_analysis"] });
    expect(await screen.findByText("The sensitivity analysis is not available")).toBeInTheDocument();
    expect(screen.getByText(/scripts\/sensitivity_analysis\.py/)).toBeInTheDocument();
  });
});
