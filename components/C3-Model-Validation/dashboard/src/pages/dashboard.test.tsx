import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { installApi, KEY, modelName, RESULTS, RUNS, runsWith } from "../test/mockApi";
import { renderApp } from "../test/render";

afterEach(() => vi.unstubAllGlobals());

const stat = (label: string) => screen.getByText(label, { selector: "[data-stat-label]" }).closest("section")!;

describe("Dashboard", () => {
  it("summarises the stored runs and the latest validation", async () => {
    installApi({ key: KEY, runs: runsWith("good") });
    renderApp("/");
    expect(await screen.findByText(/Latest validation:/)).toHaveTextContent(modelName("good"));
    const banner = await screen.findByRole("region", { name: "Decision" });
    expect(banner).toHaveTextContent("PROMOTE");
    expect(banner).toHaveTextContent("Nothing was deployed, promoted or rolled back");
    expect(stat("Validation runs")).toHaveTextContent("10");
    expect(stat("Promote")).toHaveTextContent("2");
    expect(stat("Hold")).toHaveTextContent("4");
    expect(stat("Rollback")).toHaveTextContent("4");
    expect(screen.getByRole("img", { name: "Model Quality Score 86.9 out of 100" })).toBeInTheDocument();
    for (const layer of ["Training Validity", "Model Behaviour", "Operational Readiness"]) {
      expect(screen.getByRole("region", { name: new RegExp(`${layer} score`) })).toBeInTheDocument();
    }
  });

  it("shows the charts, with their data available as tables", async () => {
    installApi({ key: KEY, runs: runsWith("good") });
    renderApp("/");
    expect(await screen.findByRole("figure", { name: "Model Quality Score by run" })).toBeInTheDocument();
    expect(screen.getByRole("figure", { name: "Recommendations" })).toBeInTheDocument();
    expect(screen.getByText(/3 runs without a quotable score/)).toBeInTheDocument();
  });

  it("shows n/a, never a number, for a latest run without a quotable score", async () => {
    installApi({ key: KEY, runs: runsWith("corrupt") });
    renderApp("/");
    expect(await screen.findByRole("img", { name: "Model Quality Score not available" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Decision" })).toHaveTextContent("ROLLBACK");
    const recent = screen.getByText("Recent validations").closest("section")!;
    expect(within(recent).getAllByText("n/a").length).toBeGreaterThan(0);
  });

  it("lists the recent runs and opens one", async () => {
    installApi({ key: KEY, runs: runsWith("good") });
    const user = userEvent.setup();
    renderApp("/");
    const recent = (await screen.findByText("Recent validations")).closest("section")!;
    expect(within(recent).getAllByRole("row")).toHaveLength(8);                   // header + 7
    await user.click(within(recent).getByRole("link", { name: modelName("good") }));
    expect(await screen.findByRole("tab", { name: "Overview" })).toBeInTheDocument();
  });

  it("invites a first validation when nothing is stored", async () => {
    installApi({ key: KEY, runs: { runs: [] } });
    renderApp("/");
    expect(await screen.findByText("No validations yet")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Run your first validation" })).toHaveAttribute("href", "/validate");
    expect(stat("Validation runs")).toHaveTextContent("0");
  });

  it("explains a latest result that cannot be loaded without hiding the rest", async () => {
    installApi({ key: KEY });                                                    // the newest stored run has no result in the mock
    renderApp("/");
    expect(await screen.findByText("Not found")).toBeInTheDocument();
    expect(screen.getByText("Recent validations")).toBeInTheDocument();
  });

  it("does not flash an empty state while the connection is being checked", async () => {
    installApi({ key: KEY, runs: runsWith("good") });
    renderApp("/");
    expect(screen.queryByText("No validations yet")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Loading")).toBeInTheDocument();
    await screen.findByText("Recent validations");
    expect(screen.queryByText("No validations yet")).not.toBeInTheDocument();
  });

  it.each([["/", "Dashboard"], ["/history", "Validation History"], ["/results", "Validation Results"], ["/hidden-failures", "Hidden Failure Analysis"],
    ["/reports", "Reports & Check Catalogue"], ["/validate", "Validate Model"]])(
    "asks for the key on %s instead of showing stale or empty data", async (path, title) => {
      installApi({ key: KEY });
      renderApp(path, "wrong-key");
      expect(await screen.findByText("Enter the API key to continue")).toBeInTheDocument();
      expect(screen.getByRole("heading", { level: 1, name: title })).toBeInTheDocument();
      expect(screen.queryByText("No validations yet")).not.toBeInTheDocument();
      await waitFor(() => expect(screen.queryByLabelText("Loading")).not.toBeInTheDocument());
    });

  it("says plainly when the API cannot be reached", async () => {
    installApi({ offline: true });
    renderApp("/history");
    expect(await screen.findByText("Cannot reach the validation API")).toBeInTheDocument();
    expect(screen.getByText(/python -m mqvg.api/)).toBeInTheDocument();
  });

  it("uses the latest run's own thresholds for the gauge", async () => {
    installApi({ key: KEY, runs: runsWith("good") });
    renderApp("/");
    await screen.findByRole("img", { name: /Model Quality Score 86.9/ });
    expect(screen.getByText(/75\+ : promotion band/)).toBeInTheDocument();
    expect(RESULTS.good.config_snapshot).toHaveProperty("decision");
    expect(RUNS.runs.length).toBe(10);
  });
});
