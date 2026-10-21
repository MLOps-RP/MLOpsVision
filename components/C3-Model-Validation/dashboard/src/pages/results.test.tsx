import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { pdfInfo } from "../test/pdf";
import { downloads } from "../test/setup";
import { installApi, KEY, modelName, RESULTS, runsWith } from "../test/mockApi";
import { renderApp } from "../test/render";

afterEach(() => vi.unstubAllGlobals());

const banner = () => screen.findByRole("region", { name: "Decision" });
const path = (name: keyof typeof RESULTS) => `/results/${RESULTS[name].run_id}`;
const blobText = (blob: Blob) => new Promise<string>((resolve) => { const r = new FileReader(); r.onload = () => resolve(String(r.result)); r.readAsText(blob); });

describe("Validation Results: overview", () => {
  it("presents a PROMOTE with its score, layers, arithmetic and key metrics", async () => {
    installApi({ key: KEY });
    renderApp(path("good"));
    const decision = await banner();
    expect(decision).toHaveTextContent("PROMOTE");
    expect(decision).toHaveTextContent("Nothing was deployed, promoted or rolled back");
    expect(screen.getByRole("img", { name: "Model Quality Score 86.9 out of 100" })).toBeInTheDocument();
    for (const layer of ["Training Validity", "Model Behaviour", "Operational Readiness"]) {
      expect(screen.getByRole("region", { name: new RegExp(`${layer} score`) })).toBeInTheDocument();
    }
    expect(screen.getByLabelText("MQS calculation")).toHaveTextContent("MQS = TVS 99.9 × 0.30 + MBS 73.8 × 0.50 + ORS 100.0 × 0.20 = 86.9");
    const metrics = screen.getByText("Key metrics").closest("section")!;
    for (const label of ["F1", "Precision", "Recall", "ROC AUC", "Calibration error", "Latency (p95)"]) {
      expect(within(metrics).getByText(label)).toBeInTheDocument();
    }
    expect(within(metrics).getByText("0.661")).toBeInTheDocument();             // the measured F1
    expect(screen.getByRole("figure", { name: "Lowest-scoring checks" })).toBeInTheDocument();
    expect(screen.getByRole("figure", { name: "Check statuses" })).toBeInTheDocument();
    expect(screen.getByText(/Hidden subgroup detection is switched off/)).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /Hidden failure analysis/ })).not.toBeInTheDocument();
  });

  it("shows that a critical rule overrode a good score (leaky model)", async () => {
    installApi({ key: KEY });
    const user = userEvent.setup();
    renderApp(path("leaky"));
    const decision = await banner();
    expect(decision).toHaveTextContent("ROLLBACK");
    expect(decision).toHaveTextContent("A critical rule overrode the score");
    expect(screen.getByRole("img", { name: "Model Quality Score 85.7 out of 100" })).toBeInTheDocument();   // the score alone would pass
    await user.click(screen.getByRole("tab", { name: /Rules & explanation/ }));
    expect(await screen.findByText(RESULTS.leaky.decision.triggered_rules![0].name)).toBeInTheDocument();
    expect(screen.getAllByText("Critical rule").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/prior_high_income_flag/).length).toBeGreaterThan(0);
  });

  it("never presents an unreliable score as a quality score (corrupt model)", async () => {
    installApi({ key: KEY });
    renderApp(path("corrupt"));
    await banner();
    expect(screen.getByRole("img", { name: "Model Quality Score not available" })).toBeInTheDocument();
    expect(screen.getByText("Recorded calculation (not reliable)")).toBeInTheDocument();
    expect(screen.getByText(/must not be quoted as a quality score/)).toBeInTheDocument();
    expect(screen.getByRole("region", { name: /Model Behaviour score/ })).toHaveTextContent("counted as 0%");
    expect(screen.getAllByText(/worst case/i).length).toBeGreaterThan(0);
    const metrics = screen.getByText("Key metrics").closest("section")!;
    expect(within(metrics).getAllByText("no evidence").length).toBeGreaterThan(0);   // F1 etc. were not measured
  });

  it("explains a model that was refused before loading (unsafe file)", async () => {
    installApi({ key: KEY });
    renderApp(path("unsafe_model"));
    expect(await banner()).toHaveTextContent("ROLLBACK");
    expect(screen.getByRole("img", { name: "Model Quality Score not available" })).toBeInTheDocument();
  });

  it("links to the hidden failure analysis when that check ran", async () => {
    installApi({ key: KEY });
    renderApp(path("hidden_group"));
    expect(await banner()).toHaveTextContent("HOLD");
    expect(screen.getByRole("link", { name: /Hidden failure analysis/ })).toHaveAttribute("href", `/hidden-failures?run=${RESULTS.hidden_group.run_id}`);
  });
});

describe("Validation Results: the other tabs", () => {
  it("lists every check and filters them", async () => {
    installApi({ key: KEY });
    const user = userEvent.setup();
    renderApp(path("leaky"));
    await banner();
    await user.click(screen.getByRole("tab", { name: /^Checks/ }));
    expect(await screen.findByText("leakage.single_feature_predictiveness")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Fail" }));
    expect(within(screen.getByRole("table")).getAllByRole("row")).toHaveLength(2);       // header + the failing check
  });

  it("shows where the evidence comes from, the gaps, and the audit trail", async () => {
    installApi({ key: KEY });
    const user = userEvent.setup();
    renderApp(path("corrupt"));
    await banner();
    await user.click(screen.getByRole("tab", { name: "Evidence" }));
    const breakdown = (await screen.findByText("Where the evidence comes from")).closest("section")!;
    const counts = (source: string) => RESULTS.corrupt.checks.filter((c) => c.evidence_source === source).length;
    expect(within(breakdown).getByText("Measured").closest("tr")).toHaveTextContent(String(counts("MEASURED")));
    expect(within(breakdown).getByText("None").closest("tr")).toHaveTextContent(String(counts("UNAVAILABLE")));
    expect(screen.getByText("Evidence gaps").closest("section")).toHaveTextContent(/never counted as a pass/);
    expect(screen.getByRole("figure", { name: "Layer scores" })).toBeInTheDocument();
    const trail = screen.getByText("Inputs and audit trail").closest("section")!;
    const model = within(trail).getByText("…/fixtures/corrupt/model.joblib");        // long server paths are shortened
    expect(model).toHaveAttribute("title", RESULTS.corrupt.candidate.model_path);        // the full path stays in the tooltip
  });

  it("renders the gate's own Markdown report and downloads it as a PDF", async () => {
    installApi({ key: KEY });
    const user = userEvent.setup();
    renderApp(path("good"));
    await banner();
    await user.click(screen.getByRole("tab", { name: "Report" }));
    expect(await screen.findByRole("heading", { level: 1, name: `Model Quality Validation Report: ${modelName("good")}` })).toBeInTheDocument();
    expect(within(screen.getByRole("article")).getAllByRole("table").length).toBeGreaterThan(0);
    expect(screen.queryByRole("button", { name: /Download \.md/ })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Download PDF" }));
    await vi.waitFor(() => expect(downloads).toHaveLength(1), { timeout: 20000 });
    expect(downloads[0].name).toBe("Model-1-Validation-Report.pdf");
    const pdf = pdfInfo(await new Promise<Uint8Array>((resolve) => { const r = new FileReader(); r.onload = () => resolve(new Uint8Array(r.result as ArrayBuffer)); r.readAsArrayBuffer(downloads[0].blob); }));
    expect(pdf.startsLikePdf).toBe(true);
    expect(pdf.text).toContain("Model Quality Validation Report: Model 1");
    expect(pdf.text).toContain("Decision: PROMOTE");
  }, 60000);

  it("downloads the full result as JSON", async () => {
    installApi({ key: KEY });
    const user = userEvent.setup();
    renderApp(path("good"));
    await banner();
    await user.click(screen.getByRole("button", { name: /Download JSON/ }));
    expect(downloads[0].name).toMatch(/\.json$/);
    expect(JSON.parse(await blobText(downloads[0].blob)).run_id).toBe(RESULTS.good.run_id);
  });

  it("shows the raw result and supports arrow-key tab navigation", async () => {
    installApi({ key: KEY });
    const user = userEvent.setup();
    renderApp(path("good"));
    await banner();
    screen.getByRole("tab", { name: "Overview" }).focus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: /^Checks/ })).toHaveAttribute("aria-selected", "true");
    await user.click(screen.getByRole("tab", { name: "Raw result" }));
    expect(screen.getByLabelText("Raw JSON result")).toHaveTextContent(RESULTS.good.run_id);
  });
});

describe("Validation Results: entry points", () => {
  it("opens the latest result from /results", async () => {
    installApi({ key: KEY, runs: runsWith("good") });
    renderApp("/results");
    expect(await banner()).toHaveTextContent("PROMOTE");
  });

  it("says there are no results yet when nothing is stored", async () => {
    installApi({ key: KEY, runs: { runs: [] } });
    renderApp("/results");
    expect(await screen.findByText("No validation results yet")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Validate a model" })).toHaveAttribute("href", "/validate");
  });

  it("explains an unknown run instead of showing a blank page", async () => {
    installApi({ key: KEY });
    renderApp(`/results/${"f".repeat(32)}`);
    expect(await screen.findByText("Not found")).toBeInTheDocument();
    expect(screen.getByText("no such validation run")).toBeInTheDocument();
  });
});
