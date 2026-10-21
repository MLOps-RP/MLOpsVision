import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { pdfInfo } from "../test/pdf";
import { downloads } from "../test/setup";
import { CHECKS, installApi, KEY, modelName, PROFILES, RESULTS, runsWith, STUDY_INFOS } from "../test/mockApi";
import { renderApp, renderPage } from "../test/render";
import { StudyReports } from "./Reports";

afterEach(() => vi.unstubAllGlobals());

const blobBytes = (blob: Blob) => new Promise<Uint8Array>((resolve) => { const r = new FileReader(); r.onload = () => resolve(new Uint8Array(r.result as ArrayBuffer)); r.readAsArrayBuffer(blob); });
const blobText = (blob: Blob) => new Promise<string>((resolve) => { const r = new FileReader(); r.onload = () => resolve(String(r.result)); r.readAsText(blob); });

describe("Reports: validation reports", () => {
  it("shows the newest run's report and offers a PDF, not a Markdown file", async () => {
    installApi({ key: KEY, runs: runsWith("good") });
    const user = userEvent.setup();
    renderApp("/reports");
    expect(await screen.findByRole("heading", { level: 1, name: `Model Quality Validation Report: ${modelName("good")}` })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Download \.md/ })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Download PDF" }));
    await vi.waitFor(() => expect(downloads).toHaveLength(1), { timeout: 20000 });
    expect(downloads[0].name).toBe("Model-1-Validation-Report.pdf");
    expect(downloads[0].blob.type).toBe("application/pdf");
    const pdf = pdfInfo(await blobBytes(downloads[0].blob));
    expect(pdf.startsLikePdf && pdf.endsLikePdf).toBe(true);                                  // a real PDF, not Markdown renamed
    expect(pdf.raw).not.toContain("# Model Quality Validation Report");
    expect(pdf.text).toContain("Model Quality Validation Report: Model 1");
    expect(pdf.text).toContain("Decision: PROMOTE");
    expect(pdf.text).toContain("Model Quality Score (MQS): 86.9");
    expect(pdf.text).toContain("Operational Readiness (ORS)");
    expect(pdf.text).not.toContain("Report: good");
    expect(pdf.text).toContain(RESULTS.good.run_id);
    expect(pdf.texts).toContain(`Page 1 of ${pdf.pages}`);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  }, 60000);

  it("names each model's PDF and report after that model, in the preview, the details table and the file", async () => {
    installApi({ key: KEY, runs: runsWith("good", "leaky", "hidden_group", "corrupt") });
    const user = userEvent.setup();
    renderApp("/reports");
    for (const id of ["good", "leaky", "hidden_group", "corrupt"] as const) {
      await user.selectOptions(await screen.findByLabelText("Validation"), RESULTS[id].run_id);
      const name = modelName(id);
      const heading = await screen.findByRole("heading", { level: 1, name: `Model Quality Validation Report: ${name}` });
      const article = heading.closest("article")!;
      expect(within(article).getByRole("cell", { name })).toBeInTheDocument();                  // the "Candidate" row of the details table
      expect(within(article).queryByRole("heading", { level: 1, name: new RegExp(`Report: ${id}$`) })).not.toBeInTheDocument();
      const before = downloads.length;
      await user.click(screen.getByRole("button", { name: "Download PDF" }));
      await vi.waitFor(() => expect(downloads).toHaveLength(before + 1), { timeout: 20000 });
      const file = downloads[downloads.length - 1];
      expect(file.name).toBe(`${name.replaceAll(" ", "-")}-Validation-Report.pdf`);
      const pdf = pdfInfo(await blobBytes(file.blob));
      expect(pdf.text).toContain(`Model Quality Validation Report: ${name}`);
      expect(pdf.text).toContain(`Decision: ${RESULTS[id].decision.decision}`);
      expect(pdf.text).toContain(RESULTS[id].run_id);
      expect(pdf.text).not.toContain(`Report: ${id}\n`);
    }
  }, 60000);

  it("tells the user, and stays usable, when the PDF cannot be created", async () => {
    installApi({ key: KEY, runs: runsWith("good") });
    vi.mocked(URL.createObjectURL).mockImplementationOnce(() => { throw new Error("out of memory"); });
    const user = userEvent.setup();
    renderApp("/reports");
    await screen.findByRole("article");
    await user.click(screen.getByRole("button", { name: "Download PDF" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The PDF could not be created: out of memory");
    expect(screen.getByRole("button", { name: "Download PDF" })).toBeEnabled();
  });

  it("downloads the run's JSON result", async () => {
    installApi({ key: KEY, runs: runsWith("good") });
    const user = userEvent.setup();
    renderApp("/reports");
    await screen.findByRole("article");
    await user.click(screen.getByRole("button", { name: /Download JSON/ }));
    expect(await screen.findByRole("article")).toBeInTheDocument();
    await vi.waitFor(() => expect(downloads).toHaveLength(1), { timeout: 20000 });
    expect(JSON.parse(await blobText(downloads[0].blob)).run_id).toBe(RESULTS.good.run_id);
  });

  it("switches to another run's report", async () => {
    installApi({ key: KEY, runs: runsWith("good", "leaky") });
    const user = userEvent.setup();
    renderApp("/reports");
    await screen.findByRole("heading", { level: 1, name: `Model Quality Validation Report: ${modelName("good")}` });
    await user.selectOptions(screen.getByLabelText("Validation"), RESULTS.leaky.run_id);
    expect(await screen.findByRole("heading", { level: 1, name: `Model Quality Validation Report: ${modelName("leaky")}` })).toBeInTheDocument();
  });

  it("says there are no reports when nothing has been validated", async () => {
    installApi({ key: KEY, runs: { runs: [] } });
    renderApp("/reports");
    expect(await screen.findByText("No validation reports yet")).toBeInTheDocument();
  });
});

describe("Reports: the study reports section is hidden", () => {
  it("has only the validation reports and the check catalogue tabs", async () => {
    installApi({ key: KEY, runs: runsWith("good") });
    renderApp("/reports");
    await screen.findByRole("article");
    expect(screen.getAllByRole("tab").map((t) => t.textContent)).toEqual(["Validation reports", "Check catalogue"]);
    expect(screen.queryByText(/Study reports?/)).not.toBeInTheDocument();
  });

  it("shows the validation reports for an old ?tab=studies link instead of a study", async () => {
    installApi({ key: KEY, runs: runsWith("good") });
    renderApp("/reports?tab=studies&study=baseline_comparison");
    expect(await screen.findByRole("heading", { level: 1, name: /Model Quality Validation Report/ })).toBeInTheDocument();
    expect(screen.queryByText("Baseline comparison", { selector: "span" })).not.toBeInTheDocument();
  });
});

describe("Reports: study reports (component kept, not shown on the page)", () => {
  const studies = async (options = {}, path = "/reports?tab=studies") => {
    installApi({ key: KEY, ...options });
    renderPage(<StudyReports />, path);
    await screen.findByText("Baseline comparison", { selector: "span" });
  };

  it("lists the studies with what exists for each", async () => {
    await studies();
    for (const s of STUDY_INFOS) expect(screen.getByText(s.title, { selector: "span" })).toBeInTheDocument();
    const hidden = screen.getByText("Hidden failure group evaluation").closest("button")!;
    expect(hidden).toHaveTextContent("Report available");
    expect(hidden).not.toHaveTextContent("Chartable data");                         // it has no machine-readable file
    expect(screen.getByText("Baseline comparison", { selector: "span" }).closest("button")).toHaveTextContent("Chartable data");
  });

  it("shows the first available study's report and lets the user pick another", async () => {
    await studies();
    expect(await screen.findByRole("heading", { level: 1, name: /Baseline comparison/ })).toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(screen.getByText("Sensitivity analysis", { selector: "span" }));
    expect(await screen.findByRole("heading", { level: 1, name: /Sensitivity of the decisions/ })).toBeInTheDocument();
  });

  it("opens the study named in the link from the comparison page, and downloads it", async () => {
    await studies({}, "/reports?tab=studies&study=hidden_groups_results");
    expect(await screen.findByRole("heading", { level: 1, name: /Hidden failure group detection/ })).toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /Download \.md/ }));
    expect(downloads[0].name).toBe("hidden_groups_results.md");
  });

  it("says a study has not been generated, and which script generates it, instead of showing an empty report", async () => {
    const infos = STUDY_INFOS.map((s) => (s.id === "sensitivity_analysis" ? { ...s, has_data: false, has_report: false, generated_at_utc: null } : s));
    await studies({ studies: infos }, "/reports?tab=studies&study=sensitivity_analysis");
    expect(await screen.findByText("Sensitivity analysis has not been generated")).toBeInTheDocument();
    expect(screen.getByText(/scripts\/sensitivity_analysis\.py/)).toBeInTheDocument();
    expect(screen.getByText(/never runs the scripts/)).toBeInTheDocument();
    expect(screen.queryByRole("article")).not.toBeInTheDocument();
  });
});

describe("Reports: check catalogue", () => {
  const open = async () => {
    installApi({ key: KEY });
    const user = userEvent.setup();
    renderApp("/reports?tab=catalogue");
    await screen.findByText("Check catalogue", { selector: "h2" });
    return user;
  };
  const catalogueRows = () => within(screen.getByRole("table", { name: "Check catalogue" })).getAllByRole("row").slice(1);

  it("lists every check the gate knows about", async () => {
    await open();
    expect(catalogueRows()).toHaveLength(CHECKS.length);
    expect(screen.getByText(`${CHECKS.length} of ${CHECKS.length} checks the gate knows about.`)).toBeInTheDocument();
  });

  it("searches and filters by layer", async () => {
    const user = await open();
    await user.type(screen.getByRole("searchbox", { name: "Search the catalogue" }), "latency");
    expect(catalogueRows()).toHaveLength(CHECKS.filter((c) => `${c.id} ${c.description}`.toLowerCase().includes("latency")).length);
    await user.clear(screen.getByRole("searchbox", { name: "Search the catalogue" }));
    await user.click(screen.getByRole("button", { name: "Operational Readiness" }));
    expect(catalogueRows()).toHaveLength(CHECKS.filter((c) => c.dimension === "operational_readiness").length);
    await user.type(screen.getByRole("searchbox", { name: "Search the catalogue" }), "no-such-check");
    expect(screen.getByText("No checks match")).toBeInTheDocument();
  });

  it("lists the profiles with their scan status, and is honest that per-check thresholds are not exposed", async () => {
    await open();
    const profiles = screen.getByText("Profiles", { selector: "h2" }).closest("section")!;
    expect(within(profiles).getAllByRole("row")).toHaveLength(PROFILES.profiles.length + 1);
    const extended = within(profiles).getByText("extended", { selector: "td" }).closest("tr")!;
    expect(extended).toHaveTextContent("Yes");
    expect(extended).toHaveTextContent("Allowed");
    expect(within(profiles).getByText("default", { selector: "td" }).closest("tr")).toHaveTextContent("Not allowed");
    expect(within(profiles).getByText(/not exposed by the API/)).toBeInTheDocument();
  });
});
