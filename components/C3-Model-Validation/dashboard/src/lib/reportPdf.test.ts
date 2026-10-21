import { describe, expect, it } from "vitest";

import { nameReport } from "./modelNames";
import { buildReportPdf, pdfText, reportPdfFilename, type PdfMeta } from "./reportPdf";
import { REPORTS } from "../test/mockApi";
import { pdfInfo } from "../test/pdf";

const meta: PdfMeta = { name: "Model 1", runId: "0123456789abcdef0123456789abcdef", createdAt: "2026-05-01T09:30:00+00:00" };
const build = async (markdown: string, m = meta) => pdfInfo(new Uint8Array((await buildReportPdf(markdown, m, { now: new Date("2026-06-02T10:00:00Z") })).output("arraybuffer")));

describe("reportPdfFilename", () => {
  it("is Model-N-Validation-Report.pdf", () => {
    expect(reportPdfFilename("Model 1")).toBe("Model-1-Validation-Report.pdf");
    expect(reportPdfFilename("Model 12")).toBe("Model-12-Validation-Report.pdf");
    expect(reportPdfFilename("../evil name?")).toBe("evil-name-Validation-Report.pdf");
    expect(reportPdfFilename("???")).toBe("Validation-Validation-Report.pdf");
  });
});

describe("pdfText", () => {
  it("keeps what the PDF fonts can draw and writes the rest in ASCII, never as a broken glyph", () => {
    expect(pdfText("MQS 86.9 – ok × 2 · “q”")).toBe("MQS 86.9 – ok × 2 · “q”");
    expect(pdfText("a → b ≥ c")).toBe("a -> b >= c");
    expect(pdfText("日本")).toBe("??");
  });
});

describe("buildReportPdf", () => {
  const report = nameReport(REPORTS.good, "good", "Model 1");

  it("produces a real PDF with several pages, not Markdown renamed", async () => {
    const info = await build(report);
    expect(info.startsLikePdf).toBe(true);
    expect(info.endsLikePdf).toBe(true);
    expect(info.pages).toBeGreaterThan(2);
    expect(info.raw).not.toContain("| Dimension | Score |");              // no Markdown table source
  });

  it("starts with the title, the candidate name, the real run date and the run id", async () => {
    const { text } = await build(report);
    expect(text).toContain("Model Quality Validation Report: Model 1");
    expect(text).toContain("Candidate:");
    expect(text).toMatch(/Model 1\n/);
    expect(text).toContain("2026-05-01 09:30 UTC");
    expect(text).toContain("2026-06-02 10:00 UTC");                         // when the PDF was created
    expect(text).toContain(meta.runId);
    expect(text).not.toContain("Report: good");
  });

  it("contains the report's recommendation, MQS, layer scores, checks, rules, explanations and evidence", async () => {
    const { text } = await build(report);
    for (const wanted of [
      "Decision: PROMOTE", "Model Quality Score (MQS): 86.9", "Training Validity (TVS)", "Model Behaviour (MBS)", "Operational Readiness (ORS)",
      "99.9", "73.8", "Why this decision", "MQS meets the promotion threshold", "Explanation", "Weakest checks",
      "training_validity.metadata_completeness", "behaviour.subgroup_f1_gap", "All checks", "Inputs and audit trail", "Model SHA-256",
      "Evidence", "Limitations",
    ]) expect(text, wanted).toContain(wanted);
    expect(text).toContain("This is a recommendation. Nothing was deployed, promoted or rolled back.");
  });

  it("includes the hidden-group findings and critical rules of a run that has them, and only those", async () => {
    const hidden = await build(nameReport(REPORTS.hidden_group, "hidden_group", "Model 12"), { ...meta, name: "Model 12" });
    expect(hidden.text).toContain("Decision: HOLD");
    expect(hidden.text).toContain("Hidden failure groups");
    expect(hidden.text).toContain("6 confirmed");
    const rollback = await build(nameReport(REPORTS.leaky, "leaky", "Model 2"), { ...meta, name: "Model 2" });
    expect(rollback.text).toContain("Decision: ROLLBACK");
    expect(rollback.text).toMatch(/critical/i);
    const good = await build(report);
    expect(good.text).not.toContain("Hidden failure groups");              // not invented for a run without them
  });

  it("numbers every page and names the model in the footer", async () => {
    const info = await build(report);
    const footers = info.texts.filter((t) => /^Page \d+ of \d+$/.test(t));
    expect(footers).toHaveLength(info.pages);
    expect(footers[0]).toBe(`Page 1 of ${info.pages}`);
    expect(footers[footers.length - 1]).toBe(`Page ${info.pages} of ${info.pages}`);
    expect(info.texts.filter((t) => t === "Model 1 - Model Quality Validation Report")).toHaveLength(info.pages);
  });

  it("draws every table row of the report (the check tables are not truncated)", async () => {
    const { text } = await build(report);
    const checkRows = REPORTS.good.split("\n").filter((l) => /^\| (training_validity|behaviour|fairness|calibration|operational)\.[a-z_0-9]+ \| (PASS|WARN|FAIL|NO_EVIDENCE)/.test(l));
    expect(checkRows.length).toBeGreaterThan(20);
    for (const row of checkRows) expect(text, row).toContain(row.split("|")[1].trim());
  });

  it("states a missing date as not recorded rather than inventing one", async () => {
    const { text } = await build(report, { ...meta, createdAt: null });
    expect(text).toContain("not recorded");
  });

  it("wraps very long unbroken text instead of running off the page", async () => {
    const long = "x".repeat(400);
    const info = await build(`# T\n\n${long}\n`);
    expect(info.texts.filter((t) => /^x+$/.test(t)).every((t) => t.length < 400)).toBe(true);
    expect(info.texts.join("").replaceAll(/[^x]/g, "").length).toBe(400);
  });
});
