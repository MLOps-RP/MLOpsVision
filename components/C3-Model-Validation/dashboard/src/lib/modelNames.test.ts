import { describe, expect, it } from "vitest";

import { modelNumbers, nameReport } from "./modelNames";

const report = (id: string, quote = "") => `# Model Quality Validation Report: ${quote}${id}${quote}\n\n## Decision: HOLD\n\nThe model good_no_sensitive and good differ.\n\n| Field | Value |\n|---|---|\n| Candidate | ${id} |\n| Profile | good |\n`;

describe("nameReport", () => {
  it("renames the title and the Candidate row, for any model, and nothing else", () => {
    for (const [id, name] of [["good", "Model 1"], ["hidden_group", "Model 12"], ["a.b-c", "Model 3"]]) {
      const out = nameReport(report(id), id, name);
      expect(out).toContain(`# Model Quality Validation Report: ${name}\n`);
      expect(out).toContain(`| Candidate | ${name} |`);
      expect(out.split("\n")).toHaveLength(report(id).split("\n").length);
    }
    const out = nameReport(report("good"), "good", "Model 1");
    expect(out).toContain("The model good_no_sensitive and good differ.");     // body text is left alone
    expect(out).toContain("| Profile | good |");
  });

  it("keeps quotes around the name when the heading has them", () => {
    expect(nameReport(report("good", '"'), "good", "Model 1")).toContain('# Model Quality Validation Report: "Model 1"\n');
  });

  it("does not rename a different model whose id merely contains this one", () => {
    const text = report("good_no_sensitive");
    expect(nameReport(text, "good", "Model 1")).toBe(text);
  });

  it("leaves the report as it is when there is no id or no new name", () => {
    const text = report("good");
    expect(nameReport(text, null, "Model 1")).toBe(text);
    expect(nameReport(text, undefined, "Model 1")).toBe(text);
    expect(nameReport(text, "good", "good")).toBe(text);
  });

  it("treats the id literally, not as a pattern", () => {
    const text = report("a+b");
    expect(nameReport(text, "a+b", "Model 2")).toContain("Report: Model 2\n");
    expect(nameReport(report("aab"), "a+b", "Model 2")).toBe(report("aab"));
  });
});

describe("modelNumbers", () => {
  it("numbers by position, from 1", () => {
    expect([...modelNumbers([{ id: "x" }, { id: "y" }])]).toEqual([["x", "Model 1"], ["y", "Model 2"]]);
  });
});
