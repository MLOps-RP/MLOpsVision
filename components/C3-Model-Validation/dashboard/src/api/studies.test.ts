import { describe, expect, it } from "vitest";

import { BASELINE, SENSITIVITY } from "../test/mockApi";
import { parseBaseline, parseSensitivity } from "./studies";

const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v));

describe("parseBaseline", () => {
  it("accepts the real study file", () => {
    const study = parseBaseline(BASELINE)!;
    expect(study).not.toBeNull();
    expect(study.gates).toEqual(["F1 only", "metrics only", "full gate (default)", "full gate (hidden_groups)"]);
    expect(study.summary["full gate (hidden_groups)"]).toMatchObject({ caught: 8, defects: 8, false_alarms: 0, clean: 2 });
    expect(study.models).toHaveLength(11);
  });

  it.each([
    ["missing gates", (d: Record<string, unknown>) => delete d.gates],
    ["a gate without a summary", (d: Record<string, unknown>) => delete (d.summary as Record<string, unknown>)["F1 only"]],
    ["a non-numeric count", (d: Record<string, unknown>) => ((d.summary as Record<string, Record<string, unknown>>)["F1 only"].caught = "8")],
    ["an unknown truth value", (d: Record<string, unknown>) => ((d.models as Record<string, unknown>[])[0].truth = "maybe")],
    ["models that are not a list", (d: Record<string, unknown>) => (d.models = {})],
    ["no caveats", (d: Record<string, unknown>) => delete d.caveats],
  ])("rejects %s", (_name, damage) => {
    const data = clone(BASELINE);
    damage(data);
    expect(parseBaseline(data)).toBeNull();
  });
});

describe("parseSensitivity", () => {
  it("accepts the real study file", () => {
    const study = parseSensitivity(SENSITIVITY)!;
    expect(study).not.toBeNull();
    expect(study.families).toEqual(["weights", "thresholds", "per-check", "bands", "missing"]);
    expect(study.cases).toHaveLength(9);
    expect(study.cases.find((c) => c.model === "slow")!.flips.length).toBeGreaterThan(0);
  });

  it.each([
    ["missing families", (d: Record<string, unknown>) => delete d.families],
    ["a case missing a family", (d: Record<string, unknown>) => delete ((d.cases as Record<string, Record<string, unknown>>[])[0].families as Record<string, unknown>).weights],
    ["a non-numeric total", (d: Record<string, unknown>) => {
      (d.cases as { families: Record<string, Record<string, unknown>> }[])[0].families.bands.total = "x";
    }],
    ["cases that are not a list", (d: Record<string, unknown>) => (d.cases = "none")],
    ["no design text", (d: Record<string, unknown>) => delete d.design],
  ])("rejects %s", (_name, damage) => {
    const data = clone(SENSITIVITY);
    damage(data);
    expect(parseSensitivity(data)).toBeNull();
  });

  it("rejects an empty object", () => {
    expect(parseSensitivity({})).toBeNull();
    expect(parseBaseline({})).toBeNull();
  });
});
