import { describe, expect, it } from "vitest";

import type { CheckResult } from "../api/types";
import { CHECKS, REPORTS, RESULTS } from "../test/mockApi";
import {
  countBy, fmtPercent, fmtScore, fmtValue, mqsCalculation, shortHash, shortPath, snapshotNumber,
} from "./format";

describe("mqsCalculation", () => {
  // The backend writes this very line into its Markdown report, so it is the ground truth.
  it.each(Object.keys(RESULTS) as (keyof typeof RESULTS)[])("matches the gate's own report line for %s", (name) => {
    const reported = REPORTS[name].match(/\*\*Calculation:\*\* (.+)/)![1].trim();
    expect(mqsCalculation(RESULTS[name].scorecard)).toBe(reported);
  });

  it("says plainly when there is no MQS", () => {
    const card = { ...RESULTS.good.scorecard, mqs: null };
    expect(mqsCalculation(card)).toMatch(/no mqs was computed/i);
  });

  it("shows a missing layer as zero under the penalize policy", () => {
    const c = RESULTS.corrupt.scorecard;
    const card = { ...c, missing_policy: "penalize" as const, effective_weights: null };
    expect(mqsCalculation(card)).toContain("MBS 0 (no score) × 0.50");
  });
});

describe("formatting", () => {
  it("shows a missing score as n/a, never as a number", () => {
    expect(fmtScore(null)).toBe("n/a");
    expect(fmtScore(undefined)).toBe("n/a");
    expect(fmtScore(86.94)).toBe("86.9");
    expect(fmtScore(0)).toBe("0.0");
  });

  it("formats percentages", () => {
    expect(fmtPercent(0.3123)).toBe("31%");
    expect(fmtPercent(0.0113, 1)).toBe("1.1%");
  });

  it("formats check values with the catalogue unit", () => {
    const catalogue = new Map(CHECKS.map((c) => [c.id, c]));
    const byId = (id: string) => RESULTS.good.checks.find((c) => c.id === id) as CheckResult;
    expect(fmtValue(byId("operational.latency"), catalogue)).toMatch(/^\d+(\.\d+)? ms$/);
    expect(fmtValue(byId("operational.model_loadable"), catalogue)).toBe("Yes");
    expect(fmtValue(byId("behaviour.f1"), catalogue)).toMatch(/^0\.\d{3}$/);
    expect(fmtValue({ ...byId("behaviour.f1"), raw_value: null } as CheckResult)).toBe("—");
  });

  it("shortens hashes and paths but keeps short paths intact", () => {
    expect(shortHash("abcdef0123456789", 6)).toBe("abcdef");
    expect(shortHash(null)).toBe("—");
    expect(shortPath("C:\\a\\b\\c\\fixtures\\good\\model.joblib")).toBe("…/fixtures/good/model.joblib");
    expect(shortPath("/srv/models/a/b.joblib")).toBe("…/models/a/b.joblib");
    expect(shortPath("a/b.joblib")).toBe("a/b.joblib");
  });

  it("counts and reads untyped config values safely", () => {
    expect(countBy(["a", "b", "a"], (x) => x)).toEqual({ a: 2, b: 1 });
    const snapshot = RESULTS.good.config_snapshot as Record<string, unknown>;
    expect(snapshotNumber(snapshot, "decision", "promote_at")).toBe(75);
    expect(snapshotNumber(snapshot, "decision", "nope")).toBeUndefined();
    expect(snapshotNumber(snapshot, "profile")).toBeUndefined();   // a string, not a number
    expect(snapshotNumber({}, "a", "b")).toBeUndefined();
  });
});
