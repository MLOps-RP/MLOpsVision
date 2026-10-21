// The study payloads are free-form JSON on the wire (docs/baseline_comparison.json and
// docs/sensitivity_analysis.json, written by the helper scripts). These parsers accept a payload
// only if it has the expected shape; otherwise the page shows the study as unavailable rather than
// charting something it does not understand.

export interface GateRates { caught: number; defects: number; false_alarms: number; clean: number }

export interface BaselineModel {
  name: string;
  truth: "defect" | "clean" | "excluded";
  f1: number | null;
  metrics_not_passing: string[];
  injected_defect: string;
  decisions: Record<string, string>;
  promoted: Record<string, boolean>;
}

export interface BaselineStudy {
  gates: string[];
  summary: Record<string, GateRates>;
  models: BaselineModel[];
  caveats: string[];
}

export interface FamilyResult { total: number; kept: number; decisions: Record<string, number> }

export interface SensitivityCase {
  model: string;
  profile: string;
  baseline_decision: string;
  baseline_mqs: number | null;
  mqs_range: [number, number] | null;
  families: Record<string, FamilyResult>;
  flips: { family: string; label: string; decision: string }[];
}

export interface SensitivityStudy {
  families: string[];
  cases: SensitivityCase[];
  design: string;
  caveats: string[];
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const isNumber = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isStrings = (v: unknown): v is string[] => Array.isArray(v) && v.every((x) => typeof x === "string");

export function parseBaseline(data: Record<string, unknown>): BaselineStudy | null {
  const { gates, summary, models, caveats } = data;
  if (!isStrings(gates) || !isObject(summary) || !Array.isArray(models) || !isStrings(caveats)) return null;
  for (const gate of gates) {
    const s = summary[gate];
    if (!isObject(s) || !["caught", "defects", "false_alarms", "clean"].every((k) => isNumber(s[k]))) return null;
  }
  const okModel = (m: unknown) =>
    isObject(m) && typeof m.name === "string" && ["defect", "clean", "excluded"].includes(String(m.truth))
    && isStrings(m.metrics_not_passing) && isObject(m.decisions) && isObject(m.promoted);
  if (!models.every(okModel)) return null;
  return data as unknown as BaselineStudy;
}

export function parseSensitivity(data: Record<string, unknown>): SensitivityStudy | null {
  const { families, cases, design, caveats } = data;
  if (!isStrings(families) || !Array.isArray(cases) || typeof design !== "string" || !isStrings(caveats)) return null;
  const okCase = (c: unknown) =>
    isObject(c) && typeof c.model === "string" && typeof c.baseline_decision === "string" && Array.isArray(c.flips)
    && isObject(c.families)
    && families.every((f) => {
      const r = (c.families as Record<string, unknown>)[f];
      return isObject(r) && isNumber(r.total) && isNumber(r.kept) && isObject(r.decisions);
    });
  if (!cases.every(okCase)) return null;
  return data as unknown as SensitivityStudy;
}
