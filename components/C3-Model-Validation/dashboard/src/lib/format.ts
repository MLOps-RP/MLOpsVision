import type {
  CheckInfo, CheckResult, CheckStatus, Decision, Dimension, EvidenceSource, ScoreCard,
} from "../api/types";

export const DIMENSION_LABEL: Record<Dimension, string> = {
  training_validity: "Training Validity",
  model_behaviour: "Model Behaviour",
  operational_readiness: "Operational Readiness",
};

export const DIMENSION_SHORT: Record<Dimension, string> = {
  training_validity: "TVS",
  model_behaviour: "MBS",
  operational_readiness: "ORS",
};

export const DECISION_MEANING: Record<Decision, string> = {
  PROMOTE: "Meets the configured quality bar",
  HOLD: "Needs review before it can be promoted",
  ROLLBACK: "Should not be promoted",
};

export const STATUS_LABEL: Record<CheckStatus, string> = {
  PASS: "Pass",
  WARN: "Warning",
  FAIL: "Fail",
  UNAVAILABLE: "No evidence",
  ERROR: "Error",
  NOT_APPLICABLE: "Not applicable",
};

export const EVIDENCE_HELP: Record<EvidenceSource, string> = {
  MEASURED: "Computed by the gate from the model and data",
  DECLARED: "Taken from producer-supplied metadata; not verified",
  SIMULATED: "Produced by a simulation",
  UNAVAILABLE: "No evidence; never counted as a pass",
};

/** A 0-100 score for display: one decimal, or "n/a" when there is none. */
export function fmtScore(value: number | null | undefined): string {
  return value === null || value === undefined ? "n/a" : value.toFixed(1);
}

export function fmtPercent(fraction: number, digits = 0): string {
  return `${(fraction * 100).toFixed(digits)}%`;
}

export function fmtDate(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? iso
    : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/** A check's measured value in a readable form. */
export function fmtValue(check: CheckResult, catalogue?: Map<string, CheckInfo>): string {
  const raw = check.raw_value;
  if (raw === null || raw === undefined) return "—";
  if (typeof raw === "boolean") return raw ? "Yes" : "No";
  if (typeof raw === "string") return raw;
  const abs = Math.abs(raw);
  const text = Number.isInteger(raw) ? String(raw) : abs >= 100 ? raw.toFixed(0) : abs >= 10 ? raw.toFixed(1) : raw.toFixed(3);
  const unit = catalogue?.get(check.id)?.unit;
  const suffix = unit && unit !== "pass/fail" ? shortUnit(unit) : "";
  return suffix ? `${text} ${suffix}` : text;
}

function shortUnit(unit: string): string {
  if (unit === "score" || unit === "ratio" || unit === "difference" || unit === "ECE") return "";
  if (unit.startsWith("fraction")) return "";
  return unit;
}

/**
 * The MQS arithmetic written out with the real numbers, the same way the report states it.
 * Uses the effective weights when a dimension without a score was left out.
 */
export function mqsCalculation(card: ScoreCard): string {
  if (card.mqs === null || card.mqs === undefined) {
    return "No dimension had enough evidence, so no MQS was computed.";
  }
  const used = card.effective_weights ?? card.weights;
  const terms: string[] = [];
  const leftOut: string[] = [];
  for (const [dimension, dim] of Object.entries(card.dimensions) as [Dimension, ScoreCard["dimensions"][Dimension]][]) {
    const short = DIMENSION_SHORT[dimension];
    if (dim.score === null || dim.score === undefined) {
      if (card.missing_policy === "penalize") terms.push(`${short} 0 (no score) × ${card.weights[dimension].toFixed(2)}`);
      else leftOut.push(DIMENSION_LABEL[dimension]);
      continue;
    }
    terms.push(`${short} ${dim.score.toFixed(1)} × ${used[dimension].toFixed(2)}`);
  }
  const text = `MQS = ${terms.join(" + ")} = ${card.mqs.toFixed(1)}`;
  return leftOut.length ? `${text} (${leftOut.join(", ")}: no score, left out)` : text;
}

export function countBy<T>(items: T[], key: (item: T) => string): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const item of items) counts[key(item)] = (counts[key(item)] ?? 0) + 1;
  return counts;
}

export function downloadText(filename: string, text: string, type = "text/plain"): void {
  downloadBlob(filename, new Blob([text], { type }));
}

export function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

/** Read a numeric threshold out of the untyped config snapshot, or undefined. */
export function snapshotNumber(snapshot: Record<string, unknown>, ...path: string[]): number | undefined {
  let node: unknown = snapshot;
  for (const part of path) {
    if (!node || typeof node !== "object") return undefined;
    node = (node as Record<string, unknown>)[part];
  }
  return typeof node === "number" ? node : undefined;
}

export const RECOMMENDATION_NOTICE = "This is a recommendation only. Nothing was deployed, promoted or rolled back.";

export function shortHash(hash: string | null | undefined, length = 12): string {
  return hash ? hash.slice(0, length) : "—";
}

/** The last few segments of a path, so a long server path does not dominate the page (the full path stays in a tooltip). */
export function shortPath(path: string, keep = 3): string {
  const parts = path.split(/[\\/]+/).filter(Boolean);
  return parts.length > keep ? `…/${parts.slice(-keep).join("/")}` : path;
}
