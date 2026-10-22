// Colours and labels used across the dashboard.
// Chart colours come from the validated categorical palette; the slot ORDER is fixed per entity,
// so a model family keeps its colour on every chart (colour follows the entity, never its rank).

export const FAMILY_COLORS: Record<string, string> = {
  random_forest: "#2a78d6",
  xgboost: "#eb6834",
  lightgbm: "#1baf7a",
  catboost: "#eda100",
  logistic_regression: "#e87ba4",
  ridge: "#008300",
};

export const FAMILY_LABELS: Record<string, string> = {
  random_forest: "Random Forest",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  catboost: "CatBoost",
  logistic_regression: "Logistic Regression",
  ridge: "Ridge",
};

export const familyColor = (f: string) => FAMILY_COLORS[f] ?? "#898781";
export const familyLabel = (f: string) => FAMILY_LABELS[f] ?? f;

// Ordinal blue ramp for rounds (validated with --ordinal)
export const ROUND_COLORS = ["#86b6ef", "#3987e5", "#184f95", "#0d366b", "#0d366b", "#0d366b"];
export const roundColor = (r: number) => ROUND_COLORS[Math.max(0, Math.min(r - 1, ROUND_COLORS.length - 1))];

export const METHOD_COLORS: Record<string, string> = {
  B2_static: "#2a78d6",
  A_full: "#eb6834",
  A_no_ranges: "#1baf7a",
  A_no_families: "#eda100",
};
export const METHOD_LABELS: Record<string, string> = {
  B2_static: "B2 · static TPE",
  A_full: "Adaptive (C2)",
  A_no_ranges: "Ablation · no range adaptation",
  A_no_families: "Ablation · no family dropping",
};

// Agents are UI identities (always shown with a text label)
export const AGENTS = {
  agent1: { name: "Agent 1", role: "Adaptive Model Generation", color: "#6250d6", tint: "#efedfc" },
  agent2: { name: "Agent 2", role: "Hyperparameter Optimization", color: "#0e7c86", tint: "#e6f4f5" },
  agent3: { name: "Agent 3", role: "Multi-Objective Selection", color: "#b45309", tint: "#fbf0e4" },
  tools: { name: "Tools", role: "Trusted computation", color: "#6b6a66", tint: "#f2f1ed" },
} as const;
export type AgentKey = keyof typeof AGENTS;

// Status palette (reserved meaning) for diagnoses - always paired with an icon + label
export const DIAGNOSIS: Record<string, { label: string; color: string; status: string }> = {
  healthy: { label: "Healthy", color: "#0ca30c", status: "good" },
  overfitting: { label: "Overfitting", color: "#fab219", status: "warning" },
  underfitting: { label: "Underfitting", color: "#ec835a", status: "serious" },
  class_imbalance: { label: "Class imbalance", color: "#ec835a", status: "serious" },
  weak_features: { label: "Weak features", color: "#d03b3b", status: "critical" },
  unsuitable_model_family: { label: "Unsuitable family", color: "#898781", status: "muted" },
};

export const fmt = (v: number | null | undefined, digits = 4) =>
  v === null || v === undefined || Number.isNaN(v) ? "–" : v.toFixed(digits);
export const pct = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined ? "–" : `${(v * 100).toFixed(digits)}%`;
export const compact = (v: number) => new Intl.NumberFormat("en", { notation: "compact" }).format(v);
export const when = (iso?: string) => {
  if (!iso) return "–";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
};
export const prettyStep = (s: string) => s.replace(/_/g, " ");
