// Typed access to the C2 FastAPI backend (proxied at /api by Vite).

export interface DatasetSummary {
  dataset_id: string;
  task_type: string;
  target: string;
  positive_class: string | null;
  n_features: number;
  n_categorical: number;
  n_train: number;
  n_test: number;
  class_distribution: Record<string, number> | null;
}

export interface TaskProfile {
  n_rows: number;
  n_features: number;
  n_categorical: number;
  categorical_ratio: number;
  size_class: string;
  metric: string;
  n_classes: number | null;
  imbalance_ratio: number | null;
}

export interface DatasetDetail {
  manifest: {
    dataset_id: string;
    task_type: string;
    target: string;
    data_dir: string;
    feature_types: Record<string, string>;
    fqs: Record<string, number>;
    class_distribution: Record<string, number> | null;
    positive_class: string | null;
    decision_log: string[];
  };
  profile: TaskProfile;
  fqs: { feature: string; score: number; type: string }[];
  sample: { columns: string[]; rows: (string | number)[][] };
}

export interface RunSummary {
  run_id: string;
  dataset_id: string;
  status: "running" | "completed" | "failed";
  created: string;
  model_family?: string;
  metric?: string;
  best_cv?: number | null;
  holdout?: Record<string, number>;
  rounds?: number;
  total_trials?: number;
  stop_reason?: string | null;
  error?: string | null;
}

export interface ReasoningEntry {
  round: number;
  agent: string;
  step: string;
  decision: Record<string, any>;
  evidence: Record<string, any>;
  decided_by: string;
  rationale: string;
}

export interface RunEvent {
  node: string;
  time: string;
  data: Record<string, any>;
  reasoning: ReasoningEntry[];
}

export interface Candidate {
  family: string;
  mean_val: number;
  gap: number | null;
  params: Record<string, any>;
  round?: number;
  n_complete: number;
  n_pruned: number;
}

export interface C2Output {
  dataset_id: string;
  run_id: string;
  task_type: string;
  metric: string;
  selected_model_path: string;
  model_family: string;
  hyperparameters: Record<string, any>;
  metrics_cv: Record<string, number>;
  metrics_holdout: Record<string, number>;
  model_size_kb: number;
  latency_ms_per_1k: number | null;
  pareto_front: any[];
  candidates: Candidate[];
  diagnoses: { model_family: string; labels: string[]; evidence: Record<string, number>; reasons: string[] }[];
  search_history_path: string;
  reasoning_log_path: string;
  total_trials: number;
  rounds: number;
  score_by_round?: number[];
  stop_reason?: string | null;
  referrals_to_c1: string[];
}

export interface RunDetail {
  run_id: string;
  dataset_id: string;
  status: "running" | "completed" | "failed";
  params: Record<string, any> | null;
  events: RunEvent[];
  current: string | null;
  output: C2Output | null;
  error: string | null;
  started?: string;
  finished?: string | null;
}

export interface Trial {
  family: string;
  round: number;
  trial: number;
  state: string;
  params: Record<string, any>;
  mean_val: number | null;
  mean_train: number | null;
  gap: number | null;
}

export interface Evaluation {
  supported: boolean;
  reason?: string;
  selected_family: string;
  prevalence: number;
  n_test: number;
  selected: { auc: number; ap: number; roc: [number, number][]; pr: [number, number][]; confusion: number[][] };
  importance: { feature: string; importance: number }[];
  candidates: Record<string, { auc: number; ap: number; roc: [number, number][] }>;
}

export interface Comparison {
  dataset_id: string;
  file: string;
  created: string;
  rows: Record<string, any>[];
}

export interface NotebookInfo {
  file: string;
  title: string;
  description: string;
  context: string;
  exists: boolean;
  url: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/health"),
  config: () => request<Record<string, any>>("/config"),
  notebooks: () => request<{ jupyter_url: string; notebooks: NotebookInfo[] }>("/notebooks"),
  datasets: () => request<DatasetSummary[]>("/datasets"),
  dataset: (id: string) => request<DatasetDetail>(`/datasets/${id}`),
  uploadDataset: (form: FormData) => request<DatasetDetail["manifest"]>("/datasets", { method: "POST", body: form }),
  runs: () => request<RunSummary[]>("/runs"),
  run: (id: string) => request<RunDetail>(`/runs/${id}`),
  trials: (id: string) => request<Trial[]>(`/runs/${id}/trials`),
  reasoning: (id: string) => request<ReasoningEntry[]>(`/runs/${id}/reasoning`),
  evaluation: (id: string) => request<Evaluation>(`/runs/${id}/evaluation`),
  comparisons: () => request<Comparison[]>("/comparisons"),
  startRun: (body: Record<string, any>) =>
    request<{ run_id: string }>("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  fileUrl: (id: string, name: string) => `/api/runs/${id}/files/${name}`,
};
