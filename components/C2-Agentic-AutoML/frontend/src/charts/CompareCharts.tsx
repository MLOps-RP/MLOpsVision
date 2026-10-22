import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { METHOD_COLORS, METHOD_LABELS, fmt } from "../theme";
import { Legend } from "../components/ui";
import { AXIS, GRID, Tip } from "./ChartTooltip";

const METRIC_LABELS: Record<string, string> = {
  roc_auc: "ROC-AUC", f1: "F1", recall: "Recall", precision: "Precision", accuracy: "Accuracy", best_cv: "Best CV",
};

/** Grouped columns: metric groups x method bars. Same 0-1 scale for all metrics, so one axis is honest. */
export function MethodBars({ rows, metrics }: { rows: Record<string, any>[]; metrics: string[] }) {
  const methods = Object.keys(METHOD_COLORS).filter((m) => rows.some((r) => r.method === m));
  const mean = (method: string, metric: string) => {
    const vs = rows.filter((r) => r.method === method && typeof r[metric] === "number").map((r) => r[metric] as number);
    return vs.length ? vs.reduce((a, b) => a + b, 0) / vs.length : null;
  };
  const data = metrics.map((m) => {
    const row: Record<string, any> = { metric: METRIC_LABELS[m] ?? m };
    methods.forEach((method) => (row[method] = mean(method, m)));
    return row;
  });
  return (
    <div>
      <Legend items={methods.map((m) => ({ color: METHOD_COLORS[m], label: METHOD_LABELS[m] ?? m }))} />
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} barGap={2} barCategoryGap="22%" margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid {...GRID} />
          <XAxis dataKey="metric" {...AXIS} />
          <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...AXIS} width={40} />
          <Tooltip cursor={{ fill: "rgba(11,11,11,0.04)" }}
            content={({ active, payload, label }) => active && payload?.length ? (
              <Tip title={label} rows={payload.map((p) => ({
                label: METHOD_LABELS[String(p.dataKey)] ?? String(p.dataKey), color: METHOD_COLORS[String(p.dataKey)], value: fmt(Number(p.value)),
              }))} />
            ) : null} />
          {methods.map((m) => (
            <Bar key={m} dataKey={m} fill={METHOD_COLORS[m]} maxBarSize={24} radius={[4, 4, 0, 0]} isAnimationActive={false} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Part-to-whole for the target classes: one 100% bar with a 2px surface gap between segments. */
export function ClassBalance({ dist, positive }: { dist: Record<string, number>; positive?: string | null }) {
  const entries = Object.entries(dist).sort((a, b) => a[0].localeCompare(b[0]));
  const colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"];
  return (
    <div>
      <div style={{ display: "flex", gap: 2, height: 22, borderRadius: 6, overflow: "hidden" }}>
        {entries.map(([k, v], i) => (
          <div key={k} title={`class ${k}: ${(v * 100).toFixed(1)}%`} style={{ width: `${v * 100}%`, background: colors[i % colors.length], minWidth: 3 }} />
        ))}
      </div>
      <Legend items={entries.map(([k, v], i) => ({
        color: colors[i % colors.length],
        label: <span>Class {k}{k === "1" && positive ? ` (${positive})` : ""} · <b style={{ color: "var(--ink)" }}>{(v * 100).toFixed(1)}%</b></span>,
      }))} />
    </div>
  );
}
