import type { ReactNode } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ErrorBar, Legend, LabelList, Pie, PieChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

import type { CheckResult, CheckStatus, Decision, DimensionScore, RunEntry } from "../api/types";
import { DIMENSIONS } from "../api/types";
import type { BaselineStudy, SensitivityCase } from "../api/studies";
import { DIMENSION_SHORT, STATUS_LABEL } from "../lib/format";
import { BLUE, BLUE_LIGHT, DECISION_COLOR, STATUS_COLOR } from "./badges";
import { Card, CardHeader } from "./ui";

const AXIS = { fontSize: 12, fill: "#64748b" } as const;
const GRID = "#e2e8f0";
const ANIMATE = false;                 // deterministic output and no motion for users who ask for none

interface TableData { headers: string[]; rows: (string | number)[][] }

/** A titled chart with its data available as a table, which is also the accessible description. */
export function ChartFrame({ title, subtitle, height = 260, table, children, footnote }: {
  title: string; subtitle?: ReactNode; height?: number; table: TableData; children: ReactNode; footnote?: ReactNode;
}) {
  return (
    <Card>
      <CardHeader title={title} subtitle={subtitle} />
      <figure className="m-0 px-4 pb-3 pt-4" aria-label={title}>
        <div style={{ height }} role="img" aria-label={`${title}. The data is in the table below the chart.`}>
          <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 640, height }}>
            {children as React.ReactElement}
          </ResponsiveContainer>
        </div>
        {footnote && <figcaption className="mt-2 text-xs text-slate-500">{footnote}</figcaption>}
        <details className="mt-2 text-[13px]">
          <summary className="font-semibold text-blue-700">View data as a table</summary>
          <div className="mt-2 overflow-x-auto">
            <table className="w-full border-collapse text-left text-[13px]">
              <thead><tr>{table.headers.map((h) => <th key={h} scope="col" className="border-b border-slate-200 px-3 py-1.5 font-semibold text-slate-600">{h}</th>)}</tr></thead>
              <tbody>{table.rows.map((r, i) => (
                <tr key={i}>{r.map((c, j) => <td key={j} className="border-b border-slate-100 px-3 py-1.5 tabular-nums">{c}</td>)}</tr>))}
              </tbody>
            </table>
          </div>
        </details>
      </figure>
    </Card>
  );
}

const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;

/* ----------------------------------------------------------- dashboard charts */
export function MqsByRunChart({ runs, nameOf }: { runs: RunEntry[]; nameOf?: (candidateId: string | null | undefined, runId: string) => string }) {
  const label = (r: RunEntry) => (nameOf ? nameOf(r.candidate_id, r.run_id) : r.candidate_id ?? r.run_id.slice(0, 6));
  const scored = runs.filter((r) => r.mqs !== null).slice(0, 14).reverse();   // newest 14, oldest on the left
  const hidden = runs.length - runs.filter((r) => r.mqs !== null).length;
  const data = scored.map((r) => ({ name: label(r), mqs: Number((r.mqs as number).toFixed(1)), decision: r.decision as Decision }));
  return (
    <ChartFrame title="Model Quality Score by run" subtitle="The latest runs that have a quotable score, coloured by recommendation."
      table={{ headers: ["Run", "MQS", "Decision"], rows: data.map((d) => [d.name, d.mqs, d.decision]) }}
      footnote={hidden > 0 ? `${hidden} run${hidden === 1 ? "" : "s"} without a quotable score (too little evidence) ${hidden === 1 ? "is" : "are"} not plotted.` : undefined}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis dataKey="name" tick={AXIS} interval={0} angle={-25} textAnchor="end" height={56} />
        <YAxis domain={[0, 100]} tick={AXIS} />
        <Tooltip />
        <Bar dataKey="mqs" name="MQS" radius={[4, 4, 0, 0]} isAnimationActive={ANIMATE}>
          {data.map((d, i) => <Cell key={i} fill={DECISION_COLOR[d.decision]} />)}
        </Bar>
      </BarChart>
    </ChartFrame>
  );
}

export function DecisionDonut({ counts }: { counts: Record<Decision, number> }) {
  const data = (Object.keys(counts) as Decision[]).map((d) => ({ name: d, value: counts[d] }));
  const total = data.reduce((s, d) => s + d.value, 0);
  return (
    <ChartFrame title="Recommendations" subtitle={`${total} stored run${total === 1 ? "" : "s"}`} height={220}
      table={{ headers: ["Decision", "Runs"], rows: data.map((d) => [d.name, d.value]) }}>
      <PieChart>
        <Pie data={data.filter((d) => d.value > 0)} dataKey="value" nameKey="name" innerRadius={55} outerRadius={85} paddingAngle={2} isAnimationActive={ANIMATE}>
          {data.filter((d) => d.value > 0).map((d) => <Cell key={d.name} fill={DECISION_COLOR[d.name]} />)}
        </Pie>
        <Tooltip />
        <Legend />
      </PieChart>
    </ChartFrame>
  );
}

/* -------------------------------------------------------------- result charts */
export function LayerBars({ dimensions }: { dimensions: Record<string, DimensionScore> }) {
  const data = DIMENSIONS.map((d) => ({ name: DIMENSION_SHORT[d], score: dimensions[d].score, coverage: dimensions[d].coverage }));
  return (
    <ChartFrame title="Layer scores" subtitle="Training validity, model behaviour and operational readiness (0 to 100)." height={200}
      table={{ headers: ["Layer", "Score", "Evidence coverage"], rows: data.map((d) => [d.name, d.score === null ? "n/a" : d.score!.toFixed(1), pct(d.coverage, 0)]) }}
      footnote={data.some((d) => d.score === null) ? "A layer with too little evidence has no score and is not drawn." : undefined}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 36, bottom: 0, left: 0 }}>
        <CartesianGrid horizontal={false} stroke={GRID} />
        <XAxis type="number" domain={[0, 100]} tick={AXIS} />
        <YAxis type="category" dataKey="name" tick={AXIS} width={44} />
        <Tooltip />
        <Bar dataKey="score" name="Score" fill={BLUE} radius={[0, 4, 4, 0]} isAnimationActive={ANIMATE}>
          <LabelList dataKey="score" position="right" formatter={(v: number) => (v === null || v === undefined ? "n/a" : v.toFixed(1))} style={{ fontSize: 12, fill: "#334155" }} />
        </Bar>
      </BarChart>
    </ChartFrame>
  );
}

export function StatusDonut({ counts }: { counts: Record<string, number> }) {
  const data = Object.entries(counts).map(([status, value]) => ({ status: status as CheckStatus, name: STATUS_LABEL[status as CheckStatus], value }));
  return (
    <ChartFrame title="Check statuses" subtitle={`${data.reduce((s, d) => s + d.value, 0)} checks`} height={200}
      table={{ headers: ["Status", "Checks"], rows: data.map((d) => [d.name, d.value]) }}>
      <PieChart>
        <Pie data={data} dataKey="value" nameKey="name" innerRadius={48} outerRadius={78} paddingAngle={2} isAnimationActive={ANIMATE}>
          {data.map((d) => <Cell key={d.status} fill={STATUS_COLOR[d.status]} />)}
        </Pie>
        <Tooltip />
        <Legend />
      </PieChart>
    </ChartFrame>
  );
}

export function CheckScoreBars({ checks }: { checks: CheckResult[] }) {
  const scored = checks.filter((c) => c.score !== null && c.score !== undefined);
  const data = [...scored].sort((a, b) => (a.score as number) - (b.score as number)).slice(0, 10)
    .map((c) => ({ name: c.id.split(".")[1], id: c.id, score: Number((c.score as number).toFixed(0)), status: c.status }));
  const missing = checks.length - scored.length;
  return (
    <ChartFrame title="Lowest-scoring checks" subtitle="The ten checks that cost the most, coloured by status." height={Math.max(180, data.length * 28 + 40)}
      table={{ headers: ["Check", "Score", "Status"], rows: data.map((d) => [d.id, d.score, STATUS_LABEL[d.status]]) }}
      footnote={missing > 0 ? `${missing} check${missing === 1 ? "" : "s"} without evidence ha${missing === 1 ? "s" : "ve"} no score and ${missing === 1 ? "is" : "are"} not drawn (never counted as a pass).` : undefined}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 30, bottom: 0, left: 0 }}>
        <CartesianGrid horizontal={false} stroke={GRID} />
        <XAxis type="number" domain={[0, 100]} tick={AXIS} />
        <YAxis type="category" dataKey="name" tick={AXIS} width={150} />
        <Tooltip />
        <Bar dataKey="score" name="Score" radius={[0, 4, 4, 0]} isAnimationActive={ANIMATE}>
          {data.map((d) => <Cell key={d.id} fill={STATUS_COLOR[d.status]} />)}
          <LabelList dataKey="score" position="right" style={{ fontSize: 12, fill: "#334155" }} />
        </Bar>
      </BarChart>
    </ChartFrame>
  );
}

/* ----------------------------------------------------------- hidden failures */
export interface GroupPoint {
  rule: string; method: string; n_rows: number; observed_error_rate: number; expected_error_rate: number;
  effect: number; effect_ci: [number, number]; adjusted_p_value: number; confirmed: boolean;
}

export function GroupErrorChart({ groups }: { groups: GroupPoint[] }) {
  const data = groups.map((g, i) => ({ name: `G${i + 1}`, observed: +(g.observed_error_rate * 100).toFixed(1), expected: +(g.expected_error_rate * 100).toFixed(1) }));
  return (
    <ChartFrame title="Error rate inside each group" subtitle="Observed errors against what the model's own probabilities predicted (held-out rows)." height={280}
      table={{ headers: ["Group", "Observed error", "Expected error"], rows: data.map((d) => [d.name, `${d.observed.toFixed(1)}%`, `${d.expected.toFixed(1)}%`]) }}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -8 }}>
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis dataKey="name" tick={AXIS} />
        <YAxis tick={AXIS} unit="%" />
        <Tooltip formatter={(v: number) => `${v}%`} />
        <Legend />
        <Bar dataKey="observed" name="Observed" fill="#dc2626" radius={[4, 4, 0, 0]} isAnimationActive={ANIMATE} />
        <Bar dataKey="expected" name="Expected" fill={BLUE_LIGHT} radius={[4, 4, 0, 0]} isAnimationActive={ANIMATE} />
      </BarChart>
    </ChartFrame>
  );
}

export function GroupEffectChart({ groups, minEffect }: { groups: GroupPoint[]; minEffect?: number }) {
  const data = groups.map((g, i) => ({
    name: `G${i + 1}`, effect: +(g.effect * 100).toFixed(1), confirmed: g.confirmed,
    error: [+((g.effect - g.effect_ci[0]) * 100).toFixed(1), +((g.effect_ci[1] - g.effect) * 100).toFixed(1)] as [number, number],
  }));
  return (
    <ChartFrame title="Excess error with its 95% interval" subtitle="How far above its expected error each group runs; the line is the minimum effect to confirm a group." height={280}
      table={{ headers: ["Group", "Excess error", "95% interval", "Confirmed"], rows: groups.map((g, i) => [`G${i + 1}`, `${(g.effect * 100).toFixed(1)}%`, `[${(g.effect_ci[0] * 100).toFixed(1)}%, ${(g.effect_ci[1] * 100).toFixed(1)}%]`, g.confirmed ? "yes" : "no"]) }}>
      <BarChart data={data} margin={{ top: 8, right: 56, bottom: 0, left: -8 }}>
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis dataKey="name" tick={AXIS} />
        <YAxis tick={AXIS} unit="%" />
        <Tooltip formatter={(v: number) => `${v}%`} />
        {typeof minEffect === "number" && <ReferenceLine y={minEffect * 100} stroke="#475569" strokeDasharray="4 4" label={{ value: `min ${(minEffect * 100).toFixed(0)}%`, fontSize: 11, fill: "#475569", position: "right" }} />}
        <Bar dataKey="effect" name="Excess error" radius={[4, 4, 0, 0]} isAnimationActive={ANIMATE}>
          {data.map((d) => <Cell key={d.name} fill={d.confirmed ? "#dc2626" : "#94a3b8"} />)}
          <ErrorBar dataKey="error" width={4} stroke="#334155" />
        </Bar>
      </BarChart>
    </ChartFrame>
  );
}

/* ---------------------------------------------------------------- comparison */
export interface CompareRow { label: string; tvs: number | null; mbs: number | null; ors: number | null; mqs: number | null }

export function RunCompareChart({ rows }: { rows: CompareRow[] }) {
  const fmt = (v: number | null) => (v === null ? "n/a" : v.toFixed(1));
  return (
    <ChartFrame title="Scores side by side" subtitle="A layer or score that is not available is left out, not drawn as zero." height={300}
      table={{ headers: ["Run", "TVS", "MBS", "ORS", "MQS"], rows: rows.map((r) => [r.label, fmt(r.tvs), fmt(r.mbs), fmt(r.ors), fmt(r.mqs)]) }}>
      <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis dataKey="label" tick={AXIS} />
        <YAxis domain={[0, 100]} tick={AXIS} />
        <Tooltip />
        <Legend />
        <Bar dataKey="tvs" name="TVS" fill="#93c5fd" radius={[3, 3, 0, 0]} isAnimationActive={ANIMATE} />
        <Bar dataKey="mbs" name="MBS" fill="#3b82f6" radius={[3, 3, 0, 0]} isAnimationActive={ANIMATE} />
        <Bar dataKey="ors" name="ORS" fill="#1d4ed8" radius={[3, 3, 0, 0]} isAnimationActive={ANIMATE} />
        <Bar dataKey="mqs" name="MQS" fill="#0f172a" radius={[3, 3, 0, 0]} isAnimationActive={ANIMATE} />
      </BarChart>
    </ChartFrame>
  );
}

export function BaselineChart({ study }: { study: BaselineStudy }) {
  const data = study.gates.map((g) => ({ gate: g, stopped: study.summary[g].caught, defects: study.summary[g].defects, alarms: study.summary[g].false_alarms, clean: study.summary[g].clean }));
  return (
    <ChartFrame title="Defects stopped and false alarms, by gate" subtitle={`Out of ${data[0]?.defects ?? 0} defective and ${data[0]?.clean ?? 0} clean sample models.`} height={300}
      table={{ headers: ["Gate", "Defects stopped", "Of defects", "False alarms", "Of clean"], rows: data.map((d) => [d.gate, d.stopped, d.defects, d.alarms, d.clean]) }}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis dataKey="gate" tick={AXIS} interval={0} />
        <YAxis allowDecimals={false} tick={AXIS} />
        <Tooltip />
        <Legend />
        <Bar dataKey="stopped" name="Defects stopped" fill={BLUE} radius={[4, 4, 0, 0]} isAnimationActive={ANIMATE} />
        <Bar dataKey="alarms" name="False alarms" fill="#f59e0b" radius={[4, 4, 0, 0]} isAnimationActive={ANIMATE} />
      </BarChart>
    </ChartFrame>
  );
}

export function SensitivityChart({ cases }: { cases: SensitivityCase[] }) {
  const data = cases.map((c) => {
    const total = Object.values(c.families).reduce((s, f) => s + f.total, 0);
    const kept = Object.values(c.families).reduce((s, f) => s + f.kept, 0);
    return { name: c.model, kept: total ? +((kept / total) * 100).toFixed(1) : 0, decision: c.baseline_decision as Decision, total, keptCount: kept };
  });
  return (
    <ChartFrame title="Share of perturbations that keep the decision" subtitle="Weights, thresholds, bands and missing-evidence policy changed; 100% means nothing flipped it." height={300}
      table={{ headers: ["Model", "Baseline decision", "Perturbations kept", "Of"], rows: data.map((d) => [d.name, d.decision, `${d.kept}% (${d.keptCount})`, d.total]) }}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis dataKey="name" tick={AXIS} interval={0} angle={-25} textAnchor="end" height={64} />
        <YAxis domain={[0, 100]} unit="%" tick={AXIS} />
        <Tooltip formatter={(v: number) => `${v}%`} />
        <Bar dataKey="kept" name="Kept" radius={[4, 4, 0, 0]} isAnimationActive={ANIMATE}>
          {data.map((d) => <Cell key={d.name} fill={DECISION_COLOR[d.decision] ?? BLUE} />)}
        </Bar>
      </BarChart>
    </ChartFrame>
  );
}
