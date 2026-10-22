import {
  Area, CartesianGrid, ComposedChart, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Evaluation } from "../api";
import { familyColor, familyLabel, fmt } from "../theme";
import { Legend } from "../components/ui";
import { AXIS, GRID, Tip } from "./ChartTooltip";

const GRID_X = Array.from({ length: 101 }, (_, i) => i / 100);

/** Step-interpolate a ROC curve onto a common FPR grid so all families share one crosshair. */
function tprAt(curve: [number, number][], x: number) {
  let tpr = 0;
  for (const [fpr, t] of curve) {
    if (fpr <= x + 1e-12) tpr = Math.max(tpr, t);
    else break;
  }
  return tpr;
}

export function RocChart({ ev }: { ev: Evaluation }) {
  const families = Object.keys(ev.candidates);
  const rows = GRID_X.map((x) => {
    const r: Record<string, number> = { fpr: x };
    families.forEach((f) => (r[f] = tprAt(ev.candidates[f].roc, x)));
    return r;
  });
  const sorted = [...families].sort((a, b) => ev.candidates[b].auc - ev.candidates[a].auc);
  return (
    <div>
      <Legend
        items={sorted.map((f) => ({
          color: familyColor(f),
          line: true,
          label: <span>{familyLabel(f)} <b style={{ color: "var(--ink)" }}>AUC {fmt(ev.candidates[f].auc, 3)}</b>{f === ev.selected_family ? " · selected" : ""}</span>,
        }))}
      />
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 22, left: 0 }}>
          <CartesianGrid {...GRID} vertical />
          <XAxis dataKey="fpr" type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...AXIS}
            label={{ value: "False positive rate", position: "insideBottom", offset: -12, className: "axis-label" }} />
          <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...AXIS} width={60}
            label={{ value: "True positive rate", angle: -90, position: "insideLeft", offset: 8, style: { textAnchor: "middle" }, className: "axis-label" }} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#c3c2b7" strokeWidth={1} />
          <Tooltip
            cursor={{ stroke: "#c3c2b7" }}
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <Tip title={`FPR ${fmt(Number(label), 2)}`}
                  rows={sorted.map((f) => ({ label: familyLabel(f), color: familyColor(f), value: `TPR ${fmt(Number(payload.find((p) => p.dataKey === f)?.value), 3)}` }))} />
              ) : null}
          />
          {families.map((f) => (
            <Line key={f} dataKey={f} type="stepAfter" stroke={familyColor(f)} strokeWidth={2} dot={false}
              strokeOpacity={f === ev.selected_family ? 1 : 0.55} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function PrChart({ ev }: { ev: Evaluation }) {
  const color = familyColor(ev.selected_family);
  const rows = ev.selected.pr.map(([recall, precision]) => ({ recall, precision }));
  return (
    <div>
      <Legend items={[
        { color, line: true, label: <span>{familyLabel(ev.selected_family)} <b style={{ color: "var(--ink)" }}>AP {fmt(ev.selected.ap, 3)}</b></span> },
        { color: "#c3c2b7", line: true, label: `No-skill = positive rate ${fmt(ev.prevalence, 3)}` },
      ]} />
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={rows} margin={{ top: 8, right: 16, bottom: 22, left: 0 }}>
          <CartesianGrid {...GRID} vertical />
          <XAxis dataKey="recall" type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...AXIS}
            label={{ value: "Recall", position: "insideBottom", offset: -12, className: "axis-label" }} />
          <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...AXIS} width={60}
            label={{ value: "Precision", angle: -90, position: "insideLeft", offset: 8, style: { textAnchor: "middle" }, className: "axis-label" }} />
          <ReferenceLine y={ev.prevalence} stroke="#c3c2b7" />
          <Tooltip cursor={{ stroke: "#c3c2b7" }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <Tip rows={[{ label: "Recall", value: fmt(payload[0].payload.recall, 3) }, { label: "Precision", value: fmt(payload[0].payload.precision, 3), color }]} />
              ) : null} />
          <Area dataKey="precision" type="stepAfter" stroke="none" fill={color} fillOpacity={0.1} isAnimationActive={false} />
          <Line dataKey="precision" type="stepAfter" stroke={color} strokeWidth={2} dot={false} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

// Sequential blue ramp (one hue, light -> dark) for the confusion-matrix counts
const SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"];

export function ConfusionMatrix({ matrix, positive }: { matrix: number[][]; positive?: string | null }) {
  const [[tn, fp], [fn, tp]] = matrix;
  const max = Math.max(tn, fp, fn, tp, 1);
  const total = tn + fp + fn + tp || 1;
  const cell = (v: number, name: string) => {
    const step = Math.min(SEQ.length - 1, Math.floor((v / max) * (SEQ.length - 1) + 0.0001));
    const dark = step >= 3;
    return (
      <div className="cell" style={{ background: SEQ[step], color: dark ? "#fff" : "#0b0b0b" }} title={`${name}: ${v}`}>
        <b>{v.toLocaleString()}</b>
        <span>{name} · {((v / total) * 100).toFixed(1)}%</span>
      </div>
    );
  };
  return (
    <div>
      <div className="matrix" style={{ gridTemplateColumns: "92px 1fr 1fr" }}>
        <div />
        <div className="hdr">Predicted 0</div>
        <div className="hdr">Predicted 1{positive ? ` (${positive})` : ""}</div>
        <div className="hdr">Actual 0</div>
        {cell(tn, "True negative")}
        {cell(fp, "False positive")}
        <div className="hdr">Actual 1</div>
        {cell(fn, "False negative")}
        {cell(tp, "True positive")}
      </div>
      <div className="small muted" style={{ marginTop: 10 }}>
        Threshold 0.5 · recall = TP / (TP + FN) = <b style={{ color: "var(--ink)" }}>{fmt(tp / Math.max(tp + fn, 1), 3)}</b>
        {" "}· precision = <b style={{ color: "var(--ink)" }}>{fmt(tp / Math.max(tp + fp, 1), 3)}</b>
      </div>
    </div>
  );
}
