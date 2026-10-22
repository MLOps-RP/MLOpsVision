import { useState } from "react";
import {
  CartesianGrid, ComposedChart, LabelList, Line, LineChart, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis,
} from "recharts";
import { Trial } from "../api";
import { familyColor, familyLabel, fmt, roundColor } from "../theme";
import { Legend } from "../components/ui";
import { AXIS, GRID, Tip } from "./ChartTooltip";

/** Best CV score after each round (single series: no legend box, the title names it). */
export function ScoreByRound({ scores }: { scores: number[] }) {
  const rows = scores.map((s, i) => ({ round: `Round ${i + 1}`, score: s }));
  const lo = Math.min(...scores), hi = Math.max(...scores);
  const pad = Math.max((hi - lo) * 0.6, 0.002);
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={rows} margin={{ top: 22, right: 40, bottom: 4, left: 0 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="round" {...AXIS} />
        <YAxis domain={[lo - pad, hi + pad]} tickFormatter={(v) => v.toFixed(3)} {...AXIS} width={52} />
        <Tooltip cursor={{ stroke: "#c3c2b7" }}
          content={({ active, payload }) => active && payload?.length ? (
            <Tip title={payload[0].payload.round} rows={[{ label: "Best CV score", value: fmt(payload[0].payload.score), color: "#2a78d6" }]} />
          ) : null} />
        <Line dataKey="score" stroke="#2a78d6" strokeWidth={2} isAnimationActive={false}
          dot={{ r: 5, fill: "#2a78d6", stroke: "#fcfcfb", strokeWidth: 2 }} activeDot={{ r: 6, stroke: "#fcfcfb", strokeWidth: 2 }}>
          <LabelList dataKey="score" position="top" formatter={(v: number) => v.toFixed(4)}
            style={{ fill: "#52514e", fontSize: 11.5, fontVariantNumeric: "tabular-nums" }} />
        </Line>
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Best trial of each family in each round. */
export function bestPerFamilyRound(trials: Trial[]) {
  const best = new Map<string, Trial>();
  for (const t of trials) {
    if (t.state !== "COMPLETE" || t.mean_val === null) continue;
    const key = `${t.family}|${t.round}`;
    const cur = best.get(key);
    if (!cur || (cur.mean_val ?? -Infinity) < t.mean_val) best.set(key, t);
  }
  return [...best.values()].sort((a, b) => a.family.localeCompare(b.family) || a.round - b.round);
}

/**
 * Dumbbell: train score (hollow ring) vs validation score (filled dot) of each family's best
 * configuration per round. The segment length IS the overfitting gap, so the effect of the
 * adaptive loop is visible as a shrinking segment.
 */
export function GapDumbbell({ trials }: { trials: Trial[] }) {
  const rows = bestPerFamilyRound(trials).filter((t) => t.mean_train !== null);
  const [hover, setHover] = useState<number | null>(null);
  if (!rows.length) return <div className="empty">No completed trials yet</div>;
  const values = rows.flatMap((r) => [r.mean_val!, r.mean_train!]);
  const lo = Math.floor((Math.min(...values) - 0.01) * 100) / 100;
  const hi = Math.min(1, Math.ceil((Math.max(...values) + 0.005) * 100) / 100);
  const W = 640, L = 170, R = 70, rowH = 30, top = 14;
  const x = (v: number) => L + ((v - lo) / (hi - lo || 1)) * (W - L - R);
  const ticks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
  const H = top + rows.length * rowH + 28;
  return (
    <div>
      <Legend items={[
        { color: "#52514e", label: "Validation score (CV)" },
        { color: "#52514e", hollow: true, label: "Training score" },
        { color: "#c3c2b7", line: true, label: "Gap = overfitting" },
      ]} />
      <svg viewBox={`0 0 ${W} ${H}`} className="wf-svg" role="img" aria-label="Train vs validation score per family and round">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={top - 6} y2={H - 24} stroke="#e1e0d9" />
            <text x={x(t)} y={H - 8} textAnchor="middle" fontSize={11.5} fill="#898781">{t.toFixed(2)}</text>
          </g>
        ))}
        {rows.map((r, i) => {
          const y = top + i * rowH + rowH / 2;
          const c = familyColor(r.family);
          const isNewFamily = i === 0 || rows[i - 1].family !== r.family;
          return (
            <g key={`${r.family}-${r.round}`} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={0} y={y - rowH / 2} width={W} height={rowH} fill={hover === i ? "#f2f1ed" : "transparent"} />
              <text x={8} y={y + 4} fontSize={12.5} fill="#0b0b0b" fontWeight={isNewFamily ? 600 : 400}>
                {isNewFamily ? familyLabel(r.family) : ""}
              </text>
              <text x={L - 10} y={y + 4} fontSize={11.5} fill="#898781" textAnchor="end">R{r.round}</text>
              <line x1={x(r.mean_val!)} x2={x(r.mean_train!)} y1={y} y2={y} stroke="#c3c2b7" strokeWidth={2} strokeLinecap="round" />
              <circle cx={x(r.mean_train!)} cy={y} r={5} fill="#fcfcfb" stroke={c} strokeWidth={2.5} />
              <circle cx={x(r.mean_val!)} cy={y} r={6} fill={c} stroke="#fcfcfb" strokeWidth={2} />
              <text x={W - 6} y={y + 4} fontSize={11.5} fill="#52514e" textAnchor="end" style={{ fontVariantNumeric: "tabular-nums" }}>
                gap {fmt(r.gap, 3)}
              </text>
            </g>
          );
        })}
      </svg>
      {hover !== null && (
        <div className="small ink2" style={{ marginTop: 4 }}>
          <b style={{ color: "var(--ink)" }}>{familyLabel(rows[hover].family)} · round {rows[hover].round}</b>
          {" "}— validation {fmt(rows[hover].mean_val)}, train {fmt(rows[hover].mean_train)}, gap {fmt(rows[hover].gap)}
        </div>
      )}
    </div>
  );
}

/**
 * Search history as small multiples (one panel per family - avoids >3 hues in one scatter).
 * Dots = trials coloured by round (ordinal blue ramp); hollow grey = pruned; ink line = best so far.
 */
export function TrialSmallMultiples({ trials, live = false }: { trials: Trial[]; live?: boolean }) {
  const families = [...new Set(trials.map((t) => t.family))];
  const done = trials.filter((t) => t.mean_val !== null);
  if (!families.length) return <div className="empty">{live ? "Waiting for the first trials…" : "No trials"}</div>;
  const rounds = [...new Set(trials.map((t) => t.round))].sort();
  const vals = done.filter((t) => t.state === "COMPLETE").map((t) => t.mean_val!);
  const sortedVals = [...vals].sort((a, b) => a - b);
  const p5 = sortedVals[Math.floor(sortedVals.length * 0.05)] ?? 0;
  const lo = Math.max(0, p5 - 0.01), hi = Math.min(1, (sortedVals[sortedVals.length - 1] ?? 1) + 0.005);
  return (
    <div>
      <Legend items={[
        ...rounds.map((r) => ({ color: roundColor(r), label: `Round ${r}` })),
        { color: "#898781", hollow: true, label: "Pruned (stopped early)" },
        { color: "#0b0b0b", line: true, label: "Best so far" },
      ]} />
      <div className={`grid ${families.length > 2 ? "g3" : "g2"}`}>
        {families.map((f) => {
          const ft = trials.filter((t) => t.family === f);
          let best = -Infinity;
          const rows = ft.map((t, i) => {
            if (t.state === "COMPLETE" && t.mean_val !== null) best = Math.max(best, t.mean_val);
            const row: Record<string, number | null> = { seq: i + 1, best: Number.isFinite(best) ? best : null };
            if (t.mean_val !== null) {
              if (t.state === "PRUNED") row.pruned = Math.max(t.mean_val, lo);
              else row[`r${t.round}`] = Math.max(t.mean_val, lo);
            }
            return row;
          });
          return (
            <div key={f} style={{ border: "1px solid var(--border)", borderRadius: 12, padding: "10px 8px 2px" }}>
              <div className="row" style={{ padding: "0 6px 4px", justifyContent: "space-between" }}>
                <span className="row" style={{ gap: 6, fontWeight: 600, fontSize: 13 }}>
                  <span style={{ width: 9, height: 9, borderRadius: 3, background: familyColor(f) }} />{familyLabel(f)}
                </span>
                <span className="small muted">best {fmt(Number.isFinite(best) ? best : null)} · {ft.length} trials</span>
              </div>
              <ResponsiveContainer width="100%" height={170}>
                <ComposedChart data={rows} margin={{ top: 6, right: 8, bottom: 0, left: -12 }}>
                  <CartesianGrid {...GRID} />
                  <XAxis dataKey="seq" type="number" domain={[1, Math.max(rows.length, 2)]} allowDecimals={false} {...AXIS} />
                  <YAxis domain={[lo, hi]} tickFormatter={(v) => v.toFixed(2)} {...AXIS} width={44} />
                  <Tooltip cursor={false}
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null;
                      const seq = payload[0].payload.seq as number;
                      const t = ft[seq - 1];
                      if (!t) return null;
                      return <Tip title={`${familyLabel(f)} · trial ${seq}`} rows={[
                        { label: "Round", value: t.round, color: roundColor(t.round) },
                        { label: "CV score", value: fmt(t.mean_val) },
                        { label: "Train score", value: fmt(t.mean_train) },
                        { label: "State", value: t.state.toLowerCase() },
                      ]} />;
                    }} />
                  <Line dataKey="best" type="stepAfter" stroke="#0b0b0b" strokeWidth={1.5} dot={false} isAnimationActive={false} connectNulls />
                  {rounds.map((r) => (
                    <Scatter key={r} dataKey={`r${r}`} fill={roundColor(r)} isAnimationActive={false}
                      shape={(p: any) => p.cy == null ? <g /> : <circle cx={p.cx} cy={p.cy} r={4.5} fill={roundColor(r)} stroke="#fcfcfb" strokeWidth={2} />} />
                  ))}
                  <Scatter dataKey="pruned" isAnimationActive={false}
                    shape={(p: any) => p.cy == null ? <g /> : <circle cx={p.cx} cy={p.cy} r={4} fill="#fcfcfb" stroke="#898781" strokeWidth={1.5} />} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Horizontal bars, one series = one colour, value at the bar tip. */
export function HBars({ items, color = "#2a78d6", max, format = (v: number) => v.toFixed(3) }: {
  items: { label: string; value: number; note?: string }[]; color?: string; max?: number; format?: (v: number) => string;
}) {
  const m = max ?? Math.max(...items.map((i) => i.value), 1e-9);
  return (
    <div>
      {items.map((it) => (
        <div className="hbar-row" key={it.label} title={`${it.label}: ${format(it.value)}`}>
          <span className="lbl">{it.label}{it.note && <span className="muted"> · {it.note}</span>}</span>
          <div className="hbar-track">
            <div className="hbar-fill" style={{ width: `${(it.value / m) * 100}%`, background: color, maxHeight: 24 }} />
          </div>
          <span className="val">{format(it.value)}</span>
        </div>
      ))}
    </div>
  );
}
