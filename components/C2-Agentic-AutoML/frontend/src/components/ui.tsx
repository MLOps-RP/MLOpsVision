import { ReactNode, useState } from "react";
import {
  AlertOctagon, AlertTriangle, BookOpen, CheckCircle2, Loader2, MinusCircle, Scale, Table2,
  BarChart3, TrendingDown,
} from "lucide-react";
import { AGENTS, AgentKey, DIAGNOSIS } from "../theme";

export function Card({ children, className = "", pad = false }: { children: ReactNode; className?: string; pad?: boolean }) {
  return <div className={`card ${pad ? "card-pad" : ""} ${className}`}>{children}</div>;
}

export function PageHead({ title, sub, actions }: { title: string; sub?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="page-head">
      <div>
        <h1>{title}</h1>
        {sub && <p>{sub}</p>}
      </div>
      {actions && <div className="actions">{actions}</div>}
    </div>
  );
}

export function Stat({ label, value, note, icon, hero = false }: {
  label: string; value: ReactNode; note?: ReactNode; icon?: ReactNode; hero?: boolean;
}) {
  return (
    <Card className="stat">
      <div className="stat-label">{icon}{label}</div>
      <div className={`stat-value ${hero ? "hero" : ""}`}>{value}</div>
      {note && <div className="stat-note">{note}</div>}
    </Card>
  );
}

export function AgentBadge({ agent }: { agent: string }) {
  const key = (agent in AGENTS ? agent : "tools") as AgentKey;
  const a = AGENTS[key];
  return (
    <span className="agent-badge" style={{ background: a.tint, color: a.color }}>
      <span style={{ width: 7, height: 7, borderRadius: 2, background: a.color }} />
      {a.name}
    </span>
  );
}

const DIAG_ICON: Record<string, ReactNode> = {
  good: <CheckCircle2 size={14} />,
  warning: <AlertTriangle size={14} />,
  serious: <Scale size={14} />,
  critical: <AlertOctagon size={14} />,
  muted: <MinusCircle size={14} />,
};

/** Diagnosis = status: icon + label, never colour alone. */
export function DiagnosisChip({ label }: { label: string }) {
  const d = DIAGNOSIS[label] ?? { label, color: "#898781", status: "muted" };
  const icon = label === "underfitting" ? <TrendingDown size={14} /> : DIAG_ICON[d.status];
  return (
    <span className="chip">
      <span style={{ color: d.color, display: "inline-flex" }}>{icon}</span>
      {d.label}
    </span>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const map: Record<string, [string, string]> = {
    running: ["#2a78d6", "Running"],
    completed: ["#0ca30c", "Completed"],
    failed: ["#d03b3b", "Failed"],
  };
  const [color, text] = map[status] ?? ["#898781", status];
  return (
    <span className="badge">
      {status === "running" ? <Loader2 size={12} className="spin" color={color} /> : <span className="dot" style={{ background: color }} />}
      {text}
    </span>
  );
}

export function Loading({ text = "Loading…" }: { text?: string }) {
  return (
    <div className="empty row" style={{ justifyContent: "center" }}>
      <Loader2 size={16} className="spin" /> {text}
    </div>
  );
}

export function ErrorBox({ error }: { error: string }) {
  return <div className="callout error">⚠ {error}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function Tabs<T extends string>({ tabs, value, onChange }: {
  tabs: { id: T; label: string; icon?: ReactNode }[]; value: T; onChange: (v: T) => void;
}) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((t) => (
        <button key={t.id} role="tab" aria-selected={t.id === value} className={`tab ${t.id === value ? "on" : ""}`} onClick={() => onChange(t.id)}>
          {t.icon}{t.label}
        </button>
      ))}
    </div>
  );
}

export function JupyterButton({ url, label = "Open in Jupyter", small = false }: { url: string; label?: string; small?: boolean }) {
  return (
    <a className={`btn jupyter ${small ? "small" : ""}`} href={url} target="_blank" rel="noreferrer" title="Opens JupyterLab (start it with scripts/start_jupyter.sh)">
      <BookOpen size={small ? 13 : 15} color="#e46e0e" /> {label}
    </a>
  );
}

export interface TableSpec {
  columns: string[];
  rows: (string | number | null | undefined)[][];
}

/** A chart card with a chart/table switch: every chart has a table view (accessibility + exact values). */
export function ChartCard({ title, sub, children, table, tools, height }: {
  title: string; sub?: ReactNode; children: ReactNode; table?: TableSpec; tools?: ReactNode; height?: number;
}) {
  const [view, setView] = useState<"chart" | "table">("chart");
  return (
    <Card>
      <div className="card-head">
        <div>
          <h3>{title}</h3>
          {sub && <div className="sub">{sub}</div>}
        </div>
        <div className="tools">
          {tools}
          {table && (
            <div className="seg" aria-label="View">
              <button className={view === "chart" ? "on" : ""} onClick={() => setView("chart")} title="Chart"><BarChart3 size={13} /></button>
              <button className={view === "table" ? "on" : ""} onClick={() => setView("table")} title="Table"><Table2 size={13} /></button>
            </div>
          )}
        </div>
      </div>
      <div className="card-body" style={height && view === "chart" ? { minHeight: height } : undefined}>
        {view === "chart" || !table ? children : <DataTable spec={table} />}
      </div>
    </Card>
  );
}

export function DataTable({ spec, maxHeight = 420 }: { spec: TableSpec; maxHeight?: number | "none" }) {
  return (
    <div className="table-wrap" style={maxHeight === "none" ? undefined : { maxHeight }}>
      <table className="t">
        <thead><tr>{spec.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>
          {spec.rows.map((r, i) => (
            <tr key={i}>{r.map((v, j) => <td key={j} className={typeof v === "number" ? "num" : ""}>{v ?? "–"}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Legend({ items }: { items: { label: ReactNode; color: string; line?: boolean; hollow?: boolean }[] }) {
  return (
    <div className="chart-legend">
      {items.map((it, i) => (
        <span key={i} className="key">
          <span
            className={it.line ? "ln" : "sw"}
            style={it.hollow ? { border: `2px solid ${it.color}`, background: "transparent" } : { background: it.color }}
          />
          {it.label}
        </span>
      ))}
    </div>
  );
}

export function JsonView({ value, maxHeight }: { value: unknown; maxHeight?: number }) {
  return <pre className="json" style={maxHeight ? { maxHeight } : undefined}>{JSON.stringify(value, null, 2)}</pre>;
}
