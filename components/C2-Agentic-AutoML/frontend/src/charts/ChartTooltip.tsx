import { ReactNode } from "react";

export interface TipRow {
  label: ReactNode;
  value: ReactNode;
  color?: string;
}

/** The single tooltip style used by every chart: title + rows with a colour key (text stays ink). */
export function Tip({ title, rows }: { title?: ReactNode; rows: TipRow[] }) {
  return (
    <div className="tooltip">
      {title && <div className="tt-title">{title}</div>}
      {rows.map((r, i) => (
        <div key={i} className="tt-row">
          <span>{r.color && <span className="sw" style={{ background: r.color }} />}{r.label}</span>
          <b>{r.value}</b>
        </div>
      ))}
    </div>
  );
}

// Shared axis / grid props: hairline, solid, recessive
export const AXIS = { stroke: "#c3c2b7", tickLine: false, tick: { fill: "#898781", fontSize: 11.5 } } as const;
export const GRID = { stroke: "#e1e0d9", strokeDasharray: "", vertical: false } as const;
