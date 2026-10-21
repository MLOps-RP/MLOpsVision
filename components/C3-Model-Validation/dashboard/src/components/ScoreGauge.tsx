interface Props {
  /** The MQS, or null when it cannot be quoted. */
  value: number | null;
  /** What the MQS would be if every missing check had scored 0. */
  lowerBound?: number | null;
  provisional?: boolean;
  promoteAt?: number;
  rejectBelow?: number;
}

const CX = 130, CY = 124, R = 92, STROKE = 16;

function point(fraction: number, radius: number): [number, number] {
  const angle = Math.PI * (1 - fraction);
  return [CX + radius * Math.cos(angle), CY - radius * Math.sin(angle)];
}

function arc(fraction: number): string {
  const f = Math.min(Math.max(fraction, 0), 0.9999);
  const [x, y] = point(f, R);
  // The sweep is at most 180 degrees, so it is never a "large arc" (flag 0).
  return `M ${CX - R} ${CY} A ${R} ${R} 0 0 1 ${x.toFixed(2)} ${y.toFixed(2)}`;
}

/** Half-circle gauge for the 0-100 MQS, with the profile's decision thresholds marked. */
export function ScoreGauge({ value, lowerBound, provisional, promoteAt, rejectBelow }: Props) {
  const has = value !== null;
  const label = has ? `Model Quality Score ${value.toFixed(1)} out of 100` : "Model Quality Score not available";
  const ticks = [
    { at: rejectBelow, name: "reject below" }, { at: promoteAt, name: "promote at" },
  ].filter((t): t is { at: number; name: string } => typeof t.at === "number");

  return (
    <div className="grid justify-items-center px-3 pb-3 pt-4 text-center">
      <svg viewBox="0 0 260 158" width="100%" className="max-w-75" role="img" aria-label={label}>
        <path d={arc(1)} fill="none" className="stroke-blue-100" strokeWidth={STROKE} strokeLinecap="round" />
        {has && value > 0 && <path d={arc(value / 100)} fill="none" className="stroke-blue-600" strokeWidth={STROKE} strokeLinecap="round" />}
        {ticks.map((t) => {
          const [x1, y1] = point(t.at / 100, R - 14);
          const [x2, y2] = point(t.at / 100, R + 11);
          const [lx, ly] = point(t.at / 100, R + 22);
          return (
            <g key={t.name}>
              <line x1={x1} y1={y1} x2={x2} y2={y2} className="stroke-slate-600" strokeWidth={2} />
              <text x={lx} y={ly} fontSize="10.5" textAnchor="middle" className="fill-slate-500">
                <title>{`${t.name} ${t.at}`}</title>
                {t.at}
              </text>
            </g>
          );
        })}
        <text x={CX} y={CY - 6} textAnchor="middle" fontSize="40" fontWeight="700" className={has ? "fill-slate-900" : "fill-slate-400"}>
          {has ? value.toFixed(1) : "n/a"}
        </text>
        <text x={CX} y={CY + 16} textAnchor="middle" fontSize="12" className="fill-slate-500">Model Quality Score</text>
      </svg>
      <div className="mt-1 flex flex-wrap justify-center gap-x-4 text-xs text-slate-500">
        {typeof rejectBelow === "number" && <span>Below {rejectBelow}: rollback band</span>}
        {typeof promoteAt === "number" && <span>{promoteAt}+ : promotion band</span>}
      </div>
      {(provisional || !has) && (
        <p className="mt-2 max-w-[34ch] text-[13px] text-slate-500">
          {has ? "Provisional: some checks had no evidence." : "Not enough evidence to quote a score."}
          {typeof lowerBound === "number" && <> Worst case if every missing check scored 0: <strong>{lowerBound.toFixed(1)}</strong>.</>}
        </p>
      )}
    </div>
  );
}
