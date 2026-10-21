import { Cpu, Database, Layers } from "lucide-react";

import type { Dimension, DimensionScore } from "../api/types";
import { DIMENSION_LABEL, DIMENSION_SHORT, fmtPercent, fmtScore } from "../lib/format";
import { Card } from "./ui";

const ICON = { training_validity: Database, model_behaviour: Layers, operational_readiness: Cpu };

export function LayerCard({ dimension, score, weight, effectiveWeight }: {
  dimension: Dimension; score: DimensionScore; weight: number; effectiveWeight?: number;
}) {
  const Icon = ICON[dimension];
  const value = score.score;
  const differs = effectiveWeight !== undefined && Math.abs(effectiveWeight - weight) > 1e-9;
  return (
    <Card aria-label={`${DIMENSION_LABEL[dimension]} score`} className="grid gap-2.5 px-4 py-4">
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-1.5 text-[13px] font-medium text-slate-500"><Icon size={15} className="text-blue-600" aria-hidden />{DIMENSION_LABEL[dimension]}</div>
          <div className="text-xs text-slate-500">{DIMENSION_SHORT[dimension]} · weight {fmtPercent(weight)}{differs && <> · counted as {fmtPercent(effectiveWeight!)}</>}</div>
        </div>
        <div className={`text-3xl font-bold tracking-tight tabular-nums ${value === null ? "text-slate-400" : "text-slate-900"}`}>{fmtScore(value)}</div>
      </div>
      <div className="h-2 overflow-hidden rounded-full border border-blue-100 bg-blue-50" role="progressbar" aria-valuemin={0} aria-valuemax={100}
        aria-valuenow={value ?? undefined} aria-label={`${DIMENSION_SHORT[dimension]} score`}>
        <span className="block h-full rounded-full bg-linear-to-r from-blue-500 to-blue-700" style={{ width: `${value ?? 0}%` }} />
      </div>
      <div className="flex flex-wrap justify-between gap-x-3 gap-y-1 text-xs text-slate-500">
        <span>{score.available_checks} of {score.total_checks} checks with evidence</span>
        <span>Evidence {fmtPercent(score.coverage)}</span>
      </div>
      {score.note && <p className="text-xs text-slate-500">{score.note}</p>}
    </Card>
  );
}
