import { Hand, RotateCcw, ShieldCheck, type LucideIcon } from "lucide-react";

import type { Decision } from "../api/types";
import { DECISION_MEANING } from "../lib/format";
import { Badge, cx } from "./ui";

const ICON: Record<Decision, LucideIcon> = { PROMOTE: ShieldCheck, HOLD: Hand, ROLLBACK: RotateCcw };
const SKIN: Record<Decision, { box: string; icon: string; word: string }> = {
  PROMOTE: { box: "border-green-300 bg-linear-to-b from-green-50 to-white", icon: "bg-green-100 text-green-700", word: "text-green-700" },
  HOLD: { box: "border-amber-300 bg-linear-to-b from-amber-50 to-white", icon: "bg-amber-100 text-amber-700", word: "text-amber-700" },
  ROLLBACK: { box: "border-red-300 bg-linear-to-b from-red-50 to-white", icon: "bg-red-100 text-red-700", word: "text-red-700" },
};

export function DecisionBanner({ decision, summary, overriddenByCritical, profile, notice }: {
  decision: Decision; summary: string; overriddenByCritical?: boolean; profile?: string; notice: string;
}) {
  const Icon = ICON[decision];
  const skin = SKIN[decision];
  return (
    <section aria-label="Decision" className={cx("flex flex-col gap-4 rounded-xl border p-5 shadow-sm sm:flex-row sm:items-center", skin.box)}>
      <div className={cx("grid size-13 shrink-0 place-items-center rounded-2xl", skin.icon)}><Icon size={26} aria-hidden /></div>
      <div className="min-w-0 flex-1">
        <div className="text-[12.5px] font-semibold uppercase tracking-wider text-slate-500">Recommendation</div>
        <div className={cx("text-[28px] font-extrabold tracking-tight", skin.word)}>
          {decision} <span className="text-[15px] font-semibold text-slate-500">· {DECISION_MEANING[decision]}</span>
        </div>
        <p className="mt-0.5 max-w-[90ch] text-slate-700">{summary}</p>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-[13px] text-slate-500">
          <span>{notice}</span>
          {overriddenByCritical && <Badge tone="red">A critical rule overrode the score</Badge>}
          {profile && <Badge tone="blue">Profile: {profile}</Badge>}
        </div>
      </div>
    </section>
  );
}
