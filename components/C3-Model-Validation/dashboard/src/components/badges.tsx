import {
  AlertTriangle, CheckCircle2, CircleSlash, HelpCircle, Hand, OctagonX, RotateCcw, ShieldCheck, XCircle,
} from "lucide-react";
import type { ReactNode } from "react";

import type { CheckStatus, Decision, EvidenceSource, RuleKind } from "../api/types";
import { EVIDENCE_HELP, STATUS_LABEL } from "../lib/format";
import { Badge, type Tone } from "./ui";

const STATUS_TONE: Record<CheckStatus, Tone> = {
  PASS: "green", WARN: "amber", FAIL: "red", ERROR: "red", UNAVAILABLE: "slate", NOT_APPLICABLE: "slate",
};
const STATUS_ICON: Record<CheckStatus, ReactNode> = {
  PASS: <CheckCircle2 size={13} aria-hidden />,
  WARN: <AlertTriangle size={13} aria-hidden />,
  FAIL: <XCircle size={13} aria-hidden />,
  ERROR: <OctagonX size={13} aria-hidden />,
  UNAVAILABLE: <HelpCircle size={13} aria-hidden />,
  NOT_APPLICABLE: <CircleSlash size={13} aria-hidden />,
};

/** Status is always icon + text, never colour alone. */
export function StatusBadge({ status }: { status: CheckStatus }) {
  return <Badge tone={STATUS_TONE[status]}>{STATUS_ICON[status]}{STATUS_LABEL[status]}</Badge>;
}

const DECISION_TONE: Record<Decision, Tone> = { PROMOTE: "green", HOLD: "amber", ROLLBACK: "red" };
const DECISION_ICON: Record<Decision, ReactNode> = {
  PROMOTE: <ShieldCheck size={13} aria-hidden />, HOLD: <Hand size={13} aria-hidden />, ROLLBACK: <RotateCcw size={13} aria-hidden />,
};

export function DecisionBadge({ decision }: { decision: Decision }) {
  return <Badge tone={DECISION_TONE[decision]}>{DECISION_ICON[decision]}{decision}</Badge>;
}

const EVIDENCE_TONE: Record<EvidenceSource, Tone> = { MEASURED: "blue", DECLARED: "orange", SIMULATED: "violet", UNAVAILABLE: "slate" };

export function EvidenceBadge({ source }: { source: EvidenceSource }) {
  return (
    <Badge tone={EVIDENCE_TONE[source]} title={EVIDENCE_HELP[source]}>
      {source === "UNAVAILABLE" ? "None" : source.charAt(0) + source.slice(1).toLowerCase()}
    </Badge>
  );
}

const KIND_LABEL: Record<RuleKind, string> = { critical: "Critical rule", evidence: "Evidence", blocking: "Must-pass", score: "Score band" };
const KIND_TONE: Record<RuleKind, Tone> = { critical: "red", evidence: "slate", blocking: "amber", score: "blue" };

export const KindBadge = ({ kind }: { kind: RuleKind }) => <Badge tone={KIND_TONE[kind]}>{KIND_LABEL[kind]}</Badge>;

/** Hex colours shared by the charts, so a decision or status has one colour everywhere. */
export const DECISION_COLOR: Record<Decision, string> = { PROMOTE: "#16a34a", HOLD: "#f59e0b", ROLLBACK: "#dc2626" };
export const STATUS_COLOR: Record<CheckStatus, string> = {
  PASS: "#16a34a", WARN: "#f59e0b", FAIL: "#dc2626", ERROR: "#b91c1c", UNAVAILABLE: "#94a3b8", NOT_APPLICABLE: "#cbd5e1",
};
export const BLUE = "#2563eb";
export const BLUE_LIGHT = "#93c5fd";
