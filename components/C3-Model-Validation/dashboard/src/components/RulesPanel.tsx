import { Lightbulb } from "lucide-react";

import type { CheckResult, TriggeredRule } from "../api/types";
import { NO_EVIDENCE } from "../api/types";
import { DecisionBadge, KindBadge, StatusBadge } from "./badges";
import { Card, CardBody, CardHeader, Notice, Table, TableWrap, Td, Th } from "./ui";

export function RulesPanel({ rules, overridden }: { rules: TriggeredRule[]; overridden: boolean }) {
  return (
    <Card>
      <CardHeader title="Rules that fired"
        subtitle="Critical rules override the score; evidence and must-pass rules hold promotion; score bands decide the rest." />
      {overridden && (
        <CardBody className="pb-0">
          <Notice tone="error" title="A critical rule overrode the score">
            The Model Quality Score does not matter here: a critical failure forces a ROLLBACK recommendation.
          </Notice>
        </CardBody>
      )}
      <ul className="m-0 list-none p-0">
        {rules.map((rule) => (
          <li key={rule.rule_id} className="border-b border-slate-100 px-5 py-3.5 last:border-b-0">
            <div className="mb-1.5 flex flex-wrap items-center gap-2">
              <KindBadge kind={rule.kind} />
              <DecisionBadge decision={rule.effect} />
              <strong>{rule.name}</strong>
              <code className="font-mono text-xs text-slate-500">{rule.rule_id}</code>
            </div>
            <p className="text-slate-700">{rule.reason}</p>
            {rule.check_ids && rule.check_ids.length > 0 && <p className="mt-1 text-[12.5px] text-slate-500">Checks: {rule.check_ids.join(", ")}</p>}
          </li>
        ))}
        {rules.length === 0 && <li className="px-5 py-4 text-slate-500">No rule fired.</li>}
      </ul>
    </Card>
  );
}

export function Explanations({ items }: { items: string[] }) {
  if (items.length === 0) return null;
  return (
    <Card>
      <CardHeader title="Why this decision" subtitle="A trace of the gate's own reasoning. It explains the decision, not the model's individual predictions."
        actions={<Lightbulb size={18} className="text-blue-600" aria-hidden />} />
      <ul className="m-0 grid gap-2 py-4 pl-10 pr-5">{items.map((text, i) => <li key={i}>{text}</li>)}</ul>
    </Card>
  );
}

/** Checks that gave no evidence. They are never counted as passes. */
export function EvidenceGaps({ checks }: { checks: CheckResult[] }) {
  const gaps = checks.filter((c) => NO_EVIDENCE.includes(c.status));
  const na = checks.filter((c) => c.status === "NOT_APPLICABLE");
  // Many checks usually fail for one reason (for example the model did not load): list the reason once.
  const groups = new Map<string, { status: CheckResult["status"]; message: string; ids: string[] }>();
  for (const c of [...gaps, ...na]) {
    const key = `${c.status}|${c.message}`;
    groups.set(key, { status: c.status, message: c.message, ids: [...(groups.get(key)?.ids ?? []), c.id] });
  }
  return (
    <Card>
      <CardHeader title="Evidence gaps" subtitle="Checks that could not run. A check without evidence is never counted as a pass."
        actions={<span className="rounded-full bg-blue-50 px-2 text-xs font-bold text-blue-800">{gaps.length}</span>} />
      {gaps.length === 0 && na.length === 0 ? (
        <p className="px-5 py-4 text-slate-500">No gaps: every check had evidence.</p>
      ) : (
        <TableWrap>
          <Table>
            <thead><tr><Th>Checks</Th><Th>Status</Th><Th>Reason</Th></tr></thead>
            <tbody>
              {[...groups.values()].map((g) => (
                <tr key={`${g.status}|${g.message}`}>
                  <Td className="font-semibold">{g.ids.length === 1 ? g.ids[0] : (
                    <details><summary>{g.ids.length} checks</summary>
                      <ul className="mt-1.5 list-disc pl-4.5 font-normal">{g.ids.map((id) => <li key={id}>{id}</li>)}</ul></details>)}</Td>
                  <Td><StatusBadge status={g.status} /></Td><Td>{g.message}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
        </TableWrap>
      )}
    </Card>
  );
}
