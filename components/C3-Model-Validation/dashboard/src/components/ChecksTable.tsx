import { ChevronDown, ChevronRight, Search } from "lucide-react";
import { Fragment, useMemo, useState } from "react";

import type { CheckContribution, CheckInfo, CheckResult, CheckStatus, Dimension } from "../api/types";
import { DIMENSIONS, NO_EVIDENCE } from "../api/types";
import { DIMENSION_LABEL, DIMENSION_SHORT, EVIDENCE_HELP, fmtValue, STATUS_LABEL } from "../lib/format";
import { EvidenceBadge, StatusBadge } from "./badges";
import { Card, cx, EmptyState, Table, TableWrap, Td, Th } from "./ui";

// Problems first: what needs a person's attention should not be buried under passes.
const ATTENTION: Record<CheckStatus, number> = { FAIL: 0, ERROR: 1, WARN: 2, UNAVAILABLE: 3, PASS: 4, NOT_APPLICABLE: 5 };
type StatusFilter = "all" | "attention" | CheckStatus;

const chip = (on: boolean) => cx(
  "rounded-full border px-3 py-1 text-[13px] font-semibold",
  on ? "border-blue-600 bg-blue-600 text-white" : "border-slate-300 bg-white text-slate-700 hover:border-blue-400");

export function ChecksTable({ checks, contributions, catalogue }: {
  checks: CheckResult[];
  contributions: Map<string, CheckContribution>;
  catalogue: Map<string, CheckInfo>;
}) {
  const [layer, setLayer] = useState<Dimension | "all">("all");
  const [status, setStatus] = useState<StatusFilter>("all");
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState<Set<string>>(new Set());

  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return checks
      .map((check, index) => ({ check, index }))
      .filter(({ check }) => layer === "all" || check.dimension === layer)
      .filter(({ check }) =>
        status === "all" ? true
        : status === "attention" ? check.status !== "PASS" && check.status !== "NOT_APPLICABLE"
        : check.status === status)
      .filter(({ check }) => !needle || check.id.toLowerCase().includes(needle) || check.message.toLowerCase().includes(needle))
      .sort((a, b) => ATTENTION[a.check.status] - ATTENTION[b.check.status] || a.index - b.index)
      .map(({ check }) => check);
  }, [checks, layer, status, query]);

  const toggle = (id: string) =>
    setOpen((current) => {
      const next = new Set(current);
      if (!next.delete(id)) next.add(id);
      return next;
    });

  const statusChips: [StatusFilter, string][] = [
    ["all", "All"], ["attention", "Needs attention"], ["PASS", "Pass"], ["WARN", "Warning"], ["FAIL", "Fail"], ["UNAVAILABLE", "No evidence"],
  ];

  return (
    <Card>
      <div className="flex flex-wrap items-center gap-3 border-b border-slate-200 px-5 py-3.5">
        <label className="relative min-w-50 max-w-xs flex-1">
          <span className="sr-only">Search checks</span>
          <Search size={15} aria-hidden className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input type="search" placeholder="Search checks" value={query} onChange={(e) => setQuery(e.target.value)}
            className="h-9 w-full rounded-lg border border-slate-300 bg-white pl-9 pr-3 placeholder:text-slate-400 hover:border-blue-400" />
        </label>
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by layer">
          <button className={chip(layer === "all")} aria-pressed={layer === "all"} onClick={() => setLayer("all")}>All layers</button>
          {DIMENSIONS.map((d) => (
            <button key={d} className={chip(layer === d)} aria-pressed={layer === d} onClick={() => setLayer(d)}>{DIMENSION_SHORT[d]}</button>
          ))}
        </div>
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by status">
          {statusChips.map(([value, label]) => (
            <button key={value} className={chip(status === value)} aria-pressed={status === value} onClick={() => setStatus(value)}>{label}</button>
          ))}
        </div>
      </div>

      {rows.length === 0 ? (
        <EmptyState title="No checks match these filters" />
      ) : (
        <TableWrap>
          <Table caption="Validation checks, problems first">
            <thead>
              <tr>
                <Th className="w-8"><span className="sr-only">Details</span></Th>
                <Th>Check</Th><Th>Layer</Th><Th>Status</Th><Th right>Value</Th><Th>Score</Th><Th right>Weight</Th><Th>Evidence</Th>
              </tr>
            </thead>
            <tbody>
              {rows.map((check) => {
                const expanded = open.has(check.id);
                const contribution = contributions.get(check.id);
                const description = catalogue.get(check.id)?.description ?? check.name;
                const Chevron = expanded ? ChevronDown : ChevronRight;
                const hasScore = check.score !== null && check.score !== undefined;
                return (
                  <Fragment key={check.id}>
                    <tr className="hover:bg-slate-50/70">
                      <Td>
                        <button className="text-blue-700" aria-expanded={expanded}
                          aria-label={`${expanded ? "Hide" : "Show"} details for ${check.id}`} onClick={() => toggle(check.id)}>
                          <Chevron size={16} aria-hidden />
                        </button>
                      </Td>
                      <Td>
                        <div className="font-semibold">{check.id}</div>
                        <div className="max-w-md text-[12.5px] text-slate-500">{description}</div>
                      </Td>
                      <Td className="whitespace-nowrap">{DIMENSION_SHORT[check.dimension]}</Td>
                      <Td><StatusBadge status={check.status} /></Td>
                      <Td right>{fmtValue(check, catalogue)}</Td>
                      <Td>
                        {hasScore ? (
                          <div className="grid grid-cols-[34px_90px] items-center gap-2 tabular-nums">
                            <span>{(check.score as number).toFixed(0)}</span>
                            <div className="h-2 overflow-hidden rounded-full border border-blue-100 bg-blue-50">
                              <span className={cx("block h-full rounded-full", check.status === "FAIL" ? "bg-red-600" : check.status === "WARN" ? "bg-amber-500" : "bg-blue-600")}
                                style={{ width: `${check.score}%` }} />
                            </div>
                          </div>
                        ) : <span className="text-slate-500">no score</span>}
                      </Td>
                      <Td right>{contribution ? contribution.weight : "—"}</Td>
                      <Td><EvidenceBadge source={check.evidence_source} /></Td>
                    </tr>
                    {expanded && (
                      <tr className="detail-row bg-slate-50/60">
                        <Td />
                        <Td colSpan={7}><CheckDetail check={check} contribution={contribution} /></Td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </Table>
        </TableWrap>
      )}
    </Card>
  );
}

function CheckDetail({ check, contribution }: { check: CheckResult; contribution?: CheckContribution }) {
  const t = check.thresholds ?? {};
  const hasThresholds = Object.keys(t).length > 0;
  return (
    <div className="grid gap-2.5">
      <p>{check.message}</p>
      <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-[13px]">
        <dt className="text-slate-500">Layer</dt><dd>{DIMENSION_LABEL[check.dimension]}</dd>
        <dt className="text-slate-500">Evidence</dt><dd>{EVIDENCE_HELP[check.evidence_source]}</dd>
        {check.required && (<><dt className="text-slate-500">Required</dt><dd>Missing evidence for this check blocks promotion</dd></>)}
        {hasThresholds && (
          <><dt className="text-slate-500">Thresholds</dt>
            <dd className="font-mono text-[12.5px]">target {t.target} · pass at {t.pass_at} · fail at {t.fail_at} · floor {t.floor}</dd></>
        )}
        {contribution && (
          <><dt className="text-slate-500">In the score</dt>
            <dd>{contribution.treatment.replaceAll("_", " ")}; contributes {contribution.points.toFixed(2)} points to its layer</dd></>
        )}
        {NO_EVIDENCE.includes(check.status) && (<><dt className="text-slate-500">Note</dt><dd>Counted as missing evidence, never as a pass.</dd></>)}
      </dl>
      <p className="text-xs text-slate-500">Status {STATUS_LABEL[check.status].toLowerCase()} is set by the thresholds above; they are configuration, not standards.</p>
    </div>
  );
}
