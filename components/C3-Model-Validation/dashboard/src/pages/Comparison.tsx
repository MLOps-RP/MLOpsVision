import { FileText } from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";

import { ApiError } from "../api/client";
import { parseBaseline, parseSensitivity, type BaselineStudy, type SensitivityStudy } from "../api/studies";
import { DIMENSIONS, type CheckStatus, type ValidationResult } from "../api/types";
import { DecisionBadge, StatusBadge } from "../components/badges";
import { BaselineChart, RunCompareChart, SensitivityChart, type CompareRow } from "../components/charts";
import { Gate } from "../components/Gate";
import {
  Badge, buttonClass, Card, CardBody, CardHeader, CardSkeleton, ConnectNotice, cx, EmptyState, ErrorNotice, Mono, PageHeader,
  TabPanel, Table, TableWrap, Tabs, Td, Th, Unavailable,
} from "../components/ui";
import { DIMENSION_SHORT, fmtDate, fmtScore } from "../lib/format";
import { useModelNames } from "../lib/modelNames";
import { useRuns } from "../lib/runs";
import { useAsync } from "../lib/useAsync";
import { useConnection } from "../state/connection";

type Tab = "runs" | "baseline" | "sensitivity";

export function Comparison() {
  const [tab, setTab] = useState<Tab>("runs");
  return (
    <>
      <PageHeader title="Model Comparison"
        subtitle="Compare validated models, see whether a plain metrics gate would have caught the same defects, and how stable each recommendation is." />
      <ConnectNotice />
      <Tabs label="Comparison" value={tab} onChange={setTab}
        tabs={[{ id: "runs", label: "Compare runs" }, { id: "baseline", label: "Baseline gate" }, { id: "sensitivity", label: "Sensitivity analysis" }]} />
      <TabPanel id={tab}>
        {tab === "runs" && <CompareRuns />}
        {tab === "baseline" && <BaselinePanel />}
        {tab === "sensitivity" && <SensitivityPanel />}
      </TabPanel>
    </>
  );
}

/* ------------------------------------------------------------ run vs run */
const MAX_RUNS = 4;

function CompareRuns() {
  const { client } = useConnection();
  const { runs, loading, error, reload } = useRuns();
  const { nameOf } = useModelNames();
  const [selected, setSelected] = useState<string[] | null>(null);
  const [results, setResults] = useState<Record<string, ValidationResult | ApiError>>({});

  // Start with the two newest runs so the page shows something real; the user can change it.
  const chosen = selected ?? runs.slice(0, 2).map((r) => r.run_id);
  useEffect(() => {
    for (const id of chosen) {
      if (results[id]) continue;
      client.run(id).then(
        (r) => setResults((prev) => ({ ...prev, [id]: r })),
        (e: unknown) => setResults((prev) => ({ ...prev, [id]: e instanceof ApiError ? e : new ApiError("server", 0, String(e)) })));
    }
  }, [chosen.join(","), client]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = (id: string) => setSelected((cur) => {
    const base = cur ?? chosen;
    return base.includes(id) ? base.filter((x) => x !== id) : base.length >= MAX_RUNS ? base : [...base, id];
  });

  const loaded = chosen.map((id) => ({ id, value: results[id] }));
  const ready = loaded.flatMap(({ id, value }) => (value && !(value instanceof ApiError) ? [{ id, result: value }] : []));
  const rows: CompareRow[] = ready.map(({ result }) => {
    const c = result.scorecard;
    const usable = c.mqs_sufficient_evidence !== false && c.mqs !== null && c.mqs !== undefined;
    return {
      label: nameOf(result.candidate.candidate_id, result.run_id),
      tvs: c.dimensions.training_validity.score ?? null, mbs: c.dimensions.model_behaviour.score ?? null,
      ors: c.dimensions.operational_readiness.score ?? null, mqs: usable ? (c.mqs ?? null) : null,
    };
  });

  // Checks whose status is not the same in every selected run (a check a run did not run counts as different).
  const differing = useMemo(() => {
    const ids = new Set(ready.flatMap(({ result }) => result.checks.map((c) => c.id)));
    return [...ids].map((id) => ({ id, statuses: ready.map(({ result }) => result.checks.find((c) => c.id === id)?.status ?? null) }))
      .filter(({ statuses }) => new Set(statuses).size > 1);
  }, [ready]);

  return (
    <Gate loading={loading} error={error} onRetry={reload} rows={5}>
      {runs.length < 2 ? (
        <Card><EmptyState title="Validate at least two models to compare them"
          action={<Link to="/validate" className={buttonClass("primary")}>Validate a model</Link>}>The comparison uses stored validation results.</EmptyState></Card>
      ) : (
        <>
          <Card>
            <CardHeader title="Choose runs" subtitle={`Select up to ${MAX_RUNS} stored validations (${chosen.length} selected).`} />
            <ul className="m-0 grid max-h-64 list-none gap-0 overflow-y-auto p-0 sm:grid-cols-2">
              {runs.map((r) => {
                const on = chosen.includes(r.run_id);
                return (
                  <li key={r.run_id} className="border-b border-slate-100">
                    <label className={cx("flex cursor-pointer items-center gap-3 px-5 py-2.5 hover:bg-slate-50", !on && chosen.length >= MAX_RUNS && "opacity-50")}>
                      <input type="checkbox" checked={on} disabled={!on && chosen.length >= MAX_RUNS} onChange={() => toggle(r.run_id)} className="size-4 accent-blue-600" />
                      <span className="min-w-0 flex-1 truncate font-semibold">{nameOf(r.candidate_id, r.run_id)}</span>
                      <DecisionBadge decision={r.decision} />
                      <span className="w-12 text-right tabular-nums text-slate-500">{fmtScore(r.mqs)}</span>
                    </label>
                  </li>
                );
              })}
            </ul>
          </Card>

          {chosen.length < 2 ? (
            <Unavailable title="Select at least two runs">Pick two or more validations above to compare them.</Unavailable>
          ) : (
            <>
              {loaded.some((l) => l.value instanceof ApiError) && <ErrorNotice error={loaded.find((l) => l.value instanceof ApiError)!.value as ApiError} />}
              {ready.length < chosen.length && !loaded.some((l) => l.value instanceof ApiError) ? <CardSkeleton rows={6} /> : (
                <>
                  <RunCompareChart rows={rows} />
                  <Card>
                    <CardHeader title="Side by side" />
                    <TableWrap>
                      <Table>
                        <thead><tr><Th>Model</Th><Th>Decision</Th><Th right>MQS</Th>{DIMENSIONS.map((d) => <Th key={d} right>{DIMENSION_SHORT[d]}</Th>)}<Th>Profile</Th><Th>Run at</Th></tr></thead>
                        <tbody>
                          {ready.map(({ id, result }, i) => (
                            <tr key={id}>
                              <Td className="font-semibold"><Link className="text-blue-700 hover:underline" to={`/results/${id}`}>{rows[i].label}</Link></Td>
                              <Td><DecisionBadge decision={result.decision.decision} /></Td>
                              <Td right>{fmtScore(rows[i].mqs)}</Td>
                              <Td right>{fmtScore(rows[i].tvs)}</Td><Td right>{fmtScore(rows[i].mbs)}</Td><Td right>{fmtScore(rows[i].ors)}</Td>
                              <Td>{typeof result.config_snapshot.profile === "string" ? result.config_snapshot.profile : "—"}</Td>
                              <Td className="whitespace-nowrap text-xs text-slate-500">{fmtDate(result.created_at_utc)}</Td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </TableWrap>
                    <p className="border-t border-slate-200 px-5 py-3 text-xs text-slate-500">
                      An MQS shown as n/a means that run had too little evidence for a quotable score. Runs under different profiles check different things, so compare scores only when the profiles match.
                    </p>
                  </Card>
                  <Card>
                    <CardHeader title="Checks that differ" subtitle="Only checks whose status is not the same in every selected run. A dash means that run did not run the check." />
                    {differing.length === 0 ? <p className="px-5 py-4 text-slate-500">Every check has the same status in all selected runs.</p> : (
                      <TableWrap>
                        <Table>
                          <thead><tr><Th>Check</Th>{ready.map((_, i) => <Th key={i}>{rows[i].label}</Th>)}</tr></thead>
                          <tbody>{differing.map(({ id, statuses }) => (
                            <tr key={id}><Td className="font-semibold">{id}</Td>
                              {statuses.map((s: CheckStatus | null, i) => <Td key={i}>{s ? <StatusBadge status={s} /> : <span className="text-slate-400" aria-label="not run">—</span>}</Td>)}</tr>))}
                          </tbody>
                        </Table>
                      </TableWrap>
                    )}
                  </Card>
                </>
              )}
            </>
          )}
        </>
      )}
    </Gate>
  );
}

/* --------------------------------------------------------------- studies */
function useStudy<T>(id: string, parse: (data: Record<string, unknown>) => T | null) {
  const { client, status } = useConnection();
  const state = useAsync(() => (status === "connected" ? client.study(id) : Promise.resolve(null)), [client, status, id]);
  const parsed = useMemo(() => (state.data ? parse(state.data.data) : null), [state.data]); // eslint-disable-line react-hooks/exhaustive-deps
  return { ...state, parsed, generated: state.data?.generated_at_utc ?? null };
}

function StudyShell({ id, title, loading, error, onRetry, hasData, parsed, generated, children }: {
  id: string; title: string; loading: boolean; error: ApiError | null; onRetry: () => void; hasData: boolean; parsed: unknown; generated: string | null; children: ReactNode;
}) {
  return (
    <Gate loading={loading} error={error && error.kind !== "notfound" ? error : null} onRetry={onRetry} rows={5}>
      {error?.kind === "notfound" ? (
        <Unavailable title={`${title} is not available`} hint={<>Generated by a helper script, not by the API. <Link className="font-semibold text-blue-700 hover:underline" to="/reports?tab=studies">See the study reports</Link>.</>}>
          {error.message}
        </Unavailable>
      ) : hasData && parsed === null ? (
        <Unavailable title={`${title} is in an unexpected format`}>The file was found but does not have the structure this page understands, so nothing is charted. Regenerate it with the helper script.</Unavailable>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-2 text-[13px] text-slate-500">
            <Badge tone="blue">Real study results</Badge>
            {generated && <span>Generated {fmtDate(generated)}</span>}
            <Link to={`/reports?tab=studies&study=${id}`} className={cx(buttonClass("ghost", true), "ml-auto")}><FileText size={14} aria-hidden />Read the full report</Link>
          </div>
          {children}
        </>
      )}
    </Gate>
  );
}

const Caveats = ({ items }: { items: string[] }) => (
  <Card><CardHeader title="Read this before quoting these numbers" /><ul className="m-0 grid gap-1.5 py-4 pl-10 pr-5 text-[13px] text-slate-600">{items.map((c, i) => <li key={i}>{c}</li>)}</ul></Card>
);

function BaselinePanel() {
  const s = useStudy<BaselineStudy>("baseline_comparison", parseBaseline);
  const { nameFor } = useModelNames();
  const study = useMemo(() => (s.parsed ? { ...s.parsed, models: s.parsed.models.map((m) => ({ ...m, name: nameFor(m.name) })) } : s.parsed), [s.parsed, nameFor]);
  return (
    <StudyShell id="baseline_comparison" title="The baseline comparison" loading={s.loading} error={s.error} onRetry={s.reload} hasData={Boolean(s.data)} parsed={study} generated={s.generated}>
      {study && (
        <>
          <p className="max-w-4xl text-slate-600">
            The same sample models were judged by a plain metrics gate (F1 only; F1, precision, recall and ROC-AUC) and by the full gate. A model counts as a
            <em> defect</em> if a defect was injected into it.
          </p>
          <BaselineChart study={study} />
          <Card>
            <CardHeader title="Per model" subtitle="What each gate decided for every sample model." />
            <TableWrap>
              <Table>
                <thead><tr><Th>Model</Th><Th>Truth</Th><Th right>F1</Th><Th>Metrics not passing</Th>{study.gates.map((g) => <Th key={g}>{g}</Th>)}</tr></thead>
                <tbody>
                  {study.models.map((m) => (
                    <tr key={m.name}>
                      <Td className="font-semibold">{m.name}</Td>
                      <Td><Badge tone={m.truth === "defect" ? "red" : m.truth === "clean" ? "green" : "slate"}>{m.truth}</Badge></Td>
                      <Td right>{m.f1 === null ? "n/a" : m.f1.toFixed(3)}</Td>
                      <Td className="text-[13px] text-slate-500">{m.metrics_not_passing.length ? m.metrics_not_passing.join(", ") : "none"}</Td>
                      {study.gates.map((g) => {
                        const d = m.decisions[g];
                        return <Td key={g}>{d === "PROMOTE" || d === "HOLD" || d === "ROLLBACK" ? <DecisionBadge decision={d} /> : <Badge tone={m.promoted[g] ? "green" : "amber"}>{m.promoted[g] ? "promote" : "stop"}</Badge>}</Td>;
                      })}
                    </tr>
                  ))}
                </tbody>
              </Table>
            </TableWrap>
          </Card>
          <Caveats items={study.caveats} />
        </>
      )}
    </StudyShell>
  );
}

const heat = (kept: number, total: number) => {
  const share = total ? kept / total : 0;
  return cx("text-center tabular-nums", share === 1 ? "bg-green-50 text-green-800" : share >= 0.75 ? "bg-amber-50 text-amber-800" : "bg-red-50 text-red-800");
};

function SensitivityPanel() {
  const s = useStudy<SensitivityStudy>("sensitivity_analysis", parseSensitivity);
  const { nameFor } = useModelNames();
  const study = useMemo(() => (s.parsed ? { ...s.parsed, cases: s.parsed.cases.map((c) => ({ ...c, model: nameFor(c.model) })) } : s.parsed), [s.parsed, nameFor]);
  return (
    <StudyShell id="sensitivity_analysis" title="The sensitivity analysis" loading={s.loading} error={s.error} onRetry={s.reload} hasData={Boolean(s.data)} parsed={study} generated={s.generated}>
      {study && (
        <>
          <p className="max-w-4xl text-slate-600">{study.design}.</p>
          <SensitivityChart cases={study.cases} />
          <Card>
            <CardHeader title="Stability by family of changes" subtitle="Per family of changes. 100% means nothing in that family flipped the recommendation." />
            <TableWrap>
              <Table>
                <thead><tr><Th>Model</Th><Th>Baseline</Th><Th right>MQS range</Th>{study.families.map((f) => <Th key={f} className="text-center">{f}</Th>)}</tr></thead>
                <tbody>
                  {study.cases.map((c) => (
                    <tr key={c.model}>
                      <Td className="font-semibold">{c.model}</Td>
                      <Td>{c.baseline_decision === "PROMOTE" || c.baseline_decision === "HOLD" || c.baseline_decision === "ROLLBACK" ? <DecisionBadge decision={c.baseline_decision} /> : c.baseline_decision}</Td>
                      <Td right>{c.mqs_range ? `${c.mqs_range[0].toFixed(1)} – ${c.mqs_range[1].toFixed(1)}` : "n/a"}</Td>
                      {study.families.map((f) => {
                        const r = c.families[f];
                        return <Td key={f} className={heat(r.kept, r.total)}>{r.total ? `${Math.round((r.kept / r.total) * 100)}%` : "n/a"} <span className="text-[11px] opacity-70">({r.kept}/{r.total})</span></Td>;
                      })}
                    </tr>
                  ))}
                </tbody>
              </Table>
            </TableWrap>
          </Card>
          <Card>
            <CardHeader title="Perturbations that change a decision" />
            <CardBody className="grid gap-2">
              {study.cases.every((c) => c.flips.length === 0) ? <p className="text-slate-500">No perturbation changed any decision.</p>
                : study.cases.filter((c) => c.flips.length > 0).map((c) => (
                  <details key={c.model} className="rounded-lg border border-slate-200 px-3.5 py-2.5">
                    <summary className="font-semibold">{c.model} <span className="font-normal text-slate-500">({c.baseline_decision}): {c.flips.length} flip{c.flips.length === 1 ? "" : "s"}</span></summary>
                    <ul className="mt-2 list-disc pl-5 text-[13px]">{c.flips.map((f, i) => <li key={i}><Mono>{f.family}</Mono>: {f.label} → <strong>{f.decision}</strong></li>)}</ul>
                  </details>))}
            </CardBody>
          </Card>
          <Caveats items={study.caveats} />
        </>
      )}
    </StudyShell>
  );
}
