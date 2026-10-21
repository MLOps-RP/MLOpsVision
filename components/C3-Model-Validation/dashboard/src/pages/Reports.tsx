import { Download, FileText, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { DIMENSIONS, type Dimension } from "../api/types";
import { Gate } from "../components/Gate";
import { ReportView } from "../components/ReportView";
import {
  Badge, Button, Card, CardHeader, CardSkeleton, ConnectNotice, cx, EmptyState, ErrorNotice, inputClass, Mono, PageHeader,
  TabPanel, Table, TableWrap, Tabs, Td, Th, Unavailable,
} from "../components/ui";
import { DecisionBadge } from "../components/badges";
import { DIMENSION_LABEL, DIMENSION_SHORT, downloadText, fmtDate, fmtScore } from "../lib/format";
import { nameReport, useModelNames } from "../lib/modelNames";
import { useRuns } from "../lib/runs";
import { useAsync } from "../lib/useAsync";
import { useConnection } from "../state/connection";

type Tab = "runs" | "catalogue";
const isTab = (v: string | null): v is Tab => v === "runs" || v === "catalogue";

export function Reports() {
  const [params, setParams] = useSearchParams();
  const tab: Tab = isTab(params.get("tab")) ? (params.get("tab") as Tab) : "runs";
  const go = (t: Tab) => { const next = new URLSearchParams(params); next.set("tab", t); setParams(next); };
  return (
    <>
      <PageHeader title="Reports & Check Catalogue"
        subtitle="Read and download the reports the gate wrote, and browse every check it can run." />
      <ConnectNotice />
      <Tabs label="Reports" value={tab} onChange={go}
        tabs={[{ id: "runs", label: "Validation reports" }, { id: "catalogue", label: "Check catalogue" }]} />
      <TabPanel id={tab}>
        {tab === "runs" && <RunReports />}
        {tab === "catalogue" && <Catalogue />}
      </TabPanel>
    </>
  );
}

/* ------------------------------------------------------------ run reports */
function RunReports() {
  const { client } = useConnection();
  const { runs, loading, error, reload } = useRuns();
  const { nameOf } = useModelNames();
  const [params, setParams] = useSearchParams();
  const chosen = params.get("run") ?? runs[0]?.run_id;
  const run = runs.find((r) => r.run_id === chosen);
  const { status } = useConnection();
  const report = useAsync(() => (status === "connected" && chosen ? client.report(chosen) : Promise.resolve(null)), [client, status, chosen]);
  const [busy, setBusy] = useState(false);

  const downloadJson = async () => {
    if (!chosen) return;
    setBusy(true);
    try {
      downloadText(`${run ? nameOf(run.candidate_id, run.run_id) : chosen.slice(0, 8)}-${chosen.slice(0, 8)}.json`, JSON.stringify(await client.run(chosen), null, 2), "application/json");
    } finally { setBusy(false); }
  };

  return (
    <Gate loading={loading} error={error} onRetry={reload} rows={4}>
      {runs.length === 0 ? (
        <Card><EmptyState icon={FileText} title="No validation reports yet">Each validation writes a Markdown report and a JSON result; they appear here.</EmptyState></Card>
      ) : (
        <>
          <Card className="flex flex-wrap items-center gap-3 px-5 py-3.5">
            <label htmlFor="report-run" className="text-[13px] font-semibold text-slate-700">Validation</label>
            <select id="report-run" className={cx(inputClass, "max-w-md")} value={chosen ?? ""} onChange={(e) => { const n = new URLSearchParams(params); n.set("run", e.target.value); setParams(n); }}>
              {runs.map((r) => <option key={r.run_id} value={r.run_id}>{nameOf(r.candidate_id, r.run_id)} · {r.profile} · {fmtDate(r.created_at_utc)}</option>)}
            </select>
            {run && <><DecisionBadge decision={run.decision} /><span className="text-[13px] text-slate-500">MQS {fmtScore(run.mqs)}</span></>}
            <Button small className="ml-auto" onClick={downloadJson} disabled={busy}><Download size={14} aria-hidden />Download JSON</Button>
            {chosen && <Link to={`/results/${chosen}`} className="text-[13px] font-semibold text-blue-700 hover:underline">Open full result</Link>}
          </Card>
          {report.error ? <ErrorNotice error={report.error} onRetry={report.reload} />
          : report.loading || report.data === null ? <CardSkeleton rows={8} />
          : <ReportView markdown={run ? nameReport(report.data, run.candidate_id, nameOf(run.candidate_id, run.run_id)) : report.data} filename={`${run ? nameOf(run.candidate_id, run.run_id) : "validation"}-${(chosen ?? "").slice(0, 8)}-report.md`}
              pdf={{ name: run ? nameOf(run.candidate_id, run.run_id) : "Validation", runId: chosen ?? "", createdAt: run?.created_at_utc ?? null }} />}
        </>
      )}
    </Gate>
  );
}

/* ---------------------------------------------------------- study reports */
/** The study-reports section. Hidden from the page (no tab), kept for when it is shown again. */
export function StudyReports() {
  const { client, status } = useConnection();
  const [params, setParams] = useSearchParams();
  const studies = useAsync(() => (status === "connected" ? client.studies() : Promise.resolve(null)), [client, status]);
  const list = studies.data?.studies ?? [];
  const chosen = params.get("study") ?? list.find((s) => s.has_report)?.id;
  const info = list.find((s) => s.id === chosen);
  const report = useAsync(() => (status === "connected" && chosen && info?.has_report ? client.studyReport(chosen) : Promise.resolve(null)), [client, status, chosen, info?.has_report]);

  return (
    <Gate loading={studies.loading} error={studies.error} onRetry={studies.reload} rows={4}>
      <div className="grid gap-4 md:grid-cols-3">
        {list.map((s) => (
          <button key={s.id} onClick={() => { const n = new URLSearchParams(params); n.set("study", s.id); setParams(n); }} aria-pressed={chosen === s.id}
            className={cx("grid gap-1.5 rounded-xl border bg-white p-4 text-left shadow-sm hover:border-blue-400", chosen === s.id ? "border-blue-600 ring-2 ring-blue-100" : "border-slate-200")}>
            <span className="font-semibold">{s.title}</span>
            <span className="text-[13px] text-slate-500">{s.description}</span>
            <span className="mt-1 flex flex-wrap items-center gap-2 text-xs">
              {s.has_report ? <Badge tone="green">Report available</Badge> : <Badge tone="slate">Not generated</Badge>}
              {s.has_data && <Badge tone="blue">Chartable data</Badge>}
              {s.generated_at_utc && <span className="text-slate-500">{fmtDate(s.generated_at_utc)}</span>}
            </span>
          </button>
        ))}
      </div>
      {info && !info.has_report ? (
        <Unavailable title={`${info.title} has not been generated`} hint={<>Generate it with <Mono>{info.script}</Mono>. The API only serves study reports; it never runs the scripts.</>}>
          There is no report file to show.
        </Unavailable>
      ) : report.error ? <ErrorNotice error={report.error} onRetry={report.reload} />
      : report.loading || report.data === null ? (chosen ? <CardSkeleton rows={8} /> : null)
      : <ReportView markdown={report.data} filename={`${chosen}.md`} title={info?.title ?? "Study report"} subtitle={`Written by ${info?.script ?? "a helper script"}.`} />}
    </Gate>
  );
}

/* -------------------------------------------------------------- catalogue */
function Catalogue() {
  const { client, status } = useConnection();
  const checks = useAsync(() => (status === "connected" ? client.checks() : Promise.resolve(null)), [client, status]);
  const profiles = useAsync(() => (status === "connected" ? client.profiles() : Promise.resolve(null)), [client, status]);
  const [layer, setLayer] = useState<Dimension | "all">("all");
  const [query, setQuery] = useState("");

  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return (checks.data ?? []).filter((c) => (layer === "all" || c.dimension === layer)
      && (!needle || c.id.toLowerCase().includes(needle) || c.description.toLowerCase().includes(needle)));
  }, [checks.data, layer, query]);
  const chip = (on: boolean) => cx("rounded-full border px-3 py-1 text-[13px] font-semibold", on ? "border-blue-600 bg-blue-600 text-white" : "border-slate-300 bg-white text-slate-700 hover:border-blue-400");

  return (
    <Gate loading={checks.loading || profiles.loading} error={checks.error ?? profiles.error} onRetry={() => { checks.reload(); profiles.reload(); }} rows={8}>
      <Card>
        <CardHeader title="Profiles" subtitle="A profile chooses which checks run and how they are weighted and judged." />
        <TableWrap>
          <Table>
            <thead><tr><Th>Profile</Th><Th>Description</Th><Th right>Checks</Th><Th>Scans model file</Th><Th>Runs on this server</Th>{DIMENSIONS.map((d) => <Th key={d} right>{DIMENSION_SHORT[d]} weight</Th>)}</tr></thead>
            <tbody>
              {profiles.data?.profiles.map((p) => (
                <tr key={p.name}>
                  <Td className="font-semibold">{p.name}{p.name === profiles.data?.default_profile && <Badge tone="blue" className="ml-2">default</Badge>}</Td>
                  <Td className="max-w-md text-[13px] text-slate-500">{p.description ?? "—"}</Td>
                  <Td right>{p.enabled_checks}</Td>
                  <Td><Badge tone={p.scans_model_files ? "green" : "slate"}>{p.scans_model_files ? "Yes" : "No"}</Badge></Td>
                  <Td><Badge tone={p.allowed ? "blue" : "amber"}>{p.allowed ? "Allowed" : "Not allowed"}</Badge></Td>
                  {DIMENSIONS.map((d) => <Td key={d} right>{Math.round((p.weights[d] ?? 0) * 100)}%</Td>)}
                </tr>
              ))}
            </tbody>
          </Table>
        </TableWrap>
        <p className="border-t border-slate-200 px-5 py-3 text-xs text-slate-500">
          Which individual checks each profile enables, and their thresholds, are not exposed by the API (they are in the profile files under <Mono>configs/</Mono>).
          A run's own thresholds are shown on its Checks tab.
        </p>
      </Card>

      <Card>
        <CardHeader title="Check catalogue" subtitle={`${rows.length} of ${checks.data?.length ?? 0} checks the gate knows about.`} />
        <div className="flex flex-wrap items-center gap-3 border-b border-slate-200 px-5 py-3.5">
          <label className="relative min-w-50 max-w-xs flex-1">
            <span className="sr-only">Search the catalogue</span>
            <Search size={15} aria-hidden className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input type="search" placeholder="Search checks" value={query} onChange={(e) => setQuery(e.target.value)} className={cx(inputClass, "pl-9")} />
          </label>
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by layer">
            <button className={chip(layer === "all")} aria-pressed={layer === "all"} onClick={() => setLayer("all")}>All layers</button>
            {DIMENSIONS.map((d) => <button key={d} className={chip(layer === d)} aria-pressed={layer === d} onClick={() => setLayer(d)}>{DIMENSION_LABEL[d]}</button>)}
          </div>
        </div>
        {rows.length === 0 ? <EmptyState title="No checks match" /> : (
          <TableWrap>
            <Table caption="Check catalogue">
              <thead><tr><Th>Check</Th><Th>Layer</Th><Th>What it measures</Th><Th>Unit</Th><Th>Better</Th></tr></thead>
              <tbody>
                {rows.map((c) => (
                  <tr key={c.id}>
                    <Td className="font-semibold">{c.id}</Td>
                    <Td className="whitespace-nowrap">{DIMENSION_SHORT[c.dimension as Dimension] ?? c.dimension}</Td>
                    <Td>{c.description}</Td>
                    <Td className="whitespace-nowrap">{c.unit}</Td>
                    <Td className="whitespace-nowrap">{c.direction === "binary" ? "pass / fail" : c.direction === "higher_is_better" ? "Higher" : "Lower"}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </TableWrap>
        )}
      </Card>
    </Gate>
  );
}
