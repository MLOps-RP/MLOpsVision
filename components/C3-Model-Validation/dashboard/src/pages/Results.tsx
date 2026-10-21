import { ClipboardCheck, Download, ScanSearch } from "lucide-react";
import { Fragment, useEffect, useMemo, useState } from "react";
import { Link, Navigate, useParams } from "react-router-dom";

import { DIMENSIONS, type CheckContribution, type CheckInfo, type CheckResult, type ValidationResult } from "../api/types";
import { EvidenceBadge, StatusBadge } from "../components/badges";
import { CheckScoreBars, LayerBars, StatusDonut } from "../components/charts";
import { ChecksTable } from "../components/ChecksTable";
import { DecisionBanner } from "../components/DecisionBanner";
import { Gate } from "../components/Gate";
import { HIDDEN_CHECK } from "../components/HiddenGroups";
import { LayerCard } from "../components/LayerCard";
import { ReportView } from "../components/ReportView";
import { EvidenceGaps, Explanations, RulesPanel } from "../components/RulesPanel";
import { ScoreGauge } from "../components/ScoreGauge";
import {
  Button, buttonClass, Card, CardBody, CardHeader, CardSkeleton, ConnectNotice, EmptyState, ErrorNotice, Mono, Notice, PageHeader,
  TabPanel, Tabs, Table, TableWrap, Td, Th,
} from "../components/ui";
import { useCatalogue } from "../lib/catalogue";
import {
  countBy, downloadText, fmtDate, fmtPercent, fmtValue, mqsCalculation, RECOMMENDATION_NOTICE, shortHash, shortPath, snapshotNumber,
} from "../lib/format";
import { nameReport, useModelNames } from "../lib/modelNames";
import { useRun, useRuns } from "../lib/runs";
import { useAsync } from "../lib/useAsync";
import { useConnection } from "../state/connection";

type Tab = "overview" | "checks" | "rules" | "evidence" | "report" | "raw";

/** /results with no id: open the latest validation, or explain that there is none. */
export function ResultsIndex() {
  const { runs, loading, error, reload } = useRuns();
  return (
    <>
      <PageHeader title="Validation Results" subtitle="The check statuses, metrics, rules, evidence and explanations behind a recommendation." />
      <ConnectNotice />
      <Gate loading={loading} error={error} onRetry={reload} rows={4}>
        {runs.length > 0
          ? <Navigate to={`/results/${runs[0].run_id}`} replace />
          : (
            <Card>
              <EmptyState icon={ClipboardCheck} title="No validation results yet"
                action={<Link to="/validate" className={buttonClass("primary")}>Validate a model</Link>}>
                Results appear here once a model has been validated.
              </EmptyState>
            </Card>
          )}
      </Gate>
    </>
  );
}

export function Results() {
  const { runId = "" } = useParams();
  const run = useRun(runId);
  const catalogue = useCatalogue();
  return (
    <>
      {!run.data && <PageHeader title="Validation Results" crumb={<Link to="/history">Validation History</Link>} />}
      <ConnectNotice />
      <Gate loading={run.loading} error={run.error} onRetry={run.reload} rows={6}>
        {run.data && <Detail result={run.data} catalogue={catalogue.map} />}
      </Gate>
    </>
  );
}

const METRICS: [string, string][] = [
  ["behaviour.f1", "F1"], ["behaviour.precision", "Precision"], ["behaviour.recall", "Recall"], ["behaviour.roc_auc", "ROC AUC"],
  ["calibration.ece", "Calibration error"], ["fairness.selection_rate_ratio", "Selection-rate ratio"],
  ["operational.latency", "Latency (p95)"], ["operational.cpu_time", "CPU time"],
];

function KeyMetrics({ checks, catalogue }: { checks: CheckResult[]; catalogue: Map<string, CheckInfo> }) {
  const present = METRICS.map(([id, label]) => ({ check: checks.find((c) => c.id === id), label })).filter((m) => m.check);
  if (present.length === 0) return null;
  return (
    <Card>
      <CardHeader title="Key metrics" subtitle="Measured values for the checks this profile ran, with their status." />
      <CardBody className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {present.map(({ check, label }) => (
          <div key={check!.id} className="rounded-lg border border-slate-200 px-3.5 py-3">
            <div className="flex items-center justify-between gap-2"><span className="text-[13px] text-slate-500">{label}</span><StatusBadge status={check!.status} /></div>
            <div className="mt-1 text-xl font-bold tabular-nums">
              {check!.status === "UNAVAILABLE" || check!.status === "ERROR" ? <span className="text-base font-semibold text-slate-400">no evidence</span> : fmtValue(check!, catalogue)}
            </div>
          </div>
        ))}
      </CardBody>
    </Card>
  );
}

function Detail({ result, catalogue }: { result: ValidationResult; catalogue: Map<string, CheckInfo> }) {
  const { client } = useConnection();
  const [tab, setTab] = useState<Tab>("overview");
  useEffect(() => setTab("overview"), [result.run_id]);

  const card = result.scorecard;
  const hidden = result.checks.find((c) => c.id === HIDDEN_CHECK);
  const snapshot = result.config_snapshot;
  const profile = typeof snapshot.profile === "string" ? snapshot.profile : undefined;
  const contributions = useMemo(() => {
    const map = new Map<string, CheckContribution>();
    for (const dim of Object.values(card.dimensions)) for (const c of dim.contributions ?? []) map.set(c.check_id, c);
    return map;
  }, [card]);
  const statusCounts = countBy(result.checks, (c) => c.status);
  const attention = result.checks.filter((c) => c.status !== "PASS" && c.status !== "NOT_APPLICABLE").length;
  const { nameOf } = useModelNames();
  const name = nameOf(result.candidate.candidate_id, result.run_id);
  const quotable = card.mqs_sufficient_evidence !== false && card.mqs !== null && card.mqs !== undefined;

  const tabs: { id: Tab; label: string; count?: number }[] = [
    { id: "overview", label: "Overview" },
    { id: "checks", label: "Checks", count: result.checks.length },
    { id: "rules", label: "Rules & explanation", count: result.decision.triggered_rules?.length ?? 0 },
    { id: "evidence", label: "Evidence" },
    { id: "report", label: "Report" },
    { id: "raw", label: "Raw result" },
  ];

  return (
    <>
      <PageHeader crumb={<><Link to="/history" className="text-blue-700 hover:underline">Validation History</Link> / {name}</>} title={name}
        subtitle={<>Run <Mono>{result.run_id}</Mono> · {fmtDate(result.created_at_utc)} · gate {result.gate_version}</>}
        actions={<>
          {hidden && <Link to={`/hidden-failures?run=${result.run_id}`} className={buttonClass("secondary")}><ScanSearch size={15} aria-hidden />Hidden failure analysis</Link>}
          <Button onClick={() => downloadText(`${name}-${result.run_id.slice(0, 8)}.json`, JSON.stringify(result, null, 2), "application/json")}>
            <Download size={15} aria-hidden />Download JSON</Button>
        </>} />

      <DecisionBanner decision={result.decision.decision} summary={result.decision.summary}
        overriddenByCritical={result.decision.overridden_by_critical} profile={profile} notice={RECOMMENDATION_NOTICE} />

      <Tabs label="Run sections" tabs={tabs} value={tab} onChange={setTab} />

      <TabPanel id={tab}>
        {tab === "overview" && (
          <>
            <div className="grid gap-4 lg:grid-cols-[340px_1fr]">
              <Card>
                <CardHeader title="Model Quality Score" />
                <ScoreGauge value={quotable ? card.mqs ?? null : null} lowerBound={card.mqs_lower_bound} provisional={card.provisional}
                  promoteAt={snapshotNumber(snapshot, "decision", "promote_at")} rejectBelow={snapshotNumber(snapshot, "decision", "reject_below")} />
                <p className="px-5 pb-4 text-center text-[13px] text-slate-500">Evidence coverage {fmtPercent(card.overall_coverage)}</p>
              </Card>
              <div className="grid content-start gap-4 sm:grid-cols-3">
                {DIMENSIONS.map((d) => (
                  <LayerCard key={d} dimension={d} score={card.dimensions[d]} weight={card.weights[d]} effectiveWeight={card.effective_weights?.[d]} />
                ))}
                <Card className="grid gap-2 px-5 py-4 sm:col-span-3">
                  <h3 className="font-semibold">{quotable ? "How the score was calculated" : "Recorded calculation (not reliable)"}</h3>
                  {!quotable && (
                    <Notice tone="warn" title="This number is recorded but must not be quoted as a quality score">
                      {card.insufficient_evidence_reason ?? "There was too little evidence."} Look at the worst case instead.
                    </Notice>
                  )}
                  <div aria-label="MQS calculation" className="overflow-x-auto rounded-lg border border-blue-100 bg-blue-50 px-3.5 py-2.5 font-mono text-[12.5px] text-blue-950">{mqsCalculation(card)}</div>
                  <p className="text-xs text-slate-500">
                    Each layer score is a weighted average of its checks, each scored 0 to 100 between the configured floor and target.
                    Weights and thresholds are configuration, not standards.
                  </p>
                </Card>
              </div>
            </div>
            <KeyMetrics checks={result.checks} catalogue={catalogue} />
            <div className="grid gap-4 lg:grid-cols-[1fr_340px]">
              <CheckScoreBars checks={result.checks} />
              <StatusDonut counts={statusCounts} />
            </div>
            <Card className="grid gap-3 px-5 py-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-base font-semibold">Check results</h2>
                <Button small onClick={() => setTab("checks")}>{attention} need attention · view all</Button>
              </div>
              <p className="text-[13px] text-slate-500">
                {hidden ? `Hidden subgroup detection: ${hidden.status === "UNAVAILABLE" || hidden.status === "ERROR" ? "not assessed" : "ran"}.`
                  : "Hidden subgroup detection is switched off in this profile."}
              </p>
            </Card>
          </>
        )}
        {tab === "checks" && <ChecksTable checks={result.checks} contributions={contributions} catalogue={catalogue} />}
        {tab === "rules" && (
          <>
            <RulesPanel rules={result.decision.triggered_rules ?? []} overridden={Boolean(result.decision.overridden_by_critical)} />
            <Explanations items={result.explanations ?? []} />
          </>
        )}
        {tab === "evidence" && (
          <>
            <LayerBars dimensions={card.dimensions} />
            <EvidenceBreakdown checks={result.checks} />
            <EvidenceGaps checks={result.checks} />
            <AuditTrail result={result} />
          </>
        )}
        {tab === "report" && <ReportTab runId={result.run_id} name={name} createdAt={result.created_at_utc} candidateId={result.candidate.candidate_id} fetchReport={client.report} />}
        {tab === "raw" && (
          <Card>
            <CardHeader title="Raw result" subtitle="The full, auditable result as returned by the API." />
            <pre tabIndex={0} aria-label="Raw JSON result" className="m-0 max-h-[640px] overflow-auto rounded-b-xl bg-slate-900 p-5 font-mono text-xs text-slate-200">{JSON.stringify(result, null, 2)}</pre>
          </Card>
        )}
      </TabPanel>
    </>
  );
}

function EvidenceBreakdown({ checks }: { checks: CheckResult[] }) {
  const counts = countBy(checks, (c) => c.evidence_source);
  const sources = ["MEASURED", "DECLARED", "SIMULATED", "UNAVAILABLE"] as const;
  return (
    <Card>
      <CardHeader title="Where the evidence comes from"
        subtitle="Measured: computed by the gate. Declared: taken from producer-supplied metadata, not verified. Unavailable: no evidence, never a pass." />
      <TableWrap>
        <Table>
          <thead><tr><Th>Evidence</Th><Th right>Checks</Th><Th right>Share</Th></tr></thead>
          <tbody>
            {sources.map((s) => (
              <tr key={s}><Td><EvidenceBadge source={s} /></Td><Td right>{counts[s] ?? 0}</Td><Td right>{fmtPercent((counts[s] ?? 0) / Math.max(1, checks.length))}</Td></tr>
            ))}
          </tbody>
        </Table>
      </TableWrap>
    </Card>
  );
}

function ReportTab({ runId, name, createdAt, candidateId, fetchReport }: { runId: string; name: string; createdAt: string; candidateId: string | null | undefined; fetchReport: (id: string) => Promise<string> }) {
  const report = useAsync(() => fetchReport(runId), [runId]);
  if (report.error) return <ErrorNotice error={report.error} onRetry={report.reload} />;
  if (report.loading || report.data === null) return <CardSkeleton rows={8} />;
  if (!report.data.trim()) return <Card><EmptyState title="The report is empty" /></Card>;
  return <ReportView markdown={nameReport(report.data, candidateId, name)} filename={`${name}-${runId.slice(0, 8)}-report.md`} pdf={{ name, runId, createdAt }} />;
}

function AuditTrail({ result }: { result: ValidationResult }) {
  const machine = (result.environment as { machine?: { platform?: string; cpu_count?: number; ram_gb?: number } }).machine;
  const c = result.candidate;
  const rows: [string, React.ReactNode][] = [
    ["Model file", <Mono key="m" title={c.model_path}>{shortPath(c.model_path)}</Mono>],
    ["Model SHA-256", <Mono key="a">{shortHash(c.model_sha256, 24)}</Mono>],
    ["Evaluation data SHA-256", <Mono key="b">{shortHash(c.eval_data_sha256, 24)}</Mono>],
    ["Metadata SHA-256", <Mono key="c">{shortHash(c.metadata_sha256, 24)}</Mono>],
    ["Configuration hash", <Mono key="d">{shortHash(result.config_hash, 24)}</Mono>],
    ["Evaluated on", machine ? `${machine.platform ?? "unknown"}, ${machine.cpu_count ?? "?"} CPUs, ${machine.ram_gb ?? "?"} GB RAM` : "—"],
    ["Mode", `${result.scenario.mode === "simulated" ? "Simulated" : "Real evidence"} · ${result.decision.action_mode ?? "RECOMMENDATION"}`],
  ];
  return (
    <Card>
      <CardHeader title="Inputs and audit trail" subtitle="What was validated, so the result can be traced and reproduced." />
      <dl className="grid grid-cols-[max-content_1fr] gap-x-6 gap-y-1.5 px-5 py-4 text-[13px]">
        {rows.map(([k, v]) => (<Fragment key={k}><dt className="text-slate-500">{k}</dt><dd className="m-0">{v}</dd></Fragment>))}
      </dl>
      {result.scenario.mode === "simulated" && <CardBody><Notice tone="warn" title="Simulated inputs">Some evidence was simulated, not measured.</Notice></CardBody>}
    </Card>
  );
}
