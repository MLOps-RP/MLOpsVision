import { ArrowRight, BarChart3, CheckCircle2, Hand, History, PlayCircle, RotateCcw } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";

import type { Decision } from "../api/types";
import { DecisionBadge } from "../components/badges";
import { DecisionDonut, MqsByRunChart } from "../components/charts";
import { DecisionBanner } from "../components/DecisionBanner";
import { Gate } from "../components/Gate";
import { LayerCard } from "../components/LayerCard";
import { ScoreGauge } from "../components/ScoreGauge";
import {
  buttonClass, Card, CardHeader, ConnectNotice, EmptyState, ErrorNotice, PageHeader, StatCard, Table, TableWrap, Td, Th,
} from "../components/ui";
import { DIMENSIONS } from "../api/types";
import { fmtDate, fmtScore, RECOMMENDATION_NOTICE, snapshotNumber } from "../lib/format";
import { useModelNames } from "../lib/modelNames";
import { useRun, useRuns } from "../lib/runs";

export function Dashboard() {
  const runs = useRuns();
  const { nameOf } = useModelNames();
  const list = runs.runs;
  const latest = list[0];
  const result = useRun(latest?.run_id);
  const navigate = useNavigate();
  const count = (d: Decision) => list.filter((r) => r.decision === d).length;
  const counts: Record<Decision, number> = { PROMOTE: count("PROMOTE"), HOLD: count("HOLD"), ROLLBACK: count("ROLLBACK") };

  return (
    <>
      <PageHeader title="Dashboard"
        subtitle="Validate a trained model before it is promoted, and review the evidence behind every recommendation."
        actions={<Link to="/validate" className={buttonClass("primary")}><PlayCircle size={16} aria-hidden />Validate a model</Link>} />
      <ConnectNotice />

      <Gate loading={runs.loading} error={runs.error} onRetry={runs.reload} rows={6}>
        <section aria-label="Summary" className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <StatCard icon={BarChart3} label="Validation runs" value={list.length} note="stored by the API" />
          <StatCard icon={CheckCircle2} label="Promote" value={counts.PROMOTE} note="meets the quality bar" />
          <StatCard icon={Hand} label="Hold" value={counts.HOLD} note="needs review" />
          <StatCard icon={RotateCcw} label="Rollback" value={counts.ROLLBACK} note="should not be promoted" />
        </section>

        {list.length === 0 ? (
          <Card>
            <EmptyState icon={History} title="No validations yet"
              action={<Link to="/validate" className={buttonClass("primary")}>Run your first validation</Link>}>
              Validate a model to see its Model Quality Score, the three layer scores, the rules that fired and the report here.
            </EmptyState>
          </Card>
        ) : (
          <>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="text-base font-semibold">Latest validation: <span className="text-blue-700">{nameOf(latest.candidate_id, latest.run_id)}</span></h2>
              <Link to={`/results/${latest.run_id}`} className={buttonClass("secondary", true)}>Open full result <ArrowRight size={14} aria-hidden /></Link>
            </div>

            {result.error ? <ErrorNotice error={result.error} onRetry={result.reload} />
            : result.loading || !result.data ? <div className="h-40 animate-pulse rounded-xl bg-slate-200/70" aria-busy="true" aria-label="Loading" />
            : (
              <>
                <DecisionBanner decision={result.data.decision.decision} summary={result.data.decision.summary}
                  overriddenByCritical={result.data.decision.overridden_by_critical}
                  profile={typeof result.data.config_snapshot.profile === "string" ? result.data.config_snapshot.profile : undefined}
                  notice={RECOMMENDATION_NOTICE} />
                <div className="grid gap-4 lg:grid-cols-[340px_1fr]">
                  <Card>
                    <CardHeader title="Model Quality Score" />
                    <ScoreGauge value={result.data.scorecard.mqs_sufficient_evidence === false ? null : result.data.scorecard.mqs ?? null}
                      lowerBound={result.data.scorecard.mqs_lower_bound} provisional={result.data.scorecard.provisional}
                      promoteAt={snapshotNumber(result.data.config_snapshot, "decision", "promote_at")}
                      rejectBelow={snapshotNumber(result.data.config_snapshot, "decision", "reject_below")} />
                  </Card>
                  <div className="grid content-start gap-4 sm:grid-cols-3">
                    {DIMENSIONS.map((d) => (
                      <LayerCard key={d} dimension={d} score={result.data!.scorecard.dimensions[d]} weight={result.data!.scorecard.weights[d]}
                        effectiveWeight={result.data!.scorecard.effective_weights?.[d]} />
                    ))}
                  </div>
                </div>
              </>
            )}

            <div className="grid gap-4 lg:grid-cols-[1fr_340px]">
              <MqsByRunChart runs={list} nameOf={nameOf} />
              <DecisionDonut counts={counts} />
            </div>

            <Card>
              <CardHeader title="Recent validations" actions={<Link to="/history" className={buttonClass("ghost", true)}>View all</Link>} />
              <TableWrap>
                <Table>
                  <thead><tr><Th>Model</Th><Th>Decision</Th><Th right>MQS</Th><Th>Profile</Th><Th>When</Th></tr></thead>
                  <tbody>
                    {list.slice(0, 7).map((r) => (
                      <tr key={r.run_id} className="cursor-pointer hover:bg-slate-50" onClick={() => navigate(`/results/${r.run_id}`)}>
                        <Td><Link to={`/results/${r.run_id}`} className="font-semibold text-blue-700 hover:underline" onClick={(e) => e.stopPropagation()}>{nameOf(r.candidate_id, r.run_id)}</Link></Td>
                        <Td><DecisionBadge decision={r.decision} /></Td>
                        <Td right>{fmtScore(r.mqs)}</Td>
                        <Td>{r.profile}</Td>
                        <Td className="whitespace-nowrap text-xs text-slate-500">{fmtDate(r.created_at_utc)}</Td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </TableWrap>
            </Card>
          </>
        )}
      </Gate>
    </>
  );
}
