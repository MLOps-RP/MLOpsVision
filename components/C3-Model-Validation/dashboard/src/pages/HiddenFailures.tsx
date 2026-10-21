import { Layers, ScanSearch, ShieldAlert, Users } from "lucide-react";
import { useMemo } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { DecisionBadge } from "../components/badges";
import { GroupEffectChart, GroupErrorChart } from "../components/charts";
import { Gate } from "../components/Gate";
import { HIDDEN_CHECK, HiddenGroups, PolicyLine, readHiddenDetails } from "../components/HiddenGroups";
import { buttonClass, Card, CardSkeleton, ConnectNotice, EmptyState, ErrorNotice, inputClass, Notice, PageHeader, StatCard, Unavailable } from "../components/ui";
import { fmtDate, fmtPercent } from "../lib/format";
import { useModelNames } from "../lib/modelNames";
import { useRun, useRuns } from "../lib/runs";

export function HiddenFailures() {
  const runs = useRuns();
  const { nameOf } = useModelNames();
  const [params, setParams] = useSearchParams();

  // Prefer a run that was actually validated with hidden-group detection; otherwise the newest run.
  const preferred = useMemo(() => runs.runs.find((r) => r.profile.includes("hidden"))?.run_id ?? runs.runs[0]?.run_id, [runs.runs]);
  const runId = params.get("run") ?? preferred;
  const selected = runs.runs.find((r) => r.run_id === runId);
  const result = useRun(runId);

  const check = result.data?.checks.find((c) => c.id === HIDDEN_CHECK);
  const details = readHiddenDetails(check?.details);
  const snapshot = result.data?.config_snapshot as Record<string, unknown> | undefined;
  const policy = snapshot?.hidden_groups as { consequence?: string; require_assessment?: boolean; min_effect?: number } | undefined;
  const profile = typeof snapshot?.profile === "string" ? snapshot.profile : selected?.profile;

  return (
    <>
      <PageHeader title="Hidden Failure Analysis"
        subtitle="Groups of rows, defined by feature values, where the model fails more often than its own probabilities predict. Found on some rows and confirmed on held-out rows." />
      <ConnectNotice />
      <Gate loading={runs.loading} error={runs.error} onRetry={runs.reload} rows={4}>
        {runs.runs.length === 0 ? (
          <Card><EmptyState icon={ScanSearch} title="No validations yet" action={<Link to="/validate" className={buttonClass("primary")}>Validate a model</Link>}>
            Validate a model with the extended_hidden_groups profile to see its failing subgroups here.</EmptyState></Card>
        ) : (
          <>
            <Card className="flex flex-wrap items-center gap-3 px-5 py-3.5">
              <label htmlFor="hidden-run" className="text-[13px] font-semibold text-slate-700">Validation</label>
              <select id="hidden-run" className={`${inputClass} max-w-md`} value={runId ?? ""} onChange={(e) => setParams({ run: e.target.value })}>
                {runs.runs.map((r) => (
                  <option key={r.run_id} value={r.run_id}>{nameOf(r.candidate_id, r.run_id)} · {r.profile} · {fmtDate(r.created_at_utc)}</option>
                ))}
              </select>
              {selected && <DecisionBadge decision={selected.decision} />}
            </Card>

            {result.error ? <ErrorNotice error={result.error} onRetry={result.reload} />
            : result.loading || !result.data ? <CardSkeleton rows={6} />
            : !check ? (
              <Unavailable title="Hidden failure detection was not run for this validation"
                hint={<>Validate the model with the <code className="font-mono">extended_hidden_groups</code> profile on the <Link className="font-semibold text-blue-700 hover:underline" to="/validate">Validate Model</Link> page.</>}>
                The profile used here ({profile ?? "unknown"}) does not include the check, so there is nothing to show. No subgroups are being implied or hidden.
              </Unavailable>
            ) : check.status === "UNAVAILABLE" || check.status === "ERROR" || !details ? (
              <>
                <Unavailable title="Hidden failure detection could not be assessed" hint="This is not a pass: the model was not shown to be free of failing subgroups.">{check.message}</Unavailable>
                <PolicyLine policy={policy} />
              </>
            ) : (
              <>
                {details.groups_confirmed > 0 ? (
                  <Notice tone="warn" title={`${details.groups_confirmed} failing subgroup${details.groups_confirmed === 1 ? "" : "s"} confirmed`}>
                    The model makes clearly more errors in these groups than its own probabilities predict, even though it may look fine overall.
                  </Notice>
                ) : (
                  <Notice tone="ok" title="No failing subgroup was confirmed">
                    No group passed both the significance and the effect-size requirement. Detection power depends on the sample size, so this does not prove none exists.
                  </Notice>
                )}
                <div className="grid gap-4 sm:grid-cols-3">
                  <StatCard icon={Layers} label="Groups tested" value={details.groups_tested} note={`methods: ${details.methods.join(", ")}`} />
                  <StatCard icon={ShieldAlert} label="Confirmed" value={details.groups_confirmed} note={`false discovery rate ${details.alpha}, minimum effect ${fmtPercent(details.min_effect)}`} />
                  <StatCard icon={Users} label="Rows in confirmed groups" value={details.share_of_rows_in_confirmed_groups == null ? "n/a" : fmtPercent(details.share_of_rows_in_confirmed_groups, 1)} note="of the evaluation rows" />
                </div>
                {details.groups.length > 0 && (
                  <div className="grid gap-4 xl:grid-cols-2">
                    <GroupErrorChart groups={details.groups} />
                    <GroupEffectChart groups={details.groups} minEffect={details.min_effect} />
                  </div>
                )}
                <HiddenGroups check={check} policy={policy} />
              </>
            )}
          </>
        )}
      </Gate>
    </>
  );
}
