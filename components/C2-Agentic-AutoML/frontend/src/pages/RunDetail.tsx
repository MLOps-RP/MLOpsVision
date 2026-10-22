import { useEffect, useRef, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { Activity, BarChart3, FileOutput, ListTree, ScrollText, Trophy } from "lucide-react";
import { api } from "../api";
import { useData } from "../hooks";
import { familyColor, familyLabel, fmt, when } from "../theme";
import { Card, ErrorBox, JupyterButton, Loading, PageHead, Stat, StatusBadge, Tabs } from "../components/ui";
import { useNotebook } from "../App";
import WorkflowTab from "./run/WorkflowTab";
import ResultsTab from "./run/ResultsTab";
import SearchTab from "./run/SearchTab";
import ReasoningTab from "./run/ReasoningTab";
import OutputsTab from "./run/OutputsTab";

type TabId = "workflow" | "results" | "search" | "reasoning" | "outputs";

export default function RunDetailPage() {
  const { id = "" } = useParams();
  const [search] = useSearchParams();
  const [tab, setTab] = useState<TabId>((search.get("tab") as TabId) || "workflow");
  const [running, setRunning] = useState(true);
  const run = useData(() => api.run(id), [id], running ? 1500 : undefined);
  const trials = useData(() => api.trials(id), [id], running ? 2000 : undefined);
  const reasoning = useData(() => api.reasoning(id), [id], running ? 3000 : undefined);
  const done = run.data?.status === "completed";
  const evaluation = useData(done ? () => api.evaluation(id) : null, [id, done]);
  const nbUrl = useNotebook("run");
  const wasRunning = useRef(false);

  // stop polling once the run is finished; refresh everything one last time
  useEffect(() => {
    const status = run.data?.status;
    if (!status) return;
    if (status === "running") wasRunning.current = true;
    else {
      setRunning(false);
      if (wasRunning.current) {
        wasRunning.current = false;
        trials.reload();
        reasoning.reload();
        setTab("results");
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run.data?.status]);

  if (run.error && !run.data) return <ErrorBox error={run.error} />;
  if (!run.data) return <Loading />;
  const r = run.data;
  const o = r.output;
  const totalBudget = r.params?.total_trials ?? o?.total_trials ?? 0;
  const nTrials = trials.data?.length ?? 0;

  return (
    <div>
      <PageHead
        title={r.run_id}
        sub={<span className="row" style={{ gap: 8 }}>
          <StatusBadge status={r.status} />
          <span className="chip">dataset {r.dataset_id}</span>
          {r.params && <>
            <span className="chip">{r.params.total_trials} trials</span>
            <span className="chip">{r.params.folds} folds</span>
            <span className="chip">max {r.params.max_rounds} rounds</span>
            {!r.params.adapt_ranges && <span className="chip">ablation: no range adaptation</span>}
            {!r.params.adapt_families && <span className="chip">ablation: no family dropping</span>}
          </>}
          {r.started && <span className="small muted">started {when(r.started)}</span>}
        </span>}
        actions={<JupyterButton url={nbUrl} label="Analyse in Jupyter" />}
      />

      {r.status === "running" && (
        <Card pad className="mt" >
          <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
            <b className="row" style={{ gap: 8 }}><Activity size={16} /> Agents are working…</b>
            <span className="small muted">{nTrials} / {totalBudget} trials</span>
          </div>
          <div className="progress"><div style={{ width: `${Math.min(100, (nTrials / Math.max(totalBudget, 1)) * 100)}%` }} /></div>
        </Card>
      )}
      {r.status === "failed" && <div className="callout error" style={{ marginBottom: 16 }}>Run failed: {r.error}</div>}

      {o && (
        <div className="grid g4" style={{ marginBottom: 18 }}>
          <Stat label="Selected model" icon={<Trophy size={14} />}
            value={<span className="row" style={{ gap: 8 }}><span style={{ width: 12, height: 12, borderRadius: 4, background: familyColor(o.model_family) }} />{familyLabel(o.model_family)}</span>}
            note={`found in round ${o.candidates.find((c) => c.family === o.model_family)?.round ?? 1} of ${o.rounds}`} />
          <Stat label="Hold-out ROC-AUC" value={fmt(o.metrics_holdout.roc_auc, 3)} note={`CV ${fmt(o.metrics_cv[o.metric], 4)} · tested once on unseen data`} />
          <Stat label="Hold-out F1 · recall" value={`${fmt(o.metrics_holdout.f1, 3)} · ${fmt(o.metrics_holdout.recall, 3)}`} note={`precision ${fmt(o.metrics_holdout.precision, 3)} · accuracy ${fmt(o.metrics_holdout.accuracy, 3)}`} />
          <Stat label="Search" value={`${o.total_trials} trials`} note={`${o.rounds} rounds · stop: ${o.stop_reason ?? "–"}`} />
        </div>
      )}

      <Tabs<TabId>
        value={tab}
        onChange={setTab}
        tabs={[
          { id: "workflow", label: "Agentic workflow", icon: <ListTree size={15} /> },
          { id: "results", label: "Results & charts", icon: <BarChart3 size={15} /> },
          { id: "search", label: "Search & adaptation", icon: <Activity size={15} /> },
          { id: "reasoning", label: "Reasoning", icon: <ScrollText size={15} /> },
          { id: "outputs", label: "Inputs & outputs", icon: <FileOutput size={15} /> },
        ]}
      />

      {tab === "workflow" && <WorkflowTab run={r} trials={trials.data ?? []} reasoning={reasoning.data ?? []} />}
      {tab === "results" && (o
        ? <ResultsTab output={o} evaluation={evaluation.data} evalError={evaluation.error} trials={trials.data ?? []} reasoning={reasoning.data ?? []} nbUrl={nbUrl} />
        : <Card><Loading text="Results appear when the run finishes…" /></Card>)}
      {tab === "search" && <SearchTab trials={trials.data ?? []} reasoning={reasoning.data ?? []} nbUrl={nbUrl} />}
      {tab === "reasoning" && <ReasoningTab reasoning={reasoning.data ?? []} />}
      {tab === "outputs" && (o ? <OutputsTab output={o} /> : <Card><Loading text="Outputs appear when the run finishes…" /></Card>)}
    </div>
  );
}
