import { useState } from "react";
import { ReasoningEntry, RunDetail, Trial } from "../../api";
import { AGENTS, AgentKey, prettyStep } from "../../theme";
import { AgentBadge, Card, Empty } from "../../components/ui";
import { NODES, WorkflowGraph, workflowView } from "../../components/WorkflowGraph";
import { TrialSmallMultiples } from "../../charts/SearchCharts";
import { describe } from "./describe";

export default function WorkflowTab({ run, trials, reasoning }: { run: RunDetail; trials: Trial[]; reasoning: ReasoningEntry[] }) {
  const view = workflowView(run);
  const [selected, setSelected] = useState<string | null>(null);
  const node = NODES.find((n) => n.id === selected);
  const lastEvent = selected ? view.last[selected] : undefined;
  const events = [...run.events].reverse();

  return (
    <div className="stack">
      <Card>
        <div className="card-head">
          <div>
            <h3>Agentic workflow (LangGraph)</h3>
            <div className="sub">Each lane is one agent. The loop arrow is the novelty: diagnosis → new ranges → tune again. Click a node to inspect it.</div>
          </div>
          <div className="tools">{run.status === "running" && <span className="badge">Round {view.round}</span>}</div>
        </div>
        <div className="card-body">
          <WorkflowGraph view={view} selected={selected} onSelect={(id) => setSelected(id === selected ? null : id)} />
        </div>
      </Card>

      {node && (
        <Card>
          <div className="card-head">
            <div>
              <div className="row"><AgentBadge agent={node.owner} /><h3>{node.title}</h3>{(view.counts[node.id] ?? 0) > 1 && <span className="chip">ran {view.counts[node.id]}×</span>}</div>
              <div className="sub">{node.hint}</div>
            </div>
          </div>
          <div className="card-body">
            {lastEvent ? (() => {
              const d = describe(lastEvent, true);
              return <><div style={{ fontWeight: 600, marginBottom: 8 }}>{d.title}</div><div className="small">{d.body}</div></>;
            })() : <span className="muted small">{run.events.length ? "Not reached yet." : "Detailed node events are only kept for runs started from this dashboard. See the Reasoning tab for the full log."}</span>}
          </div>
        </Card>
      )}

      <div className="grid g-1-2" style={{ alignItems: "start" }}>
        <Card>
          <div className="card-head"><div><h3>Agent activity</h3><div className="sub">Newest first</div></div></div>
          <div className="card-body">
            {events.length ? (
              <div className="feed">
                {events.map((e, i) => {
                  const d = describe(e);
                  const a = AGENTS[(d.agent in AGENTS ? d.agent : "tools") as AgentKey];
                  return (
                    <div className="feed-item" key={i}>
                      <div className="ic" style={{ background: a.tint }}><span style={{ width: 8, height: 8, borderRadius: 2, background: a.color }} /></div>
                      <div>
                        <div className="row" style={{ gap: 6 }}><AgentBadge agent={d.agent} /><b className="small">{d.title}</b></div>
                        <div className="t" style={{ marginTop: 5 }}>{d.body}</div>
                        <div className="meta">{new Date(e.time).toLocaleTimeString()}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : reasoning.length ? (
              <div className="feed">
                <div className="small muted" style={{ paddingBottom: 6 }}>Run started from the command line — activity rebuilt from the reasoning log.</div>
                {[...reasoning].reverse().map((e, i) => {
                  const a = AGENTS[(e.agent in AGENTS ? e.agent : "tools") as AgentKey];
                  return (
                    <div className="feed-item" key={i}>
                      <div className="ic" style={{ background: a.tint }}><span style={{ width: 8, height: 8, borderRadius: 2, background: a.color }} /></div>
                      <div>
                        <div className="row" style={{ gap: 6 }}><AgentBadge agent={e.agent} /><b className="small" style={{ textTransform: "capitalize" }}>{prettyStep(e.step)}</b><span className="small muted">round {e.round}</span></div>
                        <div className="t" style={{ marginTop: 5 }}>{e.rationale}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : <Empty>{run.status === "running" ? "Starting…" : "No activity recorded."}</Empty>}
          </div>
        </Card>
        <Card>
          <div className="card-head">
            <div>
              <h3>{run.status === "running" ? "Live search" : "Search history"}</h3>
              <div className="sub">Every Optuna trial per family; colour = round. Watch later rounds climb after the ranges adapt.</div>
            </div>
          </div>
          <div className="card-body"><TrialSmallMultiples trials={trials} live={run.status === "running"} /></div>
        </Card>
      </div>
    </div>
  );
}
