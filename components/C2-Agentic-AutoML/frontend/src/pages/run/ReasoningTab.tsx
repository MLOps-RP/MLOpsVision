import { useState } from "react";
import { ReasoningEntry } from "../../api";
import { AGENTS, AgentKey, familyLabel, prettyStep } from "../../theme";
import { AgentBadge, Card, DiagnosisChip, Empty, JsonView } from "../../components/ui";

const AGENT_FILTERS = ["all", "agent1", "agent2", "agent3"] as const;

export default function ReasoningTab({ reasoning }: { reasoning: ReasoningEntry[] }) {
  const [filter, setFilter] = useState<(typeof AGENT_FILTERS)[number]>("all");
  const entries = reasoning.filter((e) => filter === "all" || e.agent === filter);
  const rounds = [...new Set(entries.map((e) => e.round))].sort((a, b) => a - b);

  return (
    <Card>
      <div className="card-head">
        <div>
          <h3>Auditable reasoning log</h3>
          <div className="sub">Every decision of every agent, with the evidence it was based on. This is what C3 and the audit trail receive.</div>
        </div>
        <div className="tools">
          <div className="seg">
            {AGENT_FILTERS.map((f) => (
              <button key={f} className={filter === f ? "on" : ""} onClick={() => setFilter(f)}>{f === "all" ? "All agents" : AGENTS[f].name}</button>
            ))}
          </div>
        </div>
      </div>
      <div className="card-body">
        {!entries.length && <Empty>No reasoning entries yet.</Empty>}
        <div className="timeline">
          {rounds.map((r) => (
            <div key={r}>
              <div className="tl-round">Round {r}</div>
              {entries.filter((e) => e.round === r).map((e, i) => {
                const a = AGENTS[(e.agent in AGENTS ? e.agent : "tools") as AgentKey];
                return (
                  <div className="tl-item" key={i}>
                    <span className="tl-dot" style={{ background: a.color }} />
                    <div className="row" style={{ gap: 8 }}>
                      <AgentBadge agent={e.agent} />
                      <b style={{ textTransform: "capitalize" }}>{prettyStep(e.step)}</b>
                      {e.step === "diagnose" && <>
                        <span className="chip">{familyLabel(e.decision.family)}</span>
                        {(e.decision.labels as string[]).map((l) => <DiagnosisChip key={l} label={l} />)}
                      </>}
                      {e.step === "decide_next_round" && <span className="chip">{e.decision.continue ? "continue" : `stop · ${e.decision.stop_reason}`}</span>}
                      <span className="badge outline" title="Who made the decision">decided by: {e.decided_by}</span>
                    </div>
                    {e.rationale && <div className="small ink2" style={{ marginTop: 6, lineHeight: 1.55 }}>{e.rationale}</div>}
                    <details style={{ marginTop: 6 }}>
                      <summary className="small muted" style={{ cursor: "pointer" }}>Decision & evidence</summary>
                      <div className="grid g2" style={{ marginTop: 8 }}>
                        <JsonView value={e.decision} maxHeight={260} />
                        <JsonView value={e.evidence} maxHeight={260} />
                      </div>
                    </details>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}
