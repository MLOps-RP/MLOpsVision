import { ReactNode } from "react";
import { RunEvent } from "../../api";
import { familyLabel, fmt } from "../../theme";
import { DiagnosisChip } from "../../components/ui";

const OWNER: Record<string, string> = {
  load_input: "tools", profile_task: "tools", agent1_select: "agent1", agent2_hpo: "agent2", evaluate: "tools",
  agent1_diagnose: "agent1", agent2_adapt: "agent2", agent3_select: "agent3", package_output: "tools",
};

/** Human-readable description of one workflow event (used by the feed and the node inspector). */
export function describe(e: RunEvent, detailed = false): { agent: string; title: string; body: ReactNode } {
  const d = e.data;
  const agent = OWNER[e.node] ?? "tools";
  switch (e.node) {
    case "load_input":
      return { agent, title: "Loaded the C1 output", body: <>Trial budget for the whole run: <b>{d.total_trials}</b>.</> };
    case "profile_task":
      return {
        agent, title: "Profiled the dataset",
        body: <>{d.n_rows?.toLocaleString()} rows, {d.n_features} features ({d.n_categorical} categorical) · size <b>{d.size_class}</b> · metric <b>{d.metric}</b>{d.imbalance_ratio != null && <> · imbalance ratio <b>{fmt(d.imbalance_ratio, 3)}</b></>}</>,
      };
    case "agent1_select":
      return {
        agent, title: `Selected ${d.families.length} model families`,
        body: <>
          <div className="row" style={{ gap: 6, marginBottom: 6 }}>{(d.families as string[]).map((f) => <span key={f} className="chip">{familyLabel(f)} · {d.budget[f]} trials</span>)}</div>
          <span className="ink2">{e.reasoning[0]?.rationale}</span>
        </>,
      };
    case "agent2_hpo": {
      const cands = Object.entries(d.candidates as Record<string, any>).sort((a, b) => b[1].mean_val - a[1].mean_val);
      return {
        agent, title: `Round ${d.round}: tuned ${cands.length} families`,
        body: (
          <div style={{ display: "grid", gap: 3 }}>
            {cands.map(([f, c]) => (
              <div key={f} className="small">
                <b>{familyLabel(f)}</b> best <b>{fmt(c.mean_val)}</b> · gap {fmt(c.gap, 3)} · {c.n_complete} done, {c.n_pruned} pruned
                {detailed && <div className="muted mono" style={{ fontSize: 11 }}>{JSON.stringify(c.params)}</div>}
              </div>
            ))}
          </div>
        ),
      };
    }
    case "evaluate":
      return {
        agent, title: "Collected diagnosis signals",
        body: detailed ? (
          <table className="t"><thead><tr><th>Family</th><th className="num">Train</th><th className="num">Val</th><th className="num">Gap</th><th className="num">Minority recall</th><th className="num">Lift</th></tr></thead>
            <tbody>{Object.entries(d.signals as Record<string, any>).map(([f, s]) => (
              <tr key={f}><td>{familyLabel(f)}</td><td className="num">{fmt(s.mean_train, 3)}</td><td className="num">{fmt(s.mean_val, 3)}</td><td className="num">{fmt(s.gap, 3)}</td><td className="num">{fmt(s.minority_recall, 3)}</td><td className="num">{fmt(s.lift_over_baseline, 3)}</td></tr>
            ))}</tbody></table>
        ) : <>Train/validation gap, CV spread, minority recall and lift over a dummy model for {Object.keys(d.signals).length} families.</>,
      };
    case "agent1_diagnose":
      return {
        agent, title: d.stop_reason ? `Diagnosed and decided to stop (${d.stop_reason})` : "Diagnosed and decided to continue",
        body: (
          <div style={{ display: "grid", gap: 6 }}>
            {(d.diagnoses as any[]).map((x) => (
              <div key={x.family} className="row" style={{ gap: 6 }}>
                <b className="small" style={{ minWidth: 120 }}>{familyLabel(x.family)}</b>
                {x.labels.map((l: string) => <DiagnosisChip key={l} label={l} />)}
                {detailed && <span className="small muted">{x.reasons.join("; ")}</span>}
              </div>
            ))}
            {Object.entries(d.actions as Record<string, string[]>).filter(([, a]) => a.length).map(([f, a]) => (
              <div key={f} className="small ink2">→ {familyLabel(f)}: {a.map((x) => x.replace(/_/g, " ")).join(", ")}</div>
            ))}
            {d.referrals?.length > 0 && <div className="small ink2">↩ to C1: {d.referrals.join("; ")}</div>}
            <div className="small"><b>Decision:</b> {d.decision}</div>
          </div>
        ),
      };
    case "agent2_adapt": {
      const all = Object.entries(d.changes as Record<string, string[]>).flatMap(([f, cs]) => cs.map((c) => ({ f, c })));
      return {
        agent, title: `Adapted search ranges for round ${d.round} (${all.length} changes)`,
        body: all.length ? (
          <div style={{ display: "grid", gap: 3 }}>
            {all.slice(0, detailed ? 50 : 4).map(({ f, c }, i) => <div key={i} className="small"><b>{familyLabel(f)}</b> · {c.replace(/_/g, " ")}</div>)}
            {!detailed && all.length > 4 && <div className="small muted">+ {all.length - 4} more</div>}
          </div>
        ) : <span className="ink2">No diagnosis asked for a change; ranges kept (warm start continues the search).</span>,
      };
    }
    case "agent3_select":
      return {
        agent, title: `Selected ${familyLabel(d.family)} (found in round ${d.round ?? 1})`,
        body: <>Hold-out: {Object.entries(d.metrics_holdout as Record<string, number>).map(([k, v]) => <span key={k} className="chip" style={{ marginRight: 4 }}>{k} {fmt(v, 3)}</span>)}</>,
      };
    case "package_output":
      return { agent, title: "Packaged the output for C3", body: <>{d.rounds} rounds · {d.total_trials} trials · best by round {(d.score_by_round ?? []).map((s: number) => fmt(s)).join(" → ")}</> };
    default:
      return { agent, title: e.node, body: null };
  }
}
