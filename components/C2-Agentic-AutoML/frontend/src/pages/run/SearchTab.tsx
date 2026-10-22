import { ReasoningEntry, Trial } from "../../api";
import { familyLabel, fmt } from "../../theme";
import { AgentBadge, Card, ChartCard, DataTable, Empty, JupyterButton } from "../../components/ui";
import { TrialSmallMultiples, bestPerFamilyRound } from "../../charts/SearchCharts";

/** Parse "max_depth.high 10 -> 6 (shrink_depth_range)" into its parts for a readable table. */
function parseChange(c: string) {
  const m = c.match(/^(\S+)\.(low|high) (\S+) -> (\S+) \((.+)\)$/);
  if (m) return { param: m[1], side: m[2], from: m[3], to: m[4], reason: m[5] };
  const add = c.match(/^add (\S+) (.+) \((.+)\)$/);
  if (add) return { param: add[1], side: "new", from: "–", to: add[2], reason: add[3] };
  return { param: c, side: "", from: "", to: "", reason: "" };
}

export default function SearchTab({ trials, reasoning, nbUrl }: { trials: Trial[]; reasoning: ReasoningEntry[]; nbUrl: string }) {
  const adapts = reasoning.filter((e) => e.step === "adapt_spaces");
  const budgets = reasoning.filter((e) => e.step === "run_hpo");
  const best = bestPerFamilyRound(trials);
  const pruned = trials.filter((t) => t.state === "PRUNED").length;

  return (
    <div className="stack">
      <ChartCard title="Search history by family" sub={`${trials.length} trials · ${pruned} pruned early by Optuna's median pruner. Ink line = best score found so far.`}
        tools={<JupyterButton url={nbUrl} small label="Notebook" />}
        table={{ columns: ["Family", "Round", "Trial", "State", "CV score", "Train", "Gap"], rows: trials.map((t) => [familyLabel(t.family), t.round, t.trial, t.state, t.mean_val === null ? null : Number(fmt(t.mean_val)), t.mean_train === null ? null : Number(fmt(t.mean_train)), t.gap === null ? null : Number(fmt(t.gap))]) }}>
        <TrialSmallMultiples trials={trials} />
      </ChartCard>

      <div className="stack">
        <Card>
          <div className="card-head">
            <div><div className="row"><AgentBadge agent="agent2" /><h3>Range changes (Agent 2 Level 4)</h3></div>
              <div className="sub">How each diagnosis changed the search space before the next round.</div></div>
          </div>
          <div className="card-body">
            {adapts.length === 0 && <Empty>Single-round run — no adaptation.</Empty>}
            {adapts.map((e) => {
              const rows = Object.entries(e.decision.changes as Record<string, string[]>).flatMap(([f, cs]) => cs.map((c) => ({ f, ...parseChange(c) })));
              return (
                <div key={e.round} style={{ marginBottom: 14 }}>
                  <div className="small" style={{ fontWeight: 650, marginBottom: 6 }}>Before round {e.round} · {rows.length} change{rows.length === 1 ? "" : "s"}</div>
                  {rows.length ? (
                    <DataTable maxHeight="none" spec={{ columns: ["Family", "Parameter", "Bound", "From", "To", "Why"], rows: rows.map((r) => [familyLabel(r.f), r.param, r.side, r.from, r.to, r.reason.replace(/_/g, " ")]) }} />
                  ) : <div className="small muted">No changes requested — ranges kept, warm start continues.</div>}
                </div>
              );
            })}
          </div>
        </Card>
        <Card>
          <div className="card-head"><div><div className="row"><AgentBadge agent="agent1" /><h3>Budget per round (Agent 1 Level 4)</h3></div><div className="sub">Remaining trials are shared equally over the remaining rounds.</div></div></div>
          <div className="card-body grid g2" style={{ alignItems: "start" }}>
            <div>
            <DataTable spec={{
              columns: ["Round", ...[...new Set(budgets.flatMap((b) => Object.keys(b.decision.budget)))].map(familyLabel), "Total"],
              rows: budgets.map((b) => {
                const fams = [...new Set(budgets.flatMap((x) => Object.keys(x.decision.budget)))];
                const vals = fams.map((f) => b.decision.budget[f] ?? "–");
                return [b.round, ...vals, Object.values(b.decision.budget as Record<string, number>).reduce((a, c) => a + c, 0)];
              }),
            }} />
            </div>
            <div>
            <div className="small muted" style={{ margin: "0 0 6px" }}>Best CV score per family and round</div>
            <DataTable maxHeight="none" spec={{ columns: ["Family", "Round", "Best CV", "Gap"], rows: best.map((t) => [familyLabel(t.family), t.round, Number(fmt(t.mean_val)), Number(fmt(t.gap, 3))]) }} />
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
