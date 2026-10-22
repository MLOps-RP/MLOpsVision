import { C2Output, Evaluation, ReasoningEntry, Trial } from "../../api";
import { familyColor, familyLabel, fmt } from "../../theme";
import { Card, ChartCard, DiagnosisChip, ErrorBox, JupyterButton, Loading } from "../../components/ui";
import { ConfusionMatrix, PrChart, RocChart } from "../../charts/EvalCharts";
import { GapDumbbell, HBars, ScoreByRound, bestPerFamilyRound } from "../../charts/SearchCharts";

export default function ResultsTab({ output, evaluation, evalError, trials, reasoning, nbUrl }: {
  output: C2Output; evaluation: Evaluation | null; evalError: string | null; trials: Trial[];
  reasoning: ReasoningEntry[]; nbUrl: string;
}) {
  const scores = output.score_by_round ?? [];
  const best = bestPerFamilyRound(trials);
  const diagByRound = reasoning.filter((e) => e.step === "diagnose");

  return (
    <div className="stack">
      {evalError && <ErrorBox error={evalError} />}
      {!evaluation && !evalError && <Card><Loading text="Computing hold-out curves (refits each candidate once — a few seconds)…" /></Card>}
      {evaluation && !evaluation.supported && <div className="callout warn">{evaluation.reason}</div>}
      {evaluation?.supported && (
        <>
          <div className="grid g2">
            <ChartCard title="ROC curves on hold-out data" sub="Best configuration of each family, refit on the full training split. The test split is used only to report — the choice was made on CV."
              table={{ columns: ["Family", "ROC-AUC", "Average precision"], rows: Object.entries(evaluation.candidates).map(([f, c]) => [familyLabel(f), Number(c.auc.toFixed(4)), Number(c.ap.toFixed(4))]) }}>
              <RocChart ev={evaluation} />
            </ChartCard>
            <ChartCard title="Precision–recall (selected model)" sub="More informative than ROC when the positive class is rare."
              table={{ columns: ["Recall", "Precision"], rows: evaluation.selected.pr.filter((_, i) => i % 5 === 0).map(([r, p]) => [Number(r.toFixed(3)), Number(p.toFixed(3))]) }}>
              <PrChart ev={evaluation} />
            </ChartCard>
          </div>
          <div className="grid g2">
            <ChartCard title="Confusion matrix (hold-out, threshold 0.5)" sub={`${evaluation.n_test.toLocaleString()} test rows · positive rate ${fmt(evaluation.prevalence, 3)}`}>
              <ConfusionMatrix matrix={evaluation.selected.confusion} />
            </ChartCard>
            <ChartCard title="Feature importance (selected model)" sub="Share of the model's total importance (top 15)."
              table={{ columns: ["Feature", "Importance"], rows: evaluation.importance.map((i) => [i.feature, Number(i.importance.toFixed(4))]) }}>
              {evaluation.importance.length ? <HBars items={evaluation.importance.map((i) => ({ label: i.feature, value: i.importance }))} format={(v) => `${(v * 100).toFixed(1)}%`} />
                : <div className="muted small">This model type does not expose importances.</div>}
            </ChartCard>
          </div>
        </>
      )}

      <div className="grid g2">
        <ChartCard title="Best CV score after each round" sub={`Stop reason: ${output.stop_reason ?? "–"}. The best-so-far can only stay or rise.`}
          table={{ columns: ["Round", "Best CV score"], rows: scores.map((s, i) => [i + 1, Number(s.toFixed(4))]) }}>
          {scores.length ? <ScoreByRound scores={scores} /> : <div className="muted small">Not recorded for this run.</div>}
        </ChartCard>
        <ChartCard title="Overfitting gap per family and round" sub="Hollow = training score, filled = validation score. A shorter segment after adaptation = less overfitting."
          tools={<JupyterButton url={nbUrl} small label="Notebook" />}
          table={{ columns: ["Family", "Round", "Validation", "Train", "Gap"], rows: best.map((t) => [familyLabel(t.family), t.round, Number(fmt(t.mean_val)), Number(fmt(t.mean_train)), Number(fmt(t.gap))]) }}>
          <GapDumbbell trials={trials} />
        </ChartCard>
      </div>

      <Card>
        <div className="card-head"><div><h3>Candidates and diagnoses</h3><div className="sub">Best configuration per family over all rounds, with Agent 1's diagnosis in every round.</div></div></div>
        <div className="card-body table-wrap">
          <table className="t">
            <thead><tr><th>Family</th><th className="num">Best CV</th><th className="num">Gap</th><th className="num">Found in round</th><th className="num">Trials</th>{[...new Set(diagByRound.map((d) => d.round))].map((r) => <th key={r}>Round {r} diagnosis</th>)}</tr></thead>
            <tbody>
              {[...output.candidates].sort((a, b) => b.mean_val - a.mean_val).map((c) => (
                <tr key={c.family}>
                  <td><span className="row" style={{ gap: 6 }}><span style={{ width: 9, height: 9, borderRadius: 3, background: familyColor(c.family) }} /><b>{familyLabel(c.family)}</b>{c.family === output.model_family && <span className="chip">selected</span>}</span></td>
                  <td className="num">{fmt(c.mean_val)}</td>
                  <td className="num">{fmt(c.gap, 3)}</td>
                  <td className="num">{c.round ?? 1}</td>
                  <td className="num">{trials.filter((t) => t.family === c.family).length || "–"}</td>
                  {[...new Set(diagByRound.map((d) => d.round))].map((r) => {
                    const e = diagByRound.find((d) => d.round === r && d.decision.family === c.family);
                    return <td key={r}>{e ? <span className="row" style={{ gap: 4 }}>{(e.decision.labels as string[]).map((l) => <DiagnosisChip key={l} label={l} />)}</span> : <span className="muted small">dropped</span>}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
