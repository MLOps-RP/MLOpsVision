import { useState } from "react";
import { api } from "../api";
import { useData } from "../hooks";
import { METHOD_COLORS, METHOD_LABELS, familyLabel, fmt, when } from "../theme";
import { Card, ChartCard, Empty, ErrorBox, JupyterButton, Loading, PageHead } from "../components/ui";
import { MethodBars } from "../charts/CompareCharts";
import { useNotebook } from "../App";

export default function Compare() {
  const data = useData(() => api.comparisons(), []);
  const [idx, setIdx] = useState(0);
  const nbUrl = useNotebook("comparison");
  const baselineNb = useNotebook("baselines");
  const c = data.data?.[idx];
  const metricCols = ["roc_auc", "f1", "recall", "precision"].filter((m) => c?.rows.some((r) => typeof r[m] === "number"));

  return (
    <div>
      <PageHead
        title="Baseline comparison"
        sub="Static Optuna (B2) vs the adaptive C2 method and two ablations, all with the same trial budget, folds and seed. Created with scripts/compare_adaptive.py."
        actions={<><JupyterButton url={baselineNb} label="Baselines B0–B2" /><JupyterButton url={nbUrl} label="Comparison notebook" /></>}
      />
      {data.loading && !data.data && <Loading />}
      {data.error && <ErrorBox error={data.error} />}
      {data.data && !data.data.length && (
        <Card><Empty>No comparisons yet. Run <code>python scripts/compare_adaptive.py diabetes --trials 90</code></Empty></Card>
      )}
      {c && (
        <div className="stack">
          <div className="row">
            <span className="small muted">Comparison:</span>
            <div className="seg">
              {data.data!.map((x, i) => (
                <button key={x.file} className={i === idx ? "on" : ""} onClick={() => setIdx(i)}>{x.dataset_id} · {when(x.created)}</button>
              ))}
            </div>
          </div>
          <div className="grid g-2-1">
            <ChartCard title={`Hold-out metrics · ${c.dataset_id}`} sub="All metrics share the 0–1 scale. On imbalanced data, recall and F1 show whether the rare class is found."
              table={{ columns: ["Method", ...metricCols], rows: c.rows.map((r) => [METHOD_LABELS[r.method] ?? r.method, ...metricCols.map((m) => Number(fmt(r[m])))]) }}>
              <MethodBars rows={c.rows} metrics={metricCols} />
            </ChartCard>
            <Card>
              <div className="card-head"><div><h3>How to read this</h3></div></div>
              <div className="card-body small ink2" style={{ display: "grid", gap: 10, lineHeight: 1.55 }}>
                {Object.entries(METHOD_LABELS).filter(([k]) => c.rows.some((r) => r.method === k)).map(([k, label]) => (
                  <div key={k} className="row" style={{ alignItems: "flex-start", flexWrap: "nowrap" }}>
                    <span style={{ width: 10, height: 10, borderRadius: 3, background: METHOD_COLORS[k], flex: "none", marginTop: 5 }} />
                    <div><b style={{ color: "var(--ink)" }}>{label}</b><br />{{
                      B2_static: "Optuna TPE over all families with fixed ranges, one round — standard AutoML.",
                      A_full: "Agent 1 + Agent 2 adaptive loop: diagnosis-conditioned families and ranges.",
                      A_no_ranges: "Same loop, but ranges never change → isolates the effect of range adaptation.",
                      A_no_families: "Same loop, but no family is dropped → isolates the effect of family adaptation.",
                    }[k]}</div>
                  </div>
                ))}
                <div className="callout warn" style={{ marginTop: 6 }}>Single-seed results are indicative only. Use <code>--seeds 5</code> on several datasets and a Wilcoxon test for thesis claims.</div>
              </div>
            </Card>
          </div>
          <Card>
            <div className="card-head"><div><h3>All results</h3><div className="sub">{c.file}</div></div></div>
            <div className="card-body table-wrap">
              <table className="t">
                <thead><tr><th>Method</th><th className="num">Seed</th><th>Selected</th><th className="num">Rounds</th><th className="num">Trials</th><th className="num">Best CV</th>{metricCols.map((m) => <th key={m} className="num">{m}</th>)}<th>Stop</th><th className="num">Time (s)</th></tr></thead>
                <tbody>
                  {c.rows.map((r, i) => {
                    const bestOf = (m: string) => Math.max(...c.rows.map((x) => Number(x[m]) || 0));
                    return (
                      <tr key={i}>
                        <td><span className="row" style={{ gap: 6 }}><span style={{ width: 9, height: 9, borderRadius: 3, background: METHOD_COLORS[r.method] ?? "#898781" }} /><b>{METHOD_LABELS[r.method] ?? r.method}</b></span></td>
                        <td className="num">{r.seed}</td>
                        <td>{familyLabel(r.family)}</td>
                        <td className="num">{r.rounds}</td>
                        <td className="num">{r.trials}</td>
                        <td className="num">{fmt(r.best_cv)}</td>
                        {metricCols.map((m) => <td key={m} className="num" style={{ fontWeight: Number(r[m]) === bestOf(m) ? 700 : 400 }}>{fmt(r[m])}</td>)}
                        <td className="small ink2">{r.stop}</td>
                        <td className="num">{r.time_s}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <div className="small muted" style={{ marginTop: 8 }}>Bold = best value in the column.</div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
