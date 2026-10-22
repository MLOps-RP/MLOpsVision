import { Link } from "react-router-dom";
import { ArrowRight, Download, FileJson, Package, Undo2 } from "lucide-react";
import { api, C2Output } from "../../api";
import { familyLabel, fmt } from "../../theme";
import { Card, JsonView } from "../../components/ui";

const FILES = [
  { name: "c2_output.json", label: "C2Output (contract for C3)" },
  { name: "model.joblib", label: "Selected model" },
  { name: "reasoning_log.json", label: "Reasoning log" },
  { name: "search_history.csv", label: "Search history (all trials)" },
];

export default function OutputsTab({ output }: { output: C2Output }) {
  return (
    <div className="stack">
      <div className="grid g3">
        <Card pad>
          <div className="row"><Undo2 size={16} /><b>Input · from C1</b></div>
          <dl className="kv" style={{ marginTop: 12, gridTemplateColumns: "110px 1fr" }}>
            <dt>Dataset</dt><dd><Link to={`/data/${output.dataset_id}`} style={{ textDecoration: "underline" }}>{output.dataset_id}</Link></dd>
            <dt>Task</dt><dd>{output.task_type}</dd>
            <dt>Metric</dt><dd>{output.metric}</dd>
          </dl>
        </Card>
        <Card pad>
          <div className="row"><Package size={16} /><b>Output · to C3</b></div>
          <dl className="kv" style={{ marginTop: 12, gridTemplateColumns: "110px 1fr" }}>
            <dt>Model</dt><dd>{familyLabel(output.model_family)} · {output.model_size_kb} KB</dd>
            <dt>Hold-out AUC</dt><dd>{fmt(output.metrics_holdout.roc_auc)}</dd>
            <dt>Evidence</dt><dd>{output.total_trials} trials · {output.rounds} rounds</dd>
          </dl>
        </Card>
        <Card pad>
          <div className="row"><ArrowRight size={16} /><b>Feedback · referrals to C1</b></div>
          <div style={{ marginTop: 12 }} className="small">
            {output.referrals_to_c1.length ? output.referrals_to_c1.map((r, i) => <div key={i} className="callout warn" style={{ padding: "8px 10px", marginBottom: 6 }}>{r}</div>)
              : <span className="muted">No feature issues referred to C1 in this run.</span>}
          </div>
        </Card>
      </div>

      <div className="grid g-1-2" style={{ alignItems: "start" }}>
        <Card>
          <div className="card-head"><div><h3>Downloads</h3><div className="sub" style={{ overflowWrap: "anywhere" }}>Files written to outputs/{output.dataset_id}/{output.run_id}/</div></div></div>
          <div className="card-body" style={{ display: "grid", gap: 8 }}>
            {FILES.map((f) => (
              <a key={f.name} className="btn" href={api.fileUrl(output.run_id, f.name)} style={{ justifyContent: "space-between" }}>
                <span className="row" style={{ gap: 8 }}><FileJson size={15} /> {f.label}</span>
                <Download size={15} />
              </a>
            ))}
            <div className="small muted" style={{ marginTop: 6 }}>Tuned hyperparameters</div>
            <JsonView value={output.hyperparameters} maxHeight={220} />
          </div>
        </Card>
        <Card>
          <div className="card-head"><div><h3>C2Output (what C3 receives)</h3><div className="sub">Validated by the Pydantic contract in schemas.py.</div></div></div>
          <div className="card-body"><JsonView value={output} maxHeight={560} /></div>
        </Card>
      </div>
    </div>
  );
}
