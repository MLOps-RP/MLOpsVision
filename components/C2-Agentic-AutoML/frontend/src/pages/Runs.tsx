import { Link, useNavigate } from "react-router-dom";
import { Play } from "lucide-react";
import { api } from "../api";
import { useData } from "../hooks";
import { familyColor, familyLabel, fmt, when } from "../theme";
import { Card, Empty, ErrorBox, Loading, PageHead, StatusBadge } from "../components/ui";

export default function Runs() {
  const runs = useData(() => api.runs(), [], 4000);
  const navigate = useNavigate();
  return (
    <div>
      <PageHead title="Runs & results" sub="Every C2 run, live or finished. Open a run to see the agentic workflow, charts, search history, reasoning and outputs."
        actions={<Link className="btn primary" to="/runs/new"><Play size={15} /> New run</Link>} />
      <Card>
        {runs.loading && !runs.data && <Loading />}
        {runs.error && <div className="card-body"><ErrorBox error={runs.error} /></div>}
        {runs.data && !runs.data.length && <Empty>No runs yet — start one.</Empty>}
        {runs.data && runs.data.length > 0 && (
          <div className="table-wrap">
            <table className="t">
              <thead>
                <tr>
                  <th>Status</th><th>Run</th><th>Dataset</th><th>Selected model</th>
                  <th className="num">Best CV</th><th className="num">Hold-out ROC-AUC</th><th className="num">Recall</th>
                  <th className="num">Rounds</th><th className="num">Trials</th><th>Stop</th><th>Created</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.map((r) => (
                  <tr key={r.run_id} className="click" onClick={() => navigate(`/runs/${r.run_id}`)}>
                    <td><StatusBadge status={r.status} /></td>
                    <td className="mono small">{r.run_id}</td>
                    <td>{r.dataset_id}</td>
                    <td>{r.model_family ? <span className="row" style={{ gap: 6 }}><span style={{ width: 9, height: 9, borderRadius: 3, background: familyColor(r.model_family) }} />{familyLabel(r.model_family)}</span> : r.error ? <span className="small" style={{ color: "#a32" }}>{r.error.slice(0, 50)}</span> : "–"}</td>
                    <td className="num">{fmt(r.best_cv)}</td>
                    <td className="num">{fmt(r.holdout?.roc_auc)}</td>
                    <td className="num">{fmt(r.holdout?.recall, 3)}</td>
                    <td className="num">{r.rounds ?? "–"}</td>
                    <td className="num">{r.total_trials ?? "–"}</td>
                    <td className="small ink2">{r.stop_reason ?? "–"}</td>
                    <td className="small muted">{when(r.created)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
