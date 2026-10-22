import { FormEvent, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Database, Play, Upload } from "lucide-react";
import { api } from "../api";
import { useData } from "../hooks";
import { compact, pct } from "../theme";
import { Card, ChartCard, DataTable, ErrorBox, JupyterButton, Loading, PageHead, Stat } from "../components/ui";
import { ClassBalance } from "../charts/CompareCharts";
import { HBars } from "../charts/SearchCharts";
import { useNotebook } from "../App";

function UploadCard({ onDone }: { onDone: (id: string) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [target, setTarget] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pick = async (f: File | null) => {
    setFile(f);
    setColumns([]);
    setError(null);
    if (!f) return;
    setDatasetId(f.name.replace(/\.[^.]+$/, "").toLowerCase().replace(/[^a-z0-9_-]+/g, "_").slice(0, 40));
    if (f.name.toLowerCase().endsWith(".csv")) {
      const head = (await f.slice(0, 64 * 1024).text()).split(/\r?\n/)[0];
      const cols = head.split(",").map((c) => c.trim().replace(/^"|"$/g, ""));
      setColumns(cols);
      setTarget(cols[cols.length - 1] ?? "");
    }
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError(null);
    const form = new FormData();
    form.append("file", file);
    form.append("target", target);
    form.append("task_type", "binary");
    form.append("dataset_id", datasetId);
    try {
      const m = await api.uploadDataset(form);
      onDone(m.dataset_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <div className="card-head"><div><h3>Add a dataset</h3><div className="sub">CSV or Excel with a labelled target column. C2 builds a dummy C1 output (split, basic cleaning, FQS) until C1 is integrated.</div></div></div>
      <form className="card-body stack" onSubmit={submit}>
        <div className="field">
          <label>File</label>
          <input className="input" type="file" accept=".csv,.xlsx,.xls" onChange={(e) => pick(e.target.files?.[0] ?? null)} style={{ paddingTop: 7 }} />
        </div>
        <div className="grid g2">
          <div className="field">
            <label>Target column</label>
            {columns.length ? (
              <select className="select" value={target} onChange={(e) => setTarget(e.target.value)}>
                {columns.map((c) => <option key={c}>{c}</option>)}
              </select>
            ) : (
              <input className="input" value={target} onChange={(e) => setTarget(e.target.value)} placeholder="e.g. default" />
            )}
          </div>
          <div className="field">
            <label>Dataset id</label>
            <input className="input" value={datasetId} onChange={(e) => setDatasetId(e.target.value)} placeholder="loan_default" />
          </div>
        </div>
        <div className="small muted">Task type: binary classification (multiclass and regression come in Step 7).</div>
        {error && <ErrorBox error={error} />}
        <div><button className="btn primary" disabled={!file || !target || !datasetId || busy}><Upload size={15} /> {busy ? "Preparing…" : "Upload & prepare"}</button></div>
      </form>
    </Card>
  );
}

export default function DataPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const list = useData(() => api.datasets(), []);
  const selectedId = id ?? list.data?.[0]?.dataset_id;
  const detail = useData(selectedId ? () => api.dataset(selectedId) : null, [selectedId]);
  const nbUrl = useNotebook("input");
  const d = detail.data;

  return (
    <div>
      <PageHead
        title="Input data from C1"
        sub="C2 consumes C1's output: cleaned features, a train/test split and Feature Quality Scores. The test split is never used for tuning."
        actions={<JupyterButton url={nbUrl} label="Explore in Jupyter" />}
      />
      <div className="grid g-1-2" style={{ alignItems: "start" }}>
        <div className="stack">
          <Card>
            <div className="card-head"><div><h3>Datasets</h3><div className="sub">{list.data?.length ?? 0} prepared C1 outputs</div></div></div>
            <div className="card-body" style={{ display: "grid", gap: 8 }}>
              {list.error && <ErrorBox error={list.error} />}
              {list.data?.map((ds) => (
                <Link key={ds.dataset_id} to={`/data/${ds.dataset_id}`} className="agent-card"
                  style={{ borderColor: ds.dataset_id === selectedId ? "var(--ink)" : undefined, display: "block" }}>
                  <div className="row" style={{ justifyContent: "space-between" }}>
                    <b className="row" style={{ gap: 6 }}><Database size={14} /> {ds.dataset_id}</b>
                    <span className="chip">{ds.task_type}</span>
                  </div>
                  <div className="small muted" style={{ marginTop: 6 }}>
                    {compact(ds.n_train)} train · {compact(ds.n_test)} test · {ds.n_features} features · positive “{ds.positive_class}” {pct(ds.class_distribution?.["1"])}
                  </div>
                </Link>
              ))}
            </div>
          </Card>
          <UploadCard onDone={(newId) => { list.reload(); navigate(`/data/${newId}`); }} />
        </div>

        <div className="stack">
          {detail.loading && !d && <Loading />}
          {detail.error && <ErrorBox error={detail.error} />}
          {d && (
            <>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <h2 style={{ fontSize: 18 }}>{d.manifest.dataset_id} <span className="muted" style={{ fontWeight: 400 }}>· target “{d.manifest.target}”</span></h2>
                <Link className="btn primary" to={`/runs/new?dataset=${d.manifest.dataset_id}`}><Play size={15} /> Run C2 on this dataset</Link>
              </div>
              <div className="grid g4">
                <Stat label="Training rows" value={d.profile.n_rows.toLocaleString()} note={`size class: ${d.profile.size_class}`} />
                <Stat label="Features" value={d.profile.n_features} note={`${d.profile.n_categorical} categorical (${pct(d.profile.categorical_ratio, 0)})`} />
                <Stat label="Imbalance ratio" value={d.profile.imbalance_ratio?.toFixed(3) ?? "–"} note="minority ÷ majority" />
                <Stat label="Metric" value={d.profile.metric.replace("_", "-").toUpperCase()} note="chosen by task type" />
              </div>
              {d.manifest.class_distribution && (
                <Card>
                  <div className="card-head"><div><h3>Class balance (training split)</h3><div className="sub">Below 0.2 imbalance ratio, Agent 1 checks minority recall and can diagnose class imbalance.</div></div></div>
                  <div className="card-body"><ClassBalance dist={d.manifest.class_distribution} positive={d.manifest.positive_class} /></div>
                </Card>
              )}
              <ChartCard title="Feature Quality Score (from C1)" sub="Dummy C1: mutual information scaled to 0–1. Low scores across all features support a “weak features” diagnosis."
                table={{ columns: ["Feature", "Type", "FQS"], rows: d.fqs.map((f) => [f.feature, f.type, f.score]) }}>
                <HBars items={d.fqs.map((f) => ({ label: f.feature, value: f.score, note: f.type === "categorical" ? "cat" : undefined }))} max={1} />
              </ChartCard>
              <Card>
                <div className="card-head"><div><h3>Sample rows</h3><div className="sub">First 12 training rows after the dummy C1 cleaning (categoricals are integer-coded).</div></div></div>
                <div className="card-body"><DataTable spec={{ columns: d.sample.columns, rows: d.sample.rows }} /></div>
              </Card>
              <Card pad>
                <div className="small muted">C1 decision log</div>
                {d.manifest.decision_log.map((l, i) => <div key={i} className="small" style={{ marginTop: 4 }}>• {l}</div>)}
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
