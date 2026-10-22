import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Database, Play } from "lucide-react";
import { api } from "../api";
import { useData } from "../hooks";
import { compact, pct } from "../theme";
import { Card, ErrorBox, PageHead } from "../components/ui";

function Slider({ label, value, min, max, step = 1, onChange, hint }: {
  label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void; hint: string;
}) {
  return (
    <div className="field">
      <label className="row" style={{ justifyContent: "space-between" }}>{label}<b style={{ color: "var(--ink)", fontSize: 14 }}>{value}</b></label>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      <span className="hint">{hint}</span>
    </div>
  );
}

export default function NewRun() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const datasets = useData(() => api.datasets(), []);
  const [dataset, setDataset] = useState(params.get("dataset") ?? "");
  const [trials, setTrials] = useState(90);
  const [folds, setFolds] = useState(5);
  const [rounds, setRounds] = useState(3);
  const [adaptRanges, setAdaptRanges] = useState(true);
  const [adaptFamilies, setAdaptFamilies] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!dataset && datasets.data?.length) setDataset(datasets.data.find((d) => d.dataset_id === "diabetes")?.dataset_id ?? datasets.data[0].dataset_id);
  }, [datasets.data, dataset]);

  const ds = datasets.data?.find((d) => d.dataset_id === dataset);
  const perRound = Math.floor(trials / rounds);
  // rough guide from test runs on an Apple M4: ~0.13 s per trial-fold on small data, ~0.6 s on 30k rows
  const secs = ds ? Math.round(trials * folds * (ds.n_train > 5000 ? 0.6 : 0.13)) : 0;

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const { run_id } = await api.startRun({
        dataset_id: dataset, total_trials: trials, folds, max_rounds: rounds,
        adapt_ranges: adaptRanges, adapt_families: adaptFamilies,
      });
      navigate(`/runs/${run_id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  return (
    <div>
      <PageHead title="Start a C2 run" sub="Runs the full LangGraph workflow: Agent 1 selects families, Agent 2 tunes and adapts, Agent 1 diagnoses and decides, Agent 3 selects. You can watch every step live." />
      <div className="grid g-2-1" style={{ alignItems: "start" }}>
        <div className="stack">
          <Card>
            <div className="card-head"><div><h3>1 · Input dataset (C1 output)</h3></div></div>
            <div className="card-body grid g2">
              {datasets.data?.map((d) => (
                <button key={d.dataset_id} onClick={() => setDataset(d.dataset_id)} className="agent-card"
                  style={{ textAlign: "left", cursor: "pointer", borderColor: d.dataset_id === dataset ? "var(--ink)" : undefined, borderWidth: d.dataset_id === dataset ? 1.5 : 1 }}>
                  <b className="row" style={{ gap: 6 }}><Database size={14} /> {d.dataset_id}</b>
                  <div className="small muted" style={{ marginTop: 6 }}>
                    {compact(d.n_train)} rows · {d.n_features} features · positive {pct(d.class_distribution?.["1"])}
                  </div>
                </button>
              ))}
            </div>
          </Card>
          <Card>
            <div className="card-head"><div><h3>2 · Budget</h3><div className="sub">The same total budget is used by the static baseline in comparisons, so results stay fair.</div></div></div>
            <div className="card-body grid g3">
              <Slider label="Total trials" value={trials} min={15} max={300} step={15} onChange={setTrials} hint={`≈ ${perRound} trials per round`} />
              <Slider label="CV folds" value={folds} min={3} max={5} onChange={setFolds} hint="3 is faster on large data" />
              <Slider label="Max rounds" value={rounds} min={1} max={5} onChange={setRounds} hint="1 = no adaptive loop" />
            </div>
          </Card>
          <Card>
            <div className="card-head"><div><h3>3 · Adaptive behaviour</h3><div className="sub">Switch parts off to run an ablation and measure what each part contributes.</div></div></div>
            <div className="card-body grid g2">
              <label className="switch">
                <input type="checkbox" checked={adaptRanges} onChange={(e) => setAdaptRanges(e.target.checked)} />
                <div><b>Adapt search ranges</b><span>Agent 2 changes ranges after each diagnosis (e.g. shallower trees after overfitting, class weights after imbalance).</span></div>
              </label>
              <label className="switch">
                <input type="checkbox" checked={adaptFamilies} onChange={(e) => setAdaptFamilies(e.target.checked)} />
                <div><b>Adapt model families</b><span>Agent 1 may drop a family that is clearly worse than the best one and give its trials to others.</span></div>
              </label>
            </div>
          </Card>
        </div>
        <Card className="card-pad" >
          <h3 style={{ fontSize: 15 }}>Summary</h3>
          <dl className="kv" style={{ marginTop: 14 }}>
            <dt>Dataset</dt><dd><b>{dataset || "–"}</b></dd>
            <dt>Trials</dt><dd>{trials} in up to {rounds} round{rounds > 1 ? "s" : ""}</dd>
            <dt>Cross-validation</dt><dd>{folds} folds (training split only)</dd>
            <dt>Range adaptation</dt><dd>{adaptRanges ? "on" : "off (ablation)"}</dd>
            <dt>Family adaptation</dt><dd>{adaptFamilies ? "on" : "off (ablation)"}</dd>
            <dt>Estimated time</dt><dd>{secs ? `~${Math.max(1, Math.round(secs / 60))} min` : "–"}</dd>
          </dl>
          {error && <div className="mt"><ErrorBox error={error} /></div>}
          <button className="btn primary mt" style={{ width: "100%", justifyContent: "center", height: 42 }} disabled={!dataset || busy} onClick={start}>
            <Play size={16} /> {busy ? "Starting…" : "Start run"}
          </button>
          <div className="small muted mt">Only one run at a time. Tip for demos: Pima diabetes with 90 trials takes about one minute.</div>
        </Card>
      </div>
    </div>
  );
}
