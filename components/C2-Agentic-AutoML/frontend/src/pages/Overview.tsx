import { Link } from "react-router-dom";
import {
  ArrowRight, BookOpen, Brain, CheckCircle2, Circle, Database, Gauge, Play, SlidersHorizontal, Trophy,
} from "lucide-react";
import { api } from "../api";
import { useData } from "../hooks";
import { AGENTS, familyLabel, fmt, when } from "../theme";
import { AgentBadge, Card, Stat } from "../components/ui";
import { useJupyter } from "../App";

const ROUND_STEPS = [
  { agent: "agent2", title: "Tune", text: "Optuna searches each family inside its current ranges." },
  { agent: "tools", title: "Measure", text: "Trusted tools compute evidence: train–val gap, minority recall, lift." },
  { agent: "agent1", title: "Diagnose", text: "Names the problem: overfitting, imbalance, weak features…" },
  { agent: "agent1", title: "Decide", text: "Continue with a plan, or stop (converged, budget, max rounds)." },
  { agent: "agent2", title: "Adapt", text: "Turns the plan into new search ranges, then the loop repeats." },
];

const PROGRESS: [string, string, "done" | "next" | "todo"][] = [
  ["Step 0–1", "Setup, dummy C1 data, Agent 1 levels 1–3", "done"],
  ["Step 2", "Agent 2: Optuna, pruning, baselines B0/B1/B2", "done"],
  ["Step 3", "LangGraph workflow + Agent 3 level 1", "done"],
  ["Step 4", "Adaptive loop: diagnosis-conditioned ranges ⭐", "done"],
  ["Step 8", "API + this dashboard", "done"],
  ["Step 5", "Bounded LLM reasoning (rules as fallback)", "next"],
  ["Step 6", "Agent 3: latency, size, Pareto front", "todo"],
  ["Step 7", "Multiclass + regression", "todo"],
  ["Step 9–10", "C1/C3/C4 integration, full evaluation", "todo"],
];

export default function Overview() {
  const runs = useData(() => api.runs(), []);
  const datasets = useData(() => api.datasets(), []);
  const { jupyterUrl } = useJupyter();
  const latest = runs.data?.find((r) => r.status === "completed" && r.holdout);

  return (
    <div className="stack">
      <Card className="hero">
        <div className="row" style={{ gap: 8, marginBottom: 10 }}>
          <span className="badge outline">Component C2</span>
          <span className="badge outline">Supervised · tabular · binary first</span>
        </div>
        <h1>Agentic AutoML Model Generation & Selection Engine</h1>
        <p>
          C2 trains and tunes candidate models and <b>reasons about why a round succeeded or failed</b>.
          Agents diagnose problems from structured evidence and <b>change the next search</b> — which model
          families to try and which hyperparameter ranges to search. Trusted tools do every computation; every
          decision is logged with its evidence.
        </p>
        <div className="row" style={{ marginTop: 18 }}>
          <Link className="btn primary" to="/runs/new"><Play size={15} /> Start a run</Link>
          <Link className="btn" to="/data"><Database size={15} /> Input data</Link>
          {latest && <Link className="btn" to={`/runs/${latest.run_id}`}><Trophy size={15} /> Latest result</Link>}
          <a className="btn jupyter" href={`${jupyterUrl}/lab`} target="_blank" rel="noreferrer"><BookOpen size={15} color="#e46e0e" /> JupyterLab</a>
        </div>
      </Card>

      <Card>
        <div className="card-head"><div><h3>Where C2 sits in the platform</h3><div className="sub">Inputs come from C1, the selected model goes to C3 for validation. C3 rejections and C4 retrain requests come back to C2.</div></div></div>
        <div className="card-body">
          <div className="flow">
            <div className="comp-box">
              <b>C1 · Feature Intelligence</b>
              <p className="small muted" style={{ marginTop: 6 }}>Cleaned features, preprocessing pipeline, Feature Quality Scores.</p>
              <p className="small" style={{ marginTop: 8 }}><span className="chip">dummy until integration</span></p>
            </div>
            <div className="arrow"><ArrowRight /></div>
            <div className="comp-box main">
              <b>C2 · Agentic AutoML</b>
              <div className="grid g3" style={{ marginTop: 10, gap: 10 }}>
                {[
                  { k: "agent1" as const, icon: <Brain size={18} />, text: "Chooses model families, diagnoses failures (overfitting, imbalance, weak features), decides to continue or stop." },
                  { k: "agent2" as const, icon: <SlidersHorizontal size={18} />, text: "Tunes with Optuna (TPE + pruning) and adapts the search ranges to the diagnosis, warm-starting each round." },
                  { k: "agent3" as const, icon: <Gauge size={18} />, text: "Selects the best model over all rounds, tests it once on hold-out data, packages the output for C3." },
                ].map(({ k, icon, text }) => (
                  <div className="agent-card" key={k} style={{ borderTop: `3px solid ${AGENTS[k].color}` }}>
                    <div className="row" style={{ color: AGENTS[k].color }}>{icon}<AgentBadge agent={k} /></div>
                    <h4>{AGENTS[k].role}</h4>
                    <p>{text}</p>
                  </div>
                ))}
              </div>
            </div>
            <div className="arrow"><ArrowRight /></div>
            <div className="comp-box">
              <b>C3 · Validation Gate</b>
              <p className="small muted" style={{ marginTop: 6 }}>Receives the model, tuned parameters, evidence, search history and reasoning.</p>
              <p className="small" style={{ marginTop: 8 }}><span className="chip">C2Output contract ready</span></p>
            </div>
          </div>
        </div>
      </Card>

      <Card>
        <div className="card-head"><div><h3>One round of the adaptive loop — the novelty</h3><div className="sub">Diagnosis-conditioned changes to candidate selection and hyperparameter ranges.</div></div></div>
        <div className="card-body">
          <div className="grid" style={{ gridTemplateColumns: "repeat(5, minmax(0,1fr))", gap: 10 }}>
            {ROUND_STEPS.map((s, i) => (
              <div key={i} className="agent-card" style={{ background: AGENTS[s.agent as keyof typeof AGENTS].tint, border: 0 }}>
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <span style={{ fontSize: 22, fontWeight: 700, color: AGENTS[s.agent as keyof typeof AGENTS].color }}>{i + 1}</span>
                  <AgentBadge agent={s.agent} />
                </div>
                <h4>{s.title}</h4>
                <p>{s.text}</p>
              </div>
            ))}
          </div>
        </div>
      </Card>

      <div className="grid g-2-1" style={{ alignItems: "start" }}>
        <div className="stack">
          <div className="grid g4">
            <Stat label="Datasets ready" value={datasets.data?.length ?? "–"} note="dummy C1 outputs" />
            <Stat label="C2 runs" value={runs.data?.filter((r) => r.status === "completed").length ?? "–"} note="completed" />
            <Stat label="Latest model" value={<span style={{ fontSize: 19 }}>{latest ? familyLabel(latest.model_family ?? "") : "–"}</span>} note={latest ? latest.dataset_id : ""} />
            <Stat label="Hold-out ROC-AUC" value={fmt(latest?.holdout?.roc_auc, 3)} note={latest ? `recall ${fmt(latest.holdout?.recall, 3)}` : ""} />
          </div>
          {latest && (
            <Card pad>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <div>
                  <div className="small muted">Latest completed run · {when(latest.created)}</div>
                  <div style={{ fontWeight: 620, marginTop: 4 }} className="mono">{latest.run_id}</div>
                  <div className="small ink2" style={{ marginTop: 4 }}>
                    {latest.rounds} rounds · {latest.total_trials} trials · stop: {latest.stop_reason ?? "–"}
                  </div>
                </div>
                <Link className="btn" to={`/runs/${latest.run_id}`}>Open run <ArrowRight size={15} /></Link>
              </div>
            </Card>
          )}
        </div>
        <Card>
          <div className="card-head"><div><h3>Implementation progress</h3><div className="sub">Roadmap steps of the component</div></div></div>
          <div className="card-body" style={{ display: "grid", gap: 8 }}>
            {PROGRESS.map(([step, text, status]) => (
              <div key={step} className="row" style={{ alignItems: "flex-start", flexWrap: "nowrap" }}>
                {status === "done" ? <CheckCircle2 size={16} color="#0ca30c" style={{ flex: "none", marginTop: 1 }} />
                  : <Circle size={16} color={status === "next" ? "#2a78d6" : "#c3c2b7"} style={{ flex: "none", marginTop: 1 }} />}
                <div className="small"><b>{step}</b> <span className="ink2">{text}</span>{status === "next" && <span className="chip" style={{ marginLeft: 6 }}>next</span>}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}
