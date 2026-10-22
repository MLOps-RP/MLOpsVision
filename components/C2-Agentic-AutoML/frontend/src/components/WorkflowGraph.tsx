import { RunDetail, RunEvent } from "../api";
import { AGENTS, AgentKey, DIAGNOSIS, familyLabel, fmt } from "../theme";

type NodeState = "pending" | "active" | "done";

export const NODES: { id: string; title: string; owner: AgentKey; col: number; hint: string }[] = [
  { id: "load_input", title: "Load input", owner: "tools", col: 1, hint: "Read the C1 output and the config" },
  { id: "profile_task", title: "Profile data", owner: "tools", col: 2, hint: "Rows, features, imbalance, metric" },
  { id: "agent1_select", title: "Select families", owner: "agent1", col: 3, hint: "Agent 1 picks model families for the dataset" },
  { id: "agent2_hpo", title: "Tune (Optuna)", owner: "agent2", col: 4, hint: "Agent 2 tunes every family inside its search ranges" },
  { id: "evaluate", title: "Collect signals", owner: "tools", col: 5, hint: "Train/val gap, minority recall, lift, ..." },
  { id: "agent1_diagnose", title: "Diagnose & decide", owner: "agent1", col: 6, hint: "Agent 1 names the problem, plans actions, continues or stops" },
  { id: "agent2_adapt", title: "Adapt ranges", owner: "agent2", col: 6, hint: "Agent 2 turns actions into new search ranges" },
  { id: "agent3_select", title: "Select best model", owner: "agent3", col: 7, hint: "Agent 3 picks the best over all rounds, tests it once" },
  { id: "package_output", title: "Package output", owner: "tools", col: 8, hint: "C2Output, model, history, reasoning log" },
];

const LANES: AgentKey[] = ["agent1", "agent2", "agent3", "tools"];
const EDGES: [string, string, string?][] = [
  ["load_input", "profile_task"], ["profile_task", "agent1_select"], ["agent1_select", "agent2_hpo"],
  ["agent2_hpo", "evaluate"], ["evaluate", "agent1_diagnose"], ["agent1_diagnose", "agent2_adapt", "continue"],
  ["agent2_adapt", "agent2_hpo", "loop"], ["agent1_diagnose", "agent3_select", "stop"], ["agent3_select", "package_output"],
];
const NEXT: Record<string, (e: RunEvent) => string | null> = {
  load_input: () => "profile_task",
  profile_task: () => "agent1_select",
  agent1_select: () => "agent2_hpo",
  agent2_hpo: () => "evaluate",
  evaluate: () => "agent1_diagnose",
  agent1_diagnose: (e) => (e.data.stop_reason ? "agent3_select" : "agent2_adapt"),
  agent2_adapt: () => "agent2_hpo",
  agent3_select: () => "package_output",
  package_output: () => null,
};
const SHORT: Record<string, string> = {
  random_forest: "RF", xgboost: "XGB", lightgbm: "LGBM", catboost: "CatBoost", logistic_regression: "LR", ridge: "Ridge",
};

export interface WorkflowView {
  state: Record<string, NodeState>;
  counts: Record<string, number>;
  subs: Record<string, string>;
  activeEdge: [string, string] | null;
  last: Record<string, RunEvent>;
  round: number;
}

/** Derive node states and live subtitles from the run's events. */
export function workflowView(run: RunDetail): WorkflowView {
  const state: Record<string, NodeState> = {};
  const counts: Record<string, number> = {};
  const subs: Record<string, string> = {};
  const last: Record<string, RunEvent> = {};
  NODES.forEach((n) => (state[n.id] = "pending"));
  let round = 1;

  if (!run.events.length && run.output) {
    // finished run loaded from disk: everything ran
    const r = run.output.rounds;
    NODES.forEach((n) => (state[n.id] = n.id === "agent2_adapt" && r < 2 ? "pending" : "done"));
    counts.agent2_hpo = r; counts.evaluate = r; counts.agent1_diagnose = r; counts.agent2_adapt = Math.max(r - 1, 0);
    subs.agent3_select = familyLabel(run.output.model_family);
    subs.agent1_diagnose = `stop: ${run.output.stop_reason ?? "–"}`;
    subs.agent2_hpo = `${run.output.total_trials} trials`;
    return { state, counts, subs, activeEdge: null, last, round: r };
  }

  for (const e of run.events) {
    state[e.node] = "done";
    counts[e.node] = (counts[e.node] ?? 0) + 1;
    last[e.node] = e;
    const d = e.data;
    if (e.node === "load_input") subs.load_input = `budget ${d.total_trials} trials`;
    if (e.node === "profile_task") subs.profile_task = `${d.n_rows?.toLocaleString()} rows · ${d.size_class}`;
    if (e.node === "agent1_select") subs.agent1_select = (d.families as string[]).map((f) => SHORT[f] ?? f).join(", ");
    if (e.node === "agent2_hpo") {
      round = d.round;
      const n = Object.values(d.budget as Record<string, number>).reduce((a, b) => a + b, 0);
      subs.agent2_hpo = `round ${d.round} · ${n} trials`;
    }
    if (e.node === "evaluate") subs.evaluate = `${Object.keys(d.signals).length} families measured`;
    if (e.node === "agent1_diagnose") {
      const problems = (d.diagnoses as { labels: string[] }[]).flatMap((x) => x.labels).filter((l) => l !== "healthy");
      const top = problems.length ? (DIAGNOSIS[problems[0]]?.label ?? problems[0]).toLowerCase() : "all healthy";
      subs.agent1_diagnose = d.stop_reason ? `${top} → stop` : `${top} → continue`;
    }
    if (e.node === "agent2_adapt") {
      const n = Object.values(d.changes as Record<string, string[]>).reduce((a, c) => a + c.length, 0);
      subs.agent2_adapt = `${n} range changes`;
    }
    if (e.node === "agent3_select") subs.agent3_select = `${familyLabel(d.family)} · AUC ${fmt(d.metrics_holdout?.roc_auc, 3)}`;
    if (e.node === "package_output") subs.package_output = "C2Output saved";
  }

  let activeEdge: [string, string] | null = null;
  if (run.status === "running") {
    const lastEvent = run.events[run.events.length - 1];
    const next = lastEvent ? NEXT[lastEvent.node](lastEvent) : "load_input";
    if (next) {
      state[next] = "active";
      if (lastEvent) activeEdge = [lastEvent.node, next];
      if (next === "agent2_hpo" && lastEvent?.node === "agent2_adapt") {
        round += 1;
        subs.agent2_hpo = `round ${round} · tuning…`;
      }
      else if (next === "agent2_hpo") subs.agent2_hpo = "tuning…";
    }
  }
  return { state, counts, subs, activeEdge, last, round };
}

// ---------- geometry ----------
const LEFT = 150, COLW = 138, NW = 126, NH = 54, LANE_H = 80, TOP = 26;
const W = LEFT + 20 + 8 * COLW + 30;
const H = TOP + LANES.length * LANE_H + 8;
const nodeById = Object.fromEntries(NODES.map((n) => [n.id, n]));
const nx = (id: string) => LEFT + 20 + (nodeById[id].col - 1) * COLW;
const ny = (id: string) => TOP + LANES.indexOf(nodeById[id].owner) * LANE_H + (LANE_H - NH) / 2;

function edgePath(from: string, to: string): { d: string; lx: number; ly: number } {
  const fx = nx(from), fy = ny(from), tx = nx(to), ty = ny(to);
  if (from === "agent2_adapt" && to === "agent2_hpo") {
    const y = fy + NH + 12;
    return { d: `M ${fx + NW / 2} ${fy + NH} V ${y} H ${tx + NW / 2} V ${ty + NH + 6}`, lx: (fx + tx + NW) / 2, ly: y - 4 };
  }
  if (fx === tx) return { d: `M ${fx + NW / 2} ${fy + NH} V ${ty - 6}`, lx: fx + NW / 2 - 6, ly: (fy + NH + ty) / 2 + 4 };
  const x1 = fx + NW, y1 = fy + NH / 2, x2 = tx - 6, y2 = ty + NH / 2;
  if (y1 === y2) return { d: `M ${x1} ${y1} H ${x2}`, lx: (x1 + x2) / 2, ly: y1 - 6 };
  const mx = (x1 + tx) / 2;
  return { d: `M ${x1} ${y1} H ${mx} V ${y2} H ${x2}`, lx: mx + 6, ly: (y1 + y2) / 2 };
}

export function WorkflowGraph({ view, selected, onSelect }: {
  view: WorkflowView; selected?: string | null; onSelect?: (id: string) => void;
}) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="wf-svg" role="img" aria-label="C2 agentic workflow">
      <defs>
        {[["pending", "#cfcec7"], ["done", "#6b6a66"], ["active", "#0b0b0b"]].map(([k, c]) => (
          <marker key={k} id={`arrow-${k}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill={c} />
          </marker>
        ))}
      </defs>

      {/* swimlanes: who does what */}
      {LANES.map((lane, i) => {
        const a = AGENTS[lane];
        const y = TOP + i * LANE_H;
        return (
          <g key={lane}>
            <rect x={0} y={y + 3} width={W} height={LANE_H - 6} rx={12} fill={i % 2 ? "#fbfbf9" : "#f6f6f3"} />
            <rect x={10} y={y + LANE_H / 2 - 13} width={4} height={26} rx={2} fill={a.color} />
            <text x={22} y={y + LANE_H / 2 - 2} fontSize={13} fontWeight={650} fill="#0b0b0b">{a.name}</text>
            <text x={22} y={y + LANE_H / 2 + 14} fontSize={10.5} fill="#898781">{a.role}</text>
          </g>
        );
      })}

      {/* input / output of the component */}
      <g>
        <text x={LEFT - 6} y={ny("load_input") - 6} fontSize={11} fill="#52514e" fontWeight={600}>from C1</text>
        <path d={`M ${LEFT - 8} ${ny("load_input") + NH / 2} H ${nx("load_input") - 6}`} stroke="#6b6a66" strokeWidth={1.5} markerEnd="url(#arrow-done)" />
        <text x={W - 34} y={ny("package_output") - 6} fontSize={11} fill="#52514e" fontWeight={600}>to C3</text>
        <path d={`M ${nx("package_output") + NW} ${ny("package_output") + NH / 2} H ${W - 6}`} stroke="#6b6a66" strokeWidth={1.5} markerEnd="url(#arrow-done)" />
      </g>

      {EDGES.map(([from, to, label]) => {
        const { d, lx, ly } = edgePath(from, to);
        const isActive = view.activeEdge?.[0] === from && view.activeEdge?.[1] === to;
        const done = view.state[from] === "done" && (view.state[to] === "done" || isActive);
        const k = isActive ? "active" : done ? "done" : "pending";
        const color = isActive ? "#0b0b0b" : done ? "#6b6a66" : "#cfcec7";
        return (
          <g key={`${from}-${to}`}>
            <path d={d} fill="none" stroke={color} strokeWidth={isActive ? 2 : 1.5} markerEnd={`url(#arrow-${k})`}
              className={isActive ? "wf-edge-active" : undefined} />
            {label && (
              <text x={lx} y={ly} fontSize={10.5} fill={isActive ? "#0b0b0b" : "#898781"} fontWeight={600}
                textAnchor={label === "loop" ? "middle" : label === "continue" ? "end" : "start"}>
                {label === "loop" ? "↺ next round" : label}
              </text>
            )}
          </g>
        );
      })}

      {NODES.map((n) => {
        const a = AGENTS[n.owner];
        const s = view.state[n.id];
        const x = nx(n.id), y = ny(n.id);
        const count = view.counts[n.id] ?? 0;
        return (
          <g key={n.id} className="wf-node" onClick={() => onSelect?.(n.id)} role="button" aria-label={`${n.title}: ${s}`}>
            <title>{n.hint}</title>
            {s === "active" && <rect className="wf-halo" x={x - 6} y={y - 6} width={NW + 12} height={NH + 12} rx={14} fill={a.color} opacity={0.35} />}
            <rect className="box" x={x} y={y} width={NW} height={NH} rx={10}
              fill={s === "done" ? a.tint : "#ffffff"}
              stroke={selected === n.id ? "#0b0b0b" : s === "pending" ? "#d6d5ce" : a.color}
              strokeWidth={selected === n.id ? 2.5 : s === "active" ? 2.5 : 1.5} />
            <text x={x + 10} y={y + 21} fontSize={12.5} fontWeight={640} fill={s === "pending" ? "#898781" : "#0b0b0b"}>{n.title}</text>
            <text x={x + 10} y={y + 39} fontSize={10.5} fill={s === "pending" ? "#b5b4ad" : "#52514e"}>
              {(view.subs[n.id] ?? (s === "pending" ? "waiting" : s === "active" ? "working…" : "")).slice(0, 24)}
            </text>
            {count > 1 && (
              <g>
                <rect x={x + NW - 22} y={y - 13} width={30} height={18} rx={9} fill="#0b0b0b" stroke="#fff" strokeWidth={2} />
                <text x={x + NW - 7} y={y} fontSize={10.5} fill="#fff" textAnchor="middle" fontWeight={650}>×{count}</text>
              </g>
            )}
            {s === "done" && count <= 1 && <circle cx={x + NW} cy={y} r={6} fill={a.color} stroke="#fff" strokeWidth={2} />}
          </g>
        );
      })}
    </svg>
  );
}
