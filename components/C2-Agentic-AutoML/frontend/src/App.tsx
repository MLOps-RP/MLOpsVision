import { createContext, useContext } from "react";
import { BrowserRouter, NavLink, Route, Routes, useLocation } from "react-router-dom";
import {
  BookOpen, Database, FlaskConical, GitCompareArrows, History, LayoutDashboard, Play, Settings2,
} from "lucide-react";
import { api, NotebookInfo } from "./api";
import { useData } from "./hooks";
import Overview from "./pages/Overview";
import DataPage from "./pages/DataPage";
import NewRun from "./pages/NewRun";
import Runs from "./pages/Runs";
import RunDetailPage from "./pages/RunDetail";
import Compare from "./pages/Compare";
import Notebooks from "./pages/Notebooks";
import ConfigPage from "./pages/ConfigPage";

interface NbCtx { jupyterUrl: string; notebooks: NotebookInfo[] }
const NotebookContext = createContext<NbCtx>({ jupyterUrl: "http://localhost:8888", notebooks: [] });

/** URL of a notebook by its context ("input", "baselines", "run", "comparison"). */
export function useNotebook(context: string): string {
  const { notebooks, jupyterUrl } = useContext(NotebookContext);
  return notebooks.find((n) => n.context === context)?.url ?? `${jupyterUrl}/lab`;
}
export const useJupyter = () => useContext(NotebookContext);

const NAV = [
  { section: "Component" },
  { to: "/", label: "Overview", icon: <LayoutDashboard size={17} />, end: true },
  { to: "/data", label: "Input data (C1)", icon: <Database size={17} /> },
  { section: "Agentic AutoML" },
  { to: "/runs/new", label: "New run", icon: <Play size={17} /> },
  { to: "/runs", label: "Runs & results", icon: <History size={17} />, end: true },
  { to: "/compare", label: "Baseline comparison", icon: <GitCompareArrows size={17} /> },
  { section: "Research" },
  { to: "/notebooks", label: "Jupyter notebooks", icon: <BookOpen size={17} /> },
  { to: "/config", label: "Rules & config", icon: <Settings2 size={17} /> },
] as const;

const TITLES: [RegExp, string][] = [
  [/^\/$/, "Overview"], [/^\/data/, "Input data"], [/^\/runs\/new/, "New run"], [/^\/runs\/.+/, "Run detail"],
  [/^\/runs/, "Runs & results"], [/^\/compare/, "Baseline comparison"], [/^\/notebooks/, "Jupyter notebooks"],
  [/^\/config/, "Rules & config"],
];

function TopBar() {
  const { pathname } = useLocation();
  const { jupyterUrl } = useJupyter();
  const health = useData(() => api.health(), [], 10000);
  const title = TITLES.find(([re]) => re.test(pathname))?.[1] ?? "";
  return (
    <header className="topbar">
      <div className="crumbs">C2 · Agentic AutoML / <b>{title}</b></div>
      <div className="spacer" />
      <span className="badge" title="FastAPI backend at :8000">
        <span className="dot" style={{ background: health.data ? "#0ca30c" : "#d03b3b" }} />
        {health.data ? "API connected" : "API offline"}
      </span>
      <a className="btn small jupyter" href={`${jupyterUrl}/lab`} target="_blank" rel="noreferrer">
        <BookOpen size={13} color="#e46e0e" /> JupyterLab
      </a>
    </header>
  );
}

function Layout() {
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">C2</div>
          <div>
            <div className="brand-title">Agentic AutoML</div>
            <div className="brand-sub">MLOps platform · J26-SE-324</div>
          </div>
        </div>
        {NAV.map((item, i) =>
          "section" in item ? (
            <div key={i} className="nav-section">{item.section}</div>
          ) : (
            <NavLink key={item.to} to={item.to} end={"end" in item ? item.end : false} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              {item.icon}{item.label}
            </NavLink>
          ),
        )}
        <div className="sidebar-foot">
          <div className="row" style={{ gap: 6 }}><FlaskConical size={13} /> C1 → <b style={{ color: "var(--ink)" }}>C2</b> → C3 → C4</div>
          <div>Model generation & selection engine</div>
        </div>
      </aside>
      <div className="main">
        <TopBar />
        <main className="content">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/data" element={<DataPage />} />
            <Route path="/data/:id" element={<DataPage />} />
            <Route path="/runs/new" element={<NewRun />} />
            <Route path="/runs" element={<Runs />} />
            <Route path="/runs/:id" element={<RunDetailPage />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/notebooks" element={<Notebooks />} />
            <Route path="/config" element={<ConfigPage />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

export default function App() {
  const nb = useData(() => api.notebooks(), []);
  const value: NbCtx = { jupyterUrl: nb.data?.jupyter_url ?? "http://localhost:8888", notebooks: nb.data?.notebooks ?? [] };
  return (
    <NotebookContext.Provider value={value}>
      <BrowserRouter>
        <Layout />
      </BrowserRouter>
    </NotebookContext.Provider>
  );
}
