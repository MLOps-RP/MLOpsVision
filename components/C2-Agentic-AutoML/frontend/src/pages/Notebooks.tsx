import { BookOpen, Terminal } from "lucide-react";
import { useJupyter } from "../App";
import { Card, JupyterButton, PageHead } from "../components/ui";

const WHEN: Record<string, string> = {
  input: "Before a run: understand the data C1 provides.",
  baselines: "Step 2 evidence: how much does tuning help?",
  run: "After a run: deep-dive into rounds, gaps and range changes.",
  comparison: "Thesis evidence: adaptive vs static, with ablations and statistics.",
};

export default function Notebooks() {
  const { notebooks, jupyterUrl } = useJupyter();
  return (
    <div>
      <PageHead title="Jupyter notebooks" sub="For research scenarios that need free exploration or thesis figures. All notebooks import the same c2_automl package and read the same output files as this dashboard."
        actions={<a className="btn jupyter" href={`${jupyterUrl}/lab`} target="_blank" rel="noreferrer"><BookOpen size={15} color="#e46e0e" /> Open JupyterLab</a>} />
      <div className="grid g2">
        {notebooks.map((nb, i) => (
          <Card key={nb.file} pad>
            <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start" }}>
              <div>
                <div className="small muted mono">{nb.file}</div>
                <h3 style={{ fontSize: 17, marginTop: 4 }}>{String(i + 1).padStart(2, "0")} · {nb.title}</h3>
              </div>
              {!nb.exists && <span className="chip">missing</span>}
            </div>
            <p className="small ink2" style={{ marginTop: 8, lineHeight: 1.55 }}>{nb.description}</p>
            <p className="small muted" style={{ marginTop: 6 }}>{WHEN[nb.context]}</p>
            <div style={{ marginTop: 14 }}><JupyterButton url={nb.url} /></div>
          </Card>
        ))}
      </div>
      <Card pad className="mt">
        <div className="row"><Terminal size={16} /><b>Start JupyterLab for the buttons to work</b></div>
        <pre className="json" style={{ marginTop: 10 }}>{`cd components/C2-Agentic-AutoML
./scripts/start_jupyter.sh        # JupyterLab on ${jupyterUrl} (local only, no token)`}</pre>
        <p className="small muted" style={{ marginTop: 8 }}>The notebooks use the project's .venv, so <code>c2_automl</code> imports work directly.</p>
      </Card>
    </div>
  );
}
