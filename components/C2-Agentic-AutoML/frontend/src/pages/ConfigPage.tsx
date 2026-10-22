import { useState } from "react";
import { api } from "../api";
import { useData } from "../hooks";
import { familyLabel } from "../theme";
import { AgentBadge, Card, DataTable, DiagnosisChip, ErrorBox, JsonView, Loading, PageHead, Tabs } from "../components/ui";

type T = "diagnosis" | "adaptation" | "spaces" | "general";

export default function ConfigPage() {
  const cfg = useData(() => api.config(), []);
  const [tab, setTab] = useState<T>("diagnosis");
  if (cfg.error) return <ErrorBox error={cfg.error} />;
  if (!cfg.data) return <Loading />;
  const { diagnosis_rules: dr, space_adaptation: sa, search_spaces: ss, model_registry: mr } = cfg.data;
  const ruleText: Record<string, string> = {
    overfitting: `gap > ${dr.overfitting.min_gap} and train score > ${dr.overfitting.min_train_score}`,
    underfitting: `gap < ${dr.underfitting.max_gap} and train score < ${dr.underfitting.max_train_score}`,
    class_imbalance: `imbalance ratio < ${dr.class_imbalance.max_imbalance_ratio} and minority recall < ${dr.class_imbalance.max_minority_recall}`,
    weak_features: `lift over dummy model < ${dr.weak_features.max_lift}`,
    unsuitable_model_family: `≥ ${dr.unsuitable_model_family.min_gap_to_best} below the best family`,
  };

  return (
    <div>
      <PageHead title="Rules & configuration" sub="The agents' knowledge is data, not code: thresholds, actions and range rules live in YAML files under configs/. Changing them changes behaviour without touching Python." />
      <Tabs<T> value={tab} onChange={setTab} tabs={[
        { id: "diagnosis", label: "Diagnosis rules (Agent 1)" },
        { id: "adaptation", label: "Range adaptation (Agent 2)" },
        { id: "spaces", label: "Base search spaces" },
        { id: "general", label: "Budget & loop" },
      ]} />
      {tab === "diagnosis" && (
        <Card>
          <div className="card-head"><div><div className="row"><AgentBadge agent="agent1" /><h3>configs/diagnosis_rules.yaml</h3></div><div className="sub">Evidence → diagnosis → actions sent to Agent 2.</div></div></div>
          <div className="card-body table-wrap">
            <table className="t">
              <thead><tr><th>Diagnosis</th><th>Rule (evidence)</th><th>Actions for the next round</th></tr></thead>
              <tbody>
                {Object.keys(ruleText).map((k) => (
                  <tr key={k}><td><DiagnosisChip label={k} /></td><td className="mono small">{ruleText[k]}</td>
                    <td>{(dr.actions[k] as string[]).map((a) => <span key={a} className="chip" style={{ marginRight: 4 }}>{a.replace(/_/g, " ")}</span>)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
      {tab === "adaptation" && (
        <div className="grid g2" style={{ alignItems: "start" }}>
          <Card>
            <div className="card-head"><div><div className="row"><AgentBadge agent="agent2" /><h3>Action effects</h3></div><div className="sub">Multipliers applied to the current range (configs/space_adaptation.yaml).</div></div></div>
            <div className="card-body">
              <DataTable spec={{
                columns: ["Action", "Parameter", "Effect"],
                rows: Object.entries(sa.actions as Record<string, any>).flatMap(([action, eff]) =>
                  Object.entries(eff).flatMap(([param, spec]: [string, any]) =>
                    param === "add_params"
                      ? Object.entries(spec as Record<string, any>).map(([fam, p]) => [action.replace(/_/g, " "), `${familyLabel(fam)}: ${Object.keys(p)[0]}`, "add parameter"])
                      : [[action.replace(/_/g, " "), param, Object.entries(spec).map(([s, m]) => `${s} × ${m}`).join(", ")]])),
              }} />
            </div>
          </Card>
          <Card>
            <div className="card-head"><div><h3>Hard limits & edge rule</h3><div className="sub">Ranges can never leave these limits. Best value in the outer {sa.edge.tolerance * 100}% → extend by ×{sa.edge.factor}.</div></div></div>
            <div className="card-body"><DataTable spec={{ columns: ["Parameter", "Min", "Max"], rows: Object.entries(sa.limits as Record<string, number[]>).map(([p, [lo, hi]]) => [p, lo, hi]) }} /></div>
          </Card>
        </div>
      )}
      {tab === "spaces" && (
        <div className="grid g3" style={{ alignItems: "start" }}>
          {Object.entries(ss as Record<string, Record<string, any>>).map(([fam, space]) => (
            <Card key={fam}>
              <div className="card-head"><div><h3>{familyLabel(fam)}</h3><div className="sub">tasks: {(mr.families[fam]?.tasks ?? []).join(", ")}</div></div></div>
              <div className="card-body"><DataTable spec={{ columns: ["Parameter", "Range"], rows: Object.entries(space).map(([p, r]) => [p, r.type === "cat" ? `{${r.choices.join(", ")}}` : `${r.low} – ${r.high}${r.log ? " (log)" : ""}`]) }} /></div>
            </Card>
          ))}
        </div>
      )}
      {tab === "general" && <Card pad><JsonView value={cfg.data.default} /></Card>}
    </div>
  );
}
