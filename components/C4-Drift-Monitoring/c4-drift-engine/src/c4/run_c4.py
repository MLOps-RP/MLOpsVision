"""
C4 END-TO-END RUNNER

INPUT       deployed model (MLflow alias 'production') + its versioned baseline/thresholds,
            a production stream (one scenario), delayed labels
PROCESSING  for every window: monitoring -> diagnosis -> severity -> decision
OUTPUT      Drift Reports (JSON), actions (incl. real rollback), audit log (SQLite),
            window log CSVs for the dashboard

Run from the project root:
    python -m src.c4.run_c4 --scenario S3_concept
    python -m src.c4.run_c4 --scenario S6_sudden_drop --quiet
"""
import argparse
import uuid

import numpy as np
import pandas as pd

from src.common.config import (DATA_SCENARIOS, OUTPUTS, LABEL, all_features, load_config,
                               model_dir, load_json, save_json)
from src.common import registry
from src.c4.monitoring.monitor import compute_signals
from src.c4.monitoring.performance import PerformanceTracker
from src.c4.diagnosis.diagnose import (diagnose, rank_affected_features, affected_segments,
                                       performance_is_valid)
from src.c4.severity import severity as sev
from src.c4.controller.policy import decide, explain
from src.c4.controller.rollback import rollback
from src.c4.audit.drift_report import DriftReport
from src.c4.audit import audit_log
from src.tools import export_grafana


class C4Engine:
    def __init__(self, scenario: str, verbose: bool = True):
        self.scenario = scenario
        self.verbose = verbose
        self.mon = load_config("monitoring")
        self.sev_cfg = load_config("severity")
        self.policy = load_config("policy")
        self.truth = load_json(DATA_SCENARIOS / scenario / "ground_truth.json")
        self.stream = pd.read_csv(DATA_SCENARIOS / scenario / "windows.csv")

        self.perf = PerformanceTracker(self.mon["label_delay"], self.mon["performance_windows"],
                                       self.mon["adwin_delta"], self.mon["adwin_recent_windows"])
        self.loaded_version = None
        self.served_X = {}
        self.onset = None             # first window of the current input-drift episode
        self.quiet_windows = 0
        self.last_diagnosis = None
        self.streak = 0
        self.cooldown_until = -1
        self.last_logged_action = None
        self.actions_sent = set()      # each significant action is sent once per run (one drift episode)
        self.rollback_attempted = False
        self.records, self.feature_rows = [], []

        self.run_id = uuid.uuid4().hex[:8]
        self.db = audit_log.connect()
        audit_log.start_run(self.db, self.run_id, scenario)

    # ------------------------------------------------------------------ INPUT
    def _ensure_model(self) -> None:
        """Load the model behind the 'production' alias (reloads after a rollback)."""
        version = registry.get_alias_version("production")
        if version != self.loaded_version:
            folder = model_dir(version)
            self.model = registry.load_model(version)
            self.baseline = load_json(folder / "baseline.json")
            self.thresholds = load_json(folder / "thresholds.json")
            self.reference = pd.read_csv(folder / "reference_numeric.csv")
            self.loaded_version = version
            self._say(f"   [model] production = version {version} ({self.baseline['model_type']})")

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------ one window
    def step(self, w: int, rows: pd.DataFrame) -> dict:
        self._ensure_model()
        X = rows[all_features()]
        conf = self.model.predict_proba(X)[:, 1]
        pred = (conf >= 0.5).astype(int)
        self.perf.add_served(w, rows[LABEL].to_numpy(), pred)
        self.served_X[w] = X.reset_index(drop=True)

        # 1. MONITORING
        perf = self.perf.evaluate(w, self.baseline)
        signals = compute_signals(w, X, conf, self.baseline, self.thresholds,
                                  self.reference, perf, self.mon)

        input_drift = bool(signals["fired_features"]) or signals["prediction"]["fired"]
        if input_drift:
            self.quiet_windows = 0
            if self.onset is None:
                self.onset = w
        else:
            self.quiet_windows += 1
            if self.quiet_windows >= 3:
                self.onset = None

        # 2. DIAGNOSIS
        dg = diagnose(signals, self.onset)
        diagnosis = dg["diagnosis"]
        self.streak = self.streak + 1 if diagnosis == self.last_diagnosis else 1
        self.last_diagnosis = diagnosis
        if diagnosis in ("CONCEPT_DRIFT", "LABEL_SHIFT"):
            perf_valid = bool(perf.get("available"))
        else:
            perf_valid = performance_is_valid(perf, self.onset)

        # 3. SEVERITY
        sv = sev.compute(signals, diagnosis, self.streak, perf_valid, self.sev_cfg)
        active = diagnosis != "NO_DRIFT" and (self.streak >= self.sev_cfg["min_persistence"]
                                              or sv["level"] == "Critical")

        # 4. DECISION
        action, target, rule_no = decide(diagnosis, sv["level"], self.policy) if active else ("NONE", "none", 0)
        if action == "ROLLBACK" and self.rollback_attempted:
            # rollback was the first-aid step; if the problem persists, route it normally
            action, target, rule_no = decide(diagnosis, "High", self.policy)
        version_before = self.loaded_version
        executed, note, action_result = False, "", {}

        if action in self.policy["significant_actions"]:
            in_cooldown = w <= self.cooldown_until
            escalation = action == "ROLLBACK" and not self.rollback_attempted
            if action in self.actions_sent and not escalation:
                note = f"{action} already sent - awaiting remediation"
                action_taken = "AWAITING"
            elif in_cooldown and not escalation:
                note = f"cooldown until window {self.cooldown_until}"
                action_taken = "COOLDOWN"
            else:
                action_taken = action
                if action == "ROLLBACK":
                    self.rollback_attempted = True
                    action_result = rollback(f"{diagnosis} {sv['level']} at window {w}")
                    if not action_result["ok"]:
                        # nothing to roll back to -> fall back to the normal route
                        action_taken = "ROUTE_C2" if diagnosis != "COVARIATE_DEGRADING" else "ROUTE_C1"
                        target = "C2" if action_taken == "ROUTE_C2" else "C1"
                        note = "rollback unavailable: " + action_result["error"]
                executed = action_taken not in self.actions_sent
                if executed:
                    self.actions_sent.add(action_taken)
                    self.cooldown_until = w + self.policy["cooldown_windows"]
                else:
                    action_taken, note = "AWAITING", note + "; already sent - awaiting remediation"
        else:
            action_taken = action
            executed = action in ("WARN_ONLY", "WATCH") and action != self.last_logged_action

        # 5. OUTPUT: Drift Report + audit log
        report_path = ""
        if executed:
            reason = explain(diagnosis, sv["level"], action_taken, dg["evidence"])
            segments = []
            if diagnosis in ("CONCEPT_DRIFT", "COVARIATE_DEGRADING") and perf.get("available"):
                lw = perf["labelled_windows"]
                segments = affected_segments(
                    pd.concat([self.served_X[i] for i in lw], ignore_index=True),
                    np.concatenate([self.perf.served[i][0] for i in lw]),
                    np.concatenate([self.perf.served[i][1] for i in lw]), self.baseline)
            report = DriftReport(
                model_name=self.baseline["model_name"], model_version=version_before,
                baseline_version=f"model_v{version_before}/baseline.json",
                preprocessing_version=self.baseline["preprocessing_version"],
                scenario=self.scenario, window=w, drift_type=diagnosis,
                diagnosis_confidence=dg["confidence"], severity_score=sv["score"],
                severity_level=sv["level"], severity_breakdown=sv["components"],
                recommended_action=action_taken, target_component=target, reason=reason,
                affected_features=rank_affected_features(signals, self.baseline["feature_importance"]),
                affected_segments=segments,
                performance={k: perf.get(k) for k in ("accuracy", "f1", "perf_drop",
                                                      "label_share_delta", "labelled_windows")}
                if perf.get("available") else {"available": False},
                data_quality_issues=signals["data_quality"]["issues"],
                action_result=action_result,
            )
            report_path = report.save()
            version_after = registry.get_alias_version("production")
            audit_log.log_decision(self.db, self.run_id, self.scenario, w, version_before, version_after,
                                   diagnosis, dg["confidence"], sv["level"], sv["score"],
                                   action_taken, target, reason, report_path)
            self.last_logged_action = action_taken

        audit_log.log_window(self.db, self.run_id, self.scenario, w, version_before, diagnosis,
                             sv["level"], sv["score"], action_taken, signals)

        top = max(signals["features"].items(), key=lambda kv: kv[1]["ratio"])
        rec = {
            "window": w, "model_version": version_before, "injected": bool(rows["injected"].any()),
            "n_fired": len(signals["fired_features"]),
            "fired_features": ";".join(signals["fired_features"]),
            "top_feature": top[0], "top_ratio": round(top[1]["ratio"], 3),
            "prediction_share": round(signals["prediction"]["prediction_share"], 4),
            "prediction_ratio": round(signals["prediction"]["ratio"], 3),
            "dq_fired": signals["data_quality"]["fired"],
            "dq_issues": " | ".join(signals["data_quality"]["issues"]),
            "labels_available": bool(perf.get("available")),
            "latest_labelled_window": perf.get("latest_labelled_window"),
            "accuracy": perf.get("accuracy"), "f1": perf.get("f1"),
            "perf_drop": perf.get("perf_drop"), "label_share_delta": perf.get("label_share_delta"),
            "adwin_alarm": perf.get("adwin_alarm"),
            "diagnosis": diagnosis, "confidence": dg["confidence"], "streak": self.streak,
            "active": active, "severity_score": sv["score"], "severity_level": sv["level"],
            **{f"sev_{k}": v for k, v in sv["components"].items()},
            "action": action_taken, "executed": executed, "note": note,
            "report_path": str(report_path),
        }
        self.records.append(rec)
        self.feature_rows.append({"window": w, **{f: round(r["score"], 5) for f, r in signals["features"].items()},
                                  **{f"thr__{f}": round(r["threshold"], 5) for f, r in signals["features"].items()}})

        flag = " <== ACTION" if executed and action_taken not in ("WATCH",) else ""
        perf_txt = f"F1 {perf['f1']:.3f}" if perf.get("available") else "F1  n/a "
        self._say(f"w{w:02d} {'*' if rec['injected'] else ' '} v{version_before} | fired {rec['n_fired']:>2} | "
                  f"{perf_txt} | {diagnosis:<19} {dg['confidence'][:4]:<4} | {sv['level']:<8} "
                  f"{sv['score']:.2f} | {action_taken}{flag}")
        return rec

    # ------------------------------------------------------------------ whole scenario
    def run(self) -> pd.DataFrame:
        self._say(f"\n=== C4 run {self.run_id} | scenario {self.scenario} ===")
        self._say(f"    {self.truth['description']}")
        self._say("    (* = drift injected in this window)")
        for w, rows in self.stream.groupby("window", sort=True):
            self.step(int(w), rows)
        self.db.commit()              # windows after the last decision are not committed yet
        self.db.close()
        export_grafana.export()

        out = OUTPUTS / "runs" / self.scenario
        out.mkdir(parents=True, exist_ok=True)
        log = pd.DataFrame(self.records)
        log.to_csv(out / "window_log.csv", index=False)
        pd.DataFrame(self.feature_rows).to_csv(out / "feature_scores.csv", index=False)
        save_json({"run_id": self.run_id, "scenario": self.scenario, "ground_truth": self.truth},
                  out / "run_info.json")
        self._say(f"Saved window log to {out}")
        return log


def run_scenario(scenario: str, reset: bool = True, verbose: bool = True) -> pd.DataFrame:
    if reset:
        registry.reset_production()
    return C4Engine(scenario, verbose=verbose).run()


def main() -> None:
    p = argparse.ArgumentParser(description="Run C4 on one scenario")
    p.add_argument("--scenario", default="S3_concept",
                   help="folder name under data/scenarios, e.g. S1_covariate_benign")
    p.add_argument("--no-reset", action="store_true",
                   help="do not reset the production alias to v2 before running")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    run_scenario(a.scenario, reset=not a.no_reset, verbose=not a.quiet)


if __name__ == "__main__":
    main()
