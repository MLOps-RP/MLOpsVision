"""The HTTP wrapper: it must expose the workflow, add no deployment surface, and stay inside its sandbox."""
import importlib.util
import os
import shutil
import sys
import threading
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from mqvg.api.app import create_app
from mqvg.api.models import NOTICE
from mqvg.api.settings import Settings, SettingsError
from mqvg.config import load_config
from mqvg.schemas import CandidateBundle, Decision, ValidationResult
from mqvg.workflow import validate_and_report

from conftest import CONFIGS, ROOT

KEY = "test-key-123"
AUTH = {"X-API-Key": KEY}
SAFE = "extended"


@pytest.fixture
def artifacts(tmp_path, tiny_files):
    """An artifact root holding the tiny model, its data and metadata."""
    root = tmp_path / "artifacts"
    root.mkdir()
    for name, path in tiny_files.items():
        shutil.copy(path, root / path.name)
    return root


def make_settings(tmp_path, artifacts, **overrides):
    values = dict(artifact_root=artifacts, report_dir=tmp_path / "reports",
                  config_dir=CONFIGS, api_key=KEY)
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def client(tmp_path, artifacts):
    return TestClient(create_app(make_settings(tmp_path, artifacts)))


def body(artifacts, **overrides):
    names = {p.suffix: p.name for p in artifacts.iterdir()}
    payload = {"candidate_id": "tiny", "model_path": names[".joblib"],
               "eval_data_path": names[".csv"], "metadata_path": names[".json"]}
    payload.update(overrides)
    return payload


def post(client, artifacts, **overrides):
    return client.post("/v1/validations", json=body(artifacts, **overrides), headers=AUTH)


# ---------------------------------------------------------------------- service
def test_health_is_open_and_reports_no_secrets(client):
    reply = client.get("/health")
    assert reply.status_code == 200 and reply.json()["status"] == "ok"
    assert reply.json()["authentication"] == "api key" and KEY not in reply.text


def test_every_other_endpoint_needs_the_key(client):
    for method, url in (("get", "/v1/profiles"), ("get", "/v1/checks"), ("get", "/v1/validations"),
                        ("get", "/v1/validations/" + "0" * 32), ("post", "/v1/validations")):
        kwargs = {"json": {}} if method == "post" else {}
        assert getattr(client, method)(url, **kwargs).status_code == 401, url
        assert getattr(client, method)(url, headers={"X-API-Key": "wrong"}, **kwargs).status_code == 401


def test_without_a_configured_key_the_api_fails_closed(tmp_path, artifacts):
    client = TestClient(create_app(make_settings(tmp_path, artifacts, api_key=None)))
    assert client.get("/health").json()["authentication"] == "not configured"
    reply = client.get("/v1/profiles", headers=AUTH)
    assert reply.status_code == 503 and "MQVG_API_KEY" in reply.json()["detail"]


def test_local_development_can_switch_authentication_off_explicitly(tmp_path, artifacts):
    client = TestClient(create_app(make_settings(tmp_path, artifacts, api_key=None, allow_no_auth=True)))
    assert client.get("/v1/checks").status_code == 200


def test_the_settings_never_show_the_key():
    assert KEY not in repr(Settings(artifact_root=Path("."), report_dir=Path("."), api_key=KEY))


def test_startup_rejects_bad_settings(tmp_path):
    with pytest.raises(SettingsError):
        create_app(Settings(artifact_root=tmp_path / "missing", report_dir=tmp_path / "r"))
    with pytest.raises(SettingsError):
        create_app(Settings(artifact_root=tmp_path, report_dir=tmp_path / "r",
                            config_dir=tmp_path / "missing"))


def test_the_environment_must_name_an_artifact_root(monkeypatch):
    monkeypatch.delenv("MQVG_ARTIFACT_ROOT", raising=False)
    with pytest.raises(SettingsError):
        Settings.from_env()
    monkeypatch.setenv("MQVG_ARTIFACT_ROOT", "/data")
    monkeypatch.setenv("MQVG_API_KEY", "k")
    assert Settings.from_env().api_key == "k"


# ------------------------------------------------------------------ configuration
def test_the_profiles_say_which_ones_scan_model_files(client):
    reply = client.get("/v1/profiles", headers=AUTH).json()
    by_name = {p["name"]: p for p in reply["profiles"]}
    assert reply["default_profile"] == "extended"
    for name in ("extended", "extended_hidden_groups"):
        assert by_name[name]["scans_model_files"] and by_name[name]["allowed"]
    for name in ("default", "hidden_groups", "strict"):
        assert not by_name[name]["scans_model_files"] and not by_name[name]["allowed"]
    assert abs(sum(by_name["extended"]["weights"].values()) - 1) < 1e-9


def test_the_checks_are_the_catalogue(client):
    checks = client.get("/v1/checks", headers=AUTH).json()
    ids = {c["id"] for c in checks}
    assert {"behaviour.f1", "operational.model_file_safety", "behaviour.hidden_failure_groups"} <= ids
    assert len(checks) == len(ids) == 27


# ---------------------------------------------------------------------- validation
def test_a_validation_returns_a_recommendation_and_stores_the_run(client, artifacts, tmp_path):
    reply = post(client, artifacts)
    assert reply.status_code == 201, reply.text
    s = reply.json()
    assert s["decision"] in {d.value for d in Decision} and s["action_mode"] == "RECOMMENDATION"
    assert s["notice"] == NOTICE and s["profile"] == "extended" and s["candidate_id"] == "tiny"
    assert reply.headers["location"] == s["result_url"] == f"/v1/validations/{s['run_id']}"
    assert set(s["dimension_scores"]) == {"training_validity", "model_behaviour", "operational_readiness"}
    assert sum(s["check_counts"].values()) == 25            # the 21 default checks plus the four extended ones
    assert sorted(p.name for p in (tmp_path / "reports" / s["run_id"]).iterdir()) == [
        "report.md", "result.json", "summary.json"]


def test_the_stored_result_and_report_can_be_fetched(client, artifacts):
    s = post(client, artifacts).json()
    full = client.get(s["result_url"], headers=AUTH)
    assert full.status_code == 200
    result = ValidationResult.model_validate(full.json())
    assert result.run_id == s["run_id"] and result.decision.decision.value == s["decision"]
    assert result.decision.action_mode.value == "RECOMMENDATION"
    assert client.get(s["result_url"] + "/summary", headers=AUTH).json() == s
    report = client.get(s["report_url"], headers=AUTH)
    assert report.status_code == 200 and report.headers["content-type"].startswith("text/markdown")
    assert "Nothing was deployed, promoted or rolled back" in report.text
    assert f"Decision: {s['decision']}" in report.text


def test_the_api_gives_the_same_answer_as_calling_the_workflow(client, artifacts, tiny_files):
    s = post(client, artifacts).json()
    direct = validate_and_report(
        CandidateBundle(candidate_id="tiny", model_path=artifacts / tiny_files["model"].name,
                        eval_data_path=artifacts / tiny_files["eval"].name,
                        metadata_path=artifacts / tiny_files["metadata"].name),
        load_config(CONFIGS / "extended.yaml"))
    assert s["decision"] == direct.decision
    assert [r["rule_id"] for r in s["triggered_rules"]] == [
        r.rule_id for r in direct.result.decision.triggered_rules]
    for dimension, score in direct.result.scorecard.dimensions.items():
        assert s["dimension_scores"][dimension.value] == pytest.approx(score.score, abs=1.0)


def test_runs_are_listed_newest_first(client, artifacts):
    first = post(client, artifacts, candidate_id="first").json()
    second = post(client, artifacts, candidate_id="second").json()
    assert first["run_id"] != second["run_id"]
    runs = client.get("/v1/validations", headers=AUTH).json()["runs"]
    assert [r["candidate_id"] for r in runs][:2] == ["second", "first"]
    assert len(client.get("/v1/validations?limit=1", headers=AUTH).json()["runs"]) == 1
    assert client.get("/v1/validations?limit=0", headers=AUTH).status_code == 422


def test_hidden_group_detection_can_be_requested_through_the_api(client, artifacts):
    s = post(client, artifacts, profile="extended_hidden_groups").json()
    full = client.get(s["result_url"], headers=AUTH).json()
    assert any(c["id"] == "behaviour.hidden_failure_groups" for c in full["checks"])


def test_concurrent_requests_are_serialised_and_all_answered(client, artifacts):
    replies = []

    def go():
        replies.append(post(client, artifacts))
    threads = [threading.Thread(target=go) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [r.status_code for r in replies] == [201, 201, 201]
    assert len({r.json()["run_id"] for r in replies}) == 3


def test_a_busy_gate_answers_503_instead_of_hanging(tmp_path, artifacts):
    app_client = TestClient(create_app(make_settings(tmp_path, artifacts, queue_timeout_seconds=0.05)))
    from mqvg.api import app as app_module
    original = app_module.validate_and_report
    started, release = threading.Event(), threading.Event()

    def slow(*args, **kwargs):
        started.set()
        release.wait(5)
        return original(*args, **kwargs)
    app_module.validate_and_report = slow
    try:
        first = []
        t = threading.Thread(target=lambda: first.append(post(app_client, artifacts)))
        t.start()
        assert started.wait(5)
        busy = post(app_client, artifacts)
        release.set()
        t.join()
    finally:
        app_module.validate_and_report = original
    assert busy.status_code == 503 and busy.headers["retry-after"] == "30"
    assert first[0].status_code == 201


# ----------------------------------------------------------------------- the sandbox
@pytest.mark.parametrize("path", ["../outside.joblib", "..\\outside.joblib", "sub/../../outside.joblib",
                                  "C:\\Windows\\win.ini", "/etc/passwd", "\\\\server\\share\\m.joblib",
                                  "model\x00.joblib"])
def test_paths_outside_the_artifact_root_are_rejected(client, artifacts, tmp_path, path):
    (tmp_path / "outside.joblib").write_bytes(b"x")
    for field in ("model_path", "eval_data_path", "metadata_path", "train_row_hashes_path"):
        reply = post(client, artifacts, **{field: path})
        assert reply.status_code == 422, (field, path, reply.text)
        assert "outside the artifact root" in reply.text or "invalid path" in reply.text or "no such file" in reply.text


def test_an_absolute_path_inside_the_root_is_accepted_but_outside_is_not(client, artifacts, tmp_path):
    inside = str((artifacts / next(p.name for p in artifacts.iterdir() if p.suffix == ".joblib")).resolve())
    assert post(client, artifacts, model_path=inside).status_code == 201
    outside = tmp_path / "elsewhere.joblib"
    outside.write_bytes(b"x")
    assert post(client, artifacts, model_path=str(outside)).status_code == 422


def test_a_symlink_that_leaves_the_root_is_rejected(client, artifacts, tmp_path):
    target = tmp_path / "secret.joblib"
    target.write_bytes(b"x")
    link = artifacts / "link.joblib"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks cannot be created here")
    reply = post(client, artifacts, model_path="link.joblib")
    assert reply.status_code == 422 and "outside the artifact root" in reply.text


def test_a_directory_junction_that_leaves_the_root_is_rejected(client, artifacts, tmp_path):
    """Windows junctions need no privilege and escape the root like a symlink would."""
    import subprocess
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (outside / "m.joblib").write_bytes(b"x")
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(artifacts / "jdir"), str(outside)],
                            capture_output=True, text=True) if os.name == "nt" else None
    if result is None or result.returncode != 0:
        pytest.skip("directory junctions cannot be created here")
    reply = post(client, artifacts, model_path="jdir/m.joblib")
    assert reply.status_code == 422 and "outside the artifact root" in reply.text


def test_a_missing_file_is_a_request_error_not_a_verdict(client, artifacts):
    reply = post(client, artifacts, model_path="nope.joblib")
    assert reply.status_code == 422 and "no such file" in reply.text
    assert client.get("/v1/validations", headers=AUTH).json()["runs"] == []     # nothing was run


@pytest.mark.parametrize("field, value", [
    ("profile", "../default"), ("profile", "extended.yaml"), ("profile", "no_such_profile"),
    ("candidate_id", "../x"), ("candidate_id", "a" * 200), ("model_path", ""),
    ("sensitive_columns", ["x"] * 80)])
def test_bad_values_are_rejected(client, artifacts, field, value):
    assert post(client, artifacts, **{field: value}).status_code == 422


def test_unknown_fields_are_rejected(client, artifacts):
    assert post(client, artifacts, config_path="/etc/x.yaml").status_code == 422     # no custom configs
    assert post(client, artifacts, deploy=True).status_code == 422


def test_profiles_that_load_models_without_scanning_are_refused_by_default(client, artifacts):
    for name in ("default", "hidden_groups", "strict"):
        reply = post(client, artifacts, profile=name)
        assert reply.status_code == 422 and "does not scan model files" in reply.text, name
        assert "extended" in reply.text


def test_an_operator_can_allow_unscanned_profiles(tmp_path, artifacts):
    client = TestClient(create_app(make_settings(tmp_path, artifacts, allow_unscanned_profiles=True)))
    assert post(client, artifacts, profile="default").status_code == 201


def test_a_profile_that_would_run_the_scan_in_report_only_mode_does_not_count_as_scanning(tmp_path, artifacts):
    config = tmp_path / "configs"
    shutil.copytree(CONFIGS, config)
    (config / "lax.yaml").write_text(
        "extends: extended.yaml\nprofile: lax\nmodel_safety:\n  refuse_at: never\n", encoding="utf-8")
    client = TestClient(create_app(make_settings(tmp_path, artifacts, config_dir=config)))
    assert post(client, artifacts, profile="lax").status_code == 422


@pytest.mark.parametrize("run_id", ["../../etc/passwd", "ZZZ", "0" * 31, "g" * 32, "%2e%2e"])
def test_run_ids_cannot_be_used_to_read_other_files(client, run_id):
    for suffix in ("", "/summary", "/report"):
        assert client.get(f"/v1/validations/{run_id}{suffix}", headers=AUTH).status_code in (404, 422)


def test_an_unknown_run_is_a_404(client):
    assert client.get("/v1/validations/" + "a" * 32, headers=AUTH).status_code == 404


# -------------------------------------------------------------- dangerous model files
class Payload:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (os.mkdir, (self.marker,))


def test_a_malicious_model_is_recommended_against_and_never_run(client, artifacts, tmp_path):
    marker = tmp_path / "pwned"
    joblib.dump(Payload(marker), artifacts / "evil.joblib")
    reply = post(client, artifacts, model_path="evil.joblib")
    assert reply.status_code == 201
    s = reply.json()
    assert s["decision"] == "ROLLBACK" and s["action_mode"] == "RECOMMENDATION"
    assert "unsafe_model_file" in [r["rule_id"] for r in s["triggered_rules"]]
    assert not marker.exists(), "the API executed the model's code"
    full = client.get(s["result_url"], headers=AUTH).json()
    safety = next(c for c in full["checks"] if c["id"] == "operational.model_file_safety")
    assert safety["status"] == "FAIL" and safety["details"]["loaded"] is False


def test_a_model_with_unrecognised_code_waits_for_review(client, artifacts):
    from test_model_safety import CustomModel
    joblib.dump(CustomModel(), artifacts / "custom.joblib")
    s = post(client, artifacts, model_path="custom.joblib").json()
    assert s["decision"] == "HOLD"


# --------------------------------------------------------------------- the contract
def test_the_api_has_no_deployment_surface(client):
    routes = {(m, r.path) for r in client.app.routes if hasattr(r, "methods") for m in r.methods
              if r.path.startswith(("/v1", "/health"))}
    assert {m for m, _ in routes} == {"GET", "POST"}
    assert [p for m, p in routes if m == "POST"] == ["/v1/validations"]
    assert not any(word in p for _, p in routes for word in ("deploy", "promote", "rollback", "rollout"))


def _script():
    spec = importlib.util.spec_from_file_location("export_openapi", ROOT / "scripts" / "export_openapi.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_openapi"] = module
    spec.loader.exec_module(module)
    return module


def test_the_committed_openapi_contract_is_up_to_date():
    script = _script()
    committed = script.TARGET.read_text(encoding="utf-8") if script.TARGET.exists() else ""
    assert committed == script.build_contract(), "run python scripts/export_openapi.py"


def test_the_contract_documents_the_security_scheme_and_the_recommendation_notice():
    import json
    contract = json.loads(_script().build_contract())
    assert "APIKeyHeader" in contract["components"]["securitySchemes"]
    assert "recommendation" in contract["info"]["description"].lower()
    assert "/v1/validations" in contract["paths"] and "/health" in contract["paths"]


# ------------------------------------------------------------------ failure paths
def test_a_misconfigured_profile_is_a_500_that_hides_the_details_and_is_skipped_in_the_listing(tmp_path, artifacts):
    config = tmp_path / "configs"
    shutil.copytree(CONFIGS, config)
    (config / "broken.yaml").write_text("profile: broken\nweights: {}\n", encoding="utf-8")
    client = TestClient(create_app(make_settings(tmp_path, artifacts, config_dir=config)))
    reply = post(client, artifacts, profile="broken")
    assert reply.status_code == 500 and "misconfigured" in reply.text and "weights" not in reply.text
    names = [p["name"] for p in client.get("/v1/profiles", headers=AUTH).json()["profiles"]]
    assert "broken" not in names and "extended" in names


def test_a_crash_inside_the_validation_is_a_generic_500(tmp_path, artifacts, monkeypatch):
    from mqvg.api import app as app_module

    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail /srv/keys")
    monkeypatch.setattr(app_module, "validate_and_report", boom)
    client = TestClient(create_app(make_settings(tmp_path, artifacts)))
    reply = post(client, artifacts)
    assert reply.status_code == 500 and "see the server log" in reply.text
    assert "secret internal detail" not in reply.text and "/srv/keys" not in reply.text
    # the lock was released by the crash: a later request is not blocked
    monkeypatch.undo()
    assert post(client, artifacts).status_code == 201


def test_a_path_the_os_cannot_resolve_is_rejected(client, artifacts, monkeypatch):
    real = Path.resolve

    def refuse(self, *args, **kwargs):
        if "unresolvable" in str(self):
            raise OSError("boom")
        return real(self, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", refuse)
    assert post(client, artifacts, model_path="unresolvable.joblib").status_code == 422


# ----------------------------------------------------------- starting the server
def run_main(monkeypatch, capsys, **env):
    from mqvg.api import __main__ as entry
    for name in ("MQVG_ARTIFACT_ROOT", "MQVG_API_KEY", "MQVG_HOST", "MQVG_PORT", "MQVG_ALLOW_NO_AUTH"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    started = {}
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: started.update(kw, app=app))
    code = entry.main()
    return code, started, capsys.readouterr().err


def test_the_server_refuses_to_start_without_an_artifact_root(monkeypatch, capsys):
    code, started, err = run_main(monkeypatch, capsys)
    assert code == 2 and not started and "MQVG_ARTIFACT_ROOT" in err


def test_the_server_refuses_a_missing_artifact_root(monkeypatch, capsys, tmp_path):
    code, started, err = run_main(monkeypatch, capsys, MQVG_ARTIFACT_ROOT=str(tmp_path / "nope"),
                                  MQVG_API_KEY="k")
    assert code == 2 and not started and "not a folder" in err


def test_the_server_will_not_listen_beyond_loopback_without_a_key(monkeypatch, capsys, tmp_path):
    code, started, err = run_main(monkeypatch, capsys, MQVG_ARTIFACT_ROOT=str(tmp_path),
                                  MQVG_HOST="0.0.0.0")
    assert code == 2 and not started and "MQVG_API_KEY" in err
    code, started, _ = run_main(monkeypatch, capsys, MQVG_ARTIFACT_ROOT=str(tmp_path),
                                MQVG_HOST="0.0.0.0", MQVG_API_KEY="k")
    assert code == 0 and started["host"] == "0.0.0.0"


def test_the_server_binds_to_loopback_by_default_and_warns_when_no_key_is_set(monkeypatch, capsys, tmp_path):
    code, started, err = run_main(monkeypatch, capsys, MQVG_ARTIFACT_ROOT=str(tmp_path),
                                  MQVG_PORT="9001")
    assert code == 0 and started["host"] == "127.0.0.1" and started["port"] == 9001
    assert "answer 503" in err
