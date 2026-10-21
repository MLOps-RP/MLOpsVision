"""Uploads (opt-in), capabilities and the study reports: the API additions for the dashboard."""
import json
import os
import subprocess

import joblib
import pytest
from fastapi.testclient import TestClient

from mqvg.api.app import create_app
from mqvg.api.settings import Settings

from conftest import CONFIGS, ROOT
from test_api import AUTH, KEY, Payload, artifacts, make_settings, post  # noqa: F401  (artifacts is a fixture)

DOCS = ROOT / "docs"


def server(tmp_path, artifacts, **overrides):
    overrides.setdefault("docs_dir", DOCS)
    return TestClient(create_app(make_settings(tmp_path, artifacts, **overrides)))


@pytest.fixture
def uploads_on(tmp_path, artifacts):
    return server(tmp_path, artifacts, allow_uploads=True)


def send(client, kind, name, content, headers=AUTH):
    return client.post("/v1/uploads", data={"kind": kind}, files={"file": (name, content)}, headers=headers)


def tiny_bytes(artifacts, suffix):
    return next(p for p in artifacts.iterdir() if p.suffix == suffix).read_bytes()


# ------------------------------------------------------------------ capabilities
def test_capabilities_need_the_key(tmp_path, artifacts):
    assert server(tmp_path, artifacts).get("/v1/capabilities").status_code == 401


def test_uploads_are_off_by_default_and_the_server_says_so(tmp_path, artifacts):
    caps = server(tmp_path, artifacts).get("/v1/capabilities", headers=AUTH).json()
    assert caps["uploads_enabled"] is False and caps["default_profile"] == "extended"
    assert caps["unscanned_profiles_allowed"] is False
    assert caps["upload_limits"]["extensions"]["model"] == [".joblib", ".pkl", ".pickle"]
    assert caps["upload_limits"]["max_bytes"]["metadata"] == 2 * 1024 * 1024
    assert caps["upload_limits"]["max_bytes"]["model"] == 100 * 1024 * 1024


def test_the_operator_can_turn_uploads_on(uploads_on):
    assert uploads_on.get("/v1/capabilities", headers=AUTH).json()["uploads_enabled"] is True


def test_the_upload_limit_is_configurable(tmp_path, artifacts):
    caps = server(tmp_path, artifacts, upload_max_mb=5).get("/v1/capabilities", headers=AUTH).json()
    assert caps["upload_limits"]["max_bytes"]["eval"] == 5 * 1024 * 1024
    assert caps["upload_limits"]["max_bytes"]["hashes"] == 5 * 1024 * 1024      # never above the global limit


# ----------------------------------------------------------------------- uploads
def test_uploads_are_refused_unless_the_operator_enabled_them(tmp_path, artifacts):
    client = server(tmp_path, artifacts)
    reply = send(client, "eval", "e.csv", b"a,b\n1,2\n")
    assert reply.status_code == 403 and "MQVG_ALLOW_UPLOADS" in reply.json()["detail"]
    assert not (artifacts / "uploads").exists()                 # nothing was written


def test_uploading_needs_the_key(uploads_on):
    assert send(uploads_on, "eval", "e.csv", b"x", headers={}).status_code == 401
    assert send(uploads_on, "eval", "e.csv", b"x", headers={"X-API-Key": "wrong"}).status_code == 401


def test_an_uploaded_candidate_can_be_validated_end_to_end(uploads_on, artifacts, tmp_path):
    model = send(uploads_on, "model", "my model.joblib", tiny_bytes(artifacts, ".joblib"))
    data = send(uploads_on, "eval", "eval data.csv", tiny_bytes(artifacts, ".csv"))
    meta = send(uploads_on, "metadata", "meta.json", tiny_bytes(artifacts, ".json"))
    assert (model.status_code, data.status_code, meta.status_code) == (201, 201, 201)
    info = model.json()
    assert info["kind"] == "model" and info["bytes"] == len(tiny_bytes(artifacts, ".joblib"))
    assert info["path"] == f"uploads/{info['upload_id']}/model.joblib"      # the server chose the name
    import hashlib
    assert info["sha256"] == hashlib.sha256(tiny_bytes(artifacts, ".joblib")).hexdigest()
    assert (artifacts / info["path"]).is_file()

    result = uploads_on.post("/v1/validations", headers=AUTH, json={
        "candidate_id": "uploaded", "model_path": info["path"], "eval_data_path": data.json()["path"],
        "metadata_path": meta.json()["path"]})
    assert result.status_code == 201
    assert result.json()["action_mode"] == "RECOMMENDATION" and result.json()["profile"] == "extended"


def test_the_client_file_name_never_reaches_the_disk(uploads_on, artifacts):
    reply = send(uploads_on, "model", "../../evil.joblib", b"x")
    assert reply.status_code == 201
    assert reply.json()["path"].endswith("/model.joblib") and ".." not in reply.json()["path"]
    assert not (artifacts.parent / "evil.joblib").exists()


@pytest.mark.parametrize("kind, name", [
    ("model", "model.exe"), ("model", "model"), ("model", "model.csv"), ("eval", "e.joblib"),
    ("eval", "e.csv.exe"), ("metadata", "m.csv"), ("hashes", "h.json"), ("model", "x" * 300 + ".joblib")])
def test_only_the_file_types_a_kind_accepts_are_stored(uploads_on, artifacts, kind, name):
    assert send(uploads_on, kind, name, b"x").status_code == 422
    assert not (artifacts / "uploads").exists() or not any((artifacts / "uploads").iterdir())


def test_an_unknown_kind_is_rejected(uploads_on):
    assert send(uploads_on, "script", "s.py", b"x").status_code == 422
    assert uploads_on.post("/v1/uploads", files={"file": ("e.csv", b"x")}, headers=AUTH).status_code == 422   # no kind
    assert uploads_on.post("/v1/uploads", data={"kind": "eval"}, headers=AUTH).status_code == 422             # no file


def test_a_file_over_the_limit_is_refused_and_leaves_nothing_behind(tmp_path, artifacts):
    client = server(tmp_path, artifacts, allow_uploads=True, upload_max_mb=1)
    reply = send(client, "eval", "big.csv", b"a,b\n" + b"1,2\n" * 600_000)       # about 2.4 MB
    assert reply.status_code == 413 and "1 MB" in reply.json()["detail"]
    uploads = artifacts / "uploads"
    assert not uploads.exists() or not any(uploads.iterdir())


def test_a_small_kind_has_a_smaller_limit(tmp_path, artifacts):
    client = server(tmp_path, artifacts, allow_uploads=True)
    assert send(client, "metadata", "m.json", b"{" + b" " * (3 * 1024 * 1024) + b"}").status_code == 413


def test_an_upload_folder_that_leaves_the_root_is_refused(uploads_on, artifacts, tmp_path):
    outside = tmp_path / "outside_uploads"
    outside.mkdir()
    link = artifacts / "uploads"
    made = (subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True).returncode == 0
            if os.name == "nt" else False)
    if not made:
        try:
            os.symlink(outside, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("neither a junction nor a symlink can be created here")
    reply = send(uploads_on, "eval", "e.csv", b"a\n1\n")
    assert reply.status_code == 500 and "outside the artifact root" in reply.json()["detail"]
    assert not any(outside.iterdir())


def test_the_security_restrictions_still_apply_to_uploaded_models(uploads_on, artifacts, tmp_path):
    marker = tmp_path / "pwned"
    joblib.dump(Payload(marker), tmp_path / "evil.joblib")
    model = send(uploads_on, "model", "evil.joblib", (tmp_path / "evil.joblib").read_bytes()).json()
    data = send(uploads_on, "eval", "e.csv", tiny_bytes(artifacts, ".csv")).json()
    reply = uploads_on.post("/v1/validations", headers=AUTH, json={"model_path": model["path"], "eval_data_path": data["path"]})
    assert reply.status_code == 201 and reply.json()["decision"] == "ROLLBACK"
    assert "unsafe_model_file" in [r["rule_id"] for r in reply.json()["triggered_rules"]]
    assert not marker.exists(), "an uploaded model's code ran"
    # an unscanned profile is still refused, whatever the file
    refused = uploads_on.post("/v1/validations", headers=AUTH, json={
        "model_path": model["path"], "eval_data_path": data["path"], "profile": "default"})
    assert refused.status_code == 422 and "does not scan model files" in refused.text


def test_uploading_does_not_widen_what_a_validation_request_may_read(uploads_on, tmp_path):
    outside = tmp_path / "secret.joblib"
    outside.write_bytes(b"x")
    reply = uploads_on.post("/v1/validations", headers=AUTH, json={
        "model_path": str(outside), "eval_data_path": "uploads/none.csv"})
    assert reply.status_code == 422 and "outside the artifact root" in reply.text


# ------------------------------------------------------------------------ studies
def test_the_studies_are_listed_with_what_exists(tmp_path, artifacts):
    client = server(tmp_path, artifacts)
    reply = client.get("/v1/studies", headers=AUTH)
    assert reply.status_code == 200
    studies = {s["id"]: s for s in reply.json()["studies"]}
    assert set(studies) == {"baseline_comparison", "sensitivity_analysis", "hidden_groups_results"}
    for name in ("baseline_comparison", "sensitivity_analysis"):
        assert studies[name]["has_data"] and studies[name]["has_report"] and studies[name]["generated_at_utc"]
        assert studies[name]["script"].startswith("scripts/")
    assert studies["hidden_groups_results"]["has_report"] and not studies["hidden_groups_results"]["has_data"]


def test_the_baseline_study_is_served_exactly_as_the_script_wrote_it(tmp_path, artifacts):
    reply = server(tmp_path, artifacts).get("/v1/studies/baseline_comparison", headers=AUTH)
    assert reply.status_code == 200
    body = reply.json()
    assert body["data"] == json.loads((DOCS / "baseline_comparison.json").read_text(encoding="utf-8"))
    summary = body["data"]["summary"]
    assert set(summary) == set(body["data"]["gates"]) and body["data"]["models"]
    for gate in summary.values():
        assert gate["caught"] <= gate["defects"] and gate["false_alarms"] <= gate["clean"]


def test_the_sensitivity_study_is_served(tmp_path, artifacts):
    data = server(tmp_path, artifacts).get("/v1/studies/sensitivity_analysis", headers=AUTH).json()["data"]
    assert data["families"] == ["weights", "thresholds", "per-check", "bands", "missing"]
    for case in data["cases"]:
        assert set(case["families"]) == set(data["families"])
        for family in case["families"].values():
            assert 0 <= family["kept"] <= family["total"]


def test_study_reports_are_markdown(tmp_path, artifacts):
    client = server(tmp_path, artifacts)
    for name in ("baseline_comparison", "sensitivity_analysis", "hidden_groups_results"):
        reply = client.get(f"/v1/studies/{name}/report", headers=AUTH)
        assert reply.status_code == 200 and reply.headers["content-type"].startswith("text/markdown")
        assert reply.text.startswith("# ")


def test_a_study_without_machine_readable_data_says_so(tmp_path, artifacts):
    reply = server(tmp_path, artifacts).get("/v1/studies/hidden_groups_results", headers=AUTH)
    assert reply.status_code == 404 and "only a report" in reply.json()["detail"]


@pytest.mark.parametrize("study_id", ["nope", "../etc/passwd", "..%2Fsecret", "BASELINE", "a" * 60, "baseline.json"])
def test_study_ids_cannot_name_files(tmp_path, artifacts, study_id):
    client = server(tmp_path, artifacts)
    for suffix in ("", "/report"):
        assert client.get(f"/v1/studies/{study_id}{suffix}", headers=AUTH).status_code in (404, 422)


def test_the_studies_need_the_key(tmp_path, artifacts):
    client = server(tmp_path, artifacts)
    for url in ("/v1/studies", "/v1/studies/baseline_comparison", "/v1/studies/baseline_comparison/report"):
        assert client.get(url).status_code == 401, url


def test_missing_or_broken_studies_are_unavailable_not_errors(tmp_path, artifacts):
    empty = tmp_path / "empty_docs"
    empty.mkdir()
    client = server(tmp_path, artifacts, docs_dir=empty)
    listed = {s["id"]: s for s in client.get("/v1/studies", headers=AUTH).json()["studies"]}
    assert not any(s["has_data"] or s["has_report"] for s in listed.values())
    missing = client.get("/v1/studies/baseline_comparison", headers=AUTH)
    assert missing.status_code == 404 and "scripts/baseline_comparison.py" in missing.json()["detail"]
    assert client.get("/v1/studies/baseline_comparison/report", headers=AUTH).status_code == 404

    (empty / "baseline_comparison.json").write_text("{not json", encoding="utf-8")
    (empty / "sensitivity_analysis.json").write_text("[1, 2]", encoding="utf-8")
    listed = {s["id"]: s for s in client.get("/v1/studies", headers=AUTH).json()["studies"]}
    assert not listed["baseline_comparison"]["has_data"] and not listed["sensitivity_analysis"]["has_data"]
    assert client.get("/v1/studies/baseline_comparison", headers=AUTH).status_code == 404


def test_the_committed_study_files_match_the_documented_shape():
    baseline = json.loads((DOCS / "baseline_comparison.json").read_text(encoding="utf-8"))
    assert baseline["study"] == "baseline_comparison" and len(baseline["models"]) == 11
    assert {m["truth"] for m in baseline["models"]} <= {"defect", "clean", "excluded"}
    sensitivity = json.loads((DOCS / "sensitivity_analysis.json").read_text(encoding="utf-8"))
    assert sensitivity["study"] == "sensitivity_analysis" and len(sensitivity["cases"]) == 9
