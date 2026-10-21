"""The candidate catalogue: explicit file associations, served read-only, never inferred."""
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from mqvg.api.app import create_app
from mqvg.api.candidates import load_catalogue
from mqvg.api.settings import Settings

from conftest import CONFIGS, FIXTURES, ROOT

KEY = "k"
AUTH = {"X-API-Key": KEY}
CATALOGUE = ROOT / "candidates.yaml"
MANIFESTS = ["manifest.yaml", "hidden_groups_manifest.yaml", "extended_manifest.yaml"]


def write(tmp_path, entries, name="candidates.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"candidates": entries}), encoding="utf-8")
    return path


def entry(**kw):
    base = {"id": "m1", "description": "A model.", "model": "m/model.joblib", "eval_data": "d/eval.csv",
            "metadata": "m/metadata.json", "train_row_hashes": "d/hashes.txt"}
    base.update(kw)
    return base


@pytest.fixture
def files(tmp_path):
    """A small artifact root with every file an entry can name."""
    root = tmp_path / "root"
    for rel in ("m/model.joblib", "d/eval.csv", "m/metadata.json", "d/hashes.txt", "m/other.joblib"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"x")
    return root


# ----------------------------------------------------------------------- the loader
def test_a_valid_catalogue_is_served_exactly_as_declared_and_in_order(tmp_path, files):
    path = write(tmp_path, [entry(id="b", suggested_profile="extended"), entry(id="a", model="m/other.joblib")])
    cat = load_catalogue(path, files.resolve())
    assert cat.found and cat.error is None
    assert [c.id for c in cat.candidates] == ["b", "a"]                       # the file's order, not sorted
    first = cat.candidates[0]
    assert (first.model_path, first.eval_data_path, first.metadata_path, first.train_row_hashes_path) == (
        "m/model.joblib", "d/eval.csv", "m/metadata.json", "d/hashes.txt")
    assert first.suggested_profile == "extended" and first.available and first.unavailable_reason is None
    assert cat.candidates[1].model_path == "m/other.joblib"


def test_no_file_means_no_catalogue_rather_than_a_guess(tmp_path, files):
    assert load_catalogue(tmp_path / "nope.yaml", files.resolve()).found is False
    assert load_catalogue(None, files.resolve()).found is False
    assert load_catalogue(tmp_path / "nope.yaml", files.resolve()).candidates == []


def test_null_metadata_and_hashes_are_allowed_when_written_explicitly(tmp_path, files):
    cat = load_catalogue(write(tmp_path, [entry(metadata=None, train_row_hashes=None)]), files.resolve())
    c = cat.candidates[0]
    assert c.available and c.metadata_path is None and c.train_row_hashes_path is None


@pytest.mark.parametrize("missing", ["metadata", "train_row_hashes", "model", "eval_data", "description", "id"])
def test_a_missing_key_is_an_error_not_silently_none(tmp_path, files, missing):
    bad = entry()
    del bad[missing]
    cat = load_catalogue(write(tmp_path, [bad]), files.resolve())
    assert cat.found and cat.error and missing in cat.error and cat.candidates == []


def test_an_unknown_key_is_an_error(tmp_path, files):
    cat = load_catalogue(write(tmp_path, [entry(evaluation="d/eval.csv")]), files.resolve())
    assert cat.error and cat.candidates == []


@pytest.mark.parametrize("bad_id", ["", "has space", "../x", "a/b", "x" * 81])
def test_ids_are_restricted(tmp_path, files, bad_id):
    assert load_catalogue(write(tmp_path, [entry(id=bad_id)]), files.resolve()).error


def test_duplicate_ids_reject_the_whole_catalogue(tmp_path, files):
    cat = load_catalogue(write(tmp_path, [entry(), entry(model="m/other.joblib")]), files.resolve())
    assert cat.error and "m1" in cat.error and cat.candidates == []


@pytest.mark.parametrize("text", ["candidates: [", ": : :", "", "- just a list", "candidates: []", "candidates: not-a-list"])
def test_malformed_catalogues_offer_nothing(tmp_path, files, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    cat = load_catalogue(path, files.resolve())
    assert cat.found and cat.error and cat.candidates == []


def test_a_missing_file_makes_that_entry_unavailable_with_the_reason(tmp_path, files):
    (files / "d" / "eval.csv").unlink()
    (files / "m" / "metadata.json").unlink()
    cat = load_catalogue(write(tmp_path, [entry(), entry(id="ok", model="m/other.joblib", eval_data="d/hashes.txt", metadata=None)]), files.resolve())
    bad, ok = cat.candidates
    assert not bad.available and "evaluation data was not found" in bad.unavailable_reason
    assert "metadata file was not found" in bad.unavailable_reason
    assert ok.available                                                         # others are unaffected


def test_a_declared_but_missing_optional_file_is_not_treated_as_absent(tmp_path, files):
    (files / "d" / "hashes.txt").unlink()
    c = load_catalogue(write(tmp_path, [entry()]), files.resolve()).candidates[0]
    assert not c.available and "training row-hash file was not found" in c.unavailable_reason


@pytest.mark.parametrize("escape", ["../outside.joblib", "m/../../outside.joblib", "/etc/passwd"])
def test_a_path_outside_the_artifact_root_is_never_available(tmp_path, files, escape):
    (tmp_path / "outside.joblib").write_bytes(b"x")
    c = load_catalogue(write(tmp_path, [entry(model=escape)]), files.resolve()).candidates[0]
    assert not c.available and "outside the artifact root" in c.unavailable_reason


# ------------------------------------------------------------------------ the endpoint
def server(tmp_path, root, **overrides):
    values = dict(artifact_root=root, report_dir=tmp_path / "reports", config_dir=CONFIGS, api_key=KEY)
    values.update(overrides)
    return TestClient(create_app(Settings(**values)))


def test_the_endpoint_needs_the_key(tmp_path, files):
    assert server(tmp_path, files).get("/v1/candidates").status_code == 401


def test_the_catalogue_is_found_in_the_artifact_root_by_default(tmp_path, files):
    (files / "candidates.yaml").write_text(yaml.safe_dump({"candidates": [entry()]}), encoding="utf-8")
    body = server(tmp_path, files).get("/v1/candidates", headers=AUTH).json()
    assert body["catalogue_found"] is True and body["error"] is None
    assert [c["id"] for c in body["candidates"]] == ["m1"]
    assert body["candidates"][0]["model_path"] == "m/model.joblib"


def test_a_server_without_a_catalogue_says_so(tmp_path, files):
    body = server(tmp_path, files).get("/v1/candidates", headers=AUTH).json()
    assert body == {"catalogue_found": False, "error": None, "candidates": []}


def test_the_catalogue_location_is_configurable(tmp_path, files):
    elsewhere = write(tmp_path, [entry(id="elsewhere")], name="my_models.yaml")
    body = server(tmp_path, files, candidates_file=elsewhere).get("/v1/candidates", headers=AUTH).json()
    assert [c["id"] for c in body["candidates"]] == ["elsewhere"]


def test_edits_to_the_catalogue_are_picked_up_without_a_restart(tmp_path, files):
    path = write(tmp_path, [entry(id="one")])
    client = server(tmp_path, files, candidates_file=path)
    assert [c["id"] for c in client.get("/v1/candidates", headers=AUTH).json()["candidates"]] == ["one"]
    write(tmp_path, [entry(id="one"), entry(id="two", model="m/other.joblib")])
    assert [c["id"] for c in client.get("/v1/candidates", headers=AUTH).json()["candidates"]] == ["one", "two"]
    (files / "m" / "other.joblib").unlink()
    two = client.get("/v1/candidates", headers=AUTH).json()["candidates"][1]
    assert two["available"] is False


def test_a_broken_catalogue_is_reported_not_served(tmp_path, files):
    path = tmp_path / "bad.yaml"
    path.write_text("candidates: [", encoding="utf-8")
    body = server(tmp_path, files, candidates_file=path).get("/v1/candidates", headers=AUTH).json()
    assert body["catalogue_found"] is True and body["error"] and body["candidates"] == []


def test_the_endpoint_is_read_only_and_adds_no_route_that_can_change_anything(tmp_path, files):
    client = server(tmp_path, files)
    assert client.post("/v1/candidates", headers=AUTH).status_code == 405
    assert client.delete("/v1/candidates", headers=AUTH).status_code == 405


# ------------------------------------------------ the shipped catalogue: no wrong association
def shipped():
    return yaml.safe_load(CATALOGUE.read_text(encoding="utf-8"))["candidates"]


def manifest_entries():
    out = {}
    for name in MANIFESTS:
        out.update(yaml.safe_load((FIXTURES / name).read_text(encoding="utf-8"))["fixtures"])
    return out


def need_fixtures():
    if not (FIXTURES / "good" / "model.joblib").exists():
        pytest.skip("sample fixtures not generated")


def test_the_shipped_catalogue_is_valid_and_complete():
    entries = shipped()
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)) == 15
    assert set(ids) == set(manifest_entries()), "every manifest fixture is offered, and nothing else"
    cat = load_catalogue(CATALOGUE, ROOT)
    assert cat.error is None and len(cat.candidates) == 15


def test_every_association_agrees_with_the_fixture_manifests():
    manifests = manifest_entries()
    for e in shipped():
        m = manifests[e["id"]]
        assert e["model"] == m["model"], e["id"]
        assert e["eval_data"] == m["eval_data"], e["id"]
        assert e["metadata"] == m["metadata"], e["id"]
        if "train_row_hashes" in m:                                              # only some manifests state it
            assert e["train_row_hashes"] == m["train_row_hashes"], e["id"]


def test_every_training_hash_file_is_the_one_the_metadata_itself_declares():
    """The metadata names its own hash file, so the association can be checked rather than assumed."""
    for e in shipped():
        declared = json.loads((ROOT / e["metadata"]).read_text(encoding="utf-8")).get("train_row_hashes_file")
        assert (declared or None) == (e["train_row_hashes"] or None), (e["id"], declared, e["train_row_hashes"])


def test_the_evaluation_data_has_the_columns_the_metadata_says_the_model_uses():
    need_fixtures()
    for e in shipped():
        meta = json.loads((ROOT / e["metadata"]).read_text(encoding="utf-8"))
        header = list(pd.read_csv(ROOT / e["eval_data"], nrows=0).columns)
        if meta.get("feature_columns"):
            assert set(meta["feature_columns"]) <= set(header), e["id"]
        if meta.get("target_column"):
            assert meta["target_column"] in header, e["id"]


@pytest.mark.parametrize("name", ["good", "leaky", "overfit", "biased", "miscalibrated", "hidden_group", "hidden_group_null"])
def test_the_model_expects_features_its_evaluation_data_provides(name):
    need_fixtures()
    e = next(x for x in shipped() if x["id"] == name)
    model = joblib.load(ROOT / e["model"])
    header = set(pd.read_csv(ROOT / e["eval_data"], nrows=0).columns)
    assert set(getattr(model, "feature_names_in_", header)) <= header


def test_the_suggested_profiles_exist():
    names = {p.stem for p in CONFIGS.glob("*.yaml")}
    for e in shipped():
        assert e.get("suggested_profile") is None or e["suggested_profile"] in names, e["id"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", ["good", "leaky", "hidden_group", "good_no_sensitive"])
def test_validating_a_listed_candidate_uses_exactly_its_declared_files(tmp_path, name):
    """End to end: the files a validation actually read are the ones the catalogue declares."""
    need_fixtures()
    client = server(tmp_path, ROOT)
    listed = {c["id"]: c for c in client.get("/v1/candidates", headers=AUTH).json()["candidates"]}[name]
    body = {"candidate_id": name, "model_path": listed["model_path"], "eval_data_path": listed["eval_data_path"],
            "metadata_path": listed["metadata_path"], "train_row_hashes_path": listed["train_row_hashes_path"]}
    if listed["suggested_profile"]:
        body["profile"] = listed["suggested_profile"]
    reply = client.post("/v1/validations", headers=AUTH, json=body)
    assert reply.status_code == 201, reply.text
    result = client.get(reply.json()["result_url"], headers=AUTH).json()
    used = result["candidate"]
    assert used["model_path"].replace("\\", "/").endswith(listed["model_path"])
    assert used["model_sha256"] == sha(ROOT / listed["model_path"])
    assert used["eval_data_sha256"] == sha(ROOT / listed["eval_data_path"])
    assert used["metadata_sha256"] == sha(ROOT / listed["metadata_path"])


def test_a_restricted_profile_stays_blocked_whatever_candidate_is_chosen(tmp_path):
    need_fixtures()
    client = server(tmp_path, ROOT)
    listed = client.get("/v1/candidates", headers=AUTH).json()["candidates"][0]
    reply = client.post("/v1/validations", headers=AUTH, json={
        "model_path": listed["model_path"], "eval_data_path": listed["eval_data_path"], "profile": "default"})
    assert reply.status_code == 422 and "does not scan model files" in reply.text
