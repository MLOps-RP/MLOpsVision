import json

from src.c4.audit import drift_report
from src.c4.audit.drift_report import DriftReport


def make(**kw):
    args = dict(model_name="income-classifier", model_version="2", baseline_version="v2",
                preprocessing_version="prep-1", scenario="S3_concept", window=21,
                drift_type="CONCEPT_DRIFT", diagnosis_confidence="confirmed", severity_score=0.41,
                severity_level="Medium", severity_breakdown={}, recommended_action="ROUTE_C2",
                target_component="C2", reason="test")
    args.update(kw)
    return DriftReport(**args)


def test_report_contains_contract_fields():
    d = make().to_dict()
    for key in ["drift_type", "severity_level", "recommended_action", "target_component",
                "affected_features", "affected_segments", "performance", "reason",
                "report_id", "created_at", "schema_version"]:
        assert key in d
    assert d["schema_version"] == "1.0"


def test_report_ids_are_unique():
    assert make().report_id != make().report_id


def test_save_writes_readable_json(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_report, "OUTPUTS", tmp_path)
    path = make().save()
    assert path.exists() and "ROUTE_C2" in path.name and "S3_concept" in str(path)
    assert json.loads(path.read_text())["window"] == 21
