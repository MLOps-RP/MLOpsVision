"""The FastAPI application. See `mqvg.api` for the security model and `docs/api.md`."""
from __future__ import annotations

import hashlib
import logging
import secrets
import threading
import uuid
from collections import Counter
from pathlib import Path
from typing import Annotated

from fastapi import (
    Depends, FastAPI, File, Form, HTTPException, Path as PathParam, Query, Request, Response, UploadFile,
)
from fastapi.responses import PlainTextResponse
from fastapi.security import APIKeyHeader

from mqvg import __version__
from mqvg.api import profile_guide, studies as study_store
from mqvg.api.candidates import load_catalogue
from mqvg.api.models import (
    Capabilities, CandidateInfo, CandidateList, CheckInfo, Health, ProfileInfo, ProfileList, RuleSummary, RunEntry, RunList, StudyData,
    StudyInfo, StudyList, UploadInfo, UploadKind, UploadLimits, ValidationRequest, ValidationSummary,
)
from mqvg.api.settings import Settings, SettingsError
from mqvg.catalogue import CATALOGUE
from mqvg.config import ConfigError, GateConfig, load_config
from mqvg.schemas import CandidateBundle, ValidationResult
from mqvg.workflow import validate_and_report

log = logging.getLogger("mqvg.api")
SAFETY_CHECK = "operational.model_file_safety"
RUN_ID = r"^[0-9a-f]{32}$"
STUDY_ID = r"^[a-z_]{1,40}$"
MB = 1024 * 1024
CHUNK = MB
# What an upload may be called. The stored name is chosen by the server (kind + a validated extension),
# never taken from the client.
UPLOAD_EXTENSIONS: dict[str, list[str]] = {
    "model": [".joblib", ".pkl", ".pickle"], "eval": [".csv"],
    "metadata": [".json", ".yaml", ".yml"], "hashes": [".txt"],
}
UPLOAD_CAP_MB = {"metadata": 2, "hashes": 50}      # smaller kinds never need the full limit
PROFILE_SUFFIX = ".yaml"

StudyIdPath = Annotated[str, PathParam(pattern=STUDY_ID, description="one of the ids listed by GET /v1/studies")]
RunIdPath = Annotated[str, PathParam(
    pattern=RUN_ID, description="the run_id returned by POST /v1/validations")]

key_header = APIKeyHeader(name="X-API-Key", auto_error=False,
                          description="the key the server was started with (MQVG_API_KEY)")


def _scans_files(config: GateConfig) -> bool:
    """True if the profile scans the model file and will refuse to load a flagged one."""
    return config.checks[SAFETY_CHECK].enabled and config.model_safety.refuse_at != "never"


def summarize(result: ValidationResult, profile: str) -> ValidationSummary:
    card, decision = result.scorecard, result.decision
    return ValidationSummary(
        run_id=result.run_id, candidate_id=result.candidate.candidate_id, profile=profile,
        decision=decision.decision, action_mode=decision.action_mode, summary=decision.summary,
        mqs=card.mqs if card.mqs_sufficient_evidence else None, mqs_lower_bound=card.mqs_lower_bound,
        mqs_sufficient_evidence=card.mqs_sufficient_evidence,
        overall_coverage=card.overall_coverage, provisional=card.provisional,
        dimension_scores={d.value: s.score for d, s in card.dimensions.items()},
        triggered_rules=[RuleSummary(rule_id=r.rule_id, kind=r.kind, effect=r.effect, reason=r.reason)
                         for r in decision.triggered_rules],
        check_counts=dict(Counter(c.status.value for c in result.checks)),
        created_at_utc=result.created_at_utc,
        result_url=f"/v1/validations/{result.run_id}",
        report_url=f"/v1/validations/{result.run_id}/report")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    root = settings.artifact_root.resolve()
    if not root.is_dir():
        raise SettingsError(f"the artifact root {root} is not a folder")
    configs = settings.config_dir.resolve()
    if not configs.is_dir():
        raise SettingsError(f"the config folder {configs} does not exist")
    reports = settings.report_dir.resolve()
    reports.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()                 # validations are timed; run one at a time
    profiles: dict[str, GateConfig] = {}

    app = FastAPI(
        title="C3 Model Quality & Validation Gate",
        version=__version__,
        description=(
            "Validates a trained model and answers with a PROMOTE / HOLD / ROLLBACK "
            "**recommendation**. Nothing is deployed, promoted or rolled back. "
            "Loading a model runs the code inside its file: read the security notes in "
            "docs/api.md before exposing this API."))

    # ------------------------------------------------------------ helpers
    def authenticate(key: Annotated[str | None, Depends(key_header)] = None) -> None:
        if settings.api_key is None:
            if settings.allow_no_auth:
                return
            raise HTTPException(503, "the API key is not configured: set MQVG_API_KEY "
                                     "(or MQVG_ALLOW_NO_AUTH=true for local development only)")
        if key is None or not secrets.compare_digest(key.encode(), settings.api_key.encode()):
            raise HTTPException(401, "missing or wrong API key", headers={"WWW-Authenticate": "ApiKey"})

    def profile_names() -> list[str]:
        return sorted(p.stem for p in configs.glob(f"*{PROFILE_SUFFIX}") if p.is_file())

    def load_profile(name: str) -> GateConfig:
        if name not in profiles:
            path = (configs / f"{name}{PROFILE_SUFFIX}").resolve()
            if path.parent != configs or not path.is_file():
                raise HTTPException(422, f"unknown profile '{name}'; available: {profile_names()}")
            try:
                profiles[name] = load_config(path)
            except ConfigError as exc:
                log.error("profile %s is invalid: %s", name, exc)
                raise HTTPException(500, f"profile '{name}' is misconfigured on the server") from None
        return profiles[name]

    def artifact(field: str, value: str) -> Path:
        """A file under the artifact root; anything that leaves it is rejected."""
        if "\x00" in value:
            raise HTTPException(422, f"{field}: invalid path")
        try:
            resolved = (root / value).resolve()     # an absolute `value` replaces `root`
        except (OSError, ValueError):
            raise HTTPException(422, f"{field}: invalid path") from None
        if not resolved.is_relative_to(root):
            raise HTTPException(422, f"{field}: the path is outside the artifact root")
        if not resolved.is_file():
            raise HTTPException(422, f"{field}: no such file under the artifact root")
        return resolved

    def run_folder(run_id: str) -> Path:
        folder = reports / run_id
        if not (folder / "summary.json").is_file():
            raise HTTPException(404, "no such validation run")
        return folder

    # ---------------------------------------------------------- endpoints
    @app.get("/health", response_model=Health, tags=["service"])
    def health() -> Health:
        auth = ("api key" if settings.api_key else
                "disabled (local development)" if settings.allow_no_auth else "not configured")
        return Health(status="ok", gate_version=__version__, authentication=auth)

    @app.get("/v1/profiles", response_model=ProfileList, tags=["configuration"],
             dependencies=[Depends(authenticate)])
    def list_profiles() -> ProfileList:
        infos = []
        for name in profile_names():
            try:
                config = load_profile(name)
            except HTTPException:
                continue
            scans = _scans_files(config)
            allowed = scans or settings.allow_unscanned_profiles
            guide = profile_guide.describe(config, scans)
            infos.append(ProfileInfo(
                name=name, description=config.description,
                enabled_checks=sum(c.enabled for c in config.checks.values()),
                scans_model_files=scans, allowed=allowed,
                weights={d.value: w for d, w in config.weights.as_dict().items()},
                summary=guide.summary, highlights=guide.highlights, strictness=guide.strictness,
                strictness_text=guide.strictness_text,
                unavailable_reason=None if allowed else profile_guide.unavailable_reason(scans)))
        return ProfileList(default_profile=settings.default_profile, profiles=infos)

    @app.get("/v1/candidates", response_model=CandidateList, tags=["configuration"],
             dependencies=[Depends(authenticate)])
    def list_candidates() -> CandidateList:
        """The models this server offers, each with the evaluation data and metadata that belong to it.

        Read from an explicit catalogue (candidates.yaml); no association is ever inferred from file
        names. A validation request names the paths returned here.
        """
        path = settings.candidates_file or root / "candidates.yaml"
        catalogue = load_catalogue(path, root)
        return CandidateList(
            catalogue_found=catalogue.found, error=catalogue.error,
            candidates=[CandidateInfo(**c.model_dump()) for c in catalogue.candidates])

    @app.get("/v1/checks", response_model=list[CheckInfo], tags=["configuration"],
             dependencies=[Depends(authenticate)])
    def list_checks() -> list[CheckInfo]:
        return [CheckInfo(id=s.id, dimension=s.dimension.value, direction=s.direction,
                          unit=s.unit, description=s.description) for s in CATALOGUE]

    @app.post("/v1/validations", response_model=ValidationSummary, status_code=201,
              tags=["validation"], dependencies=[Depends(authenticate)],
              responses={401: {"description": "wrong API key"},
                         422: {"description": "bad path, profile or body"},
                         503: {"description": "another validation kept the gate busy"}})
    def create_validation(request: ValidationRequest, response: Response) -> ValidationSummary:
        """Validate one candidate and return the recommendation. Synchronous: it can take a minute."""
        name = request.profile or settings.default_profile
        config = load_profile(name)
        if not (_scans_files(config) or settings.allow_unscanned_profiles):
            allowed = [p for p in profile_names()
                       if _scans_files(load_profile(p))]
            raise HTTPException(422, (
                f"profile '{name}' does not scan model files before loading them, and "
                "loading a model runs the code inside it. This server only runs profiles "
                f"that scan: {allowed}."))
        optional = {k: artifact(k, v) for k, v in (
            ("metadata_path", request.metadata_path),
            ("train_row_hashes_path", request.train_row_hashes_path)) if v is not None}
        bundle = CandidateBundle(
            candidate_id=request.candidate_id,
            model_path=artifact("model_path", request.model_path),
            eval_data_path=artifact("eval_data_path", request.eval_data_path),
            target_column=request.target_column, positive_label=request.positive_label,
            sensitive_columns=request.sensitive_columns, **optional)

        run_id = uuid.uuid4().hex
        if not lock.acquire(timeout=settings.queue_timeout_seconds):
            raise HTTPException(503, "the gate is busy with another validation; try again",
                                headers={"Retry-After": "30"})
        try:
            output = validate_and_report(bundle, config, reports / run_id, run_id=run_id)
        except Exception:                       # noqa: BLE001 - never leak internals to the client
            log.exception("validation %s crashed", run_id)
            raise HTTPException(500, f"the validation crashed (run {run_id}); see the server log") from None
        finally:
            lock.release()
        summary = summarize(output.result, name)
        (reports / run_id / "summary.json").write_text(
            summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
        response.headers["Location"] = summary.result_url
        return summary

    @app.get("/v1/validations", response_model=RunList, tags=["validation"],
             dependencies=[Depends(authenticate)])
    def list_runs(limit: Annotated[int, Query(ge=1, le=200)] = 20) -> RunList:
        files = sorted(reports.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        entries = []
        for path in files[:limit]:
            s = ValidationSummary.model_validate_json(path.read_text(encoding="utf-8"))
            entries.append(RunEntry(run_id=s.run_id, candidate_id=s.candidate_id, profile=s.profile,
                                    decision=s.decision, mqs=s.mqs, created_at_utc=s.created_at_utc))
        return RunList(runs=entries)

    @app.get("/v1/validations/{run_id}", response_model=ValidationResult, tags=["validation"],
             dependencies=[Depends(authenticate)])
    def get_result(run_id: RunIdPath) -> ValidationResult:
        """The full, auditable result: every check, the scorecard, the rules and the configuration used."""
        return ValidationResult.model_validate_json(
            (run_folder(run_id) / "result.json").read_text(encoding="utf-8"))

    @app.get("/v1/validations/{run_id}/summary", response_model=ValidationSummary, tags=["validation"],
             dependencies=[Depends(authenticate)])
    def get_summary(run_id: RunIdPath) -> ValidationSummary:
        return ValidationSummary.model_validate_json(
            (run_folder(run_id) / "summary.json").read_text(encoding="utf-8"))

    @app.get("/v1/validations/{run_id}/report", response_class=PlainTextResponse, tags=["validation"],
             dependencies=[Depends(authenticate)])
    def get_report(run_id: RunIdPath) -> PlainTextResponse:
        """The unified Markdown report."""
        return PlainTextResponse((run_folder(run_id) / "report.md").read_text(encoding="utf-8"),
                                 media_type="text/markdown; charset=utf-8")

    # -------------------------------------------------- capabilities, uploads
    def upload_limits() -> UploadLimits:
        top = settings.upload_max_mb * MB
        return UploadLimits(
            max_bytes={k: min(top, UPLOAD_CAP_MB.get(k, settings.upload_max_mb) * MB) for k in UPLOAD_EXTENSIONS},
            extensions=UPLOAD_EXTENSIONS)

    @app.get("/v1/capabilities", response_model=Capabilities, tags=["configuration"],
             dependencies=[Depends(authenticate)])
    def capabilities() -> Capabilities:
        """What this server allows, so a client can show a feature as unavailable instead of failing."""
        return Capabilities(gate_version=__version__, uploads_enabled=settings.allow_uploads,
                            upload_limits=upload_limits(),
                            unscanned_profiles_allowed=settings.allow_unscanned_profiles,
                            default_profile=settings.default_profile)

    @app.post("/v1/uploads", response_model=UploadInfo, status_code=201, tags=["uploads"],
              dependencies=[Depends(authenticate)],
              responses={403: {"description": "uploads are disabled on this server"},
                         413: {"description": "the file is larger than the limit"},
                         422: {"description": "unknown kind or a file type that kind does not accept"}})
    def upload(request: Request, kind: Annotated[UploadKind, Form()], file: Annotated[UploadFile, File()]) -> UploadInfo:
        """Store one file under the artifact root, then name it in a validation request.

        Disabled unless the operator sets MQVG_ALLOW_UPLOADS. A model file is code: it is still scanned
        before it is loaded, and only scanning profiles run, exactly as for any other model.
        """
        if not settings.allow_uploads:
            raise HTTPException(403, "uploads are disabled on this server (the operator must set MQVG_ALLOW_UPLOADS=true)")
        name = file.filename or ""
        if "\x00" in name or len(name) > 255:
            raise HTTPException(422, "file: invalid file name")
        suffix = Path(name).suffix.lower()
        if suffix not in UPLOAD_EXTENSIONS[kind]:
            raise HTTPException(422, f"file: a '{kind}' upload must end in one of {UPLOAD_EXTENSIONS[kind]}")
        limit = upload_limits().max_bytes[kind]
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit + MB:
            raise HTTPException(413, f"the file is larger than the {limit // MB} MB limit for '{kind}' uploads")
        base = root / "uploads"
        base.mkdir(exist_ok=True)
        if not base.resolve().is_relative_to(root):
            raise HTTPException(500, "the upload folder is outside the artifact root")
        upload_id = uuid.uuid4().hex
        folder = base / upload_id
        folder.mkdir()
        target = folder / f"{kind}{suffix}"
        digest, size = hashlib.sha256(), 0
        try:
            with open(target, "wb") as out:
                while chunk := file.file.read(CHUNK):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"the file is larger than the {limit // MB} MB limit for '{kind}' uploads")
                    digest.update(chunk)
                    out.write(chunk)
        except BaseException:
            target.unlink(missing_ok=True)
            folder.rmdir()
            raise
        return UploadInfo(upload_id=upload_id, kind=kind, path=f"uploads/{upload_id}/{target.name}",
                          bytes=size, sha256=digest.hexdigest())

    # ------------------------------------------------------------- studies
    docs = settings.docs_dir.resolve()

    def study_spec(study_id: str) -> study_store.StudySpec:
        spec = study_store.STUDIES.get(study_id)
        if spec is None:
            raise HTTPException(404, "no such study")
        return spec

    @app.get("/v1/studies", response_model=StudyList, tags=["studies"], dependencies=[Depends(authenticate)])
    def list_studies() -> StudyList:
        """The study reports (baseline comparison, sensitivity analysis, ...) and whether each exists."""
        return StudyList(studies=[
            StudyInfo(id=s.id, title=s.title, description=s.description, script=s.script,
                      has_data=study_store.load_data(docs, s) is not None,
                      has_report=study_store.load_report(docs, s) is not None,
                      generated_at_utc=study_store.generated_at(docs, s))
            for s in study_store.STUDIES.values()])

    @app.get("/v1/studies/{study_id}", response_model=StudyData, tags=["studies"], dependencies=[Depends(authenticate)])
    def get_study(study_id: StudyIdPath) -> StudyData:
        """The machine-readable result of a study, exactly as its script wrote it."""
        spec = study_spec(study_id)
        data = study_store.load_data(docs, spec)
        if data is None:
            raise HTTPException(404, f"this study has no machine-readable result; generate it with {spec.script}"
                                if spec.data_file else "this study has no machine-readable result, only a report")
        return StudyData(id=spec.id, title=spec.title, generated_at_utc=study_store.generated_at(docs, spec), data=data)

    @app.get("/v1/studies/{study_id}/report", response_class=PlainTextResponse, tags=["studies"],
             dependencies=[Depends(authenticate)])
    def get_study_report(study_id: StudyIdPath) -> PlainTextResponse:
        spec = study_spec(study_id)
        text = study_store.load_report(docs, spec)
        if text is None:
            raise HTTPException(404, f"the report has not been generated; run {spec.script}")
        return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")

    return app
