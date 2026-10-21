"""API settings, read from the environment (nothing secret is ever logged or returned)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROFILE = "extended"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
DEFAULT_DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
TRUE = {"1", "true", "yes", "on"}


class SettingsError(Exception):
    """The API cannot start with these settings."""


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUE


@dataclass(frozen=True)
class Settings:
    """Everything the API needs to know about its environment."""
    artifact_root: Path                 # model, data and metadata files must live under this
    report_dir: Path                    # where each run's result.json, report.md, summary.json go
    config_dir: Path = DEFAULT_CONFIG_DIR
    api_key: str | None = None
    allow_no_auth: bool = False         # local development only: serve without an API key
    allow_unscanned_profiles: bool = False   # permit profiles that load models without scanning
    default_profile: str = DEFAULT_PROFILE
    queue_timeout_seconds: float = 300.0     # how long a request waits for the validation lock
    allow_uploads: bool = False         # accept model/data uploads (a model file is code): off by default
    upload_max_mb: int = 100            # per uploaded file
    docs_dir: Path = DEFAULT_DOCS_DIR   # where the study reports (baseline, sensitivity, ...) are read from
    candidates_file: Path | None = None  # the models offered for validation; default <artifact root>/candidates.yaml

    @classmethod
    def from_env(cls) -> "Settings":
        root = os.environ.get("MQVG_ARTIFACT_ROOT", "").strip()
        if not root:
            raise SettingsError("MQVG_ARTIFACT_ROOT must name the folder the API may read "
                                "model and data files from")
        return cls(
            artifact_root=Path(root),
            report_dir=Path(os.environ.get("MQVG_REPORT_DIR", "mqvg_api_reports")),
            config_dir=Path(os.environ.get("MQVG_CONFIG_DIR", str(DEFAULT_CONFIG_DIR))),
            api_key=os.environ.get("MQVG_API_KEY") or None,
            allow_no_auth=_flag("MQVG_ALLOW_NO_AUTH"),
            allow_unscanned_profiles=_flag("MQVG_ALLOW_UNSCANNED_PROFILES"),
            default_profile=os.environ.get("MQVG_DEFAULT_PROFILE", DEFAULT_PROFILE),
            queue_timeout_seconds=float(os.environ.get("MQVG_QUEUE_TIMEOUT", "300")),
            allow_uploads=_flag("MQVG_ALLOW_UPLOADS"),
            upload_max_mb=int(os.environ.get("MQVG_UPLOAD_MAX_MB", "100")),
            docs_dir=Path(os.environ.get("MQVG_DOCS_DIR", str(DEFAULT_DOCS_DIR))),
            candidates_file=Path(os.environ["MQVG_CANDIDATES_FILE"]) if os.environ.get("MQVG_CANDIDATES_FILE") else None)

    def __repr__(self) -> str:      # never print the key
        return (f"Settings(artifact_root={self.artifact_root}, report_dir={self.report_dir}, "
                f"config_dir={self.config_dir}, api_key={'set' if self.api_key else 'unset'}, "
                f"allow_no_auth={self.allow_no_auth}, "
                f"allow_unscanned_profiles={self.allow_unscanned_profiles}, "
                f"default_profile={self.default_profile}, allow_uploads={self.allow_uploads})")
