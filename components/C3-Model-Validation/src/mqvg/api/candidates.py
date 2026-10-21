"""The catalogue of models the server offers for validation.

The catalogue is an explicit YAML file (`candidates.yaml`). It states, for every model, which
evaluation data, metadata and training-hash file belong to it. Nothing here infers an association
from a file name, a folder or a convention: a client is only ever given what the file declares.

Read-only and side-effect free. Paths are relative to the artifact root, and a path that would leave
the root, or a file that does not exist, makes the entry unavailable (with the reason), never usable.
A malformed catalogue is reported as an error and offers nothing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

CandidateId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9._-]{1,80}$")]
RelPath = Annotated[str, StringConstraints(min_length=1, max_length=400)]


class _Entry(BaseModel):
    """One line of the catalogue. Every key except `suggested_profile` is required (null is allowed
    for metadata and hashes), so a typo cannot silently turn into "no metadata"."""
    model_config = ConfigDict(extra="forbid")
    id: CandidateId
    description: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    model: RelPath
    eval_data: RelPath
    metadata: RelPath | None
    train_row_hashes: RelPath | None
    suggested_profile: Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]{1,60}$")] | None = None


class _Catalogue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidates: list[_Entry] = Field(min_length=1)


class Candidate(BaseModel):
    """A catalogue entry as served: its files, and whether they can be used on this server."""
    id: str
    description: str
    model_path: str
    eval_data_path: str
    metadata_path: str | None
    train_row_hashes_path: str | None
    suggested_profile: str | None = None
    available: bool
    unavailable_reason: str | None = None


class Catalogue(BaseModel):
    found: bool
    error: str | None = None
    candidates: list[Candidate] = Field(default_factory=list)


def _problem(root: Path, label: str, relative: str) -> str | None:
    """Why one declared file cannot be used, or None if it exists under the root."""
    try:
        resolved = (root / relative).resolve()
    except (OSError, ValueError):
        return f"the {label} path is invalid"
    if not resolved.is_relative_to(root):
        return f"the {label} path is outside the artifact root"
    if not resolved.is_file():
        return f"the {label} was not found"
    return None


def load_catalogue(path: Path | None, root: Path) -> Catalogue:
    """Read the catalogue and check every file it names. Never raises."""
    if path is None or not path.is_file():
        return Catalogue(found=False)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        parsed = _Catalogue.model_validate(raw)
    except (OSError, yaml.YAMLError) as exc:
        return Catalogue(found=True, error=f"the candidate list could not be read ({type(exc).__name__})")
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        return Catalogue(found=True, error=f"the candidate list is invalid at {where}: {first['msg']}")
    ids = [e.id for e in parsed.candidates]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        return Catalogue(found=True, error=f"the candidate list repeats these ids: {', '.join(duplicates)}")

    candidates = []
    for entry in parsed.candidates:
        checks = [("model file", entry.model), ("evaluation data", entry.eval_data)]
        if entry.metadata is not None:
            checks.append(("metadata file", entry.metadata))
        if entry.train_row_hashes is not None:
            checks.append(("training row-hash file", entry.train_row_hashes))
        problems = [p for label, rel in checks if (p := _problem(root, label, rel))]
        candidates.append(Candidate(
            id=entry.id, description=entry.description, model_path=entry.model, eval_data_path=entry.eval_data,
            metadata_path=entry.metadata, train_row_hashes_path=entry.train_row_hashes,
            suggested_profile=entry.suggested_profile, available=not problems,
            unavailable_reason=None if not problems else "; ".join(problems)))
    return Catalogue(found=True, candidates=candidates)
