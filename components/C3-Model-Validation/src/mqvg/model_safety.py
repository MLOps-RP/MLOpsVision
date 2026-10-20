"""Static scan of a pickled model file: what code would loading it run?

Loading a joblib/pickle file executes whatever the file asks for. This module
reads the pickle stream WITHOUT running it and lists every global (module.name)
the stream refers to, so the gate can decline to load a file that references
dangerous code.

How the scan avoids executing the file: it uses joblib's own unpickler (so
joblib's array layout, compression and nested object streams are parsed exactly
as `joblib.load` would parse them) but replaces `find_class`, the single place
where a pickle obtains a callable. Every global is recorded and answered with an
inert stub, so no file-supplied function is ever called. Only a short fixed list
of numpy/joblib classes needed to keep the stream readable is resolved for real.
Object-dtype arrays hold a nested pickle stream; the scan reads that with the
same restricted rules (it does not rely on joblib's own handling of it).

What a clean result means: every reference is on the configured allowlist.
It does not mean the model is safe. A static allowlist is not a sandbox: a
trusted library can contain a harmful function, and a file this scan cannot
parse is reported as such rather than as clean. Keep the requirement that only
trusted files are validated.
"""
from __future__ import annotations

import importlib
import inspect
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mqvg.config import ModelSafetyConfig

CLEAN, UNRECOGNISED, DANGEROUS = 0, 1, 2
LEVEL_NAMES = {CLEAN: "recognised only", UNRECOGNISED: "unrecognised code",
               DANGEROUS: "dangerous code"}
MAX_REFS_REPORTED = 40

# Resolved for real (never file-supplied behaviour): these only build arrays.
_REAL_GLOBALS = {
    "numpy.ndarray", "numpy.dtype",
    "numpy.core.multiarray._reconstruct", "numpy._core.multiarray._reconstruct",
    "numpy.core.multiarray.scalar", "numpy._core.multiarray.scalar",
}
_WRAPPER = "joblib.numpy_pickle.NumpyArrayWrapper"


class ScanError(Exception):
    """The file could not be scanned.

    `parse` is True when the bytes are not a valid pickle stream (a corrupt
    file), False when the scanner itself cannot handle this file or joblib
    version, which says nothing about the file.
    """

    def __init__(self, message: str, parse: bool):
        super().__init__(message)
        self.parse = parse


@dataclass
class ScanReport:
    references: list[str] = field(default_factory=list)
    dangerous: list[str] = field(default_factory=list)
    unrecognised: list[str] = field(default_factory=list)
    severity: int = CLEAN
    file_bytes: int = 0

    @property
    def verdict(self) -> str:
        return LEVEL_NAMES[self.severity]


class _Stub:
    """Inert stand-in for any callable or object a pickle asks for."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Stub()

    def __setstate__(self, state):
        pass

    def append(self, *args):
        pass

    def extend(self, *args):
        pass

    def add(self, *args):
        pass

    def update(self, *args, **kwargs):
        pass

    def __setitem__(self, key, value):
        pass


def classify(reference: str, policy: ModelSafetyConfig) -> int:
    """CLEAN, UNRECOGNISED or DANGEROUS for one 'module.name' reference."""
    module, _, name = reference.rpartition(".")
    if module == "builtins":
        if name in policy.blocked_builtins:
            return DANGEROUS
        return CLEAN if name in policy.allowed_builtins else UNRECOGNISED
    if reference in policy.blocked_globals or any(
            module == prefix or module.startswith(prefix + ".")
            for prefix in policy.blocked_module_prefixes):
        return DANGEROUS
    if any(module == prefix or module.startswith(prefix + ".")
           for prefix in policy.allowed_module_prefixes):
        return CLEAN
    return UNRECOGNISED


def _machinery():
    """Build the restricted unpickler classes; raises ScanError if joblib differs."""
    try:
        import numpy as np
        from joblib import numpy_pickle as joblib_pickle
        from joblib import numpy_pickle_utils as joblib_utils
        unpickler_base = joblib_pickle.NumpyUnpickler
        wrapper_base = joblib_pickle.NumpyArrayWrapper
        opener = getattr(joblib_pickle, "_validate_fileobject_and_memmap", None)
        if opener is None:
            opener = getattr(joblib_utils, "_read_fileobject")
            plain = True
        else:
            plain = False
    except (ImportError, AttributeError) as exc:
        raise ScanError(f"this joblib version is not supported by the scanner ({exc})",
                        parse=False) from None

    def resolve(refs: set[str], module: str, name: str, guarded):
        reference = f"{module}.{name}"
        refs.add(reference)
        if reference == _WRAPPER:
            return guarded
        if reference in _REAL_GLOBALS:
            return getattr(importlib.import_module(module), name)
        return _Stub

    class Inner(pickle._Unpickler):          # nested stream inside an object-dtype array
        def __init__(self, file, refs, guarded):
            super().__init__(file)
            self.refs, self.guarded = refs, guarded

        def find_class(self, module, name):
            return resolve(self.refs, module, name, self.guarded)

    class Guarded(wrapper_base):
        def read_array(self, unpickler, *args, **kwargs):
            if getattr(self.dtype, "hasobject", False):
                Inner(unpickler.file_handle, unpickler.refs, type(self)).load()
                return np.empty((0,), dtype=object)       # placeholder, never used
            return super().read_array(unpickler, *args, **kwargs)

    parameters = inspect.signature(unpickler_base.__init__).parameters

    class Outer(unpickler_base):
        def __init__(self, filename, file_handle, refs):
            self.refs = refs
            kwargs: dict[str, Any] = {}
            if "ensure_native_byte_order" in parameters:
                kwargs["ensure_native_byte_order"] = True
            super().__init__(filename, file_handle, **kwargs)

        def find_class(self, module, name):
            return resolve(self.refs, module, name, Guarded)

    return Outer, opener, plain


def scan_model_file(path: Path, policy: ModelSafetyConfig) -> ScanReport:
    """List and classify every global the pickle stream refers to. Never executes it."""
    outer, opener, plain = _machinery()
    refs: set[str] = set()
    try:
        with open(path, "rb") as handle:
            context = opener(handle, str(path), None)
            with context as opened:
                fobj = opened if plain else opened[0]
                if isinstance(fobj, str):
                    raise ScanError("legacy joblib format (joblib < 0.10) is not scanned",
                                    parse=False)
                outer(str(path), fobj, refs).load()
    except ScanError:
        raise
    except ImportError as exc:
        raise ScanError(f"scanner could not import a module needed to read the file: {exc}",
                        parse=False) from None
    except Exception as exc:  # noqa: BLE001 - any parse failure means "not scannable"
        raise ScanError(f"not a readable pickle stream: {type(exc).__name__}: {exc}",
                        parse=True) from None

    report = ScanReport(file_bytes=Path(path).stat().st_size)
    for reference in sorted(refs):
        level = classify(reference, policy)
        report.references.append(reference)
        if level == DANGEROUS:
            report.dangerous.append(reference)
        elif level == UNRECOGNISED:
            report.unrecognised.append(reference)
        report.severity = max(report.severity, level)
    return report
