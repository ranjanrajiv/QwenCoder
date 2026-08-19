"""The experiment-wide artifact manifest (spec 12 sections 69, 70).

``artifacts.json`` is the pointer table the plan's "canonical stores plus pointers"
decision relies on: stage outputs stay in their existing `data/<stage>/runs/<run_id>/`
directories, and this module records each one's path and SHA-256 without copying it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..atomic_io import atomic_write_json, read_json
from .hashing import sha256_file, sha256_tree


class ArtifactError(Exception):
    """Raised when an artifact reference or the artifact manifest is malformed."""


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ArtifactRef:
    """One entry in ``artifacts.json``: a name pointing at a path, its SHA-256, and size."""

    name: str
    path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_text(self.path, "path")
        _require_text(self.sha256, "sha256")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes < 0:
            raise ArtifactError("bytes must be an integer of 0 or greater")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}

    @classmethod
    def from_dict(cls, name: str, data: Any) -> ArtifactRef:
        if not isinstance(data, dict):
            raise ArtifactError(f"artifact {name!r}: expected a JSON object")
        unknown = sorted(set(data) - {"path", "sha256", "bytes"})
        if unknown:
            raise ArtifactError(f"artifact {name!r}: unknown field(s): {', '.join(unknown)}")
        missing = sorted({"path", "sha256", "bytes"} - set(data))
        if missing:
            raise ArtifactError(
                f"artifact {name!r}: missing required field(s): {', '.join(missing)}"
            )
        return cls(name=name, path=data["path"], sha256=data["sha256"], bytes=data["bytes"])


def make_artifact_ref(name: str, path: Path, *, relative_to: Path | None = None) -> ArtifactRef:
    """Build an :class:`ArtifactRef` for a file or a directory.

    A directory is hashed with :func:`~python_dpo.pipeline.hashing.sha256_tree` and its
    size is the sum of every regular file under it; a file uses
    :func:`~python_dpo.pipeline.hashing.sha256_file` directly.
    """
    path = Path(path)
    if path.is_dir():
        digest = sha256_tree(path)
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    elif path.is_file():
        digest = sha256_file(path)
        size = path.stat().st_size
    else:
        raise ArtifactError(f"artifact {name!r}: no such file or directory: {path}")

    display_path = path.relative_to(relative_to) if relative_to is not None else path
    return ArtifactRef(name=name, path=str(display_path), sha256=digest, bytes=size)


def write_artifact_manifest(path: Path, refs: dict[str, ArtifactRef]) -> None:
    atomic_write_json(path, {name: ref.to_dict() for name, ref in sorted(refs.items())})


def read_artifact_manifest(path: Path) -> dict[str, ArtifactRef]:
    if not Path(path).is_file():
        return {}
    data = read_json(path)
    return {name: ArtifactRef.from_dict(name, value) for name, value in data.items()}


__all__ = [
    "ArtifactError",
    "ArtifactRef",
    "make_artifact_ref",
    "read_artifact_manifest",
    "write_artifact_manifest",
]
