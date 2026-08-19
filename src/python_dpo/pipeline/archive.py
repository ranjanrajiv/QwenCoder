"""Experiment archiving (spec 12 sections 73, 74).

``experiment archive`` tars and gzips the whole experiment run directory, plus an
``archive_manifest.json`` member holding every file's path and SHA-256 -- so
``experiment inspect --archive`` can answer "what is in this archive" by reading one
small JSON member, never by extracting the (potentially multi-gigabyte, adapter-carrying)
archive. Restoring an archive back into a live experiment run is deliberately out of scope
here (plan Phase 4 deferral, matching CLAUDE.md's Scope Control) -- the spec itself says
the initial implementation need not support it.
"""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ArchiveError
from .hashing import sha256_file
from .repository import ExperimentRunRepository

ARCHIVE_MANIFEST_FILENAME = "archive_manifest.json"
ARCHIVE_MANIFEST_VERSION = "archive_manifest_v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class ArchiveManifest:
    """One archive's table of contents (spec section 73)."""

    experiment_run_id: str
    created_at: str
    file_count: int
    total_bytes: int
    files: dict[str, str] = field(default_factory=dict)
    manifest_version: str = ARCHIVE_MANIFEST_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "experiment_run_id": self.experiment_run_id,
            "created_at": self.created_at,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "files": dict(self.files),
        }

    @classmethod
    def from_dict(cls, data: Any) -> ArchiveManifest:
        if not isinstance(data, dict):
            raise ArchiveError("archive manifest: expected a JSON object")
        missing = sorted({"experiment_run_id", "created_at", "file_count", "total_bytes"} - set(data))
        if missing:
            raise ArchiveError(f"archive manifest: missing required field(s): {', '.join(missing)}")
        return cls(
            manifest_version=data.get("manifest_version", ARCHIVE_MANIFEST_VERSION),
            experiment_run_id=data["experiment_run_id"],
            created_at=data["created_at"],
            file_count=data["file_count"],
            total_bytes=data["total_bytes"],
            files=data.get("files") or {},
        )


def archive_experiment(
    repo: ExperimentRunRepository, experiment_run_id: str, dest_dir: Path
) -> Path:
    """Write ``dest_dir/<experiment_run_id>.tar.gz`` and return its path."""
    source_dir = repo.run_dir(experiment_run_id)
    if not source_dir.is_dir():
        raise ArchiveError(f"no experiment run directory at {source_dir}")

    files: dict[str, str] = {}
    total_bytes = 0
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source_dir).as_posix()
        files[relative] = sha256_file(path)
        total_bytes += path.stat().st_size

    manifest = ArchiveManifest(
        experiment_run_id=experiment_run_id,
        created_at=_utc_now_iso(),
        file_count=len(files),
        total_bytes=total_bytes,
        files=files,
    )
    manifest_bytes = json.dumps(manifest.to_dict(), sort_keys=True, indent=2).encode("utf-8")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / f"{experiment_run_id}.tar.gz"

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname=experiment_run_id)
        info = tarfile.TarInfo(name=f"{experiment_run_id}/{ARCHIVE_MANIFEST_FILENAME}")
        info.size = len(manifest_bytes)
        info.mtime = int(datetime.now(timezone.utc).timestamp())
        tar.addfile(info, io.BytesIO(manifest_bytes))

    return archive_path


def inspect_archive(archive_path: Path) -> ArchiveManifest:
    """Read ``archive_manifest.json`` out of the archive without extracting anything
    else (spec section 74)."""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise ArchiveError(f"no archive at {archive_path}")

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            member = next(
                (m for m in tar.getmembers() if m.name.endswith(f"/{ARCHIVE_MANIFEST_FILENAME}")),
                None,
            )
            if member is None:
                raise ArchiveError(f"{archive_path}: no {ARCHIVE_MANIFEST_FILENAME} inside")
            extracted = tar.extractfile(member)
            if extracted is None:
                raise ArchiveError(f"{archive_path}: {ARCHIVE_MANIFEST_FILENAME} could not be read")
            data = json.loads(extracted.read().decode("utf-8"))
    except tarfile.TarError as exc:
        raise ArchiveError(f"{archive_path}: not a valid archive: {exc}") from exc

    return ArchiveManifest.from_dict(data)


__all__ = [
    "ARCHIVE_MANIFEST_FILENAME",
    "ARCHIVE_MANIFEST_VERSION",
    "ArchiveManifest",
    "archive_experiment",
    "inspect_archive",
]
