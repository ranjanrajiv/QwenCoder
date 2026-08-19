"""Analysis run directories (spec 11 sections 8, 9, 121).

The **seventh** copy of the run-repository shape in this project (candidates, evaluations,
rankings, preferences, training, model_evaluations, experiments, and now analysis). The
duplication has been carried deliberately since Stage 7 and is re-flagged here rather than
quietly repeated: extracting a shared base class would touch eight stages at once, and each
repository's manifest type and lifecycle differ enough that the shared part is thinner than
it looks. The debt is real; this docstring is the marker.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..atomic_io import JsonlError, append_jsonl, atomic_write_json, read_json
from .errors import AnalysisRunError, AnalysisRunNotFoundError, AnalysisStoreError
from .models import AnalysisManifest, ExperimentLineage, utc_now_iso

MANIFEST_FILENAME = "manifest.json"
CONFIG_FILENAME = "config.yaml"
SUMMARY_FILENAME = "summary.json"
LOG_FILENAME = "analysis.log"


class AnalysisRunRepository:
    """Owns analysis run directories under ``analysis_root``."""

    def __init__(self, analysis_root: Path) -> None:
        self.analysis_root = Path(analysis_root)

    # ------------------------------------------------------------------------- paths

    def run_dir(self, analysis_run_id: str) -> Path:
        return self.analysis_root / analysis_run_id

    def subdir(self, analysis_run_id: str, name: str) -> Path:
        path = self.run_dir(analysis_run_id) / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def log_path(self, analysis_run_id: str) -> Path:
        return self.run_dir(analysis_run_id) / "logs" / LOG_FILENAME

    def _manifest_path(self, analysis_run_id: str) -> Path:
        return self.run_dir(analysis_run_id) / MANIFEST_FILENAME

    # -------------------------------------------------------------------- run ids

    def existing_run_ids(self) -> set[str]:
        if not self.analysis_root.is_dir():
            return set()
        return {
            p.name for p in self.analysis_root.iterdir()
            if p.is_dir() and (p / MANIFEST_FILENAME).is_file()
        }

    def new_run_id(self, now: datetime | None = None) -> str:
        """``analysis_YYYYMMDD_HHMMSS_xxxx`` (spec section 8)."""
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
        existing = self.existing_run_ids()
        for _ in range(10):
            candidate = f"analysis_{stamp}_{secrets.token_hex(2)}"
            if candidate not in existing:
                return candidate
        raise AnalysisRunError("could not mint a unique analysis run id after 10 attempts")

    # ---------------------------------------------------------------------- lifecycle

    def create_run(
        self,
        *,
        lineage: ExperimentLineage,
        benchmark_version: str | None = None,
        analysis_run_id: str | None = None,
    ) -> AnalysisManifest:
        analysis_run_id = analysis_run_id or self.new_run_id()
        manifest = AnalysisManifest(
            analysis_run_id=analysis_run_id,
            status="created",
            created_at=utc_now_iso(),
            lineage=lineage,
            benchmark_version=benchmark_version,
        )
        self._write_manifest(manifest)
        return manifest

    def get_run(self, analysis_run_id: str) -> AnalysisManifest:
        path = self._manifest_path(analysis_run_id)
        if not path.is_file():
            raise AnalysisRunNotFoundError(
                f"no analysis run {analysis_run_id!r} at {self.run_dir(analysis_run_id)}"
            )
        try:
            return AnalysisManifest.from_dict(read_json(path))
        except (JsonlError, AnalysisStoreError) as exc:
            raise AnalysisRunError(f"{path}: {exc}") from exc

    def list_runs(self) -> list[AnalysisManifest]:
        if not self.analysis_root.is_dir():
            return []
        manifests = [
            self.get_run(p.name) for p in sorted(self.analysis_root.iterdir())
            if p.is_dir() and (p / MANIFEST_FILENAME).is_file()
        ]
        return sorted(manifests, key=lambda m: m.created_at, reverse=True)

    def _write_manifest(self, manifest: AnalysisManifest) -> AnalysisManifest:
        atomic_write_json(self._manifest_path(manifest.analysis_run_id), manifest.to_dict())
        return manifest

    def update_status(self, analysis_run_id: str, status: str, **changes: Any) -> AnalysisManifest:
        return self._write_manifest(self.get_run(analysis_run_id).with_status(status, **changes))

    def start_run(self, analysis_run_id: str) -> AnalysisManifest:
        return self.update_status(analysis_run_id, "running", started_at=utc_now_iso())

    def complete_run(self, analysis_run_id: str) -> AnalysisManifest:
        return self.update_status(analysis_run_id, "completed", completed_at=utc_now_iso())

    def fail_run(self, analysis_run_id: str, *, error: dict[str, Any] | None = None) -> AnalysisManifest:
        return self.update_status(
            analysis_run_id, "failed", completed_at=utc_now_iso(), error=error
        )

    # ------------------------------------------------------------------------ artifacts

    def write_config(self, analysis_run_id: str, configuration: dict[str, Any]) -> None:
        path = self.run_dir(analysis_run_id) / CONFIG_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(configuration, sort_keys=True, default_flow_style=False),
            encoding="utf-8",
        )

    def write_json(self, analysis_run_id: str, relative: str, payload: Any) -> Path:
        path = self.run_dir(analysis_run_id) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, payload) if isinstance(payload, dict) else _write_list(path, payload)
        return path

    def write_jsonl(self, analysis_run_id: str, relative: str, rows: list[dict[str, Any]]) -> Path:
        path = self.run_dir(analysis_run_id) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        for row in rows:
            append_jsonl(path, row)
        path.touch(exist_ok=True)
        return path

    def write_text(self, analysis_run_id: str, relative: str, text: str) -> Path:
        path = self.run_dir(analysis_run_id) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_yaml(self, analysis_run_id: str, relative: str, payload: dict[str, Any]) -> Path:
        path = self.run_dir(analysis_run_id) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, default_flow_style=False), encoding="utf-8"
        )
        return path

    def read_summary(self, analysis_run_id: str) -> dict[str, Any] | None:
        path = self.run_dir(analysis_run_id) / SUMMARY_FILENAME
        return read_json(path) if path.is_file() else None


def _write_list(path: Path, payload: Any) -> None:
    """``atomic_write_json`` takes a mapping; a top-level JSON array needs its own writer."""
    import json
    import os

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


__all__ = [
    "CONFIG_FILENAME",
    "LOG_FILENAME",
    "MANIFEST_FILENAME",
    "SUMMARY_FILENAME",
    "AnalysisRunRepository",
]
