"""Central owner of experiment run state and its directory tree (spec 12 sections 12, 14, 28).

The seventh instance of the run-repository shape already used by candidates, evaluations,
rankings, preferences, training and model_evaluations: one directory per run under
``experiments_root``, this repository the only code that mints experiment run ids or
writes the run's JSON/YAML artifacts.

Per the plan's "canonical stores plus pointers" decision, this repository never writes a
stage's actual output data -- only the experiment-level bookkeeping (the manifest, the
resolved config, the environment capture, the artifact pointer table, per-stage manifests,
lineage, and logs). The one deliberate exception is ``model/``, the packaged deliverable,
which Stage 12's own packaging stage writes directly into this tree.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..atomic_io import JsonlError, atomic_write_json, read_json
from .artifacts import ArtifactRef, read_artifact_manifest, write_artifact_manifest
from .errors import ExperimentRunNotFoundError
from .manifest import ExperimentManifest, ManifestError, StageManifest, utc_now_iso

MANIFEST_FILENAME = "manifest.json"
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"
ENVIRONMENT_FILENAME = "environment.json"
ARTIFACTS_FILENAME = "artifacts.json"
LINEAGE_FILENAME = "lineage.json"
STAGES_DIRNAME = "stages"
STAGE_MANIFEST_FILENAME = "stage_manifest.json"
MODEL_DIRNAME = "model"
REPORTS_DIRNAME = "reports"
LOGS_DIRNAME = "logs"
EXPERIMENT_LOG_FILENAME = "experiment.log"


class ExperimentRunError(Exception):
    """Raised when an experiment run's artifacts cannot be read or written."""


class ExperimentRunRepository:
    """Owns experiment run directories under ``experiments_root``."""

    def __init__(self, experiments_root: Path) -> None:
        self.experiments_root = Path(experiments_root)

    # ------------------------------------------------------------------------- paths

    def run_dir(self, experiment_run_id: str) -> Path:
        return self.experiments_root / experiment_run_id

    def stages_dir(self, experiment_run_id: str) -> Path:
        return self.run_dir(experiment_run_id) / STAGES_DIRNAME

    def stage_dir(self, experiment_run_id: str, stage_name: str) -> Path:
        return self.stages_dir(experiment_run_id) / stage_name

    def stage_manifest_path(self, experiment_run_id: str, stage_name: str) -> Path:
        return self.stage_dir(experiment_run_id, stage_name) / STAGE_MANIFEST_FILENAME

    def model_dir(self, experiment_run_id: str) -> Path:
        return self.run_dir(experiment_run_id) / MODEL_DIRNAME

    def reports_dir(self, experiment_run_id: str) -> Path:
        return self.run_dir(experiment_run_id) / REPORTS_DIRNAME

    def logs_dir(self, experiment_run_id: str) -> Path:
        return self.run_dir(experiment_run_id) / LOGS_DIRNAME

    def log_path(self, experiment_run_id: str, stage_name: str | None = None) -> Path:
        """``logs/experiment.log`` overall, or ``logs/<stage>.log`` per stage (section 64)."""
        filename = f"{stage_name}.log" if stage_name is not None else EXPERIMENT_LOG_FILENAME
        return self.logs_dir(experiment_run_id) / filename

    def _manifest_path(self, experiment_run_id: str) -> Path:
        return self.run_dir(experiment_run_id) / MANIFEST_FILENAME

    def _resolved_config_path(self, experiment_run_id: str) -> Path:
        return self.run_dir(experiment_run_id) / RESOLVED_CONFIG_FILENAME

    def _environment_path(self, experiment_run_id: str) -> Path:
        return self.run_dir(experiment_run_id) / ENVIRONMENT_FILENAME

    def _artifacts_path(self, experiment_run_id: str) -> Path:
        return self.run_dir(experiment_run_id) / ARTIFACTS_FILENAME

    def _lineage_path(self, experiment_run_id: str) -> Path:
        return self.run_dir(experiment_run_id) / LINEAGE_FILENAME

    # -------------------------------------------------------------------- run ids

    def existing_run_ids(self) -> set[str]:
        if not self.experiments_root.is_dir():
            return set()
        return {
            path.name
            for path in self.experiments_root.iterdir()
            if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
        }

    def new_run_id(self, now: datetime | None = None) -> str:
        """``exp_YYYYMMDD_HHMMSS_xxxx`` (spec section 11)."""
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
        existing = self.existing_run_ids()
        for _ in range(10):
            candidate = f"exp_{stamp}_{secrets.token_hex(2)}"
            if candidate not in existing:
                return candidate
        raise ExperimentRunError("could not mint a unique experiment run id after 10 attempts")

    # ---------------------------------------------------------------------- create

    def create_run(
        self,
        *,
        experiment_name: str,
        configuration_hash: str,
        experiment_run_id: str | None = None,
    ) -> ExperimentManifest:
        experiment_run_id = experiment_run_id or self.new_run_id()
        manifest = ExperimentManifest(
            experiment_run_id=experiment_run_id,
            experiment_name=experiment_name,
            status="created",
            configuration_hash=configuration_hash,
        )
        self._write_manifest(manifest)
        return manifest

    # ------------------------------------------------------------------------ read

    def get_run(self, experiment_run_id: str) -> ExperimentManifest:
        path = self._manifest_path(experiment_run_id)
        if not path.is_file():
            raise ExperimentRunNotFoundError(
                f"no experiment run {experiment_run_id!r} at {self.run_dir(experiment_run_id)}"
            )
        try:
            return ExperimentManifest.from_dict(read_json(path))
        except (JsonlError, ManifestError) as exc:
            raise ExperimentRunError(f"{path}: {exc}") from exc

    def list_runs(self) -> list[ExperimentManifest]:
        if not self.experiments_root.is_dir():
            return []
        manifests = [
            self.get_run(path.name)
            for path in sorted(self.experiments_root.iterdir())
            if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
        ]
        return sorted(
            manifests, key=lambda m: (m.start_time or "", m.experiment_run_id), reverse=True
        )

    # ---------------------------------------------------------------------- write

    def _write_manifest(self, manifest: ExperimentManifest) -> ExperimentManifest:
        atomic_write_json(self._manifest_path(manifest.experiment_run_id), manifest.to_dict())
        return manifest

    def save(self, manifest: ExperimentManifest) -> ExperimentManifest:
        """Persist an already-updated manifest (e.g. one built via ``with_status`` or
        with new ``stage_runs``/``final_model`` fields set by the orchestrator)."""
        return self._write_manifest(manifest)

    def update_status(
        self,
        experiment_run_id: str,
        status: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> ExperimentManifest:
        manifest = self.get_run(experiment_run_id)
        updated = manifest.with_status(status, start_time=start_time, end_time=end_time)
        return self._write_manifest(updated)

    def start_run(self, experiment_run_id: str) -> ExperimentManifest:
        return self.update_status(experiment_run_id, "running", start_time=utc_now_iso())

    def complete_run(self, experiment_run_id: str) -> ExperimentManifest:
        return self.update_status(experiment_run_id, "completed", end_time=utc_now_iso())

    def interrupt_run(self, experiment_run_id: str) -> ExperimentManifest:
        return self.update_status(experiment_run_id, "interrupted", end_time=utc_now_iso())

    def fail_run(self, experiment_run_id: str) -> ExperimentManifest:
        return self.update_status(experiment_run_id, "failed", end_time=utc_now_iso())

    def cancel_run(self, experiment_run_id: str) -> ExperimentManifest:
        return self.update_status(experiment_run_id, "cancelled", end_time=utc_now_iso())

    # ------------------------------------------------------------- resolved config

    def write_resolved_config(self, experiment_run_id: str, configuration: dict[str, Any]) -> None:
        """Write the immutable resolved config (section 10) as YAML, atomically.

        Written once, at experiment start, and never again -- callers must not invoke
        this a second time for the same run; the orchestrator enforces that by only
        calling it from ``create_run``'s caller, before the run starts.
        """
        path = self._resolved_config_path(experiment_run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            yaml.safe_dump(configuration, sort_keys=True, default_flow_style=False),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def read_resolved_config(self, experiment_run_id: str) -> dict[str, Any] | None:
        path = self._resolved_config_path(experiment_run_id)
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    # ----------------------------------------------------------------- environment

    def write_environment(self, experiment_run_id: str, environment: dict[str, Any]) -> None:
        atomic_write_json(self._environment_path(experiment_run_id), environment)

    def read_environment(self, experiment_run_id: str) -> dict[str, Any] | None:
        path = self._environment_path(experiment_run_id)
        if not path.is_file():
            return None
        return read_json(path)

    # ------------------------------------------------------------------- artifacts

    def write_artifacts(self, experiment_run_id: str, refs: dict[str, ArtifactRef]) -> None:
        write_artifact_manifest(self._artifacts_path(experiment_run_id), refs)

    def read_artifacts(self, experiment_run_id: str) -> dict[str, ArtifactRef]:
        return read_artifact_manifest(self._artifacts_path(experiment_run_id))

    # ---------------------------------------------------------------------- lineage

    def write_lineage(self, experiment_run_id: str, lineage: dict[str, Any]) -> None:
        atomic_write_json(self._lineage_path(experiment_run_id), lineage)

    def read_lineage(self, experiment_run_id: str) -> dict[str, Any] | None:
        path = self._lineage_path(experiment_run_id)
        if not path.is_file():
            return None
        return read_json(path)

    # ----------------------------------------------------------------- stage manifests

    def write_stage_manifest(self, experiment_run_id: str, manifest: StageManifest) -> None:
        atomic_write_json(
            self.stage_manifest_path(experiment_run_id, manifest.stage_name), manifest.to_dict()
        )

    def read_stage_manifest(
        self, experiment_run_id: str, stage_name: str
    ) -> StageManifest | None:
        path = self.stage_manifest_path(experiment_run_id, stage_name)
        if not path.is_file():
            return None
        try:
            return StageManifest.from_dict(read_json(path))
        except (JsonlError, ManifestError) as exc:
            raise ExperimentRunError(f"{path}: {exc}") from exc

    def find_reusable_stage(
        self, stage_name: str, cache_key_value: str, *, exclude_run_id: str | None = None
    ) -> StageManifest | None:
        """Section 17: find a COMPLETED stage manifest for ``stage_name`` with a matching
        cache key in *any* experiment run, newest first -- reuse is not limited to
        resuming the same experiment, since an identical prior experiment's work is
        exactly as valid as this run's own earlier attempt.
        """
        if not self.experiments_root.is_dir():
            return None
        candidates = sorted(
            (
                path.name
                for path in self.experiments_root.iterdir()
                if path.is_dir()
                and path.name != exclude_run_id
                and (path / MANIFEST_FILENAME).is_file()
            ),
            reverse=True,
        )
        for run_id in candidates:
            manifest = self.read_stage_manifest(run_id, stage_name)
            if manifest is not None and manifest.status == "COMPLETED" and manifest.cache_key == cache_key_value:
                return manifest
        return None


__all__ = [
    "ARTIFACTS_FILENAME",
    "ENVIRONMENT_FILENAME",
    "EXPERIMENT_LOG_FILENAME",
    "ExperimentRunError",
    "ExperimentRunRepository",
    "LINEAGE_FILENAME",
    "LOGS_DIRNAME",
    "MANIFEST_FILENAME",
    "MODEL_DIRNAME",
    "REPORTS_DIRNAME",
    "RESOLVED_CONFIG_FILENAME",
    "STAGES_DIRNAME",
    "STAGE_MANIFEST_FILENAME",
]
