"""Central owner of training run state: creation, status transitions, listing, resume.

Mirrors :class:`python_dpo.preferences.run_repository.PreferenceRunRepository`: every
training run is one directory under ``training_root``, and this repository is the only
code that mints training run ids or writes the run's JSON artifacts.

A fifth copy of this plumbing (after Stages 4, 6, 7, 8) rather than a shared base. The
extraction has been deferred at every stage since Stage 7; doing it now would touch five
stages at once, so the debt is carried deliberately and recorded rather than paid here.

Resume (spec sections 90-92) lives here because it is fundamentally a *manifest*
comparison: does the checkpoint's recorded configuration still match what we are about to
run? Nothing about that needs torch, so it is testable offline.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..atomic_io import JsonlError, atomic_write_json, read_json
from .errors import (
    CheckpointCompatibilityError,
    TrainingRunError,
    TrainingRunNotFoundError,
)
from .models import (
    DatasetManifest,
    FinalReport,
    TrainingManifest,
    TrainingModelError,
    utc_now_iso,
)

MANIFEST_FILENAME = "manifest.json"
DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
HARDWARE_FILENAME = "hardware.json"
FINAL_REPORT_FILENAME = "final_report.json"
CONFIG_FILENAME = "config.yaml"
METRICS_DIRNAME = "metrics"
METRICS_FILENAME = "metrics.jsonl"
LOGS_DIRNAME = "logs"
LOG_FILENAME = "training.log"
ADAPTER_DIRNAME = "adapter"
CHECKPOINTS_DIRNAME = "checkpoints"
TOKENIZER_DIRNAME = "tokenizer"

# Spec section 92: a difference in any of these makes a checkpoint incompatible, because
# the saved optimizer and adapter state simply do not describe the same training problem.
CRITICAL_CONFIG_PATHS = (
    ("model", "name"),
    ("model", "revision"),
    ("lora", "r"),
    ("lora", "alpha"),
    ("lora", "target_modules"),
    ("quantization", "bits"),
    ("quantization", "quant_type"),
)


class TrainingRunRepository:
    """Owns training run directories under ``training_root``."""

    def __init__(self, training_root: Path) -> None:
        self.training_root = Path(training_root)

    # ------------------------------------------------------------------------- paths

    def run_dir(self, training_run_id: str) -> Path:
        return self.training_root / training_run_id

    def adapter_dir(self, training_run_id: str) -> Path:
        return self.run_dir(training_run_id) / ADAPTER_DIRNAME

    def checkpoints_dir(self, training_run_id: str) -> Path:
        return self.run_dir(training_run_id) / CHECKPOINTS_DIRNAME

    def tokenizer_dir(self, training_run_id: str) -> Path:
        return self.run_dir(training_run_id) / TOKENIZER_DIRNAME

    def metrics_path(self, training_run_id: str) -> Path:
        return self.run_dir(training_run_id) / METRICS_DIRNAME / METRICS_FILENAME

    def log_path(self, training_run_id: str) -> Path:
        return self.run_dir(training_run_id) / LOGS_DIRNAME / LOG_FILENAME

    def _manifest_path(self, training_run_id: str) -> Path:
        return self.run_dir(training_run_id) / MANIFEST_FILENAME

    # -------------------------------------------------------------------- run ids

    def existing_run_ids(self) -> set[str]:
        if not self.training_root.is_dir():
            return set()
        return {
            path.name
            for path in self.training_root.iterdir()
            if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
        }

    def new_run_id(self, now: datetime | None = None) -> str:
        """``dpo_YYYYMMDD_HHMMSS_xxxx`` (spec section 68)."""
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
        existing = self.existing_run_ids()
        for _ in range(10):
            candidate = f"dpo_{stamp}_{secrets.token_hex(2)}"
            if candidate not in existing:
                return candidate
        raise TrainingRunError("could not mint a unique training run id after 10 attempts")

    # ---------------------------------------------------------------------- create

    def create_run(
        self,
        *,
        experiment_name: str,
        model_name: str,
        model_revision: str | None,
        tokenizer_revision: str | None,
        preference_run_id: str,
        ranking_run_id: str,
        evaluation_run_id: str,
        candidate_run_id: str,
        dataset_hashes: dict[str, str],
        hardware: dict[str, Any],
        environment: dict[str, Any],
        configuration: dict[str, Any],
        seed: int,
        data_seed: int,
        trainer_version: str,
        mode: str = "train",
        training_run_id: str | None = None,
    ) -> TrainingManifest:
        training_run_id = training_run_id or self.new_run_id()
        manifest = TrainingManifest(
            training_run_id=training_run_id,
            experiment_name=experiment_name,
            status="created",
            created_at=utc_now_iso(),
            mode=mode,
            model_name=model_name,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            preference_run_id=preference_run_id,
            ranking_run_id=ranking_run_id,
            evaluation_run_id=evaluation_run_id,
            candidate_run_id=candidate_run_id,
            dataset_hashes=dataset_hashes,
            hardware=hardware,
            environment=environment,
            configuration=configuration,
            seed=seed,
            data_seed=data_seed,
            trainer_version=trainer_version,
        )
        self._write_manifest(manifest)
        return manifest

    # ------------------------------------------------------------------------ read

    def get_run(self, training_run_id: str) -> TrainingManifest:
        path = self._manifest_path(training_run_id)
        if not path.is_file():
            raise TrainingRunNotFoundError(
                f"no training run {training_run_id!r} at {self.run_dir(training_run_id)}"
            )
        try:
            return TrainingManifest.from_dict(read_json(path))
        except (JsonlError, TrainingModelError) as exc:
            raise TrainingRunError(f"{path}: {exc}") from exc

    def list_runs(self) -> list[TrainingManifest]:
        if not self.training_root.is_dir():
            return []
        manifests = [
            self.get_run(path.name)
            for path in sorted(self.training_root.iterdir())
            if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
        ]
        return sorted(
            manifests, key=lambda m: (m.created_at, m.training_run_id), reverse=True
        )

    def read_dataset_manifest(self, training_run_id: str) -> DatasetManifest | None:
        path = self.run_dir(training_run_id) / DATASET_MANIFEST_FILENAME
        if not path.is_file():
            return None
        return DatasetManifest.from_dict(read_json(path))

    def read_final_report(self, training_run_id: str) -> FinalReport | None:
        path = self.run_dir(training_run_id) / FINAL_REPORT_FILENAME
        if not path.is_file():
            return None
        return FinalReport.from_dict(read_json(path))

    def latest_checkpoint(self, training_run_id: str) -> Path | None:
        """The highest-numbered ``checkpoint-N`` directory, or ``None``."""
        root = self.checkpoints_dir(training_run_id)
        if not root.is_dir():
            return None
        candidates = [
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith("checkpoint-")
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: int(p.name.rsplit("-", 1)[-1]))

    # ---------------------------------------------------------------------- write

    def _write_manifest(self, manifest: TrainingManifest) -> TrainingManifest:
        atomic_write_json(self._manifest_path(manifest.training_run_id), manifest.to_dict())
        return manifest

    def write_dataset_manifest(
        self, training_run_id: str, dataset_manifest: DatasetManifest
    ) -> None:
        atomic_write_json(
            self.run_dir(training_run_id) / DATASET_MANIFEST_FILENAME,
            dataset_manifest.to_dict(),
        )

    def write_hardware(self, training_run_id: str, hardware: dict[str, Any]) -> None:
        atomic_write_json(self.run_dir(training_run_id) / HARDWARE_FILENAME, hardware)

    def write_final_report(self, training_run_id: str, report: FinalReport) -> None:
        atomic_write_json(
            self.run_dir(training_run_id) / FINAL_REPORT_FILENAME, report.to_dict()
        )

    def write_config(self, training_run_id: str, configuration: dict[str, Any]) -> None:
        """The resolved experiment config actually used (spec section 69).

        Written as YAML to match the file it came from, so a run can be re-executed by
        pointing ``--config`` straight at it.
        """
        import yaml

        path = self.run_dir(training_run_id) / CONFIG_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            yaml.safe_dump(configuration, sort_keys=True, default_flow_style=False),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def update_status(
        self,
        training_run_id: str,
        status: str,
        *,
        started_at: str | None = None,
        completed_at: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> TrainingManifest:
        manifest = self.get_run(training_run_id)
        try:
            updated = manifest.with_status(
                status, started_at=started_at, completed_at=completed_at, error=error
            )
        except TrainingModelError as exc:
            raise TrainingRunError(str(exc)) from exc
        return self._write_manifest(updated)

    def start_run(self, training_run_id: str) -> TrainingManifest:
        return self.update_status(training_run_id, "running", started_at=utc_now_iso())

    def complete_run(self, training_run_id: str) -> TrainingManifest:
        return self.update_status(training_run_id, "completed", completed_at=utc_now_iso())

    def interrupt_run(self, training_run_id: str) -> TrainingManifest:
        return self.update_status(training_run_id, "interrupted", completed_at=utc_now_iso())

    def fail_run(
        self,
        training_run_id: str,
        *,
        error_type: str,
        error_message: str,
        traceback_text: str | None = None,
        last_step: int | None = None,
    ) -> TrainingManifest:
        """Mark a run failed with everything needed to diagnose it (spec section 82)."""
        error: dict[str, Any] = {
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": utc_now_iso(),
        }
        if traceback_text is not None:
            error["traceback"] = traceback_text
        if last_step is not None:
            error["last_step"] = last_step
        return self.update_status(
            training_run_id, "failed", completed_at=utc_now_iso(), error=error
        )

    def cancel_run(self, training_run_id: str) -> TrainingManifest:
        return self.update_status(training_run_id, "cancelled", completed_at=utc_now_iso())

    # --------------------------------------------------------------------- resume

    def check_resume_compatibility(
        self,
        training_run_id: str,
        *,
        dataset_hashes: dict[str, str],
        configuration: dict[str, Any],
        force: bool = False,
    ) -> TrainingManifest:
        """Verify a run may be resumed against the current inputs (sections 90-92).

        A **dataset** difference is refusable but overridable with ``force`` (section 91):
        the user may knowingly want to continue against regenerated data. A **critical
        configuration** difference is never overridable (section 92) — the saved optimizer
        and adapter state do not describe the same problem, so resuming would be
        meaningless rather than merely risky.
        """
        manifest = self.get_run(training_run_id)

        for path in CRITICAL_CONFIG_PATHS:
            original = _dig(manifest.configuration, path)
            current = _dig(configuration, path)
            if _normalize(original) != _normalize(current):
                raise CheckpointCompatibilityError(
                    f"cannot resume {training_run_id!r}: {'.'.join(path)} changed from "
                    f"{original!r} to {current!r}. Critical configuration differences make "
                    "a checkpoint meaningless; start a new run instead"
                )

        changed = sorted(
            split
            for split, digest in dataset_hashes.items()
            if manifest.dataset_hashes.get(split) != digest
        )
        if changed and not force:
            raise CheckpointCompatibilityError(
                f"cannot resume {training_run_id!r}: the {', '.join(changed)} split "
                "hash(es) differ from the original run. Pass --force-resume to continue "
                "against the changed dataset"
            )
        if changed:
            # Recorded rather than merely logged: a resumed run trained on two different
            # datasets, and the manifest is where that has to be discoverable.
            atomic_write_json(
                self.run_dir(training_run_id) / "resume_dataset_override.json",
                {
                    "timestamp": utc_now_iso(),
                    "changed_splits": changed,
                    "original_hashes": manifest.dataset_hashes,
                    "new_hashes": dataset_hashes,
                },
            )
        return manifest


def _dig(data: Any, path: tuple[str, ...]) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _normalize(value: Any) -> Any:
    """Compare list/tuple orderings structurally, so YAML round-tripping never trips a
    false incompatibility."""
    if isinstance(value, (list, tuple)):
        return sorted(str(item) for item in value)
    return value


class run_log_file:  # noqa: N801 - a context manager used like a function
    """Tee the ``python_dpo`` logger into ``logs/training.log`` (spec section 93).

    Console output is not a record: a run that scrolled past in a terminal must still be
    diagnosable afterwards. The handler is removed on exit so repeated CLI invocations in
    one process never accumulate handlers or write into a finished run's log.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handler: Any = None

    def __enter__(self) -> Path:
        import logging

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(self.path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logging.getLogger("python_dpo").addHandler(handler)
        self._handler = handler
        return self.path

    def __exit__(self, *exc_info: object) -> None:
        import logging

        if self._handler is not None:
            logging.getLogger("python_dpo").removeHandler(self._handler)
            self._handler.close()
            self._handler = None


def read_metrics(path: Path) -> list[dict[str, Any]]:
    """Read a metrics.jsonl written by :mod:`python_dpo.training.callbacks`."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


__all__ = [
    "ADAPTER_DIRNAME",
    "CHECKPOINTS_DIRNAME",
    "CONFIG_FILENAME",
    "CRITICAL_CONFIG_PATHS",
    "DATASET_MANIFEST_FILENAME",
    "FINAL_REPORT_FILENAME",
    "HARDWARE_FILENAME",
    "LOGS_DIRNAME",
    "LOG_FILENAME",
    "MANIFEST_FILENAME",
    "METRICS_DIRNAME",
    "METRICS_FILENAME",
    "TOKENIZER_DIRNAME",
    "TrainingRunRepository",
    "read_metrics",
    "run_log_file",
]
