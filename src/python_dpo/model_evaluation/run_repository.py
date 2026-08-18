"""Central owner of Stage 10 evaluation-run state (spec sections 79, 80, 148).

Mirrors :class:`python_dpo.training.run_repository.TrainingRunRepository`: one directory
per run, this repository is the only code that mints run ids or writes the run's JSON/YAML
artifacts. A sixth copy of this plumbing (after Stages 4, 6, 7, 8, 9) rather than a shared
base -- the extraction has been deferred at every stage since Stage 7; doing it now would
touch six stages at once, so the debt is carried deliberately again.
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..atomic_io import JsonlError, append_jsonl, atomic_write_json, iter_jsonl, read_json
from ..evaluation.repository import EvaluationRepository
from .errors import EvaluationRunError, EvaluationRunNotFoundError
from .models import (
    EvaluationRecord,
    GenerationRecord,
    ModelEvaluationManifest,
    ModelEvaluationModelError,
    utc_now_iso,
)

MANIFEST_FILENAME = "manifest.json"
CONFIG_FILENAME = "config.yaml"
BENCHMARK_MANIFEST_FILENAME = "benchmark_manifest.json"
GENERATIONS_DIRNAME = "generations"
EVALUATIONS_DIRNAME = "evaluations"
SANDBOX_DIRNAME = "_sandbox"
METRICS_DIRNAME = "metrics"
REPORTS_DIRNAME = "reports"
LOGS_DIRNAME = "logs"
LOG_FILENAME = "evaluation.log"


class ModelEvaluationRunRepository:
    """Owns Stage 10 evaluation run directories under ``model_evaluations_root``."""

    def __init__(self, model_evaluations_root: Path) -> None:
        self.model_evaluations_root = Path(model_evaluations_root)

    # ------------------------------------------------------------------------- paths

    def run_dir(self, evaluation_run_id: str) -> Path:
        return self.model_evaluations_root / evaluation_run_id

    def generations_path(self, evaluation_run_id: str, variant: str) -> Path:
        return self.run_dir(evaluation_run_id) / GENERATIONS_DIRNAME / f"{variant}.jsonl"

    def evaluations_path(self, evaluation_run_id: str, variant: str) -> Path:
        return self.run_dir(evaluation_run_id) / EVALUATIONS_DIRNAME / f"{variant}.jsonl"

    def sandbox_dir(self, evaluation_run_id: str, variant: str) -> Path:
        """Where the unmodified Stage 6 ``EvaluationRepository`` writes for this variant --
        full stdout/stderr, kept for failure-analysis forensics (spec section 126) without
        bloating the curated ``EvaluationRecord`` schema (spec section 82)."""
        return self.run_dir(evaluation_run_id) / EVALUATIONS_DIRNAME / SANDBOX_DIRNAME / variant

    def sandbox_repository(self, evaluation_run_id: str, variant: str) -> EvaluationRepository:
        return EvaluationRepository(self.sandbox_dir(evaluation_run_id, variant))

    def metrics_path(self, evaluation_run_id: str, name: str) -> Path:
        return self.run_dir(evaluation_run_id) / METRICS_DIRNAME / f"{name}.json"

    def reports_dir(self, evaluation_run_id: str) -> Path:
        return self.run_dir(evaluation_run_id) / REPORTS_DIRNAME

    def log_path(self, evaluation_run_id: str) -> Path:
        return self.run_dir(evaluation_run_id) / LOGS_DIRNAME / LOG_FILENAME

    def _manifest_path(self, evaluation_run_id: str) -> Path:
        return self.run_dir(evaluation_run_id) / MANIFEST_FILENAME

    def _benchmark_manifest_path(self, evaluation_run_id: str) -> Path:
        return self.run_dir(evaluation_run_id) / BENCHMARK_MANIFEST_FILENAME

    # -------------------------------------------------------------------- run ids

    def existing_run_ids(self) -> set[str]:
        if not self.model_evaluations_root.is_dir():
            return set()
        return {
            path.name
            for path in self.model_evaluations_root.iterdir()
            if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
        }

    def new_run_id(self, now: datetime | None = None) -> str:
        """``eval_YYYYMMDD_HHMMSS_xxxx`` (spec section 79)."""
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
        existing = self.existing_run_ids()
        for _ in range(10):
            candidate = f"eval_{stamp}_{secrets.token_hex(2)}"
            if candidate not in existing:
                return candidate
        raise EvaluationRunError("could not mint a unique evaluation run id after 10 attempts")

    # ---------------------------------------------------------------------- create

    def create_run(
        self,
        *,
        benchmark_version: str,
        benchmark_hash: str,
        base_model_name: str,
        adapter_path: str,
        training_run_id: str,
        models_requested: tuple[str, ...],
        generation_config: dict[str, Any],
        quantization: dict[str, Any],
        statistics_config: dict[str, Any],
        seeds: dict[str, Any],
        hardware: dict[str, Any],
        environment: dict[str, Any],
        base_model_revision: str | None = None,
        evaluation_run_id: str | None = None,
    ) -> ModelEvaluationManifest:
        evaluation_run_id = evaluation_run_id or self.new_run_id()
        manifest = ModelEvaluationManifest(
            evaluation_run_id=evaluation_run_id,
            benchmark_version=benchmark_version,
            benchmark_hash=benchmark_hash,
            base_model_name=base_model_name,
            base_model_revision=base_model_revision,
            adapter_path=adapter_path,
            training_run_id=training_run_id,
            models_requested=models_requested,
            generation_config=generation_config,
            quantization=quantization,
            statistics_config=statistics_config,
            seeds=seeds,
            hardware=hardware,
            environment=environment,
            created_at=utc_now_iso(),
        )
        self._write_manifest(manifest)
        return manifest

    # ------------------------------------------------------------------------ read

    def get_run(self, evaluation_run_id: str) -> ModelEvaluationManifest:
        path = self._manifest_path(evaluation_run_id)
        if not path.is_file():
            raise EvaluationRunNotFoundError(
                f"no evaluation run {evaluation_run_id!r} at {self.run_dir(evaluation_run_id)}"
            )
        try:
            return ModelEvaluationManifest.from_dict(read_json(path))
        except (JsonlError, ModelEvaluationModelError) as exc:
            raise EvaluationRunError(f"{path}: {exc}") from exc

    def list_runs(self) -> list[ModelEvaluationManifest]:
        if not self.model_evaluations_root.is_dir():
            return []
        manifests = [
            self.get_run(path.name)
            for path in sorted(self.model_evaluations_root.iterdir())
            if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
        ]
        return sorted(manifests, key=lambda m: (m.created_at, m.evaluation_run_id), reverse=True)

    def read_benchmark_manifest(self, evaluation_run_id: str) -> dict[str, Any] | None:
        path = self._benchmark_manifest_path(evaluation_run_id)
        if not path.is_file():
            return None
        return read_json(path)

    # ---------------------------------------------------------------------- write

    def _write_manifest(self, manifest: ModelEvaluationManifest) -> ModelEvaluationManifest:
        atomic_write_json(self._manifest_path(manifest.evaluation_run_id), manifest.to_dict())
        return manifest

    def write_config(self, evaluation_run_id: str, configuration: dict[str, Any]) -> None:
        """The resolved evaluation config actually used (spec section 148)."""
        path = self.run_dir(evaluation_run_id) / CONFIG_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            yaml.safe_dump(configuration, sort_keys=True, default_flow_style=False),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def write_benchmark_manifest(self, evaluation_run_id: str, benchmark_manifest: dict[str, Any]) -> None:
        atomic_write_json(self._benchmark_manifest_path(evaluation_run_id), benchmark_manifest)

    def read_config(self, evaluation_run_id: str) -> dict[str, Any] | None:
        """The resolved evaluation config as written by :meth:`write_config`."""
        path = self.run_dir(evaluation_run_id) / CONFIG_FILENAME
        if not path.is_file():
            return None
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def update_status(
        self,
        evaluation_run_id: str,
        status: str,
        *,
        started_at: str | None = None,
        completed_at: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> ModelEvaluationManifest:
        manifest = self.get_run(evaluation_run_id)
        try:
            updated = manifest.with_status(
                status, started_at=started_at, completed_at=completed_at, error=error
            )
        except ModelEvaluationModelError as exc:
            raise EvaluationRunError(str(exc)) from exc
        return self._write_manifest(updated)

    def start_run(self, evaluation_run_id: str) -> ModelEvaluationManifest:
        return self.update_status(evaluation_run_id, "running", started_at=utc_now_iso())

    def complete_run(self, evaluation_run_id: str) -> ModelEvaluationManifest:
        return self.update_status(evaluation_run_id, "completed", completed_at=utc_now_iso())

    def interrupt_run(self, evaluation_run_id: str) -> ModelEvaluationManifest:
        return self.update_status(evaluation_run_id, "interrupted", completed_at=utc_now_iso())

    def fail_run(
        self, evaluation_run_id: str, *, error_type: str, error_message: str, traceback_text: str | None = None
    ) -> ModelEvaluationManifest:
        error: dict[str, Any] = {
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": utc_now_iso(),
        }
        if traceback_text is not None:
            error["traceback"] = traceback_text
        return self.update_status(evaluation_run_id, "failed", completed_at=utc_now_iso(), error=error)

    def cancel_run(self, evaluation_run_id: str) -> ModelEvaluationManifest:
        return self.update_status(evaluation_run_id, "cancelled", completed_at=utc_now_iso())

    # ------------------------------------------------------------- generation records

    def append_generation_record(self, evaluation_run_id: str, variant: str, record: GenerationRecord) -> None:
        append_jsonl(self.generations_path(evaluation_run_id, variant), record.to_dict())

    def load_generation_records(self, evaluation_run_id: str, variant: str) -> list[GenerationRecord]:
        path = self.generations_path(evaluation_run_id, variant)
        try:
            return [GenerationRecord.from_dict(raw) for _, raw in iter_jsonl(path)]
        except (JsonlError, ModelEvaluationModelError) as exc:
            raise EvaluationRunError(f"{path}: {exc}") from exc

    # ------------------------------------------------------------- evaluation records

    def append_evaluation_record(self, evaluation_run_id: str, variant: str, record: EvaluationRecord) -> None:
        append_jsonl(self.evaluations_path(evaluation_run_id, variant), record.to_dict())

    def load_evaluation_records(self, evaluation_run_id: str, variant: str) -> list[EvaluationRecord]:
        path = self.evaluations_path(evaluation_run_id, variant)
        try:
            return [EvaluationRecord.from_dict(raw) for _, raw in iter_jsonl(path)]
        except (JsonlError, ModelEvaluationModelError) as exc:
            raise EvaluationRunError(f"{path}: {exc}") from exc

    # ------------------------------------------------------------------------ metrics

    def write_metrics(self, evaluation_run_id: str, name: str, payload: dict[str, Any]) -> None:
        atomic_write_json(self.metrics_path(evaluation_run_id, name), payload)

    def read_metrics(self, evaluation_run_id: str, name: str) -> dict[str, Any] | None:
        path = self.metrics_path(evaluation_run_id, name)
        if not path.is_file():
            return None
        return read_json(path)

    # ------------------------------------------------------------------------ reports

    def write_report_json(self, evaluation_run_id: str, name: str, payload: dict[str, Any]) -> None:
        atomic_write_json(self.reports_dir(evaluation_run_id) / f"{name}.json", payload)

    def write_report_text(self, evaluation_run_id: str, name: str, text: str) -> None:
        path = self.reports_dir(evaluation_run_id) / f"{name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def write_report_jsonl(self, evaluation_run_id: str, name: str, records: list[dict[str, Any]]) -> None:
        """Overwrite ``reports/<name>.jsonl`` with exactly ``records``.

        Unlike the generation/evaluation JSONL files, improvements/regressions/ties are
        **derived** from those records, fully recomputed on every ``report``/``stats``
        invocation -- appending here would duplicate rows on a second run, which
        `append_jsonl`'s incremental-fact semantics are wrong for. Written even when
        ``records`` is empty, so the file's presence (or absence) is never ambiguous.
        """
        path = self.reports_dir(evaluation_run_id) / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        tmp_path.replace(path)


class run_log_file:  # noqa: N801 - a context manager used like a function
    """Tee the ``python_dpo`` logger into ``logs/evaluation.log`` (matches every prior
    stage's run-log convention). The handler is removed on exit."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handler: Any = None

    def __enter__(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(self.path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logging.getLogger("python_dpo").addHandler(handler)
        self._handler = handler
        return self.path

    def __exit__(self, *exc_info: object) -> None:
        if self._handler is not None:
            logging.getLogger("python_dpo").removeHandler(self._handler)
            self._handler.close()
            self._handler = None


__all__ = [
    "BENCHMARK_MANIFEST_FILENAME",
    "CONFIG_FILENAME",
    "EVALUATIONS_DIRNAME",
    "GENERATIONS_DIRNAME",
    "LOG_FILENAME",
    "LOGS_DIRNAME",
    "MANIFEST_FILENAME",
    "METRICS_DIRNAME",
    "REPORTS_DIRNAME",
    "SANDBOX_DIRNAME",
    "ModelEvaluationRunRepository",
    "run_log_file",
]
