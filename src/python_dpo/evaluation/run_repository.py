"""Central owner of evaluation run state: creation, status transitions, listing, resume.

Mirrors :class:`python_dpo.runs.repository.RunRepository`: every evaluation run is one
directory under ``evaluations_root``; this repository is the only code that mints
evaluation run ids, writes ``manifest.json``, or writes ``statistics.json`` (spec sections
48-51).
"""

from __future__ import annotations

import secrets
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..atomic_io import JsonlError, atomic_write_json, read_json
from .errors import EvaluationError
from .models import EvaluationManifest, EvaluationModelError, EvaluationStatistics, utc_now_iso
from .repository import EvaluationRepository

MANIFEST_FILENAME = "manifest.json"
STATISTICS_FILENAME = "statistics.json"


class EvaluationRunError(Exception):
    """Raised for evaluation-run-level problems (bad manifest, bad transition)."""


class EvaluationRunNotFoundError(EvaluationRunError):
    """Raised when an evaluation run id has no corresponding directory."""


class EvaluationRunRepository:
    """Owns evaluation run directories under ``evaluations_root``."""

    def __init__(self, evaluations_root: Path) -> None:
        self.evaluations_root = Path(evaluations_root)

    # ------------------------------------------------------------------------- paths

    def run_dir(self, evaluation_run_id: str) -> Path:
        return self.evaluations_root / evaluation_run_id

    def _manifest_path(self, evaluation_run_id: str) -> Path:
        return self.run_dir(evaluation_run_id) / MANIFEST_FILENAME

    def _statistics_path(self, evaluation_run_id: str) -> Path:
        return self.run_dir(evaluation_run_id) / STATISTICS_FILENAME

    def results(self, evaluation_run_id: str) -> EvaluationRepository:
        """The evaluation-run-scoped results repository for ``evaluation_run_id``."""
        return EvaluationRepository(self.run_dir(evaluation_run_id))

    # -------------------------------------------------------------------- run ids

    def existing_run_ids(self) -> set[str]:
        if not self.evaluations_root.is_dir():
            return set()
        return {
            path.name
            for path in self.evaluations_root.iterdir()
            if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
        }

    def new_run_id(self, now: datetime | None = None) -> str:
        """``eval_YYYYMMDD_HHMMSS_xxxx`` (spec section 49)."""
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
        candidate_run_id: str,
        evaluator_version: str,
        test_generator_version: str,
        pytest_version: str,
        python_version: str,
        sandbox_config: dict[str, Any],
        requested_candidate_ids: Sequence[str],
        requested_problem_id: str | None = None,
        evaluation_run_id: str | None = None,
    ) -> EvaluationManifest:
        """Mint (or accept) an evaluation run id, write ``manifest.json``, and return it."""
        evaluation_run_id = evaluation_run_id or self.new_run_id()
        manifest = EvaluationManifest(
            evaluation_run_id=evaluation_run_id,
            candidate_run_id=candidate_run_id,
            status="created",
            created_at=utc_now_iso(),
            evaluator_version=evaluator_version,
            test_generator_version=test_generator_version,
            pytest_version=pytest_version,
            python_version=python_version,
            sandbox_config=sandbox_config,
            requested_candidate_ids=tuple(requested_candidate_ids),
            requested_problem_id=requested_problem_id,
        )
        self._write_manifest(manifest)
        return manifest

    # ------------------------------------------------------------------------ read

    def get_run(self, evaluation_run_id: str) -> EvaluationManifest:
        path = self._manifest_path(evaluation_run_id)
        if not path.is_file():
            raise EvaluationRunNotFoundError(
                f"no evaluation run {evaluation_run_id!r} at {self.run_dir(evaluation_run_id)}"
            )
        try:
            return EvaluationManifest.from_dict(read_json(path))
        except (JsonlError, EvaluationModelError) as exc:
            raise EvaluationRunError(f"{path}: {exc}") from exc

    def list_runs(self) -> list[EvaluationManifest]:
        """All evaluation runs, newest first (same tie-break as ``RunRepository``:
        ``created_at`` has only second resolution, and the run id embeds the same
        timestamp plus a random suffix)."""
        if not self.evaluations_root.is_dir():
            return []
        manifests = [
            self.get_run(path.name)
            for path in sorted(self.evaluations_root.iterdir())
            if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
        ]
        return sorted(manifests, key=lambda m: (m.created_at, m.evaluation_run_id), reverse=True)

    def latest_run_for_candidate_run(self, candidate_run_id: str) -> EvaluationManifest | None:
        """The most recently created evaluation run covering ``candidate_run_id``, or
        ``None`` if this generation run has never been evaluated. The resume target for
        ``evaluate run --run-id R`` (spec sections 56-57).
        """
        matches = [m for m in self.list_runs() if m.candidate_run_id == candidate_run_id]
        return matches[0] if matches else None

    def read_statistics(self, evaluation_run_id: str) -> EvaluationStatistics | None:
        path = self._statistics_path(evaluation_run_id)
        if not path.is_file():
            return None
        return EvaluationStatistics.from_dict(read_json(path))

    # ---------------------------------------------------------------------- write

    def _write_manifest(self, manifest: EvaluationManifest) -> EvaluationManifest:
        atomic_write_json(self._manifest_path(manifest.evaluation_run_id), manifest.to_dict())
        return manifest

    def write_statistics(self, statistics: EvaluationStatistics) -> None:
        atomic_write_json(self._statistics_path(statistics.evaluation_run_id), statistics.to_dict())

    def update_status(
        self,
        evaluation_run_id: str,
        status: str,
        *,
        started_at: str | None = None,
        completed_at: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> EvaluationManifest:
        manifest = self.get_run(evaluation_run_id)
        try:
            updated = manifest.with_status(
                status, started_at=started_at, completed_at=completed_at, error=error
            )
        except EvaluationModelError as exc:
            raise EvaluationRunError(str(exc)) from exc
        return self._write_manifest(updated)

    def start_run(self, evaluation_run_id: str) -> EvaluationManifest:
        """``created``/``interrupted``/``failed`` -> ``running``."""
        return self.update_status(evaluation_run_id, "running", started_at=utc_now_iso())

    def resume_run(self, evaluation_run_id: str) -> EvaluationManifest:
        """Reopen an incomplete evaluation run. Refuses a run that already completed."""
        manifest = self.get_run(evaluation_run_id)
        if manifest.status == "completed":
            raise EvaluationRunError(
                f"evaluation run {evaluation_run_id!r} is already completed; nothing to resume"
            )
        if manifest.status not in {"created", "interrupted", "failed"}:
            raise EvaluationRunError(
                f"evaluation run {evaluation_run_id!r} has status {manifest.status!r}; only "
                "created, interrupted, or failed runs can be resumed"
            )
        return self.start_run(evaluation_run_id)

    def complete_run(self, evaluation_run_id: str) -> EvaluationManifest:
        return self.update_status(evaluation_run_id, "completed", completed_at=utc_now_iso())

    def interrupt_run(self, evaluation_run_id: str) -> EvaluationManifest:
        return self.update_status(evaluation_run_id, "interrupted", completed_at=utc_now_iso())

    def fail_run(
        self, evaluation_run_id: str, *, error_type: str, error_message: str
    ) -> EvaluationManifest:
        error = {"error_type": error_type, "error_message": error_message, "timestamp": utc_now_iso()}
        return self.update_status(
            evaluation_run_id, "failed", completed_at=utc_now_iso(), error=error
        )

    def cancel_run(self, evaluation_run_id: str) -> EvaluationManifest:
        return self.update_status(evaluation_run_id, "cancelled", completed_at=utc_now_iso())


__all__ = [
    "MANIFEST_FILENAME",
    "STATISTICS_FILENAME",
    "EvaluationRunError",
    "EvaluationRunNotFoundError",
    "EvaluationRunRepository",
]
