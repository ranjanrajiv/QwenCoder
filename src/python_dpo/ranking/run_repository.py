"""Central owner of ranking run state: creation, status transitions, listing, resume.

Mirrors :class:`python_dpo.evaluation.run_repository.EvaluationRunRepository`: every
ranking run is one directory under ``rankings_root``; this repository is the only code
that mints ranking run ids, writes ``manifest.json``, or writes ``statistics.json`` (spec
sections 26, 27, 42).
"""

from __future__ import annotations

import secrets
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..atomic_io import JsonlError, atomic_write_json, read_json
from .models import RankingManifest, RankingModelError, RankingStatistics, utc_now_iso
from .repository import RankingRepository

MANIFEST_FILENAME = "manifest.json"
STATISTICS_FILENAME = "statistics.json"


class RankingRunError(Exception):
    """Raised for ranking-run-level problems (bad manifest, bad transition)."""


class RankingRunNotFoundError(RankingRunError):
    """Raised when a ranking run id has no corresponding directory."""


class RankingRunRepository:
    """Owns ranking run directories under ``rankings_root``."""

    def __init__(self, rankings_root: Path) -> None:
        self.rankings_root = Path(rankings_root)

    # ------------------------------------------------------------------------- paths

    def run_dir(self, ranking_run_id: str) -> Path:
        return self.rankings_root / ranking_run_id

    def _manifest_path(self, ranking_run_id: str) -> Path:
        return self.run_dir(ranking_run_id) / MANIFEST_FILENAME

    def _statistics_path(self, ranking_run_id: str) -> Path:
        return self.run_dir(ranking_run_id) / STATISTICS_FILENAME

    def results(self, ranking_run_id: str) -> RankingRepository:
        """The ranking-run-scoped results repository for ``ranking_run_id``."""
        return RankingRepository(self.run_dir(ranking_run_id))

    # -------------------------------------------------------------------- run ids

    def existing_run_ids(self) -> set[str]:
        if not self.rankings_root.is_dir():
            return set()
        return {
            path.name
            for path in self.rankings_root.iterdir()
            if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
        }

    def new_run_id(self, now: datetime | None = None) -> str:
        """``rank_YYYYMMDD_HHMMSS_xxxx`` (spec section 26)."""
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
        existing = self.existing_run_ids()
        for _ in range(10):
            candidate = f"rank_{stamp}_{secrets.token_hex(2)}"
            if candidate not in existing:
                return candidate
        raise RankingRunError("could not mint a unique ranking run id after 10 attempts")

    # ---------------------------------------------------------------------- create

    def create_run(
        self,
        *,
        evaluation_run_id: str,
        candidate_run_id: str,
        ranking_version: str,
        scoring_version: str,
        comparator_version: str,
        requested_problem_ids: Sequence[str],
        scoring_configuration: dict[str, Any] | None = None,
        ranking_run_id: str | None = None,
    ) -> RankingManifest:
        """Mint (or accept) a ranking run id, write ``manifest.json``, and return it."""
        ranking_run_id = ranking_run_id or self.new_run_id()
        manifest = RankingManifest(
            ranking_run_id=ranking_run_id,
            evaluation_run_id=evaluation_run_id,
            candidate_run_id=candidate_run_id,
            status="created",
            created_at=utc_now_iso(),
            ranking_version=ranking_version,
            scoring_version=scoring_version,
            comparator_version=comparator_version,
            scoring_configuration=scoring_configuration or {},
            requested_problem_ids=tuple(requested_problem_ids),
        )
        self._write_manifest(manifest)
        return manifest

    # ------------------------------------------------------------------------ read

    def get_run(self, ranking_run_id: str) -> RankingManifest:
        path = self._manifest_path(ranking_run_id)
        if not path.is_file():
            raise RankingRunNotFoundError(
                f"no ranking run {ranking_run_id!r} at {self.run_dir(ranking_run_id)}"
            )
        try:
            return RankingManifest.from_dict(read_json(path))
        except (JsonlError, RankingModelError) as exc:
            raise RankingRunError(f"{path}: {exc}") from exc

    def list_runs(self) -> list[RankingManifest]:
        """All ranking runs, newest first (same tie-break as ``EvaluationRunRepository``:
        ``created_at`` has only second resolution, and the run id embeds the same
        timestamp plus a random suffix)."""
        if not self.rankings_root.is_dir():
            return []
        manifests = [
            self.get_run(path.name)
            for path in sorted(self.rankings_root.iterdir())
            if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
        ]
        return sorted(manifests, key=lambda m: (m.created_at, m.ranking_run_id), reverse=True)

    def latest_run_for_evaluation_run(self, evaluation_run_id: str) -> RankingManifest | None:
        """The most recently created ranking run covering ``evaluation_run_id``, or
        ``None`` if it has never been ranked.
        """
        matches = [m for m in self.list_runs() if m.evaluation_run_id == evaluation_run_id]
        return matches[0] if matches else None

    def read_statistics(self, ranking_run_id: str) -> RankingStatistics | None:
        path = self._statistics_path(ranking_run_id)
        if not path.is_file():
            return None
        return RankingStatistics.from_dict(read_json(path))

    # ---------------------------------------------------------------------- write

    def _write_manifest(self, manifest: RankingManifest) -> RankingManifest:
        atomic_write_json(self._manifest_path(manifest.ranking_run_id), manifest.to_dict())
        return manifest

    def write_statistics(self, statistics: RankingStatistics) -> None:
        atomic_write_json(self._statistics_path(statistics.ranking_run_id), statistics.to_dict())

    def update_status(
        self,
        ranking_run_id: str,
        status: str,
        *,
        started_at: str | None = None,
        completed_at: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> RankingManifest:
        manifest = self.get_run(ranking_run_id)
        try:
            updated = manifest.with_status(
                status, started_at=started_at, completed_at=completed_at, error=error
            )
        except RankingModelError as exc:
            raise RankingRunError(str(exc)) from exc
        return self._write_manifest(updated)

    def start_run(self, ranking_run_id: str) -> RankingManifest:
        """``created``/``interrupted``/``failed`` -> ``running``."""
        return self.update_status(ranking_run_id, "running", started_at=utc_now_iso())

    def resume_run(self, ranking_run_id: str) -> RankingManifest:
        """Reopen an incomplete ranking run. Refuses a run that already completed (spec
        section 54).
        """
        manifest = self.get_run(ranking_run_id)
        if manifest.status == "completed":
            raise RankingRunError(
                f"ranking run {ranking_run_id!r} is already completed; nothing to resume"
            )
        if manifest.status not in {"created", "interrupted", "failed"}:
            raise RankingRunError(
                f"ranking run {ranking_run_id!r} has status {manifest.status!r}; only "
                "created, interrupted, or failed runs can be resumed"
            )
        return self.start_run(ranking_run_id)

    def complete_run(self, ranking_run_id: str) -> RankingManifest:
        return self.update_status(ranking_run_id, "completed", completed_at=utc_now_iso())

    def interrupt_run(self, ranking_run_id: str) -> RankingManifest:
        return self.update_status(ranking_run_id, "interrupted", completed_at=utc_now_iso())

    def fail_run(
        self, ranking_run_id: str, *, error_type: str, error_message: str
    ) -> RankingManifest:
        error = {"error_type": error_type, "error_message": error_message, "timestamp": utc_now_iso()}
        return self.update_status(
            ranking_run_id, "failed", completed_at=utc_now_iso(), error=error
        )

    def cancel_run(self, ranking_run_id: str) -> RankingManifest:
        return self.update_status(ranking_run_id, "cancelled", completed_at=utc_now_iso())


__all__ = [
    "MANIFEST_FILENAME",
    "STATISTICS_FILENAME",
    "RankingRunError",
    "RankingRunNotFoundError",
    "RankingRunRepository",
]
