"""Central owner of run state: creation, status transitions, listing, statistics.

Every run is one directory under ``runs_root``; this repository is the only code that
mints run ids, writes ``manifest.json``, or writes ``statistics.json`` (spec 04 section
24). Nothing outside this module manipulates those two files directly.
"""

from __future__ import annotations

import secrets
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from ..atomic_io import JsonlError, atomic_write_json, read_json
from ..candidates.models import CANDIDATE_SCHEMA_VERSION
from ..candidates.repository import CandidateRepository
from .environment import capture_environment
from .models import RunError, RunFailure, RunManifest, RunStatistics, utc_now_iso

MANIFEST_FILENAME = "manifest.json"
STATISTICS_FILENAME = "statistics.json"


class RunNotFoundError(RunError):
    """Raised when a run id has no corresponding directory."""


class RunRepository:
    """Owns run directories under ``runs_root`` (spec 04 sections 6, 24)."""

    def __init__(self, runs_root: Path) -> None:
        self.runs_root = Path(runs_root)

    # ------------------------------------------------------------------------- paths

    def run_dir(self, run_id: str) -> Path:
        return self.runs_root / run_id

    def _manifest_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / MANIFEST_FILENAME

    def _statistics_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / STATISTICS_FILENAME

    def candidates(self, run_id: str) -> CandidateRepository:
        """The run-scoped candidate repository for ``run_id``."""
        return CandidateRepository(self.run_dir(run_id))

    # -------------------------------------------------------------------- run ids

    def existing_run_ids(self) -> set[str]:
        if not self.runs_root.is_dir():
            return set()
        return {
            path.name
            for path in self.runs_root.iterdir()
            if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
        }

    def new_run_id(self, now: datetime | None = None) -> str:
        """``run_YYYYMMDD_HHMMSS_xxxx`` (spec 04 section 5) — collision-checked, not scanned
        for disambiguation like the Stage 3 bare timestamp: the random suffix makes a
        second collision astronomically unlikely, so one existence check is enough.
        """
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
        existing = self.existing_run_ids()
        for _ in range(10):
            candidate = f"run_{stamp}_{secrets.token_hex(2)}"
            if candidate not in existing:
                return candidate
        raise RunError("could not mint a unique run id after 10 attempts")

    # ---------------------------------------------------------------------- create

    def create_run(
        self,
        *,
        requested_problem_ids: Sequence[str],
        requested_candidates_per_problem: int,
        strategies: Sequence[str],
        model_config: dict,
        generation_config: dict,
        prompt_version: str,
        retry: dict,
        run_id: str | None = None,
        source: str = "generate",
        environment: dict | None = None,
        candidate_schema_version: str = CANDIDATE_SCHEMA_VERSION,
    ) -> RunManifest:
        """Mint (or accept) a run id, write ``manifest.json``, and return the manifest."""
        run_id = run_id or self.new_run_id()
        manifest = RunManifest(
            run_id=run_id,
            status="created",
            created_at=utc_now_iso(),
            candidate_schema_version=candidate_schema_version,
            prompt_version=prompt_version,
            model=model_config,
            generation_config=generation_config,
            strategies=tuple(strategies),
            requested_problem_ids=tuple(requested_problem_ids),
            requested_candidates_per_problem=requested_candidates_per_problem,
            retry=retry,
            environment=environment if environment is not None else capture_environment(),
            source=source,
        )
        self._write_manifest(manifest)
        return manifest

    def create_run_from(self, source_manifest: RunManifest) -> RunManifest:
        """Seed a brand-new run from an existing manifest's configuration.

        Used for ``--resume RUN_ID --force`` (spec 04 section 13: force never overwrites
        in place, it starts a new run) and for migrating a legacy flat file into a run
        directory. The new run gets a fresh id, ``created`` status, and a freshly
        captured environment; everything else is copied.
        """
        return self.create_run(
            requested_problem_ids=source_manifest.requested_problem_ids,
            requested_candidates_per_problem=source_manifest.requested_candidates_per_problem,
            strategies=source_manifest.strategies,
            model_config=source_manifest.model,
            generation_config=source_manifest.generation_config,
            prompt_version=source_manifest.prompt_version,
            retry=source_manifest.retry,
            source="generate",
            candidate_schema_version=source_manifest.candidate_schema_version,
        )

    # ------------------------------------------------------------------------ read

    def get_run(self, run_id: str) -> RunManifest:
        path = self._manifest_path(run_id)
        if not path.is_file():
            raise RunNotFoundError(f"no run {run_id!r} at {self.run_dir(run_id)}")
        try:
            return RunManifest.from_dict(read_json(path))
        except JsonlError as exc:
            raise RunError(f"{path}: {exc}") from exc

    def list_runs(self) -> list[RunManifest]:
        """All runs, newest first.

        Sorted by ``(created_at, run_id)``: ``created_at`` has only second resolution, and
        the run id embeds the same timestamp plus a random suffix, so it is a stable
        tie-breaker for two runs created within the same second.
        """
        if not self.runs_root.is_dir():
            return []
        manifests = [
            self.get_run(path.name)
            for path in sorted(self.runs_root.iterdir())
            if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
        ]
        return sorted(manifests, key=lambda m: (m.created_at, m.run_id), reverse=True)

    def read_statistics(self, run_id: str) -> RunStatistics | None:
        path = self._statistics_path(run_id)
        if not path.is_file():
            return None
        return RunStatistics.from_dict(read_json(path))

    # ---------------------------------------------------------------------- write

    def _write_manifest(self, manifest: RunManifest) -> RunManifest:
        atomic_write_json(self._manifest_path(manifest.run_id), manifest.to_dict())
        return manifest

    def write_statistics(self, statistics: RunStatistics) -> None:
        atomic_write_json(self._statistics_path(statistics.run_id), statistics.to_dict())

    def update_status(
        self,
        run_id: str,
        status: str,
        *,
        started_at: str | None = None,
        completed_at: str | None = None,
        error: RunFailure | None = None,
    ) -> RunManifest:
        manifest = self.get_run(run_id)
        updated = manifest.with_status(
            status, started_at=started_at, completed_at=completed_at, error=error
        )
        return self._write_manifest(updated)

    def start_run(self, run_id: str) -> RunManifest:
        """``created``/``interrupted``/``failed`` -> ``running`` (spec 04 section 8)."""
        return self.update_status(run_id, "running", started_at=utc_now_iso())

    def resume_run(self, run_id: str) -> RunManifest:
        """Reopen an incomplete run. Refuses a run that already finished successfully
        (spec 04 section 11).
        """
        manifest = self.get_run(run_id)
        if manifest.status == "completed":
            raise RunError(f"run {run_id!r} is already completed; nothing to resume")
        if manifest.status not in {"created", "interrupted", "failed"}:
            raise RunError(
                f"run {run_id!r} has status {manifest.status!r}; only created, interrupted, "
                "or failed runs can be resumed"
            )
        return self.start_run(run_id)

    def complete_run(self, run_id: str) -> RunManifest:
        return self.update_status(run_id, "completed", completed_at=utc_now_iso())

    def interrupt_run(self, run_id: str) -> RunManifest:
        return self.update_status(run_id, "interrupted", completed_at=utc_now_iso())

    def fail_run(
        self,
        run_id: str,
        *,
        error_type: str,
        error_message: str,
        problem_id: str | None = None,
        generation_index: int | None = None,
    ) -> RunManifest:
        error = RunFailure(
            error_type=error_type,
            error_message=error_message,
            timestamp=utc_now_iso(),
            problem_id=problem_id,
            generation_index=generation_index,
        )
        return self.update_status(
            run_id, "failed", completed_at=utc_now_iso(), error=error
        )

    def cancel_run(self, run_id: str) -> RunManifest:
        return self.update_status(run_id, "cancelled", completed_at=utc_now_iso())


__all__ = ["MANIFEST_FILENAME", "STATISTICS_FILENAME", "RunNotFoundError", "RunRepository"]
