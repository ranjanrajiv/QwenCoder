"""Central owner of preference run state: creation, status transitions, listing, resume.

Mirrors :class:`python_dpo.ranking.run_repository.RankingRunRepository`: every preference
run is one directory under ``preferences_root``; this repository is the only code that
mints preference run ids, writes ``manifest.json``, or writes ``statistics.json``/
``quality_report.json``/``split_manifest.json``.

A fourth copy of this run-directory plumbing (after Stages 4, 6, 7) rather than a shared
base — Stage 7 deliberately deferred that extraction to keep its own blast radius
contained, and that decision stands here for the same reason, recorded as debt in the
Stage 8 plan.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..atomic_io import JsonlError, atomic_write_json, read_json
from .errors import PreferenceRunNotFoundError
from .models import (
    PreferenceManifest,
    PreferenceModelError,
    PreferenceStatistics,
    QualityReport,
    utc_now_iso,
)
from .repository import PreferenceRepository
from .splitter import SplitManifest

MANIFEST_FILENAME = "manifest.json"
STATISTICS_FILENAME = "statistics.json"


class PreferenceRunError(Exception):
    """Raised for preference-run-level problems (bad manifest, bad transition)."""


class PreferenceRunRepository:
    """Owns preference run directories under ``preferences_root``."""

    def __init__(self, preferences_root: Path) -> None:
        self.preferences_root = Path(preferences_root)

    # ------------------------------------------------------------------------- paths

    def run_dir(self, preference_run_id: str) -> Path:
        return self.preferences_root / preference_run_id

    def _manifest_path(self, preference_run_id: str) -> Path:
        return self.run_dir(preference_run_id) / MANIFEST_FILENAME

    def _statistics_path(self, preference_run_id: str) -> Path:
        return self.run_dir(preference_run_id) / STATISTICS_FILENAME

    def results(self, preference_run_id: str) -> PreferenceRepository:
        """The preference-run-scoped results repository for ``preference_run_id``."""
        return PreferenceRepository(self.run_dir(preference_run_id))

    # -------------------------------------------------------------------- run ids

    def existing_run_ids(self) -> set[str]:
        if not self.preferences_root.is_dir():
            return set()
        return {
            path.name
            for path in self.preferences_root.iterdir()
            if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
        }

    def new_run_id(self, now: datetime | None = None) -> str:
        """``pref_YYYYMMDD_HHMMSS_xxxx`` (spec section 50)."""
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
        existing = self.existing_run_ids()
        for _ in range(10):
            candidate = f"pref_{stamp}_{secrets.token_hex(2)}"
            if candidate not in existing:
                return candidate
        raise PreferenceRunError("could not mint a unique preference run id after 10 attempts")

    # ---------------------------------------------------------------------- create

    def create_run(
        self,
        *,
        ranking_run_id: str,
        evaluation_run_id: str,
        candidate_run_id: str,
        preference_version: str,
        selection_policy: str,
        selection_policy_version: str,
        minimum_score_margin: float,
        split_ratios: dict[str, float],
        split_seed: int,
        builder_version: str,
        max_pairs_per_problem: int | None = None,
        preference_run_id: str | None = None,
    ) -> PreferenceManifest:
        """Mint (or accept) a preference run id, write ``manifest.json``, and return it."""
        preference_run_id = preference_run_id or self.new_run_id()
        manifest = PreferenceManifest(
            preference_run_id=preference_run_id,
            ranking_run_id=ranking_run_id,
            evaluation_run_id=evaluation_run_id,
            candidate_run_id=candidate_run_id,
            status="created",
            created_at=utc_now_iso(),
            preference_version=preference_version,
            selection_policy=selection_policy,
            selection_policy_version=selection_policy_version,
            minimum_score_margin=minimum_score_margin,
            max_pairs_per_problem=max_pairs_per_problem,
            split_ratios=split_ratios,
            split_seed=split_seed,
            builder_version=builder_version,
        )
        self._write_manifest(manifest)
        return manifest

    # ------------------------------------------------------------------------ read

    def get_run(self, preference_run_id: str) -> PreferenceManifest:
        path = self._manifest_path(preference_run_id)
        if not path.is_file():
            raise PreferenceRunNotFoundError(
                f"no preference run {preference_run_id!r} at {self.run_dir(preference_run_id)}"
            )
        try:
            return PreferenceManifest.from_dict(read_json(path))
        except (JsonlError, PreferenceModelError) as exc:
            raise PreferenceRunError(f"{path}: {exc}") from exc

    def list_runs(self) -> list[PreferenceManifest]:
        """All preference runs, newest first."""
        if not self.preferences_root.is_dir():
            return []
        manifests = [
            self.get_run(path.name)
            for path in sorted(self.preferences_root.iterdir())
            if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
        ]
        return sorted(
            manifests, key=lambda m: (m.created_at, m.preference_run_id), reverse=True
        )

    def latest_run_for_ranking_run(self, ranking_run_id: str) -> PreferenceManifest | None:
        matches = [m for m in self.list_runs() if m.ranking_run_id == ranking_run_id]
        return matches[0] if matches else None

    def read_statistics(self, preference_run_id: str) -> PreferenceStatistics | None:
        path = self._statistics_path(preference_run_id)
        if not path.is_file():
            return None
        return PreferenceStatistics.from_dict(read_json(path))

    def read_quality_report(self, preference_run_id: str) -> QualityReport | None:
        path = self.run_dir(preference_run_id) / "quality_report.json"
        if not path.is_file():
            return None
        return QualityReport.from_dict(read_json(path))

    def read_split_manifest(self, preference_run_id: str) -> SplitManifest | None:
        path = self.run_dir(preference_run_id) / "split_manifest.json"
        if not path.is_file():
            return None
        return SplitManifest.from_dict(read_json(path))

    # ---------------------------------------------------------------------- write

    def _write_manifest(self, manifest: PreferenceManifest) -> PreferenceManifest:
        atomic_write_json(self._manifest_path(manifest.preference_run_id), manifest.to_dict())
        return manifest

    def write_statistics(self, statistics: PreferenceStatistics) -> None:
        atomic_write_json(
            self._statistics_path(statistics.preference_run_id), statistics.to_dict()
        )

    def write_quality_report(self, report: QualityReport) -> None:
        atomic_write_json(
            self.run_dir(report.preference_run_id) / "quality_report.json", report.to_dict()
        )

    def write_split_manifest(
        self, preference_run_id: str, split_manifest: SplitManifest
    ) -> None:
        atomic_write_json(
            self.run_dir(preference_run_id) / "split_manifest.json", split_manifest.to_dict()
        )

    def update_status(
        self,
        preference_run_id: str,
        status: str,
        *,
        started_at: str | None = None,
        completed_at: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> PreferenceManifest:
        manifest = self.get_run(preference_run_id)
        try:
            updated = manifest.with_status(
                status, started_at=started_at, completed_at=completed_at, error=error
            )
        except PreferenceModelError as exc:
            raise PreferenceRunError(str(exc)) from exc
        return self._write_manifest(updated)

    def start_run(self, preference_run_id: str) -> PreferenceManifest:
        """``created``/``interrupted``/``failed`` -> ``running``."""
        return self.update_status(preference_run_id, "running", started_at=utc_now_iso())

    def resume_run(self, preference_run_id: str) -> PreferenceManifest:
        """Reopen an incomplete preference run. Refuses a run that already completed
        (spec section 83).
        """
        manifest = self.get_run(preference_run_id)
        if manifest.status == "completed":
            raise PreferenceRunError(
                f"preference run {preference_run_id!r} is already completed; nothing to resume"
            )
        if manifest.status not in {"created", "interrupted", "failed"}:
            raise PreferenceRunError(
                f"preference run {preference_run_id!r} has status {manifest.status!r}; only "
                "created, interrupted, or failed runs can be resumed"
            )
        return self.start_run(preference_run_id)

    def complete_run(self, preference_run_id: str) -> PreferenceManifest:
        return self.update_status(preference_run_id, "completed", completed_at=utc_now_iso())

    def interrupt_run(self, preference_run_id: str) -> PreferenceManifest:
        return self.update_status(preference_run_id, "interrupted", completed_at=utc_now_iso())

    def fail_run(
        self, preference_run_id: str, *, error_type: str, error_message: str
    ) -> PreferenceManifest:
        error = {
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": utc_now_iso(),
        }
        return self.update_status(
            preference_run_id, "failed", completed_at=utc_now_iso(), error=error
        )

    def cancel_run(self, preference_run_id: str) -> PreferenceManifest:
        return self.update_status(preference_run_id, "cancelled", completed_at=utc_now_iso())


__all__ = [
    "MANIFEST_FILENAME",
    "STATISTICS_FILENAME",
    "PreferenceRunError",
    "PreferenceRunRepository",
]
