"""Append-only JSONL persistence for candidates and generation failures.

Records are written the moment they exist, not batched until the end of a run
(spec 03 section 24), so a job killed halfway through leaves a usable, resumable file
behind. Each ``append`` opens, writes, and closes, which costs a few syscalls per
candidate and buys durability at the granularity that matters.

The file is **append-only**. ``--force`` mints a new ``run_id`` and appends rather than
rewriting (spec 03 sections 21.1, 28.1), so the same ``candidate_id`` may legitimately
appear more than once with different runs. Reading code that wants one record per
candidate should use :meth:`CandidateRepository.latest_by_candidate_id`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import Candidate, CandidateError, GenerationFailure

CANDIDATES_FILENAME = "candidates.jsonl"
FAILURES_FILENAME = "generation_failures.jsonl"


class CandidateStoreError(Exception):
    """Raised when a persisted candidate file is malformed."""


def _read_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Parse a JSONL file into (line number, object) pairs.

    Never silently skips a bad line — a malformed record is a hard error with its line
    number, matching ``problems/storage.py`` and CLAUDE.md's Data Integrity rule.
    """
    if not path.is_file():
        return []

    records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                raise CandidateStoreError(
                    f"{path}:{number}: blank line; JSONL must hold exactly one object per line"
                )
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise CandidateStoreError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise CandidateStoreError(f"{path}:{number}: expected a JSON object")
            records.append((number, record))
    return records


class CandidateRepository:
    """Reads and appends the two candidate artifacts in one directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.candidates_path = self.directory / CANDIDATES_FILENAME
        self.failures_path = self.directory / FAILURES_FILENAME

    def _append_line(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")

    def append(self, candidate: Candidate) -> None:
        self._append_line(self.candidates_path, candidate.to_dict())

    def append_failure(self, failure: GenerationFailure) -> None:
        self._append_line(self.failures_path, failure.to_dict())

    def load_all(self) -> list[Candidate]:
        """Load every candidate in file order. Returns ``[]`` when nothing exists yet."""
        candidates: list[Candidate] = []
        for number, record in _read_records(self.candidates_path):
            try:
                candidates.append(Candidate.from_dict(record))
            except CandidateError as exc:
                raise CandidateStoreError(f"{self.candidates_path}:{number}: {exc}") from exc
        return candidates

    def load_failures(self) -> list[GenerationFailure]:
        failures: list[GenerationFailure] = []
        for number, record in _read_records(self.failures_path):
            try:
                failures.append(GenerationFailure.from_dict(record))
            except CandidateError as exc:
                raise CandidateStoreError(f"{self.failures_path}:{number}: {exc}") from exc
        return failures

    def existing_keys(self) -> set[tuple[str, int]]:
        """``(problem_id, generation_index)`` pairs already generated, across all runs.

        This is the resume index (spec 03 section 28.1). A generation that previously
        *failed* left no candidate behind and so is absent here, which is what makes
        failures retryable on the next run.
        """
        return {
            (candidate.problem_id, candidate.generation_index)
            for candidate in self.load_all()
        }

    def code_index(self) -> dict[str, dict[str, str]]:
        """``problem_id -> {code: earliest candidate_id}`` for exact duplicate detection.

        Scoped per problem: identical code for two *different* problems is a coincidence,
        not a duplicate. File order is generation order, so ``setdefault`` keeps the
        earliest candidate as the one a later duplicate points at.
        """
        index: dict[str, dict[str, str]] = {}
        for candidate in self.load_all():
            index.setdefault(candidate.problem_id, {}).setdefault(
                candidate.code, candidate.candidate_id
            )
        return index

    def existing_run_ids(self) -> set[str]:
        runs = {candidate.run_id for candidate in self.load_all()}
        runs.update(failure.run_id for failure in self.load_failures())
        return runs

    def latest_by_candidate_id(self) -> dict[str, Candidate]:
        """One record per ``candidate_id``, preferring the most recent run.

        The file is append-only, so a later line is by construction a later run — no
        timestamp comparison needed.
        """
        latest: dict[str, Candidate] = {}
        for candidate in self.load_all():
            latest[candidate.candidate_id] = candidate
        return latest

    def new_run_id(self, now: datetime | None = None) -> str:
        """Mint a run id, disambiguating if this second already produced one.

        Format is the ``20260817_103000`` shape from spec 03 section 22 — sortable and
        readable, and deliberately not a UUID.
        """
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
        existing = self.existing_run_ids()
        if stamp not in existing:
            return stamp
        suffix = 2
        while f"{stamp}_{suffix}" in existing:
            suffix += 1
        return f"{stamp}_{suffix}"


__all__: Sequence[str] = [
    "CANDIDATES_FILENAME",
    "FAILURES_FILENAME",
    "CandidateRepository",
    "CandidateStoreError",
]
