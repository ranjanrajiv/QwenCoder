"""Durable persistence for one run's candidates, failures, and prompts.

Records are written the moment they exist, not batched until the end of a run (spec 03
section 24), so a job killed halfway through leaves a usable, resumable file behind. Each
append is a single fsynced write via :mod:`python_dpo.atomic_io` (spec 04 section 21).

The repository is **run-scoped**: one instance owns exactly one run directory (spec 04
section 6). Within a run, ``candidate_id`` is unique — duplicate detection, resume, and
lookup are all scoped to this one file, matching sections 19 (duplicates are detected
*within* a run) and 20 (duplicates are never auto-rejected *across* runs).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..atomic_io import JsonlError, append_jsonl, iter_jsonl
from .models import Candidate, CandidateError, GenerationFailure

CANDIDATES_FILENAME = "candidates.jsonl"
FAILURES_FILENAME = "failures.jsonl"
# Stage 3 used this name; the migration path reads it, nothing else does.
LEGACY_FAILURES_FILENAME = "generation_failures.jsonl"
PROMPTS_DIRNAME = "prompts"
PROMPTS_FILENAME = "prompts.jsonl"

_PROMPT_FIELDS = frozenset(
    {"run_id", "problem_id", "generation_index", "strategy", "attempt", "prompt", "prompt_sha256"}
)


class CandidateStoreError(Exception):
    """Raised when a persisted candidate file is malformed."""


class PromptRecord:
    """A prompt as it was actually sent, recorded before inference (spec 04 section 31)."""

    __slots__ = (
        "run_id",
        "problem_id",
        "generation_index",
        "strategy",
        "attempt",
        "prompt",
        "prompt_sha256",
    )

    def __init__(
        self,
        *,
        run_id: str,
        problem_id: str,
        generation_index: int,
        strategy: str,
        attempt: int,
        prompt: str,
        prompt_sha256: str,
    ) -> None:
        self.run_id = run_id
        self.problem_id = problem_id
        self.generation_index = generation_index
        self.strategy = strategy
        self.attempt = attempt
        self.prompt = prompt
        self.prompt_sha256 = prompt_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "problem_id": self.problem_id,
            "generation_index": self.generation_index,
            "strategy": self.strategy,
            "attempt": self.attempt,
            "prompt": self.prompt,
            "prompt_sha256": self.prompt_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptRecord:
        unknown = sorted(set(data) - _PROMPT_FIELDS)
        if unknown:
            raise CandidateStoreError(f"prompt record: unknown field(s): {', '.join(unknown)}")
        missing = sorted(_PROMPT_FIELDS - set(data))
        if missing:
            raise CandidateStoreError(
                f"prompt record: missing required field(s): {', '.join(missing)}"
            )
        return cls(**data)


def _load(path: Path, build: Any) -> list[Any]:
    """Parse a JSONL file with :func:`~python_dpo.atomic_io.iter_jsonl`, validating every
    record through ``build``. Never silently skips a bad line (CLAUDE.md Data Integrity).

    The raw JSON pass is fully materialized before any record is built, so a structural
    problem later in the file (a blank line, a torn tail) is reported even if an earlier
    line would otherwise fail schema validation first.
    """
    try:
        raw_records = list(iter_jsonl(path))
    except JsonlError as exc:
        raise CandidateStoreError(str(exc)) from exc

    records = []
    for number, raw in raw_records:
        try:
            records.append(build(raw))
        except CandidateError as exc:
            raise CandidateStoreError(f"{path}:{number}: {exc}") from exc
    return records


class CandidateRepository:
    """Reads and appends the candidate, failure, and prompt artifacts of one run."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.candidates_path = self.directory / CANDIDATES_FILENAME
        self.failures_path = self.directory / FAILURES_FILENAME
        self.prompts_path = self.directory / PROMPTS_DIRNAME / PROMPTS_FILENAME

    # --------------------------------------------------------------------------- write

    def save(self, candidate: Candidate) -> None:
        append_jsonl(self.candidates_path, candidate.to_dict())

    # Stage 3 name, kept as an alias so existing call sites keep working.
    append = save

    def save_failure(self, failure: GenerationFailure) -> None:
        append_jsonl(self.failures_path, failure.to_dict())

    append_failure = save_failure

    def append_prompt(self, record: PromptRecord) -> None:
        append_jsonl(self.prompts_path, record.to_dict())

    # ---------------------------------------------------------------------------- read

    def load_all(self) -> list[Candidate]:
        """Load every candidate in file order. Returns ``[]`` when nothing exists yet."""
        return _load(self.candidates_path, Candidate.from_dict)

    def load_failures(self) -> list[GenerationFailure]:
        return _load(self.failures_path, GenerationFailure.from_dict)

    def load_prompts(self) -> list[PromptRecord]:
        return _load(self.prompts_path, PromptRecord.from_dict)

    # ------------------------------------------------------------------ spec 04 section 23

    def get(self, candidate_id: str) -> Candidate | None:
        for candidate in self.load_all():
            if candidate.candidate_id == candidate_id:
                return candidate
        return None

    def exists(self, candidate_id: str) -> bool:
        return self.get(candidate_id) is not None

    def list(self) -> list[Candidate]:
        return self.load_all()

    def count(self) -> int:
        return len(self.load_all())

    def find_by_problem(self, problem_id: str) -> list[Candidate]:
        return [c for c in self.load_all() if c.problem_id == problem_id]

    def find_by_hash(self, code_sha256: str) -> list[Candidate]:
        return [c for c in self.load_all() if c.code_sha256 == code_sha256]

    # ------------------------------------------------------------------------- indexes

    def existing_keys(self) -> set[tuple[str, int]]:
        """``(problem_id, generation_index)`` pairs already generated in this run — the
        resume index (spec 04 section 12). A generation that only *failed* left no
        candidate behind, so it is absent here and will be retried on resume.
        """
        return {
            (candidate.problem_id, candidate.generation_index) for candidate in self.load_all()
        }

    def code_index(self) -> dict[str, dict[str, str]]:
        """``problem_id -> {code_sha256: earliest candidate_id}`` for duplicate detection,
        scoped to this run (spec 04 section 19). File order is generation order, so
        ``setdefault`` keeps the earliest candidate as the one a later duplicate points at.
        """
        index: dict[str, dict[str, str]] = {}
        for candidate in self.load_all():
            index.setdefault(candidate.problem_id, {}).setdefault(
                candidate.code_sha256, candidate.candidate_id
            )
        return index

    def load_index(self) -> tuple[set[tuple[str, int]], dict[str, dict[str, str]], list[Candidate]]:
        """``(existing_keys, code_index, candidates)`` from a single file pass."""
        candidates = self.load_all()
        existing: set[tuple[str, int]] = set()
        code_index: dict[str, dict[str, str]] = {}
        for candidate in candidates:
            existing.add((candidate.problem_id, candidate.generation_index))
            code_index.setdefault(candidate.problem_id, {}).setdefault(
                candidate.code_sha256, candidate.candidate_id
            )
        return existing, code_index, candidates


__all__: Sequence[str] = [
    "CANDIDATES_FILENAME",
    "FAILURES_FILENAME",
    "LEGACY_FAILURES_FILENAME",
    "PROMPTS_DIRNAME",
    "PROMPTS_FILENAME",
    "CandidateRepository",
    "CandidateStoreError",
    "PromptRecord",
]
