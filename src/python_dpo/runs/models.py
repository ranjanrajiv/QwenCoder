"""Typed schema for generation runs: the manifest and the statistics cache.

Frozen dataclasses validating in ``__post_init__``, matching the house style of
``python_dpo.problems.models`` and ``python_dpo.candidates.models``.

A :class:`RunManifest` is the historical source of truth for a run's configuration (spec
04 section 34) — it is written once at run creation and never silently reinterpreted from
today's ``config.yaml``. A :class:`RunStatistics` is always a *cache*: it must be
reconstructable from ``candidates.jsonl`` and ``failures.jsonl`` alone, via
:meth:`RunStatistics.from_records` (section 25).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..candidates.models import (
    ERROR_TYPES,
    INFRASTRUCTURE_ERROR_TYPES,
    Candidate,
    GenerationFailure,
)

MANIFEST_VERSION = "1.0"
STATISTICS_VERSION = "1.0"

RUN_STATUSES = frozenset(
    {"created", "running", "completed", "failed", "interrupted", "cancelled"}
)

# A run may only move to `completed` from `running` (spec 04 sections 8, 9). `failed`
# and `cancelled` are likewise reachable only while work is in flight or queued.
RUN_STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    "created": frozenset({"running", "cancelled"}),
    "running": frozenset({"running", "completed", "failed", "interrupted", "cancelled"}),
    "interrupted": frozenset({"running", "cancelled"}),
    "failed": frozenset({"running"}),
    "completed": frozenset(),
    "cancelled": frozenset(),
}

RUN_SOURCES = frozenset({"generate", "migrated"})


class RunError(Exception):
    """Raised when a run manifest, statistics record, or status transition is invalid."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunError(f"{label} must be a non-empty string")
    return value


def _require_optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RunError(f"{label} must be a non-empty string or null")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RunError(f"{label} must be an integer of 1 or greater")
    return value


def _require_nonneg_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunError(f"{label} must be an integer of 0 or greater")
    return value


_RUN_FAILURE_FIELDS = frozenset(
    {"error_type", "error_message", "timestamp", "problem_id", "generation_index"}
)


@dataclass(frozen=True)
class RunFailure:
    """A run-level (infrastructure) failure that ended the run (spec 04 section 10)."""

    error_type: str
    error_message: str
    timestamp: str
    problem_id: str | None = None
    generation_index: int | None = None

    def __post_init__(self) -> None:
        if self.error_type not in ERROR_TYPES:
            raise RunError(
                f"error_type must be one of {', '.join(sorted(ERROR_TYPES))}, "
                f"got {self.error_type!r}"
            )
        _require_text(self.error_message, "error_message")
        _require_text(self.timestamp, "timestamp")
        _require_optional_text(self.problem_id, "problem_id")
        if self.generation_index is not None:
            _require_positive_int(self.generation_index, "generation_index")

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "problem_id": self.problem_id,
            "generation_index": self.generation_index,
        }

    @classmethod
    def from_dict(cls, data: Any) -> RunFailure:
        if not isinstance(data, dict):
            raise RunError("run error: expected a JSON object")
        unknown = sorted(set(data) - _RUN_FAILURE_FIELDS)
        if unknown:
            raise RunError(f"run error: unknown field(s): {', '.join(unknown)}")
        missing = sorted({"error_type", "error_message", "timestamp"} - set(data))
        if missing:
            raise RunError(f"run error: missing required field(s): {', '.join(missing)}")
        return cls(
            error_type=data["error_type"],
            error_message=data["error_message"],
            timestamp=data["timestamp"],
            problem_id=data.get("problem_id"),
            generation_index=data.get("generation_index"),
        )


_MANIFEST_FIELDS = frozenset(
    {
        "manifest_version",
        "run_id",
        "status",
        "created_at",
        "started_at",
        "completed_at",
        "candidate_schema_version",
        "prompt_version",
        "model",
        "generation_config",
        "strategies",
        "requested_problem_ids",
        "requested_problems",
        "requested_candidates_per_problem",
        "retry",
        "environment",
        "error",
        "source",
    }
)


@dataclass(frozen=True)
class RunManifest:
    """The complete, historical configuration snapshot for one generation run.

    Everything a future researcher needs to answer "which model, prompt, config, and
    problems produced this run's candidates?" (spec 04 section 3) without consulting
    today's ``config.yaml`` (section 34).
    """

    run_id: str
    status: str
    created_at: str
    candidate_schema_version: str
    prompt_version: str
    model: dict[str, Any]
    generation_config: dict[str, Any]
    strategies: tuple[str, ...]
    requested_problem_ids: tuple[str, ...]
    requested_candidates_per_problem: int
    retry: dict[str, Any]
    environment: dict[str, Any]
    manifest_version: str = MANIFEST_VERSION
    started_at: str | None = None
    completed_at: str | None = None
    error: RunFailure | None = None
    source: str = "generate"

    def __post_init__(self) -> None:
        _require_text(self.manifest_version, "manifest_version")
        _require_text(self.run_id, "run_id")
        if self.status not in RUN_STATUSES:
            raise RunError(
                f"status must be one of {', '.join(sorted(RUN_STATUSES))}, got {self.status!r}"
            )
        _require_text(self.created_at, "created_at")
        _require_optional_text(self.started_at, "started_at")
        _require_optional_text(self.completed_at, "completed_at")
        _require_text(self.candidate_schema_version, "candidate_schema_version")
        _require_text(self.prompt_version, "prompt_version")

        if not isinstance(self.model, dict) or not self.model:
            raise RunError("model must be a non-empty mapping")
        if not isinstance(self.generation_config, dict) or not self.generation_config:
            raise RunError("generation_config must be a non-empty mapping")

        if isinstance(self.strategies, (str, bytes)) or not isinstance(self.strategies, (list, tuple)):
            raise RunError("strategies must be a sequence of strategy names")
        object.__setattr__(self, "strategies", tuple(self.strategies))
        if not self.strategies:
            raise RunError("strategies must not be empty")
        if len(self.strategies) != self.requested_candidates_per_problem:
            raise RunError(
                "strategies must have exactly requested_candidates_per_problem entries "
                f"({self.requested_candidates_per_problem}), got {len(self.strategies)}"
            )

        if isinstance(self.requested_problem_ids, (str, bytes)) or not isinstance(
            self.requested_problem_ids, (list, tuple)
        ):
            raise RunError("requested_problem_ids must be a sequence of problem ids")
        object.__setattr__(self, "requested_problem_ids", tuple(self.requested_problem_ids))
        if not self.requested_problem_ids:
            raise RunError("requested_problem_ids must not be empty")
        if len(set(self.requested_problem_ids)) != len(self.requested_problem_ids):
            raise RunError("requested_problem_ids must not contain duplicates")

        _require_positive_int(
            self.requested_candidates_per_problem, "requested_candidates_per_problem"
        )

        if not isinstance(self.retry, dict) or "max_attempts" not in self.retry:
            raise RunError("retry must be a mapping containing 'max_attempts'")
        _require_positive_int(self.retry["max_attempts"], "retry.max_attempts")

        if not isinstance(self.environment, dict):
            raise RunError("environment must be a mapping")

        if self.error is not None and not isinstance(self.error, RunFailure):
            raise RunError("error must be a RunFailure or null")

        if self.source not in RUN_SOURCES:
            raise RunError(
                f"source must be one of {', '.join(sorted(RUN_SOURCES))}, got {self.source!r}"
            )

    @property
    def requested_problems(self) -> int:
        return len(self.requested_problem_ids)

    def with_status(
        self,
        status: str,
        *,
        started_at: str | None = None,
        completed_at: str | None = None,
        error: RunFailure | None = None,
    ) -> RunManifest:
        """Return a copy with a new status, enforcing the allowed transition graph."""
        allowed = RUN_STATUS_TRANSITIONS.get(self.status, frozenset())
        if status != self.status and status not in allowed:
            raise RunError(f"cannot transition run status from {self.status!r} to {status!r}")

        updates: dict[str, Any] = {"status": status}
        if started_at is not None:
            updates["started_at"] = started_at
        if completed_at is not None:
            updates["completed_at"] = completed_at
        if error is not None:
            updates["error"] = error
        return dataclasses.replace(self, **updates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "run_id": self.run_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "candidate_schema_version": self.candidate_schema_version,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "generation_config": self.generation_config,
            "strategies": list(self.strategies),
            "requested_problem_ids": list(self.requested_problem_ids),
            "requested_problems": self.requested_problems,
            "requested_candidates_per_problem": self.requested_candidates_per_problem,
            "retry": self.retry,
            "environment": self.environment,
            "error": self.error.to_dict() if self.error is not None else None,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Any) -> RunManifest:
        if not isinstance(data, dict):
            raise RunError("run manifest: expected a JSON object")
        unknown = sorted(set(data) - _MANIFEST_FIELDS - {"requested_problems"})
        if unknown:
            raise RunError(f"run manifest: unknown field(s): {', '.join(unknown)}")
        required = {
            "run_id",
            "status",
            "created_at",
            "candidate_schema_version",
            "prompt_version",
            "model",
            "generation_config",
            "strategies",
            "requested_problem_ids",
            "requested_candidates_per_problem",
            "retry",
            "environment",
        }
        missing = sorted(required - set(data))
        if missing:
            raise RunError(f"run manifest: missing required field(s): {', '.join(missing)}")

        error = data.get("error")
        return cls(
            manifest_version=data.get("manifest_version", MANIFEST_VERSION),
            run_id=data["run_id"],
            status=data["status"],
            created_at=data["created_at"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            candidate_schema_version=data["candidate_schema_version"],
            prompt_version=data["prompt_version"],
            model=data["model"],
            generation_config=data["generation_config"],
            strategies=tuple(data["strategies"]),
            requested_problem_ids=tuple(data["requested_problem_ids"]),
            requested_candidates_per_problem=data["requested_candidates_per_problem"],
            retry=data["retry"],
            environment=data["environment"],
            error=RunFailure.from_dict(error) if error is not None else None,
            source=data.get("source", "generate"),
        )


_STATISTICS_FIELDS = frozenset(
    {
        "statistics_version",
        "run_id",
        "problems_requested",
        "problems_completed",
        "candidates_requested",
        "candidates_generated",
        "generation_failures",
        "retry_attempts",
        "syntax_valid",
        "syntax_invalid",
        "function_name_valid",
        "duplicates",
        "candidates_by_strategy",
        "failures_by_error_type",
        "computed_at",
    }
)


@dataclass(frozen=True)
class RunStatistics:
    """Run statistics — always reconstructable from persisted records (spec 04 section 25).

    Distinguishes ``requested`` from ``generated`` from ``valid`` from ``failed`` per
    section 26: nothing here is conflated.
    """

    run_id: str
    problems_requested: int
    problems_completed: int
    candidates_requested: int
    candidates_generated: int
    generation_failures: int
    retry_attempts: int
    syntax_valid: int
    syntax_invalid: int
    function_name_valid: int
    duplicates: int
    candidates_by_strategy: dict[str, int]
    failures_by_error_type: dict[str, int]
    computed_at: str
    statistics_version: str = STATISTICS_VERSION

    def __post_init__(self) -> None:
        _require_text(self.statistics_version, "statistics_version")
        _require_text(self.run_id, "run_id")
        for name in (
            "problems_requested",
            "problems_completed",
            "candidates_requested",
            "candidates_generated",
            "generation_failures",
            "retry_attempts",
            "syntax_valid",
            "syntax_invalid",
            "function_name_valid",
            "duplicates",
        ):
            _require_nonneg_int(getattr(self, name), name)
        if not isinstance(self.candidates_by_strategy, dict):
            raise RunError("candidates_by_strategy must be a mapping")
        if not isinstance(self.failures_by_error_type, dict):
            raise RunError("failures_by_error_type must be a mapping")
        _require_text(self.computed_at, "computed_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistics_version": self.statistics_version,
            "run_id": self.run_id,
            "problems_requested": self.problems_requested,
            "problems_completed": self.problems_completed,
            "candidates_requested": self.candidates_requested,
            "candidates_generated": self.candidates_generated,
            "generation_failures": self.generation_failures,
            "retry_attempts": self.retry_attempts,
            "syntax_valid": self.syntax_valid,
            "syntax_invalid": self.syntax_invalid,
            "function_name_valid": self.function_name_valid,
            "duplicates": self.duplicates,
            "candidates_by_strategy": self.candidates_by_strategy,
            "failures_by_error_type": self.failures_by_error_type,
            "computed_at": self.computed_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> RunStatistics:
        if not isinstance(data, dict):
            raise RunError("run statistics: expected a JSON object")
        unknown = sorted(set(data) - _STATISTICS_FIELDS)
        if unknown:
            raise RunError(f"run statistics: unknown field(s): {', '.join(unknown)}")
        required = _STATISTICS_FIELDS - {"statistics_version"}
        missing = sorted(required - set(data))
        if missing:
            raise RunError(f"run statistics: missing required field(s): {', '.join(missing)}")
        kwargs = dict(data)
        kwargs.setdefault("statistics_version", STATISTICS_VERSION)
        return cls(**kwargs)

    @classmethod
    def from_records(
        cls,
        manifest: RunManifest,
        candidates: list[Candidate],
        failures: list[GenerationFailure],
        *,
        computed_at: str | None = None,
    ) -> RunStatistics:
        """Recompute statistics from the persisted run artifacts — never from counters
        accumulated in memory during generation (spec 04 section 25).
        """
        candidates_by_strategy: dict[str, int] = {}
        syntax_valid = syntax_invalid = function_name_valid = duplicates = 0
        candidate_keys: set[tuple[str, int]] = set()
        for candidate in candidates:
            candidates_by_strategy[candidate.strategy] = (
                candidates_by_strategy.get(candidate.strategy, 0) + 1
            )
            if candidate.syntax_valid:
                syntax_valid += 1
            else:
                syntax_invalid += 1
            if candidate.function_name_valid:
                function_name_valid += 1
            if candidate.duplicate_of is not None:
                duplicates += 1
            candidate_keys.add((candidate.problem_id, candidate.generation_index))

        failures_by_error_type: dict[str, int] = {}
        retry_attempts = 0
        failure_only_keys: set[tuple[str, int]] = set()
        for failure in failures:
            failures_by_error_type[failure.error_type] = (
                failures_by_error_type.get(failure.error_type, 0) + 1
            )
            if failure.error_type in INFRASTRUCTURE_ERROR_TYPES:
                retry_attempts += 1
            key = (failure.problem_id, failure.generation_index)
            if key not in candidate_keys:
                failure_only_keys.add(key)

        attempted_keys = candidate_keys | failure_only_keys
        problems_completed = sum(
            1
            for problem_id in manifest.requested_problem_ids
            if all(
                (problem_id, index) in attempted_keys
                for index in range(1, manifest.requested_candidates_per_problem + 1)
            )
        )

        return cls(
            run_id=manifest.run_id,
            problems_requested=manifest.requested_problems,
            problems_completed=problems_completed,
            candidates_requested=(
                manifest.requested_problems * manifest.requested_candidates_per_problem
            ),
            candidates_generated=len(candidates),
            generation_failures=len(failure_only_keys),
            retry_attempts=retry_attempts,
            syntax_valid=syntax_valid,
            syntax_invalid=syntax_invalid,
            function_name_valid=function_name_valid,
            duplicates=duplicates,
            candidates_by_strategy=candidates_by_strategy,
            failures_by_error_type=failures_by_error_type,
            computed_at=computed_at or utc_now_iso(),
        )


__all__ = [
    "MANIFEST_VERSION",
    "RUN_SOURCES",
    "RUN_STATUSES",
    "RUN_STATUS_TRANSITIONS",
    "STATISTICS_VERSION",
    "RunError",
    "RunFailure",
    "RunManifest",
    "RunStatistics",
    "utc_now_iso",
]
