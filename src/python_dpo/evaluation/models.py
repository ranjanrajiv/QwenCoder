"""Typed schema for evaluation results, runs, and statistics.

Frozen dataclasses validating in ``__post_init__``, matching the house style of
``python_dpo.candidates.models`` and ``python_dpo.runs.models``.

The layer answers *"what happened when this candidate was tested?"* — never
*"is this the best candidate?"* (spec section 2). ``EvaluationResult.status`` is a fact
about execution, not a preference judgement; nothing here computes ``chosen``/``rejected``
or a reward (spec section 89).

Two failure shapes are deliberately kept apart, mirroring how ``candidates.models``
separates a ``Candidate`` from a ``GenerationFailure``:

* :class:`EvaluationResult` — a job was built and *attempted* in the sandbox. Its
  ``status`` may still be ``infrastructure_error`` (Docker died mid-run), but a job
  existed and ran. Persisted to ``evaluations.jsonl``.
* :class:`EvaluationFailure` — no job was ever attempted: the problem could not be
  found, had no test cases (spec section 45), or the generated job failed structural
  validation (spec section 47) before ever reaching Docker. These are deterministic given
  the same candidate and problem, unlike a transient Docker fault, so resume treats them
  as settled rather than retryable. Persisted to ``failures.jsonl``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Spec section 23: the candidate-level outcome set.
EVALUATION_RESULT_STATUSES = frozenset(
    {"passed", "failed", "timeout", "syntax_error", "infrastructure_error"}
)
CANDIDATE_EVALUATION_STATUSES = EVALUATION_RESULT_STATUSES - {"infrastructure_error"}

# Spec section 21: the per-test-case outcome set.
TEST_CASE_STATUSES = frozenset({"passed", "failed", "error", "skipped"})

# Mirrors spec 04's RunManifest status set/lifecycle — an evaluation run has the same
# create -> running -> completed/failed/interrupted shape as a generation run.
EVALUATION_RUN_STATUSES = frozenset(
    {"created", "running", "completed", "failed", "interrupted", "cancelled"}
)
EVALUATION_RUN_STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    "created": frozenset({"running", "cancelled"}),
    "running": frozenset({"running", "completed", "failed", "interrupted", "cancelled"}),
    "interrupted": frozenset({"running", "cancelled"}),
    "failed": frozenset({"running"}),
    "completed": frozenset(),
    "cancelled": frozenset(),
}

# Closed set for EvaluationFailure.error_type — structural reasons no job was attempted.
EVALUATION_FAILURE_TYPES = frozenset(
    {"problem_not_found", "candidate_not_found", "empty_test_suite", "test_generation_error"}
)

MANIFEST_VERSION = "1.0"
STATISTICS_VERSION = "1.0"


class EvaluationModelError(Exception):
    """Raised when an evaluation record or manifest fails schema validation."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationModelError(f"{label} must be a non-empty string")
    return value


def _require_optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, label)


def _require_nonneg_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationModelError(f"{label} must be an integer of 0 or greater")
    return value


def _require_flag(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise EvaluationModelError(f"{label} must be true or false")
    return value


# --------------------------------------------------------------------- TestCaseResult

_TEST_CASE_RESULT_FIELDS = frozenset(
    {
        "evaluation_run_id",
        "candidate_run_id",
        "candidate_id",
        "problem_id",
        "test_case_id",
        "status",
        "duration_ms",
        "error_type",
        "error_message",
        "stdout",
        "stderr",
        "traceback",
    }
)


@dataclass(frozen=True)
class TestCaseResult:
    """One test case's outcome for one candidate (spec section 30)."""

    # Not a pytest test class despite the name; keeps collection warnings away.
    __test__ = False

    evaluation_run_id: str
    candidate_run_id: str
    candidate_id: str
    problem_id: str
    test_case_id: str
    status: str
    duration_ms: int
    error_type: str | None = None
    error_message: str | None = None
    stdout: str = ""
    stderr: str = ""
    traceback: str | None = None

    def __post_init__(self) -> None:
        for name in ("evaluation_run_id", "candidate_run_id", "candidate_id", "problem_id", "test_case_id"):
            _require_text(getattr(self, name), name)
        if self.status not in TEST_CASE_STATUSES:
            raise EvaluationModelError(
                f"status must be one of {', '.join(sorted(TEST_CASE_STATUSES))}, "
                f"got {self.status!r}"
            )
        _require_nonneg_int(self.duration_ms, "duration_ms")
        _require_optional_text(self.error_type, "error_type")
        _require_optional_text(self.error_message, "error_message")
        _require_optional_text(self.traceback, "traceback")
        for name in ("stdout", "stderr"):
            if not isinstance(getattr(self, name), str):
                raise EvaluationModelError(f"{name} must be a string")

        if self.status in ("passed", "skipped") and self.error_type is not None:
            raise EvaluationModelError(
                f"error_type must be null when status is {self.status!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_run_id": self.evaluation_run_id,
            "candidate_run_id": self.candidate_run_id,
            "candidate_id": self.candidate_id,
            "problem_id": self.problem_id,
            "test_case_id": self.test_case_id,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "traceback": self.traceback,
        }

    @classmethod
    def from_dict(cls, data: Any) -> TestCaseResult:
        if not isinstance(data, dict):
            raise EvaluationModelError("test case result: expected a JSON object")
        unknown = sorted(set(data) - _TEST_CASE_RESULT_FIELDS)
        if unknown:
            raise EvaluationModelError(
                f"test case result: unknown field(s): {', '.join(unknown)}"
            )
        required = {
            "evaluation_run_id",
            "candidate_run_id",
            "candidate_id",
            "problem_id",
            "test_case_id",
            "status",
            "duration_ms",
        }
        missing = sorted(required - set(data))
        if missing:
            raise EvaluationModelError(
                f"test case result: missing required field(s): {', '.join(missing)}"
            )
        return cls(**data)


# --------------------------------------------------------------------- EvaluationResult

_EVALUATION_RESULT_FIELDS = frozenset(
    {
        "evaluation_run_id",
        "candidate_run_id",
        "candidate_id",
        "problem_id",
        "status",
        "tests_total",
        "tests_passed",
        "tests_failed",
        "tests_error",
        "tests_skipped",
        "pass_rate",
        "duration_ms",
        "timeout",
        "syntax_error",
        "runtime_error",
        "infrastructure_error",
        "stdout",
        "stderr",
        "exit_code",
        "sandbox_container_id",
        "evaluation_timestamp",
        "metadata_discrepancy",
        "discrepancy_reason",
    }
)


def compute_pass_rate(tests_passed: int, tests_total: int) -> float:
    return tests_passed / tests_total if tests_total > 0 else 0.0


@dataclass(frozen=True)
class EvaluationResult:
    """One candidate's execution outcome against its problem's test suite.

    ``status = "passed"`` is validated, not assumed (spec section 24): it requires
    ``tests_total > 0`` and every test to have passed. Exit code 0 alone never produces
    ``passed`` — that would conflate "the process exited zero" with "the tests passed",
    which pytest's own exit code does not guarantee (an empty/skipped suite can still
    exit zero).

    This is *objective execution evidence* (spec section 62): it never states
    ``correct``/``incorrect``, a reward, or a ranking. ``pass_rate`` is a metric, not a
    preference signal (spec section 61).
    """

    evaluation_run_id: str
    candidate_run_id: str
    candidate_id: str
    problem_id: str
    status: str
    tests_total: int
    tests_passed: int
    tests_failed: int
    tests_error: int
    tests_skipped: int
    pass_rate: float
    duration_ms: int
    timeout: bool = False
    syntax_error: bool = False
    runtime_error: bool = False
    infrastructure_error: bool = False
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    sandbox_container_id: str | None = None
    evaluation_timestamp: str = ""
    metadata_discrepancy: bool = False
    discrepancy_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("evaluation_run_id", "candidate_run_id", "candidate_id", "problem_id"):
            _require_text(getattr(self, name), name)

        if self.status not in EVALUATION_RESULT_STATUSES:
            raise EvaluationModelError(
                f"status must be one of {', '.join(sorted(EVALUATION_RESULT_STATUSES))}, "
                f"got {self.status!r}"
            )

        for name in ("tests_total", "tests_passed", "tests_failed", "tests_error", "tests_skipped"):
            _require_nonneg_int(getattr(self, name), name)
        _require_nonneg_int(self.duration_ms, "duration_ms")

        for name in ("timeout", "syntax_error", "runtime_error", "infrastructure_error", "metadata_discrepancy"):
            _require_flag(getattr(self, name), name)
        for name in ("stdout", "stderr"):
            if not isinstance(getattr(self, name), str):
                raise EvaluationModelError(f"{name} must be a string")
        _require_optional_text(self.sandbox_container_id, "sandbox_container_id")
        _require_optional_text(self.discrepancy_reason, "discrepancy_reason")

        if self.exit_code is not None and (
            isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int)
        ):
            raise EvaluationModelError("exit_code must be an integer or null")

        # Spec section 68: the four counts must partition tests_total exactly.
        counted = self.tests_passed + self.tests_failed + self.tests_error + self.tests_skipped
        if counted != self.tests_total:
            raise EvaluationModelError(
                "tests_passed + tests_failed + tests_error + tests_skipped "
                f"({counted}) must equal tests_total ({self.tests_total})"
            )

        # Spec section 24: passed is validated, never assumed from an exit code.
        if self.status == "passed" and not (
            self.tests_total > 0
            and self.tests_passed == self.tests_total
            and self.tests_failed == 0
            and self.tests_error == 0
            and self.tests_skipped == 0
        ):
            raise EvaluationModelError(
                "status 'passed' requires tests_total > 0 and every test to have passed"
            )

        # The four convenience booleans are fully determined by status/tests_error, kept
        # as validated (not silently overwritten) fields so a caller cannot construct an
        # internally-inconsistent record.
        expected_timeout = self.status == "timeout"
        if self.timeout != expected_timeout:
            raise EvaluationModelError(
                f"timeout must be {expected_timeout} when status is {self.status!r}"
            )
        expected_syntax_error = self.status == "syntax_error"
        if self.syntax_error != expected_syntax_error:
            raise EvaluationModelError(
                f"syntax_error must be {expected_syntax_error} when status is {self.status!r}"
            )
        expected_infrastructure_error = self.status == "infrastructure_error"
        if self.infrastructure_error != expected_infrastructure_error:
            raise EvaluationModelError(
                "infrastructure_error must be "
                f"{expected_infrastructure_error} when status is {self.status!r}"
            )
        # A candidate exception during a test — not merely a wrong-answer AssertionError —
        # sets runtime_error, independent of status (spec section 27 keeps this distinct
        # from an infrastructure failure).
        expected_runtime_error = self.tests_error > 0
        if self.runtime_error != expected_runtime_error:
            raise EvaluationModelError(
                f"runtime_error must be {expected_runtime_error} when tests_error is "
                f"{self.tests_error}"
            )

        expected_pass_rate = compute_pass_rate(self.tests_passed, self.tests_total)
        if abs(self.pass_rate - expected_pass_rate) > 1e-9:
            raise EvaluationModelError(
                f"pass_rate does not match tests_passed/tests_total: expected "
                f"{expected_pass_rate}, got {self.pass_rate}"
            )

        if self.metadata_discrepancy and self.discrepancy_reason is None:
            raise EvaluationModelError(
                "discrepancy_reason is required when metadata_discrepancy is true"
            )
        if not self.metadata_discrepancy and self.discrepancy_reason is not None:
            raise EvaluationModelError(
                "discrepancy_reason must be null when metadata_discrepancy is false"
            )

        if not self.evaluation_timestamp:
            object.__setattr__(self, "evaluation_timestamp", utc_now_iso())

    @property
    def is_candidate_outcome(self) -> bool:
        """Whether this result says something about the candidate (spec section 81's
        distinction, applied to evaluation)."""
        return self.status in CANDIDATE_EVALUATION_STATUSES

    @classmethod
    def create(
        cls,
        *,
        evaluation_run_id: str,
        candidate_run_id: str,
        candidate_id: str,
        problem_id: str,
        status: str,
        tests_passed: int,
        tests_failed: int,
        tests_error: int,
        tests_skipped: int,
        duration_ms: int,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        sandbox_container_id: str | None = None,
        metadata_discrepancy: bool = False,
        discrepancy_reason: str | None = None,
    ) -> EvaluationResult:
        """Build a result, computing every derived field from the raw counts and status.

        The counterpart to :meth:`Candidate.create` in Stage 4 — callers supply the facts,
        this computes ``tests_total``, ``pass_rate``, and the four convenience booleans so
        they can never drift from ``status``.
        """
        tests_total = tests_passed + tests_failed + tests_error + tests_skipped
        return cls(
            evaluation_run_id=evaluation_run_id,
            candidate_run_id=candidate_run_id,
            candidate_id=candidate_id,
            problem_id=problem_id,
            status=status,
            tests_total=tests_total,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_error=tests_error,
            tests_skipped=tests_skipped,
            pass_rate=compute_pass_rate(tests_passed, tests_total),
            duration_ms=duration_ms,
            timeout=status == "timeout",
            syntax_error=status == "syntax_error",
            runtime_error=tests_error > 0,
            infrastructure_error=status == "infrastructure_error",
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            sandbox_container_id=sandbox_container_id,
            metadata_discrepancy=metadata_discrepancy,
            discrepancy_reason=discrepancy_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_run_id": self.evaluation_run_id,
            "candidate_run_id": self.candidate_run_id,
            "candidate_id": self.candidate_id,
            "problem_id": self.problem_id,
            "status": self.status,
            "tests_total": self.tests_total,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_error": self.tests_error,
            "tests_skipped": self.tests_skipped,
            "pass_rate": self.pass_rate,
            "duration_ms": self.duration_ms,
            "timeout": self.timeout,
            "syntax_error": self.syntax_error,
            "runtime_error": self.runtime_error,
            "infrastructure_error": self.infrastructure_error,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "sandbox_container_id": self.sandbox_container_id,
            "evaluation_timestamp": self.evaluation_timestamp,
            "metadata_discrepancy": self.metadata_discrepancy,
            "discrepancy_reason": self.discrepancy_reason,
        }

    @classmethod
    def from_dict(cls, data: Any) -> EvaluationResult:
        if not isinstance(data, dict):
            raise EvaluationModelError("evaluation result: expected a JSON object")
        unknown = sorted(set(data) - _EVALUATION_RESULT_FIELDS)
        if unknown:
            raise EvaluationModelError(
                f"evaluation result: unknown field(s): {', '.join(unknown)}"
            )
        required = {
            "evaluation_run_id",
            "candidate_run_id",
            "candidate_id",
            "problem_id",
            "status",
            "tests_total",
            "tests_passed",
            "tests_failed",
            "tests_error",
            "tests_skipped",
            "pass_rate",
            "duration_ms",
        }
        missing = sorted(required - set(data))
        if missing:
            raise EvaluationModelError(
                f"evaluation result: missing required field(s): {', '.join(missing)}"
            )
        return cls(**data)


# -------------------------------------------------------------------- EvaluationFailure

_EVALUATION_FAILURE_FIELDS = frozenset(
    {
        "evaluation_run_id",
        "candidate_run_id",
        "candidate_id",
        "problem_id",
        "error_type",
        "error_message",
        "timestamp",
    }
)


@dataclass(frozen=True)
class EvaluationFailure:
    """A candidate for which no evaluation job was ever attempted.

    Structural and deterministic given the same candidate and problem — the problem was
    missing, had no tests (spec section 45), or the generated job failed structural
    validation (spec section 47). Unlike a transient Docker fault (recorded as an
    ``EvaluationResult`` with ``status="infrastructure_error"``), re-running would fail
    identically, so resume treats these as settled rather than retryable.
    """

    evaluation_run_id: str
    candidate_run_id: str
    candidate_id: str
    problem_id: str
    error_type: str
    error_message: str
    timestamp: str

    def __post_init__(self) -> None:
        for name in ("evaluation_run_id", "candidate_run_id", "candidate_id", "problem_id"):
            _require_text(getattr(self, name), name)
        if self.error_type not in EVALUATION_FAILURE_TYPES:
            raise EvaluationModelError(
                f"error_type must be one of {', '.join(sorted(EVALUATION_FAILURE_TYPES))}, "
                f"got {self.error_type!r}"
            )
        _require_text(self.error_message, "error_message")
        _require_text(self.timestamp, "timestamp")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_run_id": self.evaluation_run_id,
            "candidate_run_id": self.candidate_run_id,
            "candidate_id": self.candidate_id,
            "problem_id": self.problem_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Any) -> EvaluationFailure:
        if not isinstance(data, dict):
            raise EvaluationModelError("evaluation failure: expected a JSON object")
        unknown = sorted(set(data) - _EVALUATION_FAILURE_FIELDS)
        if unknown:
            raise EvaluationModelError(
                f"evaluation failure: unknown field(s): {', '.join(unknown)}"
            )
        missing = sorted(_EVALUATION_FAILURE_FIELDS - set(data))
        if missing:
            raise EvaluationModelError(
                f"evaluation failure: missing required field(s): {', '.join(missing)}"
            )
        return cls(**data)


# ------------------------------------------------------------------- EvaluationManifest

_MANIFEST_FIELDS = frozenset(
    {
        "manifest_version",
        "evaluation_run_id",
        "candidate_run_id",
        "status",
        "created_at",
        "started_at",
        "completed_at",
        "evaluator_version",
        "test_generator_version",
        "pytest_version",
        "python_version",
        "sandbox_config",
        "requested_problem_id",
        "requested_candidate_ids",
        "error",
    }
)

_MANIFEST_ERROR_FIELDS = frozenset({"error_type", "error_message", "timestamp"})


@dataclass(frozen=True)
class EvaluationManifest:
    """The historical, immutable configuration snapshot for one evaluation run.

    Mirrors ``python_dpo.runs.models.RunManifest``: it is the source of truth for what an
    evaluation run covers and how it was configured, never re-derived from today's
    ``config.yaml`` (spec sections 51, 72-75).
    """

    evaluation_run_id: str
    candidate_run_id: str
    status: str
    created_at: str
    evaluator_version: str
    test_generator_version: str
    pytest_version: str
    python_version: str
    sandbox_config: dict[str, Any]
    requested_candidate_ids: tuple[str, ...]
    manifest_version: str = MANIFEST_VERSION
    started_at: str | None = None
    completed_at: str | None = None
    requested_problem_id: str | None = None
    error: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_text(self.manifest_version, "manifest_version")
        _require_text(self.evaluation_run_id, "evaluation_run_id")
        _require_text(self.candidate_run_id, "candidate_run_id")
        if self.status not in EVALUATION_RUN_STATUSES:
            raise EvaluationModelError(
                f"status must be one of {', '.join(sorted(EVALUATION_RUN_STATUSES))}, "
                f"got {self.status!r}"
            )
        _require_text(self.created_at, "created_at")
        _require_optional_text(self.started_at, "started_at")
        _require_optional_text(self.completed_at, "completed_at")
        _require_text(self.evaluator_version, "evaluator_version")
        _require_text(self.test_generator_version, "test_generator_version")
        _require_text(self.pytest_version, "pytest_version")
        _require_text(self.python_version, "python_version")

        if not isinstance(self.sandbox_config, dict) or not self.sandbox_config:
            raise EvaluationModelError("sandbox_config must be a non-empty mapping")

        if isinstance(self.requested_candidate_ids, (str, bytes)) or not isinstance(
            self.requested_candidate_ids, (list, tuple)
        ):
            raise EvaluationModelError("requested_candidate_ids must be a sequence of candidate ids")
        object.__setattr__(self, "requested_candidate_ids", tuple(self.requested_candidate_ids))
        if not self.requested_candidate_ids:
            raise EvaluationModelError("requested_candidate_ids must not be empty")
        if len(set(self.requested_candidate_ids)) != len(self.requested_candidate_ids):
            raise EvaluationModelError("requested_candidate_ids must not contain duplicates")

        _require_optional_text(self.requested_problem_id, "requested_problem_id")

        if self.error is not None:
            if not isinstance(self.error, dict):
                raise EvaluationModelError("error must be a mapping or null")
            unknown = sorted(set(self.error) - _MANIFEST_ERROR_FIELDS)
            if unknown:
                raise EvaluationModelError(f"error: unknown field(s): {', '.join(unknown)}")
            missing = sorted(_MANIFEST_ERROR_FIELDS - set(self.error))
            if missing:
                raise EvaluationModelError(f"error: missing required field(s): {', '.join(missing)}")

    @property
    def requested_candidates(self) -> int:
        return len(self.requested_candidate_ids)

    def with_status(
        self,
        status: str,
        *,
        started_at: str | None = None,
        completed_at: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> EvaluationManifest:
        allowed = EVALUATION_RUN_STATUS_TRANSITIONS.get(self.status, frozenset())
        if status != self.status and status not in allowed:
            raise EvaluationModelError(
                f"cannot transition evaluation run status from {self.status!r} to {status!r}"
            )
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
            "evaluation_run_id": self.evaluation_run_id,
            "candidate_run_id": self.candidate_run_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "evaluator_version": self.evaluator_version,
            "test_generator_version": self.test_generator_version,
            "pytest_version": self.pytest_version,
            "python_version": self.python_version,
            "sandbox_config": self.sandbox_config,
            "requested_problem_id": self.requested_problem_id,
            "requested_candidate_ids": list(self.requested_candidate_ids),
            "requested_candidates": self.requested_candidates,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Any) -> EvaluationManifest:
        if not isinstance(data, dict):
            raise EvaluationModelError("evaluation manifest: expected a JSON object")
        unknown = sorted(set(data) - _MANIFEST_FIELDS - {"requested_candidates"})
        if unknown:
            raise EvaluationModelError(
                f"evaluation manifest: unknown field(s): {', '.join(unknown)}"
            )
        required = {
            "evaluation_run_id",
            "candidate_run_id",
            "status",
            "created_at",
            "evaluator_version",
            "test_generator_version",
            "pytest_version",
            "python_version",
            "sandbox_config",
            "requested_candidate_ids",
        }
        missing = sorted(required - set(data))
        if missing:
            raise EvaluationModelError(
                f"evaluation manifest: missing required field(s): {', '.join(missing)}"
            )
        return cls(
            manifest_version=data.get("manifest_version", MANIFEST_VERSION),
            evaluation_run_id=data["evaluation_run_id"],
            candidate_run_id=data["candidate_run_id"],
            status=data["status"],
            created_at=data["created_at"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            evaluator_version=data["evaluator_version"],
            test_generator_version=data["test_generator_version"],
            pytest_version=data["pytest_version"],
            python_version=data["python_version"],
            sandbox_config=data["sandbox_config"],
            requested_problem_id=data.get("requested_problem_id"),
            requested_candidate_ids=tuple(data["requested_candidate_ids"]),
            error=data.get("error"),
        )


# ------------------------------------------------------------------ EvaluationStatistics

_STATISTICS_FIELDS = frozenset(
    {
        "statistics_version",
        "evaluation_run_id",
        "candidates_requested",
        "candidates_evaluated",
        "evaluation_failures",
        "passed",
        "failed",
        "timeouts",
        "syntax_errors",
        "infrastructure_errors",
        "tests_total",
        "tests_passed",
        "tests_failed",
        "tests_errors",
        "tests_skipped",
        "computed_at",
    }
)


@dataclass(frozen=True)
class EvaluationStatistics:
    """Evaluation run statistics — always reconstructable from persisted records (spec
    section 58), never trusted from in-memory counters accumulated during a run.
    """

    evaluation_run_id: str
    candidates_requested: int
    candidates_evaluated: int
    evaluation_failures: int
    passed: int
    failed: int
    timeouts: int
    syntax_errors: int
    infrastructure_errors: int
    tests_total: int
    tests_passed: int
    tests_failed: int
    tests_errors: int
    tests_skipped: int
    computed_at: str
    statistics_version: str = STATISTICS_VERSION

    def __post_init__(self) -> None:
        _require_text(self.statistics_version, "statistics_version")
        _require_text(self.evaluation_run_id, "evaluation_run_id")
        for name in (
            "candidates_requested",
            "candidates_evaluated",
            "evaluation_failures",
            "passed",
            "failed",
            "timeouts",
            "syntax_errors",
            "infrastructure_errors",
            "tests_total",
            "tests_passed",
            "tests_failed",
            "tests_errors",
            "tests_skipped",
        ):
            _require_nonneg_int(getattr(self, name), name)
        _require_text(self.computed_at, "computed_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistics_version": self.statistics_version,
            "evaluation_run_id": self.evaluation_run_id,
            "candidates_requested": self.candidates_requested,
            "candidates_evaluated": self.candidates_evaluated,
            "evaluation_failures": self.evaluation_failures,
            "passed": self.passed,
            "failed": self.failed,
            "timeouts": self.timeouts,
            "syntax_errors": self.syntax_errors,
            "infrastructure_errors": self.infrastructure_errors,
            "tests_total": self.tests_total,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_errors": self.tests_errors,
            "tests_skipped": self.tests_skipped,
            "computed_at": self.computed_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> EvaluationStatistics:
        if not isinstance(data, dict):
            raise EvaluationModelError("evaluation statistics: expected a JSON object")
        unknown = sorted(set(data) - _STATISTICS_FIELDS)
        if unknown:
            raise EvaluationModelError(
                f"evaluation statistics: unknown field(s): {', '.join(unknown)}"
            )
        required = _STATISTICS_FIELDS - {"statistics_version"}
        missing = sorted(required - set(data))
        if missing:
            raise EvaluationModelError(
                f"evaluation statistics: missing required field(s): {', '.join(missing)}"
            )
        kwargs = dict(data)
        kwargs.setdefault("statistics_version", STATISTICS_VERSION)
        return cls(**kwargs)

    @classmethod
    def from_records(
        cls,
        manifest: EvaluationManifest,
        results: list[EvaluationResult],
        failures: list[EvaluationFailure],
        *,
        computed_at: str | None = None,
    ) -> EvaluationStatistics:
        passed = failed = timeouts = syntax_errors = infrastructure_errors = 0
        tests_total = tests_passed = tests_failed = tests_errors = tests_skipped = 0

        for result in results:
            if result.status == "passed":
                passed += 1
            elif result.status == "timeout":
                timeouts += 1
                failed += 1
            elif result.status == "syntax_error":
                syntax_errors += 1
                failed += 1
            elif result.status == "infrastructure_error":
                infrastructure_errors += 1
            else:  # "failed"
                failed += 1

            tests_total += result.tests_total
            tests_passed += result.tests_passed
            tests_failed += result.tests_failed
            tests_errors += result.tests_error
            tests_skipped += result.tests_skipped

        return cls(
            evaluation_run_id=manifest.evaluation_run_id,
            candidates_requested=manifest.requested_candidates,
            candidates_evaluated=len(results),
            evaluation_failures=len(failures),
            passed=passed,
            failed=failed,
            timeouts=timeouts,
            syntax_errors=syntax_errors,
            infrastructure_errors=infrastructure_errors,
            tests_total=tests_total,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_errors=tests_errors,
            tests_skipped=tests_skipped,
            computed_at=computed_at or utc_now_iso(),
        )


__all__ = [
    "CANDIDATE_EVALUATION_STATUSES",
    "EVALUATION_FAILURE_TYPES",
    "EVALUATION_RESULT_STATUSES",
    "EVALUATION_RUN_STATUSES",
    "EVALUATION_RUN_STATUS_TRANSITIONS",
    "MANIFEST_VERSION",
    "STATISTICS_VERSION",
    "TEST_CASE_STATUSES",
    "EvaluationFailure",
    "EvaluationManifest",
    "EvaluationModelError",
    "EvaluationResult",
    "EvaluationStatistics",
    "TestCaseResult",
    "compute_pass_rate",
    "utc_now_iso",
]
