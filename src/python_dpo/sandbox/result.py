"""What happened during one sandboxed execution (spec 05 sections 7, 8).

**This module makes no correctness judgement** (spec section 82). ``status="success"`` means
"the program exited zero", never "the candidate is correct" — deciding that requires running
the problem's test suite, which belongs to a later stage.

The other distinction carried here is the one spec section 81 calls critical:

* **Candidate outcomes** — ``syntax_error``, ``runtime_error``, ``timeout``,
  ``resource_exceeded``. The candidate ran (or failed to compile) and this is what it did.
* **Infrastructure outcomes** — ``infrastructure_error``. Docker was unavailable, the image
  was missing, the container could not be created. The candidate is not implicated and must
  not be judged on this.

:func:`classify` is a pure function over already-collected facts, so every branch is unit
tested without Docker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Spec section 8's controlled set.
EXECUTION_STATUSES = frozenset(
    {
        "success",
        "syntax_error",
        "runtime_error",
        "timeout",
        "resource_exceeded",
        "infrastructure_error",
        # No producer in Stage 5 — there is no cancellation API yet. Present so the closed
        # set is stable for the stages that will add one.
        "cancelled",
    }
)

CANDIDATE_STATUSES = frozenset(
    {"success", "syntax_error", "runtime_error", "timeout", "resource_exceeded"}
)
INFRASTRUCTURE_STATUSES = frozenset({"infrastructure_error"})

# Compile-time failures CPython reports before executing a single statement.
_COMPILE_ERROR_MARKERS = ("SyntaxError:", "IndentationError:", "TabError:")
# CPython prints this header for every *runtime* exception traceback. Its presence is what
# separates `raise SyntaxError(...)` (a runtime error) from a file that will not compile.
_TRACEBACK_HEADER = "Traceback (most recent call last):"

# `docker run` uses these for its own failures rather than the container's exit code:
# 125 the docker command itself failed, 126 the command could not be invoked, 127 not found.
DOCKER_CLI_EXIT_CODES = frozenset({125, 126, 127})

_RESULT_FIELDS = frozenset(
    {
        "status",
        "exit_code",
        "stdout",
        "stderr",
        "duration_ms",
        "timed_out",
        "container_id",
        "error_type",
        "error_message",
        "signal",
        "memory_limit_exceeded",
        "process_limit_exceeded",
        "network_blocked",
        "stdout_truncated",
        "stderr_truncated",
        "workspace_id",
        "created_at",
        "sandbox_config",
    }
)


class ExecutionResultError(Exception):
    """Raised when an execution result fails schema validation."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def looks_like_compile_error(stderr: str) -> bool:
    """Whether ``stderr`` shows a failure to *compile*, rather than a runtime exception.

    CPython always prints ``Traceback (most recent call last):`` before a runtime
    exception, and never before a compile-time one. So a program that deliberately does
    ``raise SyntaxError("x")`` still classifies as a runtime error, which is correct — it
    compiled fine and then chose to raise.
    """
    if _TRACEBACK_HEADER in stderr:
        return False
    return any(marker in stderr for marker in _COMPILE_ERROR_MARKERS)


def classify(
    *,
    exit_code: int | None,
    stderr: str,
    timed_out: bool = False,
    oom_killed: bool = False,
    output_limit_exceeded: bool = False,
    container_started: bool = True,
) -> str:
    """Map collected execution facts onto a spec section 8 status. Pure function.

    Order matters: an infrastructure failure outranks everything (the candidate never
    really ran), then resource ceilings, then the timeout, then the candidate's own exit.
    """
    if not container_started:
        return "infrastructure_error"
    if exit_code is not None and exit_code in DOCKER_CLI_EXIT_CODES:
        # `docker run` failed on its own terms rather than reporting the candidate's exit.
        return "infrastructure_error"

    if oom_killed or output_limit_exceeded:
        return "resource_exceeded"
    if timed_out:
        return "timeout"

    if exit_code is None:
        return "infrastructure_error"
    if exit_code == 0:
        return "success"
    if looks_like_compile_error(stderr):
        return "syntax_error"
    return "runtime_error"


def signal_from_exit_code(exit_code: int | None) -> int | None:
    """The terminating signal, when a shell-style ``128 + signum`` code says there was one.

    Spec section 51: signal termination is recorded, never hidden.
    """
    if exit_code is None or exit_code <= 128 or exit_code >= 160:
        return None
    return exit_code - 128


@dataclass(frozen=True)
class ExecutionResult:
    """One sandboxed execution's outcome (spec 05 section 7)."""

    status: str
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False
    container_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    signal: int | None = None
    memory_limit_exceeded: bool = False
    process_limit_exceeded: bool = False
    network_blocked: bool = True
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    workspace_id: str | None = None
    created_at: str = ""
    sandbox_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in EXECUTION_STATUSES:
            raise ExecutionResultError(
                f"status must be one of {', '.join(sorted(EXECUTION_STATUSES))}, "
                f"got {self.status!r}"
            )
        if self.exit_code is not None and (
            isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int)
        ):
            raise ExecutionResultError("exit_code must be an integer or null")
        for name in ("stdout", "stderr"):
            if not isinstance(getattr(self, name), str):
                raise ExecutionResultError(f"{name} must be a string")
        if (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, int)
            or self.duration_ms < 0
        ):
            raise ExecutionResultError("duration_ms must be an integer of 0 or greater")
        for name in (
            "timed_out",
            "memory_limit_exceeded",
            "process_limit_exceeded",
            "network_blocked",
            "stdout_truncated",
            "stderr_truncated",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ExecutionResultError(f"{name} must be true or false")
        if not isinstance(self.sandbox_config, dict):
            raise ExecutionResultError("sandbox_config must be a mapping")

        if self.timed_out and self.status != "timeout":
            raise ExecutionResultError(
                f"timed_out is true but status is {self.status!r}; expected 'timeout'"
            )

        if not self.created_at:
            object.__setattr__(self, "created_at", utc_now_iso())

    @property
    def is_candidate_outcome(self) -> bool:
        """Whether this result says something about the candidate (spec section 81)."""
        return self.status in CANDIDATE_STATUSES

    @property
    def is_infrastructure_failure(self) -> bool:
        return self.status in INFRASTRUCTURE_STATUSES

    @property
    def truncated(self) -> bool:
        return self.stdout_truncated or self.stderr_truncated

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
            "container_id": self.container_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "signal": self.signal,
            "memory_limit_exceeded": self.memory_limit_exceeded,
            "process_limit_exceeded": self.process_limit_exceeded,
            "network_blocked": self.network_blocked,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "workspace_id": self.workspace_id,
            "created_at": self.created_at,
            "sandbox_config": self.sandbox_config,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ExecutionResult:
        if not isinstance(data, dict):
            raise ExecutionResultError("execution result: expected a JSON object")
        unknown = sorted(set(data) - _RESULT_FIELDS)
        if unknown:
            raise ExecutionResultError(
                f"execution result: unknown field(s): {', '.join(unknown)}"
            )
        required = {"status", "exit_code", "stdout", "stderr", "duration_ms"}
        missing = sorted(required - set(data))
        if missing:
            raise ExecutionResultError(
                f"execution result: missing required field(s): {', '.join(missing)}"
            )
        return cls(**data)

    @classmethod
    def infrastructure_failure(
        cls,
        *,
        error_type: str,
        error_message: str,
        duration_ms: int = 0,
        sandbox_config: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        container_id: str | None = None,
    ) -> ExecutionResult:
        """Build a result for a failure that says nothing about the candidate."""
        return cls(
            status="infrastructure_error",
            exit_code=None,
            stdout="",
            stderr="",
            duration_ms=duration_ms,
            container_id=container_id,
            error_type=error_type,
            error_message=error_message,
            workspace_id=workspace_id,
            sandbox_config=sandbox_config or {},
        )


__all__ = [
    "CANDIDATE_STATUSES",
    "DOCKER_CLI_EXIT_CODES",
    "EXECUTION_STATUSES",
    "INFRASTRUCTURE_STATUSES",
    "ExecutionResult",
    "ExecutionResultError",
    "classify",
    "looks_like_compile_error",
    "signal_from_exit_code",
    "utc_now_iso",
]
