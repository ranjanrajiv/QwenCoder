"""Typed schema for generated candidates and generation failures.

Frozen dataclasses validating in ``__post_init__``, matching
``python_dpo.problems.models``. Construction validates, so every record reaching
``candidates.jsonl`` is already well-formed.

The two record types are deliberately disjoint (spec 03 sections 19.1, 26, 26.1):

* A :class:`Candidate` exists when the model produced code. Whether that code *parses* is
  a recorded property (``syntax_valid``), not a reason to throw it away — malformed
  output is exactly what later stages need on the rejected side of a preference pair.
* A :class:`GenerationFailure` exists when there is **no code at all** to store: an empty
  response, an inference error, or output containing nothing extractable.

One generation produces one or the other, never both.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Written to generation_failures.jsonl. Closed set so failures can be counted and
# compared across runs instead of grouped by free text (spec 03 section 27).
ERROR_TYPES = frozenset(
    {
        "model_load",
        "tokenizer",
        "inference",
        "timeout",
        "empty_output",
        "code_extraction",
    }
)

EXTRACTION_FORMATS = frozenset({"python_fence", "generic_fence", "plain", "unknown"})

_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "problem_id",
        "run_id",
        "generation_index",
        "strategy",
        "model",
        "model_revision",
        "provider",
        "prompt_version",
        "prompt",
        "raw_output",
        "code",
        "extraction_format",
        "syntax_valid",
        "syntax_error",
        "function_name_valid",
        "duplicate_of",
        "generation_config",
        "created_at",
    }
)

_FAILURE_FIELDS = frozenset(
    {
        "run_id",
        "problem_id",
        "generation_index",
        "strategy",
        "error_type",
        "error_message",
        "timestamp",
    }
)


class CandidateError(Exception):
    """Raised when a candidate or failure record fails schema validation."""


def utc_now_iso() -> str:
    """Timestamp for record creation, second-resolution UTC."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_candidate_id(problem_id: str, generation_index: int) -> str:
    """Deterministic candidate id: ``p001_c001`` (spec 03 section 21).

    Unique within a run, not within the file — ``--force`` starts a new run and appends,
    so the file-wide key is ``(run_id, candidate_id)`` (section 21.1).
    """
    return f"{problem_id}_c{generation_index:03d}"


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateError(f"{label} must be a non-empty string")
    return value


def _require_index(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CandidateError(f"{label} must be an integer of 1 or greater")
    return value


def _require_flag(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise CandidateError(f"{label} must be true or false")
    return value


def _require_optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CandidateError(f"{label} must be a non-empty string or null")
    return value


@dataclass(frozen=True)
class Candidate:
    """One generated program, with everything needed to reconstruct its provenance.

    Both ``raw_output`` and ``code`` are stored (spec 03 section 25): the raw text is the
    only way to debug an extraction that went wrong, and the extracted code is what later
    stages evaluate. ``prompt`` is stored too, so a candidate stays interpretable even
    after the prompt template moves to a new version.
    """

    candidate_id: str
    problem_id: str
    run_id: str
    generation_index: int
    strategy: str
    model: str
    provider: str
    prompt_version: str
    prompt: str
    raw_output: str
    code: str
    extraction_format: str
    syntax_valid: bool
    function_name_valid: bool
    generation_config: dict[str, Any]
    created_at: str
    model_revision: str | None = None
    syntax_error: str | None = None
    duplicate_of: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        _require_text(self.problem_id, "problem_id")
        _require_text(self.run_id, "run_id")
        _require_index(self.generation_index, "generation_index")
        _require_text(self.strategy, "strategy")
        _require_text(self.model, "model")
        _require_text(self.provider, "provider")
        _require_text(self.prompt_version, "prompt_version")
        _require_text(self.prompt, "prompt")
        _require_text(self.created_at, "created_at")
        _require_optional_text(self.model_revision, "model_revision")
        _require_optional_text(self.syntax_error, "syntax_error")

        if not isinstance(self.raw_output, str):
            raise CandidateError("raw_output must be a string")

        # A candidate exists only when code was extracted; an extraction failure is
        # recorded as a GenerationFailure instead (sections 18, 26).
        _require_text(self.code, "code")

        if self.extraction_format not in EXTRACTION_FORMATS - {"unknown"}:
            raise CandidateError(
                "extraction_format must be one of "
                f"{', '.join(sorted(EXTRACTION_FORMATS - {'unknown'}))}, "
                f"got {self.extraction_format!r}"
            )

        _require_flag(self.syntax_valid, "syntax_valid")
        _require_flag(self.function_name_valid, "function_name_valid")

        if self.syntax_valid and self.syntax_error is not None:
            raise CandidateError("syntax_error must be null when syntax_valid is true")

        if not self.candidate_id.startswith(f"{self.problem_id}_c"):
            raise CandidateError(
                f"candidate_id {self.candidate_id!r} does not belong to problem "
                f"{self.problem_id!r}"
            )

        duplicate_of = _require_optional_text(self.duplicate_of, "duplicate_of")
        if duplicate_of is not None and duplicate_of == self.candidate_id:
            raise CandidateError("duplicate_of must reference a different candidate")

        if not isinstance(self.generation_config, dict):
            raise CandidateError("generation_config must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "problem_id": self.problem_id,
            "run_id": self.run_id,
            "generation_index": self.generation_index,
            "strategy": self.strategy,
            "model": self.model,
            "model_revision": self.model_revision,
            "provider": self.provider,
            "prompt_version": self.prompt_version,
            "prompt": self.prompt,
            "raw_output": self.raw_output,
            "code": self.code,
            "extraction_format": self.extraction_format,
            "syntax_valid": self.syntax_valid,
            "syntax_error": self.syntax_error,
            "function_name_valid": self.function_name_valid,
            "duplicate_of": self.duplicate_of,
            "generation_config": self.generation_config,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Candidate:
        if not isinstance(data, dict):
            raise CandidateError("candidate: expected a JSON object")
        unknown = sorted(set(data) - _CANDIDATE_FIELDS)
        if unknown:
            raise CandidateError(f"candidate: unknown field(s): {', '.join(unknown)}")
        required = _CANDIDATE_FIELDS - {"model_revision", "syntax_error", "duplicate_of"}
        missing = sorted(required - set(data))
        if missing:
            raise CandidateError(f"candidate: missing required field(s): {', '.join(missing)}")
        return cls(**data)


@dataclass(frozen=True)
class GenerationFailure:
    """A generation that produced no candidate (spec 03 sections 26, 27)."""

    run_id: str
    problem_id: str
    generation_index: int
    strategy: str
    error_type: str
    error_message: str
    timestamp: str

    def __post_init__(self) -> None:
        _require_text(self.run_id, "run_id")
        _require_text(self.problem_id, "problem_id")
        _require_index(self.generation_index, "generation_index")
        _require_text(self.strategy, "strategy")
        _require_text(self.error_message, "error_message")
        _require_text(self.timestamp, "timestamp")

        if self.error_type not in ERROR_TYPES:
            raise CandidateError(
                f"error_type must be one of {', '.join(sorted(ERROR_TYPES))}, "
                f"got {self.error_type!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "problem_id": self.problem_id,
            "generation_index": self.generation_index,
            "strategy": self.strategy,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Any) -> GenerationFailure:
        if not isinstance(data, dict):
            raise CandidateError("generation failure: expected a JSON object")
        unknown = sorted(set(data) - _FAILURE_FIELDS)
        if unknown:
            raise CandidateError(
                f"generation failure: unknown field(s): {', '.join(unknown)}"
            )
        missing = sorted(_FAILURE_FIELDS - set(data))
        if missing:
            raise CandidateError(
                f"generation failure: missing required field(s): {', '.join(missing)}"
            )
        return cls(**data)
