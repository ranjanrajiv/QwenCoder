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

Schema versioning (spec 04 sections 46, 47): ``schema_version`` is stamped on every record.
``"1.0"`` is the Stage 3 shape (no hashes, no ``attempt``); ``"2.0"`` adds
``code_sha256``/``prompt_sha256``/``raw_output_sha256`` and ``attempt``. A record missing
``schema_version`` entirely is read as ``"1.0"`` — old data is never silently reinterpreted
as if it had provenance it doesn't.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .hashing import sha256_text

# Written to failures.jsonl. Closed set so failures can be counted and compared across
# runs instead of grouped by free text (spec 03 section 27).
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

# Infrastructure failures may be retried; candidate failures are terminal for that
# generation index (spec 04 section 28).
INFRASTRUCTURE_ERROR_TYPES = frozenset({"model_load", "tokenizer", "inference", "timeout"})
CANDIDATE_ERROR_TYPES = frozenset({"empty_output", "code_extraction"})
assert INFRASTRUCTURE_ERROR_TYPES | CANDIDATE_ERROR_TYPES == ERROR_TYPES

EXTRACTION_FORMATS = frozenset({"python_fence", "generic_fence", "plain", "unknown"})

CANDIDATE_SCHEMA_VERSIONS = frozenset({"1.0", "2.0"})
CANDIDATE_SCHEMA_VERSION = "2.0"
LEGACY_CANDIDATE_SCHEMA_VERSION = "1.0"

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
        "schema_version",
        "code_sha256",
        "prompt_sha256",
        "raw_output_sha256",
        "attempt",
    }
)

# Fields introduced in schema 2.0. A legacy 1.0 record must not carry these — reading one
# in must not silently invent provenance it never recorded (spec 04 section 46).
_SCHEMA_2_ONLY_FIELDS = frozenset(
    {"code_sha256", "prompt_sha256", "raw_output_sha256", "attempt"}
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
        "schema_version",
        "attempt",
        "prompt_sha256",
        "traceback",
    }
)
_FAILURE_SCHEMA_2_ONLY_FIELDS = frozenset({"attempt", "prompt_sha256"})


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

    ``code_sha256``/``prompt_sha256``/``raw_output_sha256`` (spec 04 sections 16-18) are
    verified, not merely stored: construction recomputes each from the corresponding text
    and rejects a mismatch, so a tampered record cannot be built or loaded. Use
    :meth:`Candidate.create` rather than the constructor directly when producing a new
    (schema 2.0) record — it computes the three hashes for you.
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
    schema_version: str = CANDIDATE_SCHEMA_VERSION
    code_sha256: str | None = None
    prompt_sha256: str | None = None
    raw_output_sha256: str | None = None
    attempt: int = 1

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

        if self.schema_version not in CANDIDATE_SCHEMA_VERSIONS:
            raise CandidateError(
                "schema_version must be one of "
                f"{', '.join(sorted(CANDIDATE_SCHEMA_VERSIONS))}, got {self.schema_version!r}"
            )
        _require_index(self.attempt, "attempt")

        if self.schema_version == LEGACY_CANDIDATE_SCHEMA_VERSION:
            for name in ("code_sha256", "prompt_sha256", "raw_output_sha256"):
                if getattr(self, name) is not None:
                    raise CandidateError(
                        f"{name} must be null on a schema_version {LEGACY_CANDIDATE_SCHEMA_VERSION} "
                        "record; hashes were introduced in schema 2.0"
                    )
        else:
            expected = {
                "code_sha256": sha256_text(self.code),
                "prompt_sha256": sha256_text(self.prompt),
                "raw_output_sha256": sha256_text(self.raw_output),
            }
            for name, expected_hash in expected.items():
                actual = getattr(self, name)
                if actual is None:
                    raise CandidateError(f"{name} is required on a schema_version 2.0 record")
                if actual != expected_hash:
                    raise CandidateError(
                        f"{name} does not match the stored {name.removesuffix('_sha256')}: "
                        f"expected {expected_hash}, got {actual}"
                    )

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        problem_id: str,
        run_id: str,
        generation_index: int,
        strategy: str,
        model: str,
        provider: str,
        prompt_version: str,
        prompt: str,
        raw_output: str,
        code: str,
        extraction_format: str,
        syntax_valid: bool,
        function_name_valid: bool,
        generation_config: dict[str, Any],
        created_at: str,
        model_revision: str | None = None,
        syntax_error: str | None = None,
        duplicate_of: str | None = None,
        attempt: int = 1,
    ) -> Candidate:
        """Build a schema 2.0 record, computing the three hashes from the text fields."""
        return cls(
            candidate_id=candidate_id,
            problem_id=problem_id,
            run_id=run_id,
            generation_index=generation_index,
            strategy=strategy,
            model=model,
            provider=provider,
            prompt_version=prompt_version,
            prompt=prompt,
            raw_output=raw_output,
            code=code,
            extraction_format=extraction_format,
            syntax_valid=syntax_valid,
            function_name_valid=function_name_valid,
            generation_config=generation_config,
            created_at=created_at,
            model_revision=model_revision,
            syntax_error=syntax_error,
            duplicate_of=duplicate_of,
            schema_version=CANDIDATE_SCHEMA_VERSION,
            code_sha256=sha256_text(code),
            prompt_sha256=sha256_text(prompt),
            raw_output_sha256=sha256_text(raw_output),
            attempt=attempt,
        )

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
            "schema_version": self.schema_version,
            "code_sha256": self.code_sha256,
            "prompt_sha256": self.prompt_sha256,
            "raw_output_sha256": self.raw_output_sha256,
            "attempt": self.attempt,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Candidate:
        if not isinstance(data, dict):
            raise CandidateError("candidate: expected a JSON object")
        unknown = sorted(set(data) - _CANDIDATE_FIELDS)
        if unknown:
            raise CandidateError(f"candidate: unknown field(s): {', '.join(unknown)}")

        schema_version = data.get("schema_version", LEGACY_CANDIDATE_SCHEMA_VERSION)
        optional = {"model_revision", "syntax_error", "duplicate_of", "schema_version"}
        if schema_version == LEGACY_CANDIDATE_SCHEMA_VERSION:
            optional |= _SCHEMA_2_ONLY_FIELDS
        required = _CANDIDATE_FIELDS - optional
        missing = sorted(required - set(data))
        if missing:
            raise CandidateError(f"candidate: missing required field(s): {', '.join(missing)}")

        kwargs = dict(data)
        kwargs["schema_version"] = schema_version
        kwargs.setdefault("attempt", 1)
        return cls(**kwargs)


@dataclass(frozen=True)
class GenerationFailure:
    """A generation that produced no candidate (spec 03 sections 26, 27).

    ``prompt_sha256`` links a failure back to its exact prompt in ``prompts.jsonl`` (spec
    04 section 31) — required on schema 2.0 records, since the generator always builds and
    persists the prompt before attempting inference. ``traceback`` exists for spec 04
    section 27 but is deliberately left unpopulated by default: a Python traceback embeds
    absolute filesystem paths, including the home directory CLAUDE.md's environment rule
    (spec 04 section 33) forbids recording.
    """

    run_id: str
    problem_id: str
    generation_index: int
    strategy: str
    error_type: str
    error_message: str
    timestamp: str
    schema_version: str = CANDIDATE_SCHEMA_VERSION
    attempt: int = 1
    prompt_sha256: str | None = None
    traceback: str | None = None

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

        if self.schema_version not in CANDIDATE_SCHEMA_VERSIONS:
            raise CandidateError(
                "schema_version must be one of "
                f"{', '.join(sorted(CANDIDATE_SCHEMA_VERSIONS))}, got {self.schema_version!r}"
            )
        _require_index(self.attempt, "attempt")
        _require_optional_text(self.prompt_sha256, "prompt_sha256")
        _require_optional_text(self.traceback, "traceback")

        if self.schema_version == LEGACY_CANDIDATE_SCHEMA_VERSION and self.prompt_sha256 is not None:
            raise CandidateError(
                f"prompt_sha256 must be null on a schema_version {LEGACY_CANDIDATE_SCHEMA_VERSION} "
                "record; hashes were introduced in schema 2.0"
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
            "schema_version": self.schema_version,
            "attempt": self.attempt,
            "prompt_sha256": self.prompt_sha256,
            "traceback": self.traceback,
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

        schema_version = data.get("schema_version", LEGACY_CANDIDATE_SCHEMA_VERSION)
        optional = {"schema_version", "traceback"}
        if schema_version == LEGACY_CANDIDATE_SCHEMA_VERSION:
            optional |= _FAILURE_SCHEMA_2_ONLY_FIELDS
        required = _FAILURE_FIELDS - optional
        missing = sorted(required - set(data))
        if missing:
            raise CandidateError(
                f"generation failure: missing required field(s): {', '.join(missing)}"
            )

        kwargs = dict(data)
        kwargs["schema_version"] = schema_version
        kwargs.setdefault("attempt", 1)
        kwargs.setdefault("prompt_sha256", None)
        return cls(**kwargs)
