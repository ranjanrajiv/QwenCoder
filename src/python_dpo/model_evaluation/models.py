"""Typed schema for Stage 10 generation records, evaluation records, the run manifest,
and the metrics summary.

Frozen dataclasses validating in ``__post_init__``, matching the house style of
``python_dpo.candidates.models`` and ``python_dpo.evaluation.models``. Two record types
mirror the Stage 3/6 split deliberately:

* :class:`GenerationRecord` (spec section 81) — one attempted generation for one
  ``(problem_id, model_variant, sample_index)``. ``raw_response`` and ``extracted_code``
  are always distinct fields (spec section 36): a failed extraction keeps the raw text
  with ``extracted_code = None``.
* :class:`EvaluationRecord` (spec section 82) — one sandboxed execution outcome for a
  generation whose code *was* extracted. ``correct`` is a derived property — spec section
  40's exact-correctness rule lives in exactly one place.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..candidates.models import EXTRACTION_FORMATS
from ..evaluation.models import EVALUATION_RESULT_STATUSES

MODEL_VARIANTS = frozenset({"base", "dpo"})
GENERATION_STATUSES = frozenset({"generated", "generation_error"})

# Spec section 124's failure taxonomy, as it applies to an already-sandboxed candidate.
# "generation_error" is deliberately excluded here -- that status lives on GenerationRecord,
# never on an EvaluationRecord, since a generation failure never reaches the sandbox.
EVALUATION_ERROR_TYPES = frozenset(
    {"syntax_error", "import_error", "runtime_error", "assertion_failure", "timeout",
     "infrastructure_error"}
)

MANIFEST_VERSION = "1.0"

# Mirrors python_dpo.evaluation.models's run lifecycle exactly -- Stage 10 runs go through
# the same create -> running -> completed/failed/interrupted shape as every prior stage.
RUN_STATUSES = frozenset(
    {"created", "running", "completed", "failed", "interrupted", "cancelled"}
)
RUN_STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    "created": frozenset({"running", "cancelled"}),
    "running": frozenset({"running", "completed", "failed", "interrupted", "cancelled"}),
    "interrupted": frozenset({"running", "cancelled"}),
    "failed": frozenset({"running"}),
    "completed": frozenset(),
    "cancelled": frozenset(),
}


class ModelEvaluationModelError(Exception):
    """Raised when a Stage 10 record or manifest fails schema validation."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelEvaluationModelError(f"{label} must be a non-empty string")
    return value


def _require_optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, label)


def _require_nonneg_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelEvaluationModelError(f"{label} must be an integer of 0 or greater")
    return value


def _require_flag(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ModelEvaluationModelError(f"{label} must be true or false")
    return value


def _require_variant(value: Any, label: str = "model_variant") -> str:
    if value not in MODEL_VARIANTS:
        raise ModelEvaluationModelError(
            f"{label} must be one of {', '.join(sorted(MODEL_VARIANTS))}, got {value!r}"
        )
    return value


# ------------------------------------------------------------------------ GenerationRecord

_GENERATION_RECORD_FIELDS = frozenset(
    {
        "evaluation_run_id",
        "problem_id",
        "model_variant",
        "sample_index",
        "seed",
        "prompt_sha256",
        "raw_response",
        "extracted_code",
        "extraction_format",
        "syntax_valid",
        "generation_time_ms",
        "generated_tokens",
        "status",
        "error",
    }
)


@dataclass(frozen=True)
class GenerationRecord:
    """One attempted generation (spec section 81).

    A code-extraction failure (spec section 35) is recorded here with
    ``status="generation_error"``, ``extracted_code=None``, and the raw response retained
    for debugging -- it never becomes a :class:`~python_dpo.candidates.models.Candidate`,
    since ``Candidate.create`` requires non-empty code.
    """

    evaluation_run_id: str
    problem_id: str
    model_variant: str
    sample_index: int
    seed: int
    prompt_sha256: str
    raw_response: str
    extraction_format: str
    generation_time_ms: int
    generated_tokens: int
    status: str
    extracted_code: str | None = None
    syntax_valid: bool | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.evaluation_run_id, "evaluation_run_id")
        _require_text(self.problem_id, "problem_id")
        _require_variant(self.model_variant)
        _require_nonneg_int(self.sample_index, "sample_index")
        _require_nonneg_int(self.seed, "seed")
        _require_text(self.prompt_sha256, "prompt_sha256")
        if not isinstance(self.raw_response, str):
            raise ModelEvaluationModelError("raw_response must be a string")
        if self.status not in GENERATION_STATUSES:
            raise ModelEvaluationModelError(
                f"status must be one of {', '.join(sorted(GENERATION_STATUSES))}, "
                f"got {self.status!r}"
            )
        if self.extraction_format not in EXTRACTION_FORMATS:
            raise ModelEvaluationModelError(
                f"extraction_format must be one of {', '.join(sorted(EXTRACTION_FORMATS))}, "
                f"got {self.extraction_format!r}"
            )
        _require_nonneg_int(self.generation_time_ms, "generation_time_ms")
        _require_nonneg_int(self.generated_tokens, "generated_tokens")
        _require_optional_text(self.error, "error")

        if self.status == "generated":
            _require_text(self.extracted_code or "", "extracted_code")
            if not isinstance(self.syntax_valid, bool):
                raise ModelEvaluationModelError(
                    "syntax_valid must be true or false when status is 'generated'"
                )
            if self.extraction_format == "unknown":
                raise ModelEvaluationModelError(
                    "extraction_format must not be 'unknown' when status is 'generated'"
                )
        else:  # generation_error
            if self.extracted_code is not None:
                raise ModelEvaluationModelError(
                    "extracted_code must be null when status is 'generation_error'"
                )
            if self.syntax_valid is not None:
                raise ModelEvaluationModelError(
                    "syntax_valid must be null when status is 'generation_error'"
                )
            _require_text(self.error or "", "error")

    @property
    def candidate_id(self) -> str:
        """``p001_c001`` per variant (spec section 30's per-(problem, sample) identity)."""
        return f"{self.problem_id}_c{self.sample_index + 1:03d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_run_id": self.evaluation_run_id,
            "problem_id": self.problem_id,
            "model_variant": self.model_variant,
            "sample_index": self.sample_index,
            "seed": self.seed,
            "prompt_sha256": self.prompt_sha256,
            "raw_response": self.raw_response,
            "extracted_code": self.extracted_code,
            "extraction_format": self.extraction_format,
            "syntax_valid": self.syntax_valid,
            "generation_time_ms": self.generation_time_ms,
            "generated_tokens": self.generated_tokens,
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Any) -> GenerationRecord:
        if not isinstance(data, dict):
            raise ModelEvaluationModelError("generation record: expected a JSON object")
        unknown = sorted(set(data) - _GENERATION_RECORD_FIELDS)
        if unknown:
            raise ModelEvaluationModelError(
                f"generation record: unknown field(s): {', '.join(unknown)}"
            )
        required = _GENERATION_RECORD_FIELDS - {"extracted_code", "syntax_valid", "error"}
        missing = sorted(required - set(data))
        if missing:
            raise ModelEvaluationModelError(
                f"generation record: missing required field(s): {', '.join(missing)}"
            )
        kwargs = dict(data)
        kwargs.setdefault("extracted_code", None)
        kwargs.setdefault("syntax_valid", None)
        kwargs.setdefault("error", None)
        return cls(**kwargs)


# ------------------------------------------------------------------------- EvaluationRecord

_EVALUATION_RECORD_FIELDS = frozenset(
    {
        "evaluation_run_id",
        "problem_id",
        "model_variant",
        "sample_index",
        "tests_total",
        "tests_passed",
        "tests_failed",
        "tests_error",
        "tests_skipped",
        "timeout",
        "status",
        "duration_ms",
        "error_type",
    }
)


@dataclass(frozen=True)
class EvaluationRecord:
    """One sandboxed execution outcome for a generated-and-extracted candidate (spec
    section 82). ``correct`` is spec section 40's exact-correctness rule, derived rather
    than trusted from the caller.
    """

    evaluation_run_id: str
    problem_id: str
    model_variant: str
    sample_index: int
    tests_total: int
    tests_passed: int
    tests_failed: int
    tests_error: int
    tests_skipped: int
    timeout: bool
    status: str
    duration_ms: int
    error_type: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.evaluation_run_id, "evaluation_run_id")
        _require_text(self.problem_id, "problem_id")
        _require_variant(self.model_variant)
        _require_nonneg_int(self.sample_index, "sample_index")
        for name in ("tests_total", "tests_passed", "tests_failed", "tests_error", "tests_skipped"):
            _require_nonneg_int(getattr(self, name), name)
        _require_flag(self.timeout, "timeout")
        _require_nonneg_int(self.duration_ms, "duration_ms")

        if self.status not in EVALUATION_RESULT_STATUSES:
            raise ModelEvaluationModelError(
                f"status must be one of {', '.join(sorted(EVALUATION_RESULT_STATUSES))}, "
                f"got {self.status!r}"
            )
        counted = self.tests_passed + self.tests_failed + self.tests_error + self.tests_skipped
        if counted != self.tests_total:
            raise ModelEvaluationModelError(
                "tests_passed + tests_failed + tests_error + tests_skipped "
                f"({counted}) must equal tests_total ({self.tests_total})"
            )
        if self.timeout != (self.status == "timeout"):
            raise ModelEvaluationModelError(
                f"timeout must be {self.status == 'timeout'} when status is {self.status!r}"
            )

        if self.error_type is not None:
            if self.error_type not in EVALUATION_ERROR_TYPES:
                raise ModelEvaluationModelError(
                    f"error_type must be one of {', '.join(sorted(EVALUATION_ERROR_TYPES))} "
                    f"or null, got {self.error_type!r}"
                )
        if self.status == "passed" and self.error_type is not None:
            raise ModelEvaluationModelError("error_type must be null when status is 'passed'")
        if self.status != "passed" and self.error_type is None:
            raise ModelEvaluationModelError(
                f"error_type is required when status is {self.status!r}"
            )

    @property
    def candidate_id(self) -> str:
        return f"{self.problem_id}_c{self.sample_index + 1:03d}"

    @property
    def correct(self) -> bool:
        """Spec section 40: correct only if every test passed. 9/10 is incorrect."""
        return self.status == "passed" and self.tests_total > 0 and self.tests_passed == self.tests_total

    @property
    def is_infrastructure_error(self) -> bool:
        """Spec section 120: excluded from candidate-level correctness statistics."""
        return self.status == "infrastructure_error"

    @property
    def pass_rate(self) -> float:
        return self.tests_passed / self.tests_total if self.tests_total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_run_id": self.evaluation_run_id,
            "problem_id": self.problem_id,
            "model_variant": self.model_variant,
            "sample_index": self.sample_index,
            "tests_total": self.tests_total,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_error": self.tests_error,
            "tests_skipped": self.tests_skipped,
            "timeout": self.timeout,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "error_type": self.error_type,
        }

    @classmethod
    def from_dict(cls, data: Any) -> EvaluationRecord:
        if not isinstance(data, dict):
            raise ModelEvaluationModelError("evaluation record: expected a JSON object")
        unknown = sorted(set(data) - _EVALUATION_RECORD_FIELDS)
        if unknown:
            raise ModelEvaluationModelError(
                f"evaluation record: unknown field(s): {', '.join(unknown)}"
            )
        required = _EVALUATION_RECORD_FIELDS - {"error_type"}
        missing = sorted(required - set(data))
        if missing:
            raise ModelEvaluationModelError(
                f"evaluation record: missing required field(s): {', '.join(missing)}"
            )
        kwargs = dict(data)
        kwargs.setdefault("error_type", None)
        return cls(**kwargs)


# -------------------------------------------------------------------- ModelEvaluationManifest

_MANIFEST_FIELDS = frozenset(
    {
        "manifest_version",
        "evaluation_run_id",
        "benchmark_version",
        "benchmark_hash",
        "base_model_name",
        "base_model_revision",
        "adapter_path",
        "training_run_id",
        "models_requested",
        "generation_config",
        "quantization",
        "statistics_config",
        "seeds",
        "hardware",
        "environment",
        "status",
        "created_at",
        "started_at",
        "completed_at",
        "error",
    }
)
_MANIFEST_ERROR_FIELDS = frozenset({"error_type", "error_message", "timestamp"})


@dataclass(frozen=True)
class ModelEvaluationManifest:
    """The historical, immutable configuration snapshot for one Stage 10 run (spec
    sections 32, 78, 79, 80). Mirrors ``python_dpo.training.models.TrainingManifest``.
    """

    evaluation_run_id: str
    benchmark_version: str
    benchmark_hash: str
    base_model_name: str
    adapter_path: str
    training_run_id: str
    models_requested: tuple[str, ...]
    generation_config: dict[str, Any]
    quantization: dict[str, Any]
    statistics_config: dict[str, Any]
    seeds: dict[str, Any]
    hardware: dict[str, Any]
    environment: dict[str, Any]
    created_at: str
    manifest_version: str = MANIFEST_VERSION
    base_model_revision: str | None = None
    status: str = "created"
    started_at: str | None = None
    completed_at: str | None = None
    error: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_text(self.manifest_version, "manifest_version")
        _require_text(self.evaluation_run_id, "evaluation_run_id")
        _require_text(self.benchmark_version, "benchmark_version")
        _require_text(self.benchmark_hash, "benchmark_hash")
        _require_text(self.base_model_name, "base_model_name")
        _require_optional_text(self.base_model_revision, "base_model_revision")
        _require_text(self.adapter_path, "adapter_path")
        _require_text(self.training_run_id, "training_run_id")

        if isinstance(self.models_requested, (str, bytes)) or not isinstance(
            self.models_requested, (list, tuple)
        ):
            raise ModelEvaluationModelError("models_requested must be a sequence")
        object.__setattr__(self, "models_requested", tuple(self.models_requested))
        if not self.models_requested:
            raise ModelEvaluationModelError("models_requested must not be empty")
        for variant in self.models_requested:
            _require_variant(variant, "models_requested entry")

        for name in ("generation_config", "quantization", "statistics_config", "seeds", "hardware", "environment"):
            if not isinstance(getattr(self, name), dict):
                raise ModelEvaluationModelError(f"{name} must be a mapping")

        if self.status not in RUN_STATUSES:
            raise ModelEvaluationModelError(
                f"status must be one of {', '.join(sorted(RUN_STATUSES))}, got {self.status!r}"
            )
        _require_text(self.created_at, "created_at")
        _require_optional_text(self.started_at, "started_at")
        _require_optional_text(self.completed_at, "completed_at")

        if self.error is not None:
            if not isinstance(self.error, dict):
                raise ModelEvaluationModelError("error must be a mapping or null")
            unknown = sorted(set(self.error) - _MANIFEST_ERROR_FIELDS)
            if unknown:
                raise ModelEvaluationModelError(f"error: unknown field(s): {', '.join(unknown)}")
            missing = sorted(_MANIFEST_ERROR_FIELDS - set(self.error))
            if missing:
                raise ModelEvaluationModelError(
                    f"error: missing required field(s): {', '.join(missing)}"
                )

    def with_status(
        self,
        status: str,
        *,
        started_at: str | None = None,
        completed_at: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> ModelEvaluationManifest:
        allowed = RUN_STATUS_TRANSITIONS.get(self.status, frozenset())
        if status != self.status and status not in allowed:
            raise ModelEvaluationModelError(
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
            "benchmark_version": self.benchmark_version,
            "benchmark_hash": self.benchmark_hash,
            "base_model_name": self.base_model_name,
            "base_model_revision": self.base_model_revision,
            "adapter_path": self.adapter_path,
            "training_run_id": self.training_run_id,
            "models_requested": list(self.models_requested),
            "generation_config": self.generation_config,
            "quantization": self.quantization,
            "statistics_config": self.statistics_config,
            "seeds": self.seeds,
            "hardware": self.hardware,
            "environment": self.environment,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ModelEvaluationManifest:
        if not isinstance(data, dict):
            raise ModelEvaluationModelError("evaluation manifest: expected a JSON object")
        unknown = sorted(set(data) - _MANIFEST_FIELDS)
        if unknown:
            raise ModelEvaluationModelError(
                f"evaluation manifest: unknown field(s): {', '.join(unknown)}"
            )
        required = _MANIFEST_FIELDS - {
            "manifest_version", "base_model_revision", "status", "started_at",
            "completed_at", "error",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ModelEvaluationModelError(
                f"evaluation manifest: missing required field(s): {', '.join(missing)}"
            )
        kwargs = dict(data)
        kwargs.setdefault("manifest_version", MANIFEST_VERSION)
        kwargs.setdefault("base_model_revision", None)
        kwargs.setdefault("status", "created")
        kwargs.setdefault("started_at", None)
        kwargs.setdefault("completed_at", None)
        kwargs.setdefault("error", None)
        kwargs["models_requested"] = tuple(kwargs["models_requested"])
        return cls(**kwargs)


# -------------------------------------------------------------------------- MetricsSummary

_METRICS_SUMMARY_FIELDS = frozenset(
    {
        "evaluation_run_id",
        "pass_at_k",
        "test_pass_rate",
        "solve_rate",
        "syntax_success_rate",
        "execution_success_rate",
        "timeout_rate",
        "generation_failure_rate",
        "test_failure_distribution",
        "computed_at",
    }
)


@dataclass(frozen=True)
class MetricsSummary:
    """Aggregate metrics for one or both model variants (spec section 83).

    ``pass_at_k`` and the rate fields are keyed by variant (``"base"``/``"dpo"``) then, for
    ``pass_at_k``, by the sample count ``k`` -- this stays correct for whichever
    ``statistics.pass_at_k`` values the config requests rather than hard-coding 1/5/10.
    :meth:`to_dict` additionally flattens the configured k's into the exact
    ``base_pass_at_1`` / ``dpo_pass_at_1`` / ... key names spec section 83 asks for.
    """

    evaluation_run_id: str
    pass_at_k: dict[str, dict[str, float]]
    test_pass_rate: dict[str, float]
    solve_rate: dict[str, float]
    syntax_success_rate: dict[str, float]
    execution_success_rate: dict[str, float]
    timeout_rate: dict[str, float]
    generation_failure_rate: dict[str, float]
    test_failure_distribution: dict[str, dict[str, int]]
    computed_at: str

    def __post_init__(self) -> None:
        _require_text(self.evaluation_run_id, "evaluation_run_id")
        for name in (
            "pass_at_k", "test_pass_rate", "solve_rate", "syntax_success_rate",
            "execution_success_rate", "timeout_rate", "generation_failure_rate",
            "test_failure_distribution",
        ):
            if not isinstance(getattr(self, name), dict):
                raise ModelEvaluationModelError(f"{name} must be a mapping")
        _require_text(self.computed_at, "computed_at")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "evaluation_run_id": self.evaluation_run_id,
            "pass_at_k": self.pass_at_k,
            "test_pass_rate": self.test_pass_rate,
            "solve_rate": self.solve_rate,
            "syntax_success_rate": self.syntax_success_rate,
            "execution_success_rate": self.execution_success_rate,
            "timeout_rate": self.timeout_rate,
            "generation_failure_rate": self.generation_failure_rate,
            "test_failure_distribution": self.test_failure_distribution,
            "computed_at": self.computed_at,
        }
        for variant, by_k in self.pass_at_k.items():
            for k, value in by_k.items():
                payload[f"{variant}_pass_at_{k}"] = value
        for variant, value in self.test_pass_rate.items():
            payload[f"{variant}_test_pass_rate"] = value
        for variant, value in self.timeout_rate.items():
            payload[f"{variant}_timeout_rate"] = value
        for variant, value in self.syntax_success_rate.items():
            payload[f"{variant}_syntax_success"] = value
        return payload

    @classmethod
    def from_dict(cls, data: Any) -> MetricsSummary:
        if not isinstance(data, dict):
            raise ModelEvaluationModelError("metrics summary: expected a JSON object")
        missing = sorted(_METRICS_SUMMARY_FIELDS - set(data))
        if missing:
            raise ModelEvaluationModelError(
                f"metrics summary: missing required field(s): {', '.join(missing)}"
            )
        return cls(**{key: data[key] for key in _METRICS_SUMMARY_FIELDS})


__all__ = [
    "EVALUATION_ERROR_TYPES",
    "GENERATION_STATUSES",
    "MANIFEST_VERSION",
    "MODEL_VARIANTS",
    "RUN_STATUSES",
    "RUN_STATUS_TRANSITIONS",
    "EvaluationRecord",
    "GenerationRecord",
    "MetricsSummary",
    "ModelEvaluationManifest",
    "ModelEvaluationModelError",
    "utc_now_iso",
]
