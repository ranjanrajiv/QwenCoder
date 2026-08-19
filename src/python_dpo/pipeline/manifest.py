"""Typed schema for the experiment and stage manifests (spec 12 sections 14, 28, 65, 83).

Frozen dataclasses validating in ``__post_init__``, with explicit ``to_dict``/``from_dict``
that reject unknown and missing fields -- matching the house style of
:mod:`python_dpo.runs.models` and :mod:`python_dpo.training.models`.

Two extra fields beyond the spec's literal lists live on :class:`StageManifest`:
``cache_key`` and ``reused``. The spec asks for reuse (section 17) and a cache key built
from four inputs (section 18) but never says where the key is recorded; without it,
:mod:`python_dpo.pipeline.cache` would have nothing on disk to compare a fresh key against.
``stage_run_id`` doubles as "the underlying stage's own run id" (a training run's
``dpo_...`` id, a candidate run's ``run_...`` id) when the stage has one, or a synthesized
id for stages with no underlying repository (``problem_dataset``, ``packaging``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .stages import STAGE_NAMES
from .state import STAGE_STATES, validate_transition

EXPERIMENT_MANIFEST_VERSION = "experiment_manifest_v1"
STAGE_MANIFEST_VERSION = "stage_manifest_v1"

EXPERIMENT_STATUSES = frozenset(
    {"created", "running", "completed", "failed", "interrupted", "cancelled"}
)

# An experiment may only reach `completed` from `running`, mirroring
# `runs.models.RUN_STATUS_TRANSITIONS`.
EXPERIMENT_STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    "created": frozenset({"running", "cancelled"}),
    "running": frozenset({"running", "completed", "failed", "interrupted", "cancelled"}),
    "interrupted": frozenset({"running", "cancelled"}),
    "failed": frozenset({"running"}),
    "completed": frozenset(),
    "cancelled": frozenset(),
}


class ManifestError(Exception):
    """Raised when a manifest is malformed, or a status transition is illegal."""


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{label} must be a non-empty string")
    return value


def _require_optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{label} must be a non-empty string or null")
    return value


def _require_str_mapping(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a mapping")
    for key, val in value.items():
        if not isinstance(key, str) or not isinstance(val, str):
            raise ManifestError(f"{label} must map strings to strings")
    return dict(value)


def _require_optional_mapping(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a mapping or null")
    return dict(value)


# ------------------------------------------------------------------------------- StageError


_STAGE_ERROR_FIELDS = frozenset(
    {"stage", "error_type", "message", "timestamp", "stack_trace", "input_artifacts"}
)


@dataclass(frozen=True)
class StageError:
    """A stage-level failure record (spec section 65)."""

    stage: str
    error_type: str
    message: str
    timestamp: str
    stack_trace: str | None = None
    input_artifacts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.stage, "error.stage")
        _require_text(self.error_type, "error.error_type")
        _require_text(self.message, "error.message")
        _require_text(self.timestamp, "error.timestamp")
        if self.stack_trace is not None:
            _require_text(self.stack_trace, "error.stack_trace")
        object.__setattr__(
            self,
            "input_artifacts",
            _require_str_mapping(self.input_artifacts, "error.input_artifacts"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "error_type": self.error_type,
            "message": self.message,
            "timestamp": self.timestamp,
            "stack_trace": self.stack_trace,
            "input_artifacts": dict(self.input_artifacts),
        }

    @classmethod
    def from_dict(cls, data: Any) -> StageError:
        if not isinstance(data, dict):
            raise ManifestError("error: expected a JSON object")
        unknown = sorted(set(data) - _STAGE_ERROR_FIELDS)
        if unknown:
            raise ManifestError(f"error: unknown field(s): {', '.join(unknown)}")
        missing = sorted({"stage", "error_type", "message", "timestamp"} - set(data))
        if missing:
            raise ManifestError(f"error: missing required field(s): {', '.join(missing)}")
        return cls(
            stage=data["stage"],
            error_type=data["error_type"],
            message=data["message"],
            timestamp=data["timestamp"],
            stack_trace=data.get("stack_trace"),
            input_artifacts=data.get("input_artifacts") or {},
        )


# ----------------------------------------------------------------------------- StageManifest


_STAGE_MANIFEST_FIELDS = frozenset(
    {
        "manifest_version",
        "stage_name",
        "stage_run_id",
        "status",
        "start_time",
        "end_time",
        "input_artifacts",
        "output_artifacts",
        "configuration_hash",
        "code_version",
        "cache_key",
        "reused",
        "error",
    }
)


@dataclass(frozen=True)
class StageManifest:
    """One stage's execution record within one experiment run (spec section 14)."""

    stage_name: str
    stage_run_id: str
    status: str
    code_version: str
    manifest_version: str = STAGE_MANIFEST_VERSION
    start_time: str | None = None
    end_time: str | None = None
    input_artifacts: dict[str, str] = field(default_factory=dict)
    output_artifacts: dict[str, str] = field(default_factory=dict)
    configuration_hash: str | None = None
    cache_key: str | None = None
    reused: bool = False
    error: StageError | None = None

    def __post_init__(self) -> None:
        _require_text(self.manifest_version, "manifest_version")
        if self.stage_name not in STAGE_NAMES:
            raise ManifestError(
                f"stage_name must be one of {', '.join(STAGE_NAMES)}, got {self.stage_name!r}"
            )
        _require_text(self.stage_run_id, "stage_run_id")
        if self.status not in STAGE_STATES:
            raise ManifestError(
                f"status must be one of {', '.join(sorted(STAGE_STATES))}, got {self.status!r}"
            )
        _require_optional_text(self.start_time, "start_time")
        _require_optional_text(self.end_time, "end_time")
        object.__setattr__(
            self, "input_artifacts", _require_str_mapping(self.input_artifacts, "input_artifacts")
        )
        object.__setattr__(
            self,
            "output_artifacts",
            _require_str_mapping(self.output_artifacts, "output_artifacts"),
        )
        if self.configuration_hash is not None:
            _require_text(self.configuration_hash, "configuration_hash")
        _require_text(self.code_version, "code_version")
        if self.cache_key is not None:
            _require_text(self.cache_key, "cache_key")
        if not isinstance(self.reused, bool):
            raise ManifestError("reused must be a boolean")
        if self.error is not None and not isinstance(self.error, StageError):
            raise ManifestError("error must be a StageError or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "stage_name": self.stage_name,
            "stage_run_id": self.stage_run_id,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "input_artifacts": dict(self.input_artifacts),
            "output_artifacts": dict(self.output_artifacts),
            "configuration_hash": self.configuration_hash,
            "code_version": self.code_version,
            "cache_key": self.cache_key,
            "reused": self.reused,
            "error": self.error.to_dict() if self.error is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Any) -> StageManifest:
        if not isinstance(data, dict):
            raise ManifestError("stage manifest: expected a JSON object")
        unknown = sorted(set(data) - _STAGE_MANIFEST_FIELDS)
        if unknown:
            raise ManifestError(f"stage manifest: unknown field(s): {', '.join(unknown)}")
        missing = sorted({"stage_name", "stage_run_id", "status", "code_version"} - set(data))
        if missing:
            raise ManifestError(
                f"stage manifest: missing required field(s): {', '.join(missing)}"
            )
        error = data.get("error")
        return cls(
            manifest_version=data.get("manifest_version", STAGE_MANIFEST_VERSION),
            stage_name=data["stage_name"],
            stage_run_id=data["stage_run_id"],
            status=data["status"],
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            input_artifacts=data.get("input_artifacts") or {},
            output_artifacts=data.get("output_artifacts") or {},
            configuration_hash=data.get("configuration_hash"),
            code_version=data["code_version"],
            cache_key=data.get("cache_key"),
            reused=data.get("reused", False),
            error=StageError.from_dict(error) if error is not None else None,
        )

    def with_status(
        self,
        status: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        output_artifacts: dict[str, str] | None = None,
        cache_key: str | None = None,
        reused: bool | None = None,
        error: StageError | None = None,
    ) -> StageManifest:
        validate_transition(self.status, status)
        return StageManifest(
            manifest_version=self.manifest_version,
            stage_name=self.stage_name,
            stage_run_id=self.stage_run_id,
            status=status,
            start_time=start_time if start_time is not None else self.start_time,
            end_time=end_time if end_time is not None else self.end_time,
            input_artifacts=self.input_artifacts,
            output_artifacts=(
                output_artifacts if output_artifacts is not None else self.output_artifacts
            ),
            configuration_hash=self.configuration_hash,
            code_version=self.code_version,
            cache_key=cache_key if cache_key is not None else self.cache_key,
            reused=reused if reused is not None else self.reused,
            error=error if error is not None else self.error,
        )


# ---------------------------------------------------------------------------- StageRunSummary


_STAGE_RUN_SUMMARY_FIELDS = frozenset({"status", "stage_run_id", "reused"})


@dataclass(frozen=True)
class StageRunSummary:
    """The experiment manifest's lightweight view of one stage (spec section 28's
    ``stage_runs``); the full record lives in that stage's own ``stage_manifest.json``."""

    status: str
    stage_run_id: str | None = None
    reused: bool = False

    def __post_init__(self) -> None:
        if self.status not in STAGE_STATES:
            raise ManifestError(
                f"status must be one of {', '.join(sorted(STAGE_STATES))}, got {self.status!r}"
            )
        _require_optional_text(self.stage_run_id, "stage_run_id")
        if not isinstance(self.reused, bool):
            raise ManifestError("reused must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stage_run_id": self.stage_run_id,
            "reused": self.reused,
        }

    @classmethod
    def from_dict(cls, data: Any) -> StageRunSummary:
        if not isinstance(data, dict):
            raise ManifestError("stage_runs entry: expected a JSON object")
        unknown = sorted(set(data) - _STAGE_RUN_SUMMARY_FIELDS)
        if unknown:
            raise ManifestError(f"stage_runs entry: unknown field(s): {', '.join(unknown)}")
        if "status" not in data:
            raise ManifestError("stage_runs entry: missing required field 'status'")
        return cls(
            status=data["status"],
            stage_run_id=data.get("stage_run_id"),
            reused=data.get("reused", False),
        )


# ------------------------------------------------------------------------- ExperimentManifest


_EXPERIMENT_MANIFEST_FIELDS = frozenset(
    {
        "manifest_version",
        "experiment_run_id",
        "experiment_name",
        "status",
        "start_time",
        "end_time",
        "git_commit",
        "configuration_hash",
        "dataset_versions",
        "model_versions",
        "stage_runs",
        "final_model",
        "final_evaluation",
        "recommendation",
    }
)


@dataclass(frozen=True)
class ExperimentManifest:
    """The experiment-wide manifest (spec section 28)."""

    experiment_run_id: str
    experiment_name: str
    status: str
    configuration_hash: str
    manifest_version: str = EXPERIMENT_MANIFEST_VERSION
    start_time: str | None = None
    end_time: str | None = None
    git_commit: dict[str, Any] | None = None
    dataset_versions: dict[str, str] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)
    stage_runs: dict[str, StageRunSummary] = field(default_factory=dict)
    final_model: dict[str, Any] | None = None
    final_evaluation: dict[str, Any] | None = None
    recommendation: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.manifest_version, "manifest_version")
        _require_text(self.experiment_run_id, "experiment_run_id")
        _require_text(self.experiment_name, "experiment_name")
        if self.status not in EXPERIMENT_STATUSES:
            raise ManifestError(
                f"status must be one of {', '.join(sorted(EXPERIMENT_STATUSES))}, "
                f"got {self.status!r}"
            )
        _require_optional_text(self.start_time, "start_time")
        _require_optional_text(self.end_time, "end_time")
        object.__setattr__(
            self, "git_commit", _require_optional_mapping(self.git_commit, "git_commit")
        )
        _require_text(self.configuration_hash, "configuration_hash")
        object.__setattr__(
            self,
            "dataset_versions",
            _require_str_mapping(self.dataset_versions, "dataset_versions"),
        )
        object.__setattr__(
            self, "model_versions", _require_str_mapping(self.model_versions, "model_versions")
        )
        if not isinstance(self.stage_runs, dict) or any(
            not isinstance(v, StageRunSummary) for v in self.stage_runs.values()
        ):
            raise ManifestError("stage_runs must map stage names to StageRunSummary")
        object.__setattr__(self, "stage_runs", dict(self.stage_runs))
        object.__setattr__(
            self, "final_model", _require_optional_mapping(self.final_model, "final_model")
        )
        object.__setattr__(
            self,
            "final_evaluation",
            _require_optional_mapping(self.final_evaluation, "final_evaluation"),
        )
        _require_optional_text(self.recommendation, "recommendation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "experiment_run_id": self.experiment_run_id,
            "experiment_name": self.experiment_name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "git_commit": self.git_commit,
            "configuration_hash": self.configuration_hash,
            "dataset_versions": dict(self.dataset_versions),
            "model_versions": dict(self.model_versions),
            "stage_runs": {k: v.to_dict() for k, v in self.stage_runs.items()},
            "final_model": self.final_model,
            "final_evaluation": self.final_evaluation,
            "recommendation": self.recommendation,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ExperimentManifest:
        if not isinstance(data, dict):
            raise ManifestError("experiment manifest: expected a JSON object")
        unknown = sorted(set(data) - _EXPERIMENT_MANIFEST_FIELDS)
        if unknown:
            raise ManifestError(f"experiment manifest: unknown field(s): {', '.join(unknown)}")
        missing = sorted(
            {"experiment_run_id", "experiment_name", "status", "configuration_hash"} - set(data)
        )
        if missing:
            raise ManifestError(
                f"experiment manifest: missing required field(s): {', '.join(missing)}"
            )
        stage_runs_raw = data.get("stage_runs") or {}
        if not isinstance(stage_runs_raw, dict):
            raise ManifestError("experiment manifest: stage_runs must be a mapping")
        return cls(
            manifest_version=data.get("manifest_version", EXPERIMENT_MANIFEST_VERSION),
            experiment_run_id=data["experiment_run_id"],
            experiment_name=data["experiment_name"],
            status=data["status"],
            configuration_hash=data["configuration_hash"],
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            git_commit=data.get("git_commit"),
            dataset_versions=data.get("dataset_versions") or {},
            model_versions=data.get("model_versions") or {},
            stage_runs={
                name: StageRunSummary.from_dict(value)
                for name, value in stage_runs_raw.items()
            },
            final_model=data.get("final_model"),
            final_evaluation=data.get("final_evaluation"),
            recommendation=data.get("recommendation"),
        )

    def with_status(
        self,
        status: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> ExperimentManifest:
        if status not in EXPERIMENT_STATUSES:
            raise ManifestError(f"unknown status {status!r}")
        if status not in EXPERIMENT_STATUS_TRANSITIONS[self.status]:
            raise ManifestError(
                f"illegal experiment status transition: {self.status!r} -> {status!r}"
            )
        changes: dict[str, Any] = {"status": status}
        if start_time is not None:
            changes["start_time"] = start_time
        if end_time is not None:
            changes["end_time"] = end_time
        return dataclasses.replace(self, **changes)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = [
    "EXPERIMENT_MANIFEST_VERSION",
    "EXPERIMENT_STATUSES",
    "EXPERIMENT_STATUS_TRANSITIONS",
    "STAGE_MANIFEST_VERSION",
    "ExperimentManifest",
    "ManifestError",
    "StageError",
    "StageManifest",
    "StageRunSummary",
    "utc_now_iso",
]
