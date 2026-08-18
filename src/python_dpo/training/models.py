"""Typed schema for training runs, dataset provenance, and final reports.

Frozen dataclasses validating in ``__post_init__``, matching the house style of
``python_dpo.preferences.models``. Nothing here imports torch — these are records about a
training run, not the run itself, so a manifest can be read and validated on a machine
with no GPU.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

TRAINING_RUN_STATUSES = frozenset(
    {"created", "running", "completed", "failed", "interrupted", "cancelled"}
)
TRAINING_RUN_STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    "created": frozenset({"running", "cancelled"}),
    "running": frozenset({"running", "completed", "failed", "interrupted", "cancelled"}),
    "interrupted": frozenset({"running", "cancelled"}),
    "failed": frozenset({"running"}),
    "completed": frozenset(),
    "cancelled": frozenset(),
}

MANIFEST_VERSION = "1.0"
DATASET_MANIFEST_VERSION = "1.0"
FINAL_REPORT_VERSION = "1.0"

SPLITS = ("train", "validation", "test")


class TrainingModelError(Exception):
    """Raised when a training record or manifest fails schema validation."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingModelError(f"{label} must be a non-empty string")
    return value


def _require_optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, label)


def _require_nonneg_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingModelError(f"{label} must be an integer of 0 or greater")
    return value


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingModelError(f"{label} must be a mapping")
    return value


# ------------------------------------------------------------------------- DatasetManifest

_DATASET_MANIFEST_FIELDS = frozenset(
    {
        "dataset_manifest_version",
        "preference_run_id",
        "preference_version",
        "selection_policy",
        "selection_policy_version",
        "dataset_schema_version",
        "ranking_run_id",
        "evaluation_run_id",
        "candidate_run_id",
        "split_hashes",
        "split_counts",
        "split_problem_ids",
        "statistics",
        "created_at",
    }
)


@dataclass(frozen=True)
class DatasetManifest:
    """Exactly which preference data a training run consumed (spec sections 25, 26).

    ``split_hashes`` covers **all three** splits including test — not because test is
    used (spec sections 21, 22 forbid it) but because reproducing a run later means
    proving the same test split was held out, which requires its hash.
    """

    preference_run_id: str
    preference_version: str
    selection_policy: str
    selection_policy_version: str
    dataset_schema_version: str
    ranking_run_id: str
    evaluation_run_id: str
    candidate_run_id: str
    split_hashes: dict[str, str]
    split_counts: dict[str, int]
    split_problem_ids: dict[str, list[str]]
    statistics: dict[str, Any] = field(default_factory=dict)
    dataset_manifest_version: str = DATASET_MANIFEST_VERSION
    created_at: str = ""

    def __post_init__(self) -> None:
        for name in (
            "preference_run_id",
            "preference_version",
            "selection_policy",
            "selection_policy_version",
            "dataset_schema_version",
            "ranking_run_id",
            "evaluation_run_id",
            "candidate_run_id",
            "dataset_manifest_version",
        ):
            _require_text(getattr(self, name), name)

        for label, mapping in (
            ("split_hashes", self.split_hashes),
            ("split_counts", self.split_counts),
            ("split_problem_ids", self.split_problem_ids),
        ):
            _require_mapping(mapping, label)
            missing = sorted(set(SPLITS) - set(mapping))
            if missing:
                raise TrainingModelError(f"{label}: missing split(s): {', '.join(missing)}")

        for split, count in self.split_counts.items():
            _require_nonneg_int(count, f"split_counts[{split!r}]")

        # Spec section 102: the splits must be problem-disjoint. Validated here as well as
        # at load time so a hand-edited manifest cannot claim otherwise.
        seen: dict[str, str] = {}
        for split, ids in self.split_problem_ids.items():
            if not isinstance(ids, list):
                raise TrainingModelError(f"split_problem_ids[{split!r}] must be a list")
            for problem_id in ids:
                if problem_id in seen:
                    raise TrainingModelError(
                        f"problem {problem_id!r} appears in both {seen[problem_id]!r} and "
                        f"{split!r}; splits must be problem-disjoint"
                    )
                seen[problem_id] = split

        _require_mapping(self.statistics, "statistics")

        if not self.created_at:
            object.__setattr__(self, "created_at", utc_now_iso())

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_manifest_version": self.dataset_manifest_version,
            "preference_run_id": self.preference_run_id,
            "preference_version": self.preference_version,
            "selection_policy": self.selection_policy,
            "selection_policy_version": self.selection_policy_version,
            "dataset_schema_version": self.dataset_schema_version,
            "ranking_run_id": self.ranking_run_id,
            "evaluation_run_id": self.evaluation_run_id,
            "candidate_run_id": self.candidate_run_id,
            "split_hashes": self.split_hashes,
            "split_counts": self.split_counts,
            "split_problem_ids": self.split_problem_ids,
            "statistics": self.statistics,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> DatasetManifest:
        if not isinstance(data, dict):
            raise TrainingModelError("dataset manifest: expected a JSON object")
        unknown = sorted(set(data) - _DATASET_MANIFEST_FIELDS)
        if unknown:
            raise TrainingModelError(
                f"dataset manifest: unknown field(s): {', '.join(unknown)}"
            )
        kwargs = dict(data)
        kwargs.setdefault("dataset_manifest_version", DATASET_MANIFEST_VERSION)
        kwargs.setdefault("statistics", {})
        return cls(**kwargs)


# ------------------------------------------------------------------------ TrainingManifest

_MANIFEST_FIELDS = frozenset(
    {
        "manifest_version",
        "training_run_id",
        "experiment_name",
        "status",
        "created_at",
        "started_at",
        "completed_at",
        "model_name",
        "model_revision",
        "tokenizer_revision",
        "preference_run_id",
        "ranking_run_id",
        "evaluation_run_id",
        "candidate_run_id",
        "dataset_hashes",
        "hardware",
        "environment",
        "configuration",
        "seed",
        "data_seed",
        "trainer_version",
        "mode",
        "error",
    }
)
_MANIFEST_ERROR_FIELDS = frozenset(
    {"error_type", "error_message", "timestamp", "traceback", "last_step"}
)
TRAINING_MODES = frozenset({"dry_run", "smoke_test", "train"})


@dataclass(frozen=True)
class TrainingManifest:
    """The immutable configuration snapshot for one training run (spec section 70)."""

    training_run_id: str
    experiment_name: str
    status: str
    created_at: str
    model_name: str
    preference_run_id: str
    ranking_run_id: str
    evaluation_run_id: str
    candidate_run_id: str
    dataset_hashes: dict[str, str]
    hardware: dict[str, Any]
    environment: dict[str, Any]
    configuration: dict[str, Any]
    seed: int
    data_seed: int
    trainer_version: str
    mode: str = "train"
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    manifest_version: str = MANIFEST_VERSION
    started_at: str | None = None
    completed_at: str | None = None
    error: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in (
            "training_run_id",
            "experiment_name",
            "created_at",
            "model_name",
            "preference_run_id",
            "ranking_run_id",
            "evaluation_run_id",
            "candidate_run_id",
            "trainer_version",
            "manifest_version",
        ):
            _require_text(getattr(self, name), name)
        _require_optional_text(self.model_revision, "model_revision")
        _require_optional_text(self.tokenizer_revision, "tokenizer_revision")
        _require_optional_text(self.started_at, "started_at")
        _require_optional_text(self.completed_at, "completed_at")
        _require_nonneg_int(self.seed, "seed")
        _require_nonneg_int(self.data_seed, "data_seed")

        if self.status not in TRAINING_RUN_STATUSES:
            raise TrainingModelError(
                f"status must be one of {', '.join(sorted(TRAINING_RUN_STATUSES))}, "
                f"got {self.status!r}"
            )
        if self.mode not in TRAINING_MODES:
            raise TrainingModelError(
                f"mode must be one of {', '.join(sorted(TRAINING_MODES))}, got {self.mode!r}"
            )

        for label, mapping in (
            ("dataset_hashes", self.dataset_hashes),
            ("hardware", self.hardware),
            ("environment", self.environment),
            ("configuration", self.configuration),
        ):
            _require_mapping(mapping, label)

        if self.error is not None:
            _require_mapping(self.error, "error")
            unknown = sorted(set(self.error) - _MANIFEST_ERROR_FIELDS)
            if unknown:
                raise TrainingModelError(f"error: unknown field(s): {', '.join(unknown)}")
            missing = sorted({"error_type", "error_message", "timestamp"} - set(self.error))
            if missing:
                raise TrainingModelError(
                    f"error: missing required field(s): {', '.join(missing)}"
                )

    def with_status(
        self,
        status: str,
        *,
        started_at: str | None = None,
        completed_at: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> TrainingManifest:
        allowed = TRAINING_RUN_STATUS_TRANSITIONS.get(self.status, frozenset())
        if status != self.status and status not in allowed:
            raise TrainingModelError(
                f"cannot transition training run status from {self.status!r} to {status!r}"
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
            "training_run_id": self.training_run_id,
            "experiment_name": self.experiment_name,
            "status": self.status,
            "mode": self.mode,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "preference_run_id": self.preference_run_id,
            "ranking_run_id": self.ranking_run_id,
            "evaluation_run_id": self.evaluation_run_id,
            "candidate_run_id": self.candidate_run_id,
            "dataset_hashes": self.dataset_hashes,
            "hardware": self.hardware,
            "environment": self.environment,
            "configuration": self.configuration,
            "seed": self.seed,
            "data_seed": self.data_seed,
            "trainer_version": self.trainer_version,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Any) -> TrainingManifest:
        if not isinstance(data, dict):
            raise TrainingModelError("training manifest: expected a JSON object")
        unknown = sorted(set(data) - _MANIFEST_FIELDS)
        if unknown:
            raise TrainingModelError(
                f"training manifest: unknown field(s): {', '.join(unknown)}"
            )
        required = {
            "training_run_id",
            "experiment_name",
            "status",
            "created_at",
            "model_name",
            "preference_run_id",
            "ranking_run_id",
            "evaluation_run_id",
            "candidate_run_id",
            "dataset_hashes",
            "hardware",
            "environment",
            "configuration",
            "seed",
            "data_seed",
            "trainer_version",
        }
        missing = sorted(required - set(data))
        if missing:
            raise TrainingModelError(
                f"training manifest: missing required field(s): {', '.join(missing)}"
            )
        kwargs = dict(data)
        kwargs.setdefault("manifest_version", MANIFEST_VERSION)
        kwargs.setdefault("mode", "train")
        return cls(**kwargs)


# ---------------------------------------------------------------------------- FinalReport

_FINAL_REPORT_FIELDS = frozenset(
    {
        "final_report_version",
        "training_run_id",
        "experiment_name",
        "status",
        "model_name",
        "model_revision",
        "preference_run_id",
        "number_of_examples",
        "epochs",
        "steps",
        "final_train_loss",
        "final_eval_loss",
        "reward_metrics",
        "peak_gpu_memory_bytes",
        "trainable_parameters",
        "total_parameters",
        "trainable_percentage",
        "effective_batch_size",
        "optimizer",
        "compute_dtype",
        "adapter_path",
        "checkpoint_path",
        "adapter_reload_ok",
        "training_duration_seconds",
        "created_at",
    }
)


@dataclass(frozen=True)
class FinalReport:
    """The spec section 94 end-of-run summary."""

    training_run_id: str
    experiment_name: str
    status: str
    model_name: str
    preference_run_id: str
    number_of_examples: dict[str, int]
    epochs: float
    steps: int
    trainable_parameters: int
    total_parameters: int
    effective_batch_size: int
    optimizer: str
    compute_dtype: str
    adapter_path: str | None = None
    checkpoint_path: str | None = None
    adapter_reload_ok: bool = False
    final_train_loss: float | None = None
    final_eval_loss: float | None = None
    reward_metrics: dict[str, float] = field(default_factory=dict)
    peak_gpu_memory_bytes: int | None = None
    training_duration_seconds: float | None = None
    model_revision: str | None = None
    final_report_version: str = FINAL_REPORT_VERSION
    created_at: str = ""

    def __post_init__(self) -> None:
        for name in (
            "training_run_id",
            "experiment_name",
            "status",
            "model_name",
            "preference_run_id",
            "optimizer",
            "compute_dtype",
            "final_report_version",
        ):
            _require_text(getattr(self, name), name)
        _require_nonneg_int(self.steps, "steps")
        _require_nonneg_int(self.trainable_parameters, "trainable_parameters")
        _require_nonneg_int(self.total_parameters, "total_parameters")
        _require_nonneg_int(self.effective_batch_size, "effective_batch_size")
        _require_mapping(self.number_of_examples, "number_of_examples")
        _require_mapping(self.reward_metrics, "reward_metrics")

        if self.trainable_parameters > self.total_parameters:
            raise TrainingModelError(
                "trainable_parameters cannot exceed total_parameters"
            )
        if not self.created_at:
            object.__setattr__(self, "created_at", utc_now_iso())

    @property
    def trainable_percentage(self) -> float:
        if self.total_parameters == 0:
            return 0.0
        return 100.0 * self.trainable_parameters / self.total_parameters

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_report_version": self.final_report_version,
            "training_run_id": self.training_run_id,
            "experiment_name": self.experiment_name,
            "status": self.status,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "preference_run_id": self.preference_run_id,
            "number_of_examples": self.number_of_examples,
            "epochs": self.epochs,
            "steps": self.steps,
            "final_train_loss": self.final_train_loss,
            "final_eval_loss": self.final_eval_loss,
            "reward_metrics": self.reward_metrics,
            "peak_gpu_memory_bytes": self.peak_gpu_memory_bytes,
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "trainable_percentage": round(self.trainable_percentage, 6),
            "effective_batch_size": self.effective_batch_size,
            "optimizer": self.optimizer,
            "compute_dtype": self.compute_dtype,
            "adapter_path": self.adapter_path,
            "checkpoint_path": self.checkpoint_path,
            "adapter_reload_ok": self.adapter_reload_ok,
            "training_duration_seconds": self.training_duration_seconds,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> FinalReport:
        if not isinstance(data, dict):
            raise TrainingModelError("final report: expected a JSON object")
        unknown = sorted(set(data) - _FINAL_REPORT_FIELDS)
        if unknown:
            raise TrainingModelError(f"final report: unknown field(s): {', '.join(unknown)}")
        kwargs = {k: v for k, v in data.items() if k != "trainable_percentage"}
        kwargs.setdefault("final_report_version", FINAL_REPORT_VERSION)
        return cls(**kwargs)


__all__ = [
    "DATASET_MANIFEST_VERSION",
    "FINAL_REPORT_VERSION",
    "MANIFEST_VERSION",
    "SPLITS",
    "TRAINING_MODES",
    "TRAINING_RUN_STATUSES",
    "TRAINING_RUN_STATUS_TRANSITIONS",
    "DatasetManifest",
    "FinalReport",
    "TrainingManifest",
    "TrainingModelError",
    "utc_now_iso",
]
