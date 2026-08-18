"""Typed schema for DPO preference pairs, rejections, preference runs, and their
statistics and quality reports.

Frozen dataclasses validating in ``__post_init__``, matching the house style of
``python_dpo.ranking.models``. Constructing an invalid :class:`PreferencePair` is
impossible — spec 08 section 16's validity rules are enforced as construction-time
invariants, not as a downstream check that a bad record might slip past.

This is the first layer in the pipeline allowed to say ``chosen``/``rejected`` (spec 07
section 67 reserved that vocabulary for this stage). Everything upstream — Stage 6's
``EvaluationResult``, Stage 7's ``CandidateAssessment``/``ComparisonResult`` — stays
neutral; this module is where an objective ordering becomes a labeled preference.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Spec section 24: the three selection policies.
POLICIES = frozenset({"strict", "margin", "all_better"})

# Spec section 45: "strong" is correct-vs-incorrect; "medium" is everything else this
# stage is willing to label. "weak" is deliberately never used in the default dataset.
PREFERENCE_STRENGTHS = frozenset({"strong", "medium"})

# Spec section 64.
SPLITS = frozenset({"train", "validation", "test"})

# A chosen/rejected pair is only ever built from a correct/incorrect candidate (spec
# section 38: indeterminate candidates never participate) — a strictly narrower set than
# python_dpo.ranking.models.CORRECTNESS_VALUES.
PAIR_CORRECTNESS_VALUES = frozenset({"correct", "incorrect"})

# Spec section 77: every reason a candidate pair failed to become (or stay) a preference
# pair, or a generated pair was collapsed into another at the training-record level.
REJECTION_REASONS = frozenset(
    {
        "tie",
        "indeterminate",
        "identical_code",
        "insufficient_margin",
        "not_correct_vs_incorrect",
        "invalid_prompt_match",
        "integrity_failure",
        "max_pairs_per_problem",
    }
)

# Mirrors python_dpo.ranking.models' run status lifecycle exactly.
PREFERENCE_RUN_STATUSES = frozenset(
    {"created", "running", "completed", "failed", "interrupted", "cancelled"}
)
PREFERENCE_RUN_STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    "created": frozenset({"running", "cancelled"}),
    "running": frozenset({"running", "completed", "failed", "interrupted", "cancelled"}),
    "interrupted": frozenset({"running", "cancelled"}),
    "failed": frozenset({"running"}),
    "completed": frozenset(),
    "cancelled": frozenset(),
}

# Spec section 85: any structural change to the JSONL artifacts must bump this.
DATASET_SCHEMA_VERSION = "dpo_preference_v1"
# Spec section 28: the pair-schema version, distinct from a policy's own version string.
PREFERENCE_VERSION = "v1"

MANIFEST_VERSION = "1.0"
STATISTICS_VERSION = "1.0"
QUALITY_REPORT_VERSION = "1.0"

_SCORE_EPSILON = 1e-9


class PreferenceModelError(Exception):
    """Raised when a preference record or manifest fails schema validation."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def compute_pass_rate(tests_passed: int, tests_total: int) -> float:
    return tests_passed / tests_total if tests_total > 0 else 0.0


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreferenceModelError(f"{label} must be a non-empty string")
    return value


def _require_optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, label)


def _require_nonneg_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PreferenceModelError(f"{label} must be an integer of 0 or greater")
    return value


def _require_flag(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise PreferenceModelError(f"{label} must be true or false")
    return value


def _require_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreferenceModelError(f"{label} must be a number")
    return float(value)


# ------------------------------------------------------------------------------ PreferencePair

_PAIR_FIELDS = frozenset(
    {
        "preference_id",
        "problem_id",
        "candidate_run_id",
        "ranking_run_id",
        "evaluation_run_id",
        "chosen_candidate_id",
        "rejected_candidate_id",
        "prompt",
        "chosen",
        "rejected",
        "chosen_score",
        "rejected_score",
        "score_margin",
        "chosen_pass_rate",
        "rejected_pass_rate",
        "chosen_tests_passed",
        "rejected_tests_passed",
        "chosen_tests_total",
        "rejected_tests_total",
        "chosen_correctness",
        "rejected_correctness",
        "preference_strength",
        "selection_policy",
        "selection_policy_version",
        "preference_version",
        "canonical_prompt_sha256",
        "prompt_version",
        "chosen_generation_prompt_sha256",
        "rejected_generation_prompt_sha256",
        "chosen_strategy",
        "rejected_strategy",
        "chosen_code_sha256",
        "rejected_code_sha256",
        "duplicate_training_record",
        "canonical_preference_id",
        "created_at",
    }
)


@dataclass(frozen=True)
class PreferencePair:
    """One DPO preference pair (spec 08 section 7), plus full audit provenance.

    ``prompt``/``chosen``/``rejected`` are exactly what a training JSONL record needs
    (spec section 52); everything else is audit metadata (spec section 53) that lets a
    reader trace the pair back to the candidates, ranking run, and evaluation run it was
    built from without re-running Qwen, Docker, or pytest.

    ``chosen``/``rejected`` are the candidates' code verbatim (spec sections 12-14): never
    reformatted, repaired, or wrapped in Markdown fences. ``prompt`` is the *canonical*,
    strategy-free problem prompt (spec 08 decision 1), not either candidate's own
    generation-time prompt — those are preserved separately as
    ``chosen_generation_prompt_sha256``/``rejected_generation_prompt_sha256``.
    """

    preference_id: str
    problem_id: str
    candidate_run_id: str
    ranking_run_id: str
    evaluation_run_id: str
    chosen_candidate_id: str
    rejected_candidate_id: str
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    score_margin: float
    chosen_pass_rate: float
    rejected_pass_rate: float
    chosen_tests_passed: int
    rejected_tests_passed: int
    chosen_tests_total: int
    rejected_tests_total: int
    chosen_correctness: str
    rejected_correctness: str
    preference_strength: str
    selection_policy: str
    selection_policy_version: str
    canonical_prompt_sha256: str
    prompt_version: str
    chosen_generation_prompt_sha256: str
    rejected_generation_prompt_sha256: str
    chosen_strategy: str
    rejected_strategy: str
    chosen_code_sha256: str
    rejected_code_sha256: str
    preference_version: str = PREFERENCE_VERSION
    duplicate_training_record: bool = False
    canonical_preference_id: str | None = None
    created_at: str = ""

    def __post_init__(self) -> None:
        for name in (
            "preference_id",
            "problem_id",
            "candidate_run_id",
            "ranking_run_id",
            "evaluation_run_id",
            "chosen_candidate_id",
            "rejected_candidate_id",
            "prompt",
            "chosen",
            "rejected",
            "selection_policy",
            "selection_policy_version",
            "preference_version",
            "canonical_prompt_sha256",
            "prompt_version",
            "chosen_generation_prompt_sha256",
            "rejected_generation_prompt_sha256",
            "chosen_strategy",
            "rejected_strategy",
            "chosen_code_sha256",
            "rejected_code_sha256",
        ):
            _require_text(getattr(self, name), name)

        if self.selection_policy not in POLICIES:
            raise PreferenceModelError(
                f"selection_policy must be one of {', '.join(sorted(POLICIES))}, "
                f"got {self.selection_policy!r}"
            )
        for name in ("chosen_correctness", "rejected_correctness"):
            value = getattr(self, name)
            if value not in PAIR_CORRECTNESS_VALUES:
                raise PreferenceModelError(
                    f"{name} must be one of {', '.join(sorted(PAIR_CORRECTNESS_VALUES))} "
                    f"(indeterminate candidates never participate in a pair), got {value!r}"
                )
        if self.preference_strength not in PREFERENCE_STRENGTHS:
            raise PreferenceModelError(
                f"preference_strength must be one of {', '.join(sorted(PREFERENCE_STRENGTHS))}, "
                f"got {self.preference_strength!r}"
            )

        for name in (
            "chosen_score",
            "rejected_score",
            "score_margin",
            "chosen_pass_rate",
            "rejected_pass_rate",
        ):
            _require_float(getattr(self, name), name)
        for name in (
            "chosen_tests_passed",
            "rejected_tests_passed",
            "chosen_tests_total",
            "rejected_tests_total",
        ):
            _require_nonneg_int(getattr(self, name), name)
        _require_flag(self.duplicate_training_record, "duplicate_training_record")

        # Spec section 16: pair validity.
        if self.chosen_candidate_id == self.rejected_candidate_id:
            raise PreferenceModelError(
                "chosen_candidate_id and rejected_candidate_id must be different candidates"
            )
        if self.chosen == self.rejected:
            raise PreferenceModelError("chosen and rejected code must differ")
        if self.chosen_code_sha256 == self.rejected_code_sha256:
            raise PreferenceModelError(
                "chosen_code_sha256 and rejected_code_sha256 must differ (spec sections 33, 34)"
            )
        for name in ("chosen_candidate_id", "rejected_candidate_id"):
            value = getattr(self, name)
            if not value.startswith(f"{self.problem_id}_c"):
                raise PreferenceModelError(
                    f"{name} {value!r} does not belong to problem {self.problem_id!r}"
                )
        if self.chosen_score <= self.rejected_score:
            raise PreferenceModelError(
                f"chosen_score ({self.chosen_score}) must be strictly greater than "
                f"rejected_score ({self.rejected_score})"
            )
        expected_margin = self.chosen_score - self.rejected_score
        if abs(self.score_margin - expected_margin) > _SCORE_EPSILON:
            raise PreferenceModelError(
                f"score_margin does not match chosen_score - rejected_score: expected "
                f"{expected_margin}, got {self.score_margin}"
            )

        for prefix in ("chosen", "rejected"):
            passed = getattr(self, f"{prefix}_tests_passed")
            total = getattr(self, f"{prefix}_tests_total")
            pass_rate = getattr(self, f"{prefix}_pass_rate")
            if passed > total:
                raise PreferenceModelError(
                    f"{prefix}_tests_passed ({passed}) cannot exceed {prefix}_tests_total ({total})"
                )
            expected_pass_rate = compute_pass_rate(passed, total)
            if abs(pass_rate - expected_pass_rate) > _SCORE_EPSILON:
                raise PreferenceModelError(
                    f"{prefix}_pass_rate does not match {prefix}_tests_passed/"
                    f"{prefix}_tests_total: expected {expected_pass_rate}, got {pass_rate}"
                )

        # Spec section 45: strong iff correct-vs-incorrect; everything else this stage
        # builds a pair for is "medium" ("weak" is never used, spec section 45).
        expected_strength = (
            "strong"
            if (self.chosen_correctness, self.rejected_correctness) == ("correct", "incorrect")
            else "medium"
        )
        if self.preference_strength != expected_strength:
            raise PreferenceModelError(
                f"preference_strength must be {expected_strength!r} given correctness "
                f"({self.chosen_correctness!r}, {self.rejected_correctness!r}), "
                f"got {self.preference_strength!r}"
            )

        # Decision 3: a duplicate training record always points at a different survivor.
        if self.duplicate_training_record:
            canonical = _require_text(
                self.canonical_preference_id or "", "canonical_preference_id"
            )
            if canonical == self.preference_id:
                raise PreferenceModelError(
                    "canonical_preference_id must reference a different, surviving preference_id"
                )
        elif self.canonical_preference_id is not None:
            raise PreferenceModelError(
                "canonical_preference_id must be null unless duplicate_training_record is true"
            )

        if not self.created_at:
            object.__setattr__(self, "created_at", utc_now_iso())

    def to_dict(self) -> dict[str, Any]:
        return {
            "preference_id": self.preference_id,
            "problem_id": self.problem_id,
            "candidate_run_id": self.candidate_run_id,
            "ranking_run_id": self.ranking_run_id,
            "evaluation_run_id": self.evaluation_run_id,
            "chosen_candidate_id": self.chosen_candidate_id,
            "rejected_candidate_id": self.rejected_candidate_id,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "chosen_score": self.chosen_score,
            "rejected_score": self.rejected_score,
            "score_margin": self.score_margin,
            "chosen_pass_rate": self.chosen_pass_rate,
            "rejected_pass_rate": self.rejected_pass_rate,
            "chosen_tests_passed": self.chosen_tests_passed,
            "rejected_tests_passed": self.rejected_tests_passed,
            "chosen_tests_total": self.chosen_tests_total,
            "rejected_tests_total": self.rejected_tests_total,
            "chosen_correctness": self.chosen_correctness,
            "rejected_correctness": self.rejected_correctness,
            "preference_strength": self.preference_strength,
            "selection_policy": self.selection_policy,
            "selection_policy_version": self.selection_policy_version,
            "preference_version": self.preference_version,
            "canonical_prompt_sha256": self.canonical_prompt_sha256,
            "prompt_version": self.prompt_version,
            "chosen_generation_prompt_sha256": self.chosen_generation_prompt_sha256,
            "rejected_generation_prompt_sha256": self.rejected_generation_prompt_sha256,
            "chosen_strategy": self.chosen_strategy,
            "rejected_strategy": self.rejected_strategy,
            "chosen_code_sha256": self.chosen_code_sha256,
            "rejected_code_sha256": self.rejected_code_sha256,
            "duplicate_training_record": self.duplicate_training_record,
            "canonical_preference_id": self.canonical_preference_id,
            "created_at": self.created_at,
        }

    def training_record(self) -> dict[str, str]:
        """The exactly-three-key record for ``preferences.jsonl``/``train.jsonl``/etc.
        (spec sections 52, 94) — never candidate ids, never run ids.
        """
        return {"prompt": self.prompt, "chosen": self.chosen, "rejected": self.rejected}

    @classmethod
    def from_dict(cls, data: Any) -> PreferencePair:
        if not isinstance(data, dict):
            raise PreferenceModelError("preference pair: expected a JSON object")
        unknown = sorted(set(data) - _PAIR_FIELDS)
        if unknown:
            raise PreferenceModelError(f"preference pair: unknown field(s): {', '.join(unknown)}")
        required = _PAIR_FIELDS - {
            "preference_version",
            "duplicate_training_record",
            "canonical_preference_id",
            "created_at",
        }
        missing = sorted(required - set(data))
        if missing:
            raise PreferenceModelError(
                f"preference pair: missing required field(s): {', '.join(missing)}"
            )
        kwargs = dict(data)
        kwargs.setdefault("preference_version", PREFERENCE_VERSION)
        kwargs.setdefault("duplicate_training_record", False)
        kwargs.setdefault("canonical_preference_id", None)
        return cls(**kwargs)


# -------------------------------------------------------------------------- PreferenceRejection

_REJECTION_FIELDS = frozenset(
    {
        "ranking_run_id",
        "problem_id",
        "candidate_a",
        "candidate_b",
        "reason",
        "detail",
        "relation",
        "score_a",
        "score_b",
        "score_margin",
        "created_at",
    }
)


@dataclass(frozen=True)
class PreferenceRejection:
    """One candidate pair that did not become a preference pair (spec section 77).

    Recorded, never silently dropped (CLAUDE.md's data-integrity rule): every pair the
    builder considers ends up either as a :class:`PreferencePair` or as one of these.
    ``candidate_a``/``candidate_b`` are the pair in ``candidate_id`` sort order, not
    chosen/rejected order — a rejected pair never had a decided direction.
    """

    ranking_run_id: str
    problem_id: str
    candidate_a: str
    candidate_b: str
    reason: str
    detail: str
    relation: str
    score_a: float
    score_b: float
    score_margin: float
    created_at: str = ""

    def __post_init__(self) -> None:
        for name in ("ranking_run_id", "problem_id", "candidate_a", "candidate_b", "reason"):
            _require_text(getattr(self, name), name)
        if not isinstance(self.detail, str):
            raise PreferenceModelError("detail must be a string")
        if self.candidate_a == self.candidate_b:
            raise PreferenceModelError("candidate_a and candidate_b must be different candidates")
        if self.reason not in REJECTION_REASONS:
            raise PreferenceModelError(
                f"reason must be one of {', '.join(sorted(REJECTION_REASONS))}, "
                f"got {self.reason!r}"
            )
        _require_text(self.relation, "relation")
        for name in ("score_a", "score_b", "score_margin"):
            _require_float(getattr(self, name), name)

        if not self.created_at:
            object.__setattr__(self, "created_at", utc_now_iso())

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranking_run_id": self.ranking_run_id,
            "problem_id": self.problem_id,
            "candidate_a": self.candidate_a,
            "candidate_b": self.candidate_b,
            "reason": self.reason,
            "detail": self.detail,
            "relation": self.relation,
            "score_a": self.score_a,
            "score_b": self.score_b,
            "score_margin": self.score_margin,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> PreferenceRejection:
        if not isinstance(data, dict):
            raise PreferenceModelError("preference rejection: expected a JSON object")
        unknown = sorted(set(data) - _REJECTION_FIELDS)
        if unknown:
            raise PreferenceModelError(
                f"preference rejection: unknown field(s): {', '.join(unknown)}"
            )
        required = _REJECTION_FIELDS - {"created_at"}
        missing = sorted(required - set(data))
        if missing:
            raise PreferenceModelError(
                f"preference rejection: missing required field(s): {', '.join(missing)}"
            )
        return cls(**data)


# --------------------------------------------------------------------------- PreferenceManifest

_MANIFEST_FIELDS = frozenset(
    {
        "manifest_version",
        "preference_run_id",
        "ranking_run_id",
        "evaluation_run_id",
        "candidate_run_id",
        "status",
        "created_at",
        "started_at",
        "completed_at",
        "preference_version",
        "selection_policy",
        "selection_policy_version",
        "minimum_score_margin",
        "max_pairs_per_problem",
        "split_ratios",
        "split_seed",
        "dataset_schema_version",
        "builder_version",
        "error",
    }
)
_MANIFEST_ERROR_FIELDS = frozenset({"error_type", "error_message", "timestamp"})


@dataclass(frozen=True)
class PreferenceManifest:
    """The historical, immutable configuration snapshot for one preference run (spec
    section 51).

    Mirrors :class:`python_dpo.ranking.models.RankingManifest`: the source of truth for
    which upstream runs a preference run covers and which policy/version produced it,
    never re-derived from the current source tree.
    """

    preference_run_id: str
    ranking_run_id: str
    evaluation_run_id: str
    candidate_run_id: str
    status: str
    created_at: str
    preference_version: str
    selection_policy: str
    selection_policy_version: str
    minimum_score_margin: float
    split_ratios: dict[str, float]
    split_seed: int
    builder_version: str
    manifest_version: str = MANIFEST_VERSION
    started_at: str | None = None
    completed_at: str | None = None
    max_pairs_per_problem: int | None = None
    dataset_schema_version: str = DATASET_SCHEMA_VERSION
    error: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_text(self.manifest_version, "manifest_version")
        _require_text(self.preference_run_id, "preference_run_id")
        _require_text(self.ranking_run_id, "ranking_run_id")
        _require_text(self.evaluation_run_id, "evaluation_run_id")
        _require_text(self.candidate_run_id, "candidate_run_id")
        if self.status not in PREFERENCE_RUN_STATUSES:
            raise PreferenceModelError(
                f"status must be one of {', '.join(sorted(PREFERENCE_RUN_STATUSES))}, "
                f"got {self.status!r}"
            )
        _require_text(self.created_at, "created_at")
        _require_optional_text(self.started_at, "started_at")
        _require_optional_text(self.completed_at, "completed_at")
        _require_text(self.preference_version, "preference_version")
        if self.selection_policy not in POLICIES:
            raise PreferenceModelError(
                f"selection_policy must be one of {', '.join(sorted(POLICIES))}, "
                f"got {self.selection_policy!r}"
            )
        _require_text(self.selection_policy_version, "selection_policy_version")
        _require_text(self.dataset_schema_version, "dataset_schema_version")
        _require_text(self.builder_version, "builder_version")
        _require_float(self.minimum_score_margin, "minimum_score_margin")
        if self.minimum_score_margin < 0:
            raise PreferenceModelError("minimum_score_margin must be 0 or greater")

        if self.max_pairs_per_problem is not None:
            if (
                isinstance(self.max_pairs_per_problem, bool)
                or not isinstance(self.max_pairs_per_problem, int)
                or self.max_pairs_per_problem < 1
            ):
                raise PreferenceModelError(
                    "max_pairs_per_problem must be a positive integer or null"
                )

        if not isinstance(self.split_ratios, dict):
            raise PreferenceModelError("split_ratios must be a mapping")
        unknown_splits = sorted(set(self.split_ratios) - SPLITS)
        if unknown_splits:
            raise PreferenceModelError(
                f"split_ratios: unknown split(s): {', '.join(unknown_splits)}"
            )
        missing_splits = sorted(SPLITS - set(self.split_ratios))
        if missing_splits:
            raise PreferenceModelError(
                f"split_ratios: missing split(s): {', '.join(missing_splits)}"
            )
        total = sum(self.split_ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise PreferenceModelError(f"split_ratios must sum to 1.0, got {total}")
        for name, value in self.split_ratios.items():
            _require_float(value, f"split_ratios[{name!r}]")
            if value < 0:
                raise PreferenceModelError(f"split_ratios[{name!r}] must be 0 or greater")

        if isinstance(self.split_seed, bool) or not isinstance(self.split_seed, int):
            raise PreferenceModelError("split_seed must be an integer")

        if self.error is not None:
            if not isinstance(self.error, dict):
                raise PreferenceModelError("error must be a mapping or null")
            unknown = sorted(set(self.error) - _MANIFEST_ERROR_FIELDS)
            if unknown:
                raise PreferenceModelError(f"error: unknown field(s): {', '.join(unknown)}")
            missing = sorted(_MANIFEST_ERROR_FIELDS - set(self.error))
            if missing:
                raise PreferenceModelError(f"error: missing required field(s): {', '.join(missing)}")

    def with_status(
        self,
        status: str,
        *,
        started_at: str | None = None,
        completed_at: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> PreferenceManifest:
        allowed = PREFERENCE_RUN_STATUS_TRANSITIONS.get(self.status, frozenset())
        if status != self.status and status not in allowed:
            raise PreferenceModelError(
                f"cannot transition preference run status from {self.status!r} to {status!r}"
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
            "preference_run_id": self.preference_run_id,
            "ranking_run_id": self.ranking_run_id,
            "evaluation_run_id": self.evaluation_run_id,
            "candidate_run_id": self.candidate_run_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "preference_version": self.preference_version,
            "selection_policy": self.selection_policy,
            "selection_policy_version": self.selection_policy_version,
            "minimum_score_margin": self.minimum_score_margin,
            "max_pairs_per_problem": self.max_pairs_per_problem,
            "split_ratios": self.split_ratios,
            "split_seed": self.split_seed,
            "dataset_schema_version": self.dataset_schema_version,
            "builder_version": self.builder_version,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Any) -> PreferenceManifest:
        if not isinstance(data, dict):
            raise PreferenceModelError("preference manifest: expected a JSON object")
        unknown = sorted(set(data) - _MANIFEST_FIELDS)
        if unknown:
            raise PreferenceModelError(
                f"preference manifest: unknown field(s): {', '.join(unknown)}"
            )
        required = {
            "preference_run_id",
            "ranking_run_id",
            "evaluation_run_id",
            "candidate_run_id",
            "status",
            "created_at",
            "preference_version",
            "selection_policy",
            "selection_policy_version",
            "minimum_score_margin",
            "split_ratios",
            "split_seed",
            "builder_version",
        }
        missing = sorted(required - set(data))
        if missing:
            raise PreferenceModelError(
                f"preference manifest: missing required field(s): {', '.join(missing)}"
            )
        return cls(
            manifest_version=data.get("manifest_version", MANIFEST_VERSION),
            preference_run_id=data["preference_run_id"],
            ranking_run_id=data["ranking_run_id"],
            evaluation_run_id=data["evaluation_run_id"],
            candidate_run_id=data["candidate_run_id"],
            status=data["status"],
            created_at=data["created_at"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            preference_version=data["preference_version"],
            selection_policy=data["selection_policy"],
            selection_policy_version=data["selection_policy_version"],
            minimum_score_margin=data["minimum_score_margin"],
            max_pairs_per_problem=data.get("max_pairs_per_problem"),
            split_ratios=dict(data["split_ratios"]),
            split_seed=data["split_seed"],
            dataset_schema_version=data.get("dataset_schema_version", DATASET_SCHEMA_VERSION),
            builder_version=data["builder_version"],
            error=data.get("error"),
        )


# ------------------------------------------------------------------------ PreferenceStatistics

_STATISTICS_FIELDS = frozenset(
    {
        "statistics_version",
        "preference_run_id",
        "problems_processed",
        "candidates_considered",
        "candidate_pairs_considered",
        "pairs_generated",
        "pairs_rejected",
        "ties",
        "duplicates",
        "indeterminate",
        "prompt_mismatches",
        "integrity_failures",
        "strong_pairs",
        "medium_pairs",
        "training_records",
        "rejections_by_reason",
        "per_problem",
        "computed_at",
    }
)


@dataclass(frozen=True)
class PreferenceStatistics:
    """Preference run statistics (spec section 54) — always reconstructable from
    persisted records, never trusted from an in-memory counter.

    ``ties``/``duplicates``/``indeterminate``/``prompt_mismatches``/``integrity_failures``
    are the spec section 54 headline counters; each is validated to equal the matching
    entry of ``rejections_by_reason`` (``duplicates`` <-> the ``identical_code`` reason,
    spec sections 33/74) rather than being tracked independently, so the two views of the
    same rejections can never silently disagree.
    """

    preference_run_id: str
    problems_processed: int
    candidates_considered: int
    candidate_pairs_considered: int
    pairs_generated: int
    pairs_rejected: int
    ties: int
    duplicates: int
    indeterminate: int
    prompt_mismatches: int
    integrity_failures: int
    strong_pairs: int
    medium_pairs: int
    training_records: int
    rejections_by_reason: dict[str, int]
    per_problem: dict[str, dict[str, int]]
    computed_at: str
    statistics_version: str = STATISTICS_VERSION

    def __post_init__(self) -> None:
        _require_text(self.statistics_version, "statistics_version")
        _require_text(self.preference_run_id, "preference_run_id")
        for name in (
            "problems_processed",
            "candidates_considered",
            "candidate_pairs_considered",
            "pairs_generated",
            "pairs_rejected",
            "ties",
            "duplicates",
            "indeterminate",
            "prompt_mismatches",
            "integrity_failures",
            "strong_pairs",
            "medium_pairs",
            "training_records",
        ):
            _require_nonneg_int(getattr(self, name), name)
        if not isinstance(self.rejections_by_reason, dict):
            raise PreferenceModelError("rejections_by_reason must be a mapping")
        unknown_reasons = sorted(set(self.rejections_by_reason) - REJECTION_REASONS)
        if unknown_reasons:
            raise PreferenceModelError(
                f"rejections_by_reason: unknown reason(s): {', '.join(unknown_reasons)}"
            )
        for reason, count in self.rejections_by_reason.items():
            _require_nonneg_int(count, f"rejections_by_reason[{reason!r}]")
        if not isinstance(self.per_problem, dict):
            raise PreferenceModelError("per_problem must be a mapping")
        _require_text(self.computed_at, "computed_at")

        if self.pairs_generated + self.pairs_rejected != self.candidate_pairs_considered:
            raise PreferenceModelError(
                "pairs_generated + pairs_rejected must equal candidate_pairs_considered"
            )
        if self.strong_pairs + self.medium_pairs != self.pairs_generated:
            raise PreferenceModelError("strong_pairs + medium_pairs must equal pairs_generated")
        if self.training_records > self.pairs_generated:
            raise PreferenceModelError("training_records cannot exceed pairs_generated")
        if sum(self.rejections_by_reason.values()) != self.pairs_rejected:
            raise PreferenceModelError(
                "sum(rejections_by_reason.values()) must equal pairs_rejected"
            )

        expectations = {
            "ties": ("tie", self.ties),
            "duplicates": ("identical_code", self.duplicates),
            "indeterminate": ("indeterminate", self.indeterminate),
            "prompt_mismatches": ("invalid_prompt_match", self.prompt_mismatches),
            "integrity_failures": ("integrity_failure", self.integrity_failures),
        }
        for field_name, (reason, value) in expectations.items():
            expected = self.rejections_by_reason.get(reason, 0)
            if value != expected:
                raise PreferenceModelError(
                    f"{field_name} ({value}) must equal rejections_by_reason[{reason!r}] "
                    f"({expected})"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistics_version": self.statistics_version,
            "preference_run_id": self.preference_run_id,
            "problems_processed": self.problems_processed,
            "candidates_considered": self.candidates_considered,
            "candidate_pairs_considered": self.candidate_pairs_considered,
            "pairs_generated": self.pairs_generated,
            "pairs_rejected": self.pairs_rejected,
            "ties": self.ties,
            "duplicates": self.duplicates,
            "indeterminate": self.indeterminate,
            "prompt_mismatches": self.prompt_mismatches,
            "integrity_failures": self.integrity_failures,
            "strong_pairs": self.strong_pairs,
            "medium_pairs": self.medium_pairs,
            "training_records": self.training_records,
            "rejections_by_reason": self.rejections_by_reason,
            "per_problem": self.per_problem,
            "computed_at": self.computed_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> PreferenceStatistics:
        if not isinstance(data, dict):
            raise PreferenceModelError("preference statistics: expected a JSON object")
        unknown = sorted(set(data) - _STATISTICS_FIELDS)
        if unknown:
            raise PreferenceModelError(
                f"preference statistics: unknown field(s): {', '.join(unknown)}"
            )
        required = _STATISTICS_FIELDS - {"statistics_version"}
        missing = sorted(required - set(data))
        if missing:
            raise PreferenceModelError(
                f"preference statistics: missing required field(s): {', '.join(missing)}"
            )
        kwargs = dict(data)
        kwargs.setdefault("statistics_version", STATISTICS_VERSION)
        return cls(**kwargs)

    @classmethod
    def from_records(
        cls,
        manifest: PreferenceManifest,
        pairs: list[PreferencePair],
        rejections: list[PreferenceRejection],
        *,
        candidates_considered: int,
        computed_at: str | None = None,
    ) -> PreferenceStatistics:
        problems: set[str] = set()
        per_problem: dict[str, dict[str, int]] = {}
        rejections_by_reason: dict[str, int] = {}

        def _bucket(problem_id: str) -> dict[str, int]:
            return per_problem.setdefault(
                problem_id,
                {"pairs_considered": 0, "pairs_generated": 0, "pairs_rejected": 0, "training_records": 0},
            )

        for pair in pairs:
            problems.add(pair.problem_id)
            bucket = _bucket(pair.problem_id)
            bucket["pairs_considered"] += 1
            bucket["pairs_generated"] += 1
            if not pair.duplicate_training_record:
                bucket["training_records"] += 1

        for rejection in rejections:
            problems.add(rejection.problem_id)
            bucket = _bucket(rejection.problem_id)
            bucket["pairs_considered"] += 1
            bucket["pairs_rejected"] += 1
            rejections_by_reason[rejection.reason] = rejections_by_reason.get(rejection.reason, 0) + 1

        strong_pairs = sum(1 for p in pairs if p.preference_strength == "strong")
        medium_pairs = sum(1 for p in pairs if p.preference_strength == "medium")
        training_records = sum(1 for p in pairs if not p.duplicate_training_record)

        return cls(
            preference_run_id=manifest.preference_run_id,
            problems_processed=len(problems),
            candidates_considered=candidates_considered,
            candidate_pairs_considered=len(pairs) + len(rejections),
            pairs_generated=len(pairs),
            pairs_rejected=len(rejections),
            ties=rejections_by_reason.get("tie", 0),
            duplicates=rejections_by_reason.get("identical_code", 0),
            indeterminate=rejections_by_reason.get("indeterminate", 0),
            prompt_mismatches=rejections_by_reason.get("invalid_prompt_match", 0),
            integrity_failures=rejections_by_reason.get("integrity_failure", 0),
            strong_pairs=strong_pairs,
            medium_pairs=medium_pairs,
            training_records=training_records,
            rejections_by_reason=rejections_by_reason,
            per_problem=per_problem,
            computed_at=computed_at or utc_now_iso(),
        )


# ----------------------------------------------------------------------------- QualityReport

_QUALITY_REPORT_FIELDS = frozenset(
    {
        "report_version",
        "preference_run_id",
        "total_pairs",
        "strong_pairs",
        "medium_pairs",
        "score_margin_distribution",
        "chosen_pass_rate_distribution",
        "rejected_pass_rate_distribution",
        "strategy_distribution",
        "problems_with_pairs",
        "problems_without_pairs",
        "computed_at",
    }
)


@dataclass(frozen=True)
class QualityReport:
    """Dataset quality signals (spec section 75) — reported, never enforced (spec section
    92): the initial dataset is far too small for a statistically meaningful gate.
    """

    preference_run_id: str
    total_pairs: int
    strong_pairs: int
    medium_pairs: int
    score_margin_distribution: dict[str, int]
    chosen_pass_rate_distribution: dict[str, int]
    rejected_pass_rate_distribution: dict[str, int]
    strategy_distribution: dict[str, dict[str, int]]
    problems_with_pairs: list[str]
    problems_without_pairs: dict[str, str]
    computed_at: str
    report_version: str = QUALITY_REPORT_VERSION

    def __post_init__(self) -> None:
        _require_text(self.report_version, "report_version")
        _require_text(self.preference_run_id, "preference_run_id")
        _require_nonneg_int(self.total_pairs, "total_pairs")
        _require_nonneg_int(self.strong_pairs, "strong_pairs")
        _require_nonneg_int(self.medium_pairs, "medium_pairs")
        if self.strong_pairs + self.medium_pairs != self.total_pairs:
            raise PreferenceModelError("strong_pairs + medium_pairs must equal total_pairs")
        for label, mapping in (
            ("score_margin_distribution", self.score_margin_distribution),
            ("chosen_pass_rate_distribution", self.chosen_pass_rate_distribution),
            ("rejected_pass_rate_distribution", self.rejected_pass_rate_distribution),
        ):
            if not isinstance(mapping, dict):
                raise PreferenceModelError(f"{label} must be a mapping")
        if not isinstance(self.strategy_distribution, dict):
            raise PreferenceModelError("strategy_distribution must be a mapping")
        if not isinstance(self.problems_with_pairs, list):
            raise PreferenceModelError("problems_with_pairs must be a list")
        if not isinstance(self.problems_without_pairs, dict):
            raise PreferenceModelError("problems_without_pairs must be a mapping")
        overlap = set(self.problems_with_pairs) & set(self.problems_without_pairs)
        if overlap:
            raise PreferenceModelError(
                f"a problem cannot be both with and without pairs: {sorted(overlap)}"
            )
        _require_text(self.computed_at, "computed_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_version": self.report_version,
            "preference_run_id": self.preference_run_id,
            "total_pairs": self.total_pairs,
            "strong_pairs": self.strong_pairs,
            "medium_pairs": self.medium_pairs,
            "score_margin_distribution": self.score_margin_distribution,
            "chosen_pass_rate_distribution": self.chosen_pass_rate_distribution,
            "rejected_pass_rate_distribution": self.rejected_pass_rate_distribution,
            "strategy_distribution": self.strategy_distribution,
            "problems_with_pairs": self.problems_with_pairs,
            "problems_without_pairs": self.problems_without_pairs,
            "computed_at": self.computed_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> QualityReport:
        if not isinstance(data, dict):
            raise PreferenceModelError("quality report: expected a JSON object")
        unknown = sorted(set(data) - _QUALITY_REPORT_FIELDS)
        if unknown:
            raise PreferenceModelError(f"quality report: unknown field(s): {', '.join(unknown)}")
        required = _QUALITY_REPORT_FIELDS - {"report_version"}
        missing = sorted(required - set(data))
        if missing:
            raise PreferenceModelError(
                f"quality report: missing required field(s): {', '.join(missing)}"
            )
        kwargs = dict(data)
        kwargs.setdefault("report_version", QUALITY_REPORT_VERSION)
        return cls(**kwargs)


def _round_bucket(value: float) -> str:
    """Bucket a [0, 1]-ish float to one decimal place, e.g. ``0.111`` -> ``"0.1"``."""
    return f"{round(value, 1):.1f}"


_WITHOUT_PAIRS_REASONS = frozenset(
    {
        "insufficient_candidates",
        "all_candidates_indeterminate",
        "all_candidates_correct",
        "all_candidates_tied",
        "no_qualifying_preference",
    }
)


def build_quality_report(
    manifest: PreferenceManifest,
    pairs: list[PreferencePair],
    problem_correctness: dict[str, list[str]],
    *,
    computed_at: str | None = None,
) -> QualityReport:
    """Build the spec section 75 quality report.

    ``problem_correctness`` maps every processed ``problem_id`` to the list of
    ``correctness`` values (``correct``/``incorrect``/``indeterminate``) its candidates
    were assessed with, which is enough to classify *why* a pairless problem produced
    nothing (spec section 76) without re-reading the ranking run.
    """
    strong_pairs = sum(1 for p in pairs if p.preference_strength == "strong")
    medium_pairs = sum(1 for p in pairs if p.preference_strength == "medium")

    score_margin_distribution: dict[str, int] = {}
    chosen_pass_rate_distribution: dict[str, int] = {}
    rejected_pass_rate_distribution: dict[str, int] = {}
    chosen_strategy_counts: dict[str, int] = {}
    rejected_strategy_counts: dict[str, int] = {}
    problems_with_pairs: set[str] = set()

    for pair in pairs:
        problems_with_pairs.add(pair.problem_id)
        key = _round_bucket(pair.score_margin)
        score_margin_distribution[key] = score_margin_distribution.get(key, 0) + 1
        key = _round_bucket(pair.chosen_pass_rate)
        chosen_pass_rate_distribution[key] = chosen_pass_rate_distribution.get(key, 0) + 1
        key = _round_bucket(pair.rejected_pass_rate)
        rejected_pass_rate_distribution[key] = rejected_pass_rate_distribution.get(key, 0) + 1
        chosen_strategy_counts[pair.chosen_strategy] = (
            chosen_strategy_counts.get(pair.chosen_strategy, 0) + 1
        )
        rejected_strategy_counts[pair.rejected_strategy] = (
            rejected_strategy_counts.get(pair.rejected_strategy, 0) + 1
        )

    problems_without_pairs: dict[str, str] = {}
    for problem_id, correctness_values in problem_correctness.items():
        if problem_id in problems_with_pairs:
            continue
        problems_without_pairs[problem_id] = _classify_pairless_problem(correctness_values)

    return QualityReport(
        preference_run_id=manifest.preference_run_id,
        total_pairs=len(pairs),
        strong_pairs=strong_pairs,
        medium_pairs=medium_pairs,
        score_margin_distribution=score_margin_distribution,
        chosen_pass_rate_distribution=chosen_pass_rate_distribution,
        rejected_pass_rate_distribution=rejected_pass_rate_distribution,
        strategy_distribution={"chosen": chosen_strategy_counts, "rejected": rejected_strategy_counts},
        problems_with_pairs=sorted(problems_with_pairs),
        problems_without_pairs=problems_without_pairs,
        computed_at=computed_at or utc_now_iso(),
    )


def derive_candidates_considered(
    pairs: list[PreferencePair], rejections: list[PreferenceRejection]
) -> int:
    """The number of distinct candidates that participated in at least one considered
    pair — every candidate in a processed problem appears in ``n - 1`` unordered pairs,
    each recorded as either a :class:`PreferencePair` or a :class:`PreferenceRejection`
    (spec section 29), so this is exactly reconstructable from those two artifacts alone
    without a separately tracked, independently-trustable counter.
    """
    ids: set[str] = set()
    for pair in pairs:
        ids.add(pair.chosen_candidate_id)
        ids.add(pair.rejected_candidate_id)
    for rejection in rejections:
        ids.add(rejection.candidate_a)
        ids.add(rejection.candidate_b)
    return len(ids)


def _classify_pairless_problem(correctness_values: list[str]) -> str:
    if len(correctness_values) < 2:
        return "insufficient_candidates"
    if all(value == "indeterminate" for value in correctness_values):
        return "all_candidates_indeterminate"
    if all(value == "correct" for value in correctness_values):
        return "all_candidates_correct"
    non_indeterminate = [v for v in correctness_values if v != "indeterminate"]
    if non_indeterminate and len(set(non_indeterminate)) == 1:
        return "all_candidates_tied"
    return "no_qualifying_preference"


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "MANIFEST_VERSION",
    "PAIR_CORRECTNESS_VALUES",
    "POLICIES",
    "PREFERENCE_RUN_STATUSES",
    "PREFERENCE_RUN_STATUS_TRANSITIONS",
    "PREFERENCE_STRENGTHS",
    "PREFERENCE_VERSION",
    "QUALITY_REPORT_VERSION",
    "REJECTION_REASONS",
    "SPLITS",
    "STATISTICS_VERSION",
    "PreferenceManifest",
    "PreferenceModelError",
    "PreferencePair",
    "PreferenceRejection",
    "PreferenceStatistics",
    "QualityReport",
    "build_quality_report",
    "compute_pass_rate",
    "derive_candidates_considered",
    "utc_now_iso",
]
