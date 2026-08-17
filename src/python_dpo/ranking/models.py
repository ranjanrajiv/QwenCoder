"""Typed schema for candidate assessments, rankings, comparisons, and ranking runs.

Frozen dataclasses validating in ``__post_init__``, matching the house style of
``python_dpo.candidates.models`` and ``python_dpo.evaluation.models``.

Step 6 answers *"what happened when this candidate was tested?"*; this layer answers
*"was that good?"* — but never *"which is the chosen/rejected pair?"* (spec section 2,
81). Nothing here computes a DPO preference pair; it produces the ordering Step 8 will
build pairs from.

Two artifacts are deliberately new rather than mutations of historical ones (spec
section 5): :class:`CandidateAssessment` derives from a Stage 6
:class:`~python_dpo.evaluation.models.EvaluationResult` without ever modifying it, and
:class:`RankingResult` derives from an assessment. Both carry the ids needed to trace
back through evaluation, generation, and prompt.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Spec section 9: the minimum classification set.
CORRECTNESS_VALUES = frozenset({"correct", "incorrect", "indeterminate"})

# Spec section 33: the pairwise comparison outcomes.
COMPARISON_RELATIONS = frozenset({"A_BETTER", "B_BETTER", "TIE", "INDETERMINATE"})

# Mirrors spec 06's EvaluationManifest status set/lifecycle — a ranking run has the same
# create -> running -> completed/failed/interrupted shape as an evaluation run.
RANKING_RUN_STATUSES = frozenset(
    {"created", "running", "completed", "failed", "interrupted", "cancelled"}
)
RANKING_RUN_STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    "created": frozenset({"running", "cancelled"}),
    "running": frozenset({"running", "completed", "failed", "interrupted", "cancelled"}),
    "interrupted": frozenset({"running", "cancelled"}),
    "failed": frozenset({"running"}),
    "completed": frozenset(),
    "cancelled": frozenset(),
}

MANIFEST_VERSION = "1.0"
STATISTICS_VERSION = "1.0"


class RankingModelError(Exception):
    """Raised when a ranking record or manifest fails schema validation."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def compute_pass_rate(tests_passed: int, tests_total: int) -> float:
    return tests_passed / tests_total if tests_total > 0 else 0.0


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RankingModelError(f"{label} must be a non-empty string")
    return value


def _require_optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, label)


def _require_nonneg_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RankingModelError(f"{label} must be an integer of 0 or greater")
    return value


def _require_optional_nonneg_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _require_nonneg_int(value, label)


def _require_flag(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise RankingModelError(f"{label} must be true or false")
    return value


def _require_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RankingModelError(f"{label} must be a number")
    return float(value)


# ---------------------------------------------------------------------- CandidateAssessment

_ASSESSMENT_FIELDS = frozenset(
    {
        "ranking_run_id",
        "evaluation_run_id",
        "candidate_run_id",
        "candidate_id",
        "problem_id",
        "correctness",
        "all_tests_passed",
        "pass_rate",
        "score",
        "tests_total",
        "tests_passed",
        "tests_failed",
        "tests_error",
        "tests_skipped",
        "timeout",
        "infrastructure_error",
        "execution_duration_ms",
        "indeterminate_reason",
        "code_sha256",
        "duplicate_of",
        "code_lines",
        "code_chars",
        "strategy",
        "syntax_valid",
        "created_at",
    }
)


@dataclass(frozen=True)
class CandidateAssessment:
    """One candidate's objective correctness classification and score (spec section 24).

    Derived entirely from a Stage 6 :class:`~python_dpo.evaluation.models.EvaluationResult`
    (or the absence of one) plus, for traceability only, secondary candidate metadata that
    must never influence ``score`` (spec sections 20, 22, 23, 57).

    ``correctness == "indeterminate"`` if and only if ``tests_total == 0`` — every path
    that produces indeterminate (no evaluation attempted, an infrastructure failure, or a
    problem with zero declared tests) collapses to no usable test evidence, so this
    invariant is validated rather than assumed.
    """

    ranking_run_id: str
    evaluation_run_id: str
    candidate_run_id: str
    candidate_id: str
    problem_id: str
    correctness: str
    all_tests_passed: bool
    pass_rate: float
    score: float
    tests_total: int
    tests_passed: int
    tests_failed: int
    tests_error: int
    tests_skipped: int
    timeout: bool
    infrastructure_error: bool
    execution_duration_ms: int | None = None
    indeterminate_reason: str | None = None
    # Secondary metadata (spec sections 20, 22, 23, 57): recorded, never scored.
    code_sha256: str | None = None
    duplicate_of: str | None = None
    code_lines: int | None = None
    code_chars: int | None = None
    strategy: str | None = None
    syntax_valid: bool | None = None
    created_at: str = ""

    def __post_init__(self) -> None:
        for name in (
            "ranking_run_id",
            "evaluation_run_id",
            "candidate_run_id",
            "candidate_id",
            "problem_id",
        ):
            _require_text(getattr(self, name), name)

        if self.correctness not in CORRECTNESS_VALUES:
            raise RankingModelError(
                f"correctness must be one of {', '.join(sorted(CORRECTNESS_VALUES))}, "
                f"got {self.correctness!r}"
            )

        for name in ("tests_total", "tests_passed", "tests_failed", "tests_error", "tests_skipped"):
            _require_nonneg_int(getattr(self, name), name)
        for name in ("all_tests_passed", "timeout", "infrastructure_error"):
            _require_flag(getattr(self, name), name)
        _require_float(self.pass_rate, "pass_rate")
        _require_float(self.score, "score")
        _require_optional_nonneg_int(self.execution_duration_ms, "execution_duration_ms")
        _require_optional_text(self.indeterminate_reason, "indeterminate_reason")

        _require_optional_text(self.code_sha256, "code_sha256")
        _require_optional_text(self.duplicate_of, "duplicate_of")
        _require_optional_nonneg_int(self.code_lines, "code_lines")
        _require_optional_nonneg_int(self.code_chars, "code_chars")
        _require_optional_text(self.strategy, "strategy")
        if self.syntax_valid is not None and not isinstance(self.syntax_valid, bool):
            raise RankingModelError("syntax_valid must be true, false, or null")

        # Spec section 68 (carried over from EvaluationResult): the four counts must
        # partition tests_total exactly.
        counted = self.tests_passed + self.tests_failed + self.tests_error + self.tests_skipped
        if counted != self.tests_total:
            raise RankingModelError(
                "tests_passed + tests_failed + tests_error + tests_skipped "
                f"({counted}) must equal tests_total ({self.tests_total})"
            )

        expected_pass_rate = compute_pass_rate(self.tests_passed, self.tests_total)
        if abs(self.pass_rate - expected_pass_rate) > 1e-9:
            raise RankingModelError(
                f"pass_rate does not match tests_passed/tests_total: expected "
                f"{expected_pass_rate}, got {self.pass_rate}"
            )
        # Spec section 18: score starts as exactly pass_rate.
        if abs(self.score - self.pass_rate) > 1e-9:
            raise RankingModelError(
                f"score must equal pass_rate: expected {self.pass_rate}, got {self.score}"
            )

        expected_all_passed = self.tests_total > 0 and self.tests_passed == self.tests_total
        if self.all_tests_passed != expected_all_passed:
            raise RankingModelError(
                f"all_tests_passed must be {expected_all_passed} given tests_passed="
                f"{self.tests_passed}, tests_total={self.tests_total}"
            )

        if self.correctness == "indeterminate":
            if self.indeterminate_reason is None:
                raise RankingModelError(
                    "indeterminate_reason is required when correctness is 'indeterminate'"
                )
            if self.tests_total != 0:
                raise RankingModelError(
                    "an indeterminate assessment must have tests_total == 0 "
                    "(no usable test evidence)"
                )
        else:
            if self.indeterminate_reason is not None:
                raise RankingModelError(
                    "indeterminate_reason must be null unless correctness is 'indeterminate'"
                )
            if self.tests_total == 0:
                raise RankingModelError(
                    f"correctness {self.correctness!r} requires tests_total > 0"
                )
            if self.correctness == "correct" and not self.all_tests_passed:
                raise RankingModelError(
                    "correctness 'correct' requires all_tests_passed to be true"
                )
            if self.correctness == "incorrect" and self.all_tests_passed:
                raise RankingModelError(
                    "correctness 'incorrect' requires all_tests_passed to be false"
                )

        if not self.created_at:
            object.__setattr__(self, "created_at", utc_now_iso())

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranking_run_id": self.ranking_run_id,
            "evaluation_run_id": self.evaluation_run_id,
            "candidate_run_id": self.candidate_run_id,
            "candidate_id": self.candidate_id,
            "problem_id": self.problem_id,
            "correctness": self.correctness,
            "all_tests_passed": self.all_tests_passed,
            "pass_rate": self.pass_rate,
            "score": self.score,
            "tests_total": self.tests_total,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_error": self.tests_error,
            "tests_skipped": self.tests_skipped,
            "timeout": self.timeout,
            "infrastructure_error": self.infrastructure_error,
            "execution_duration_ms": self.execution_duration_ms,
            "indeterminate_reason": self.indeterminate_reason,
            "code_sha256": self.code_sha256,
            "duplicate_of": self.duplicate_of,
            "code_lines": self.code_lines,
            "code_chars": self.code_chars,
            "strategy": self.strategy,
            "syntax_valid": self.syntax_valid,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> CandidateAssessment:
        if not isinstance(data, dict):
            raise RankingModelError("candidate assessment: expected a JSON object")
        unknown = sorted(set(data) - _ASSESSMENT_FIELDS)
        if unknown:
            raise RankingModelError(
                f"candidate assessment: unknown field(s): {', '.join(unknown)}"
            )
        required = {
            "ranking_run_id",
            "evaluation_run_id",
            "candidate_run_id",
            "candidate_id",
            "problem_id",
            "correctness",
            "all_tests_passed",
            "pass_rate",
            "score",
            "tests_total",
            "tests_passed",
            "tests_failed",
            "tests_error",
            "tests_skipped",
            "timeout",
            "infrastructure_error",
        }
        missing = sorted(required - set(data))
        if missing:
            raise RankingModelError(
                f"candidate assessment: missing required field(s): {', '.join(missing)}"
            )
        return cls(**data)


# --------------------------------------------------------------------------- RankingResult

_RANKING_RESULT_FIELDS = frozenset(
    {
        "ranking_run_id",
        "evaluation_run_id",
        "problem_id",
        "candidate_id",
        "rank",
        "score",
        "correctness",
        "pass_rate",
        "all_tests_passed",
        "tie_group",
        "tie_group_size",
        "tied",
        "eligible_for_preference",
        "created_at",
    }
)


@dataclass(frozen=True)
class RankingResult:
    """One candidate's position within its problem's ranking (spec section 25).

    ``rank``/``tie_group`` are ``None`` exactly for indeterminate candidates (spec
    section 39): they are recorded, not dropped, but never participate in an ordering.
    ``eligible_for_preference`` is the *per-candidate* signal spec section 32 defines
    (correct/incorrect -> true, indeterminate -> false) — distinct from
    :attr:`ComparisonResult.comparison_eligible`, which is *per-pair*.
    """

    ranking_run_id: str
    evaluation_run_id: str
    problem_id: str
    candidate_id: str
    score: float
    correctness: str
    pass_rate: float
    all_tests_passed: bool
    eligible_for_preference: bool
    rank: int | None = None
    tie_group: str | None = None
    tie_group_size: int = 0
    tied: bool = False
    created_at: str = ""

    def __post_init__(self) -> None:
        for name in ("ranking_run_id", "evaluation_run_id", "problem_id", "candidate_id"):
            _require_text(getattr(self, name), name)
        if self.correctness not in CORRECTNESS_VALUES:
            raise RankingModelError(
                f"correctness must be one of {', '.join(sorted(CORRECTNESS_VALUES))}, "
                f"got {self.correctness!r}"
            )
        _require_float(self.score, "score")
        _require_float(self.pass_rate, "pass_rate")
        _require_flag(self.all_tests_passed, "all_tests_passed")
        _require_flag(self.eligible_for_preference, "eligible_for_preference")
        _require_flag(self.tied, "tied")
        _require_nonneg_int(self.tie_group_size, "tie_group_size")

        # Spec section 32: exactly correct/incorrect are preference-eligible.
        expected_eligible = self.correctness != "indeterminate"
        if self.eligible_for_preference != expected_eligible:
            raise RankingModelError(
                f"eligible_for_preference must be {expected_eligible} when correctness is "
                f"{self.correctness!r}"
            )

        if self.correctness == "indeterminate":
            if self.rank is not None:
                raise RankingModelError("an indeterminate result must have rank == null")
            if self.tie_group is not None:
                raise RankingModelError("an indeterminate result must have tie_group == null")
            if self.tie_group_size != 0:
                raise RankingModelError("an indeterminate result must have tie_group_size == 0")
            if self.tied:
                raise RankingModelError("an indeterminate result must have tied == false")
        else:
            if self.rank is None or isinstance(self.rank, bool) or self.rank < 1:
                raise RankingModelError(
                    f"correctness {self.correctness!r} requires an integer rank >= 1"
                )
            _require_text(self.tie_group or "", "tie_group")
            if self.tie_group_size < 1:
                raise RankingModelError("tie_group_size must be >= 1 for a ranked result")
            expected_tied = self.tie_group_size > 1
            if self.tied != expected_tied:
                raise RankingModelError(
                    f"tied must be {expected_tied} when tie_group_size is {self.tie_group_size}"
                )

        if not self.created_at:
            object.__setattr__(self, "created_at", utc_now_iso())

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranking_run_id": self.ranking_run_id,
            "evaluation_run_id": self.evaluation_run_id,
            "problem_id": self.problem_id,
            "candidate_id": self.candidate_id,
            "rank": self.rank,
            "score": self.score,
            "correctness": self.correctness,
            "pass_rate": self.pass_rate,
            "all_tests_passed": self.all_tests_passed,
            "tie_group": self.tie_group,
            "tie_group_size": self.tie_group_size,
            "tied": self.tied,
            "eligible_for_preference": self.eligible_for_preference,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> RankingResult:
        if not isinstance(data, dict):
            raise RankingModelError("ranking result: expected a JSON object")
        unknown = sorted(set(data) - _RANKING_RESULT_FIELDS)
        if unknown:
            raise RankingModelError(f"ranking result: unknown field(s): {', '.join(unknown)}")
        required = {
            "ranking_run_id",
            "evaluation_run_id",
            "problem_id",
            "candidate_id",
            "score",
            "correctness",
            "pass_rate",
            "all_tests_passed",
            "eligible_for_preference",
        }
        missing = sorted(required - set(data))
        if missing:
            raise RankingModelError(
                f"ranking result: missing required field(s): {', '.join(missing)}"
            )
        return cls(**data)


# ------------------------------------------------------------------------ ComparisonResult

_COMPARISON_FIELDS = frozenset(
    {
        "ranking_run_id",
        "problem_id",
        "candidate_a",
        "candidate_b",
        "relation",
        "score_a",
        "score_b",
        "score_margin",
        "correctness_a",
        "correctness_b",
        "comparison_eligible",
        "created_at",
    }
)


@dataclass(frozen=True)
class ComparisonResult:
    """One pairwise comparison between two candidates in the same problem (spec section 67).

    Deliberately neutral terminology (spec section 67): ``relation`` is
    ``A_BETTER``/``B_BETTER``/``TIE``/``INDETERMINATE``, never ``chosen``/``rejected``.
    ``comparison_eligible`` is *per-pair* (true only for ``A_BETTER``/``B_BETTER`` — spec
    section 35 forbids inventing a preference from a tie or an indeterminate pair) and is
    a different notion from :attr:`RankingResult.eligible_for_preference`, which is
    per-candidate.
    """

    ranking_run_id: str
    problem_id: str
    candidate_a: str
    candidate_b: str
    relation: str
    score_a: float
    score_b: float
    score_margin: float
    correctness_a: str
    correctness_b: str
    comparison_eligible: bool
    created_at: str = ""

    def __post_init__(self) -> None:
        for name in ("ranking_run_id", "problem_id", "candidate_a", "candidate_b"):
            _require_text(getattr(self, name), name)
        if self.candidate_a == self.candidate_b:
            raise RankingModelError("candidate_a and candidate_b must be different candidates")
        if self.relation not in COMPARISON_RELATIONS:
            raise RankingModelError(
                f"relation must be one of {', '.join(sorted(COMPARISON_RELATIONS))}, "
                f"got {self.relation!r}"
            )
        for name in ("correctness_a", "correctness_b"):
            value = getattr(self, name)
            if value not in CORRECTNESS_VALUES:
                raise RankingModelError(
                    f"{name} must be one of {', '.join(sorted(CORRECTNESS_VALUES))}, "
                    f"got {value!r}"
                )
        _require_float(self.score_a, "score_a")
        _require_float(self.score_b, "score_b")
        _require_float(self.score_margin, "score_margin")
        _require_flag(self.comparison_eligible, "comparison_eligible")

        expected_margin = abs(self.score_a - self.score_b)
        if abs(self.score_margin - expected_margin) > 1e-9:
            raise RankingModelError(
                f"score_margin does not match abs(score_a - score_b): expected "
                f"{expected_margin}, got {self.score_margin}"
            )

        # Spec section 35: a preference exists only for a decisive relation.
        expected_eligible = self.relation in ("A_BETTER", "B_BETTER")
        if self.comparison_eligible != expected_eligible:
            raise RankingModelError(
                f"comparison_eligible must be {expected_eligible} when relation is "
                f"{self.relation!r}"
            )
        if self.relation == "INDETERMINATE" and not (
            self.correctness_a == "indeterminate" or self.correctness_b == "indeterminate"
        ):
            raise RankingModelError(
                "relation 'INDETERMINATE' requires at least one side to be indeterminate"
            )
        if "indeterminate" in (self.correctness_a, self.correctness_b) and self.relation != "INDETERMINATE":
            raise RankingModelError(
                "an indeterminate side must produce relation 'INDETERMINATE'"
            )

        if not self.created_at:
            object.__setattr__(self, "created_at", utc_now_iso())

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranking_run_id": self.ranking_run_id,
            "problem_id": self.problem_id,
            "candidate_a": self.candidate_a,
            "candidate_b": self.candidate_b,
            "relation": self.relation,
            "score_a": self.score_a,
            "score_b": self.score_b,
            "score_margin": self.score_margin,
            "correctness_a": self.correctness_a,
            "correctness_b": self.correctness_b,
            "comparison_eligible": self.comparison_eligible,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ComparisonResult:
        if not isinstance(data, dict):
            raise RankingModelError("comparison result: expected a JSON object")
        unknown = sorted(set(data) - _COMPARISON_FIELDS)
        if unknown:
            raise RankingModelError(f"comparison result: unknown field(s): {', '.join(unknown)}")
        required = _COMPARISON_FIELDS - {"created_at"}
        missing = sorted(required - set(data))
        if missing:
            raise RankingModelError(
                f"comparison result: missing required field(s): {', '.join(missing)}"
            )
        return cls(**data)


# -------------------------------------------------------------------------- RankingManifest

_MANIFEST_FIELDS = frozenset(
    {
        "manifest_version",
        "ranking_run_id",
        "evaluation_run_id",
        "candidate_run_id",
        "status",
        "created_at",
        "started_at",
        "completed_at",
        "ranking_version",
        "scoring_version",
        "comparator_version",
        "scoring_configuration",
        "requested_problem_ids",
        "error",
    }
)

_MANIFEST_ERROR_FIELDS = frozenset({"error_type", "error_message", "timestamp"})


@dataclass(frozen=True)
class RankingManifest:
    """The historical, immutable configuration snapshot for one ranking run (spec
    section 27).

    Mirrors :class:`python_dpo.evaluation.models.EvaluationManifest`: the source of truth
    for what a ranking run covers and which algorithm versions produced it, never
    re-derived from the current source tree (spec section 45).
    """

    ranking_run_id: str
    evaluation_run_id: str
    candidate_run_id: str
    status: str
    created_at: str
    ranking_version: str
    scoring_version: str
    comparator_version: str
    requested_problem_ids: tuple[str, ...]
    manifest_version: str = MANIFEST_VERSION
    started_at: str | None = None
    completed_at: str | None = None
    scoring_configuration: dict[str, Any] = dataclasses.field(default_factory=dict)
    error: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_text(self.manifest_version, "manifest_version")
        _require_text(self.ranking_run_id, "ranking_run_id")
        _require_text(self.evaluation_run_id, "evaluation_run_id")
        _require_text(self.candidate_run_id, "candidate_run_id")
        if self.status not in RANKING_RUN_STATUSES:
            raise RankingModelError(
                f"status must be one of {', '.join(sorted(RANKING_RUN_STATUSES))}, "
                f"got {self.status!r}"
            )
        _require_text(self.created_at, "created_at")
        _require_optional_text(self.started_at, "started_at")
        _require_optional_text(self.completed_at, "completed_at")
        _require_text(self.ranking_version, "ranking_version")
        _require_text(self.scoring_version, "scoring_version")
        _require_text(self.comparator_version, "comparator_version")

        if not isinstance(self.scoring_configuration, dict):
            raise RankingModelError("scoring_configuration must be a mapping")

        if isinstance(self.requested_problem_ids, (str, bytes)) or not isinstance(
            self.requested_problem_ids, (list, tuple)
        ):
            raise RankingModelError("requested_problem_ids must be a sequence of problem ids")
        object.__setattr__(self, "requested_problem_ids", tuple(self.requested_problem_ids))
        if not self.requested_problem_ids:
            raise RankingModelError("requested_problem_ids must not be empty")
        if len(set(self.requested_problem_ids)) != len(self.requested_problem_ids):
            raise RankingModelError("requested_problem_ids must not contain duplicates")

        if self.error is not None:
            if not isinstance(self.error, dict):
                raise RankingModelError("error must be a mapping or null")
            unknown = sorted(set(self.error) - _MANIFEST_ERROR_FIELDS)
            if unknown:
                raise RankingModelError(f"error: unknown field(s): {', '.join(unknown)}")
            missing = sorted(_MANIFEST_ERROR_FIELDS - set(self.error))
            if missing:
                raise RankingModelError(f"error: missing required field(s): {', '.join(missing)}")

    def with_status(
        self,
        status: str,
        *,
        started_at: str | None = None,
        completed_at: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> RankingManifest:
        allowed = RANKING_RUN_STATUS_TRANSITIONS.get(self.status, frozenset())
        if status != self.status and status not in allowed:
            raise RankingModelError(
                f"cannot transition ranking run status from {self.status!r} to {status!r}"
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
            "ranking_run_id": self.ranking_run_id,
            "evaluation_run_id": self.evaluation_run_id,
            "candidate_run_id": self.candidate_run_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "ranking_version": self.ranking_version,
            "scoring_version": self.scoring_version,
            "comparator_version": self.comparator_version,
            "scoring_configuration": self.scoring_configuration,
            "requested_problem_ids": list(self.requested_problem_ids),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Any) -> RankingManifest:
        if not isinstance(data, dict):
            raise RankingModelError("ranking manifest: expected a JSON object")
        unknown = sorted(set(data) - _MANIFEST_FIELDS)
        if unknown:
            raise RankingModelError(f"ranking manifest: unknown field(s): {', '.join(unknown)}")
        required = {
            "ranking_run_id",
            "evaluation_run_id",
            "candidate_run_id",
            "status",
            "created_at",
            "ranking_version",
            "scoring_version",
            "comparator_version",
            "requested_problem_ids",
        }
        missing = sorted(required - set(data))
        if missing:
            raise RankingModelError(
                f"ranking manifest: missing required field(s): {', '.join(missing)}"
            )
        return cls(
            manifest_version=data.get("manifest_version", MANIFEST_VERSION),
            ranking_run_id=data["ranking_run_id"],
            evaluation_run_id=data["evaluation_run_id"],
            candidate_run_id=data["candidate_run_id"],
            status=data["status"],
            created_at=data["created_at"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            ranking_version=data["ranking_version"],
            scoring_version=data["scoring_version"],
            comparator_version=data["comparator_version"],
            scoring_configuration=data.get("scoring_configuration") or {},
            requested_problem_ids=tuple(data["requested_problem_ids"]),
            error=data.get("error"),
        )


# ----------------------------------------------------------------------- RankingStatistics

_STATISTICS_FIELDS = frozenset(
    {
        "statistics_version",
        "ranking_run_id",
        "problems",
        "candidates",
        "correct",
        "incorrect",
        "indeterminate",
        "fully_correct",
        "partially_correct",
        "zero_test_pass",
        "tied_candidates",
        "preference_eligible_candidates",
        "per_problem",
        "computed_at",
    }
)


@dataclass(frozen=True)
class RankingStatistics:
    """Ranking run statistics (spec sections 40, 41) — always reconstructable from
    persisted records, never trusted from an in-memory counter.
    """

    ranking_run_id: str
    problems: int
    candidates: int
    correct: int
    incorrect: int
    indeterminate: int
    fully_correct: int
    partially_correct: int
    zero_test_pass: int
    tied_candidates: int
    preference_eligible_candidates: int
    per_problem: dict[str, dict[str, int]]
    computed_at: str
    statistics_version: str = STATISTICS_VERSION

    def __post_init__(self) -> None:
        _require_text(self.statistics_version, "statistics_version")
        _require_text(self.ranking_run_id, "ranking_run_id")
        for name in (
            "problems",
            "candidates",
            "correct",
            "incorrect",
            "indeterminate",
            "fully_correct",
            "partially_correct",
            "zero_test_pass",
            "tied_candidates",
            "preference_eligible_candidates",
        ):
            _require_nonneg_int(getattr(self, name), name)
        if not isinstance(self.per_problem, dict):
            raise RankingModelError("per_problem must be a mapping")
        _require_text(self.computed_at, "computed_at")

        if self.correct + self.incorrect + self.indeterminate != self.candidates:
            raise RankingModelError(
                "correct + incorrect + indeterminate must equal candidates"
            )
        if self.fully_correct != self.correct:
            raise RankingModelError("fully_correct must equal correct (spec section 41)")
        if self.partially_correct + self.zero_test_pass != self.incorrect:
            raise RankingModelError(
                "partially_correct + zero_test_pass must equal incorrect"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistics_version": self.statistics_version,
            "ranking_run_id": self.ranking_run_id,
            "problems": self.problems,
            "candidates": self.candidates,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "indeterminate": self.indeterminate,
            "fully_correct": self.fully_correct,
            "partially_correct": self.partially_correct,
            "zero_test_pass": self.zero_test_pass,
            "tied_candidates": self.tied_candidates,
            "preference_eligible_candidates": self.preference_eligible_candidates,
            "per_problem": self.per_problem,
            "computed_at": self.computed_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> RankingStatistics:
        if not isinstance(data, dict):
            raise RankingModelError("ranking statistics: expected a JSON object")
        unknown = sorted(set(data) - _STATISTICS_FIELDS)
        if unknown:
            raise RankingModelError(
                f"ranking statistics: unknown field(s): {', '.join(unknown)}"
            )
        required = _STATISTICS_FIELDS - {"statistics_version"}
        missing = sorted(required - set(data))
        if missing:
            raise RankingModelError(
                f"ranking statistics: missing required field(s): {', '.join(missing)}"
            )
        kwargs = dict(data)
        kwargs.setdefault("statistics_version", STATISTICS_VERSION)
        return cls(**kwargs)

    @classmethod
    def from_records(
        cls,
        manifest: RankingManifest,
        assessments: list[CandidateAssessment],
        rankings: list[RankingResult],
        *,
        computed_at: str | None = None,
    ) -> RankingStatistics:
        correct = incorrect = indeterminate = 0
        fully_correct = partially_correct = zero_test_pass = 0
        problems: set[str] = set()
        per_problem: dict[str, dict[str, int]] = {}

        for assessment in assessments:
            problems.add(assessment.problem_id)
            bucket = per_problem.setdefault(
                assessment.problem_id,
                {"total": 0, "fully_correct": 0, "partially_correct": 0, "zero_test_pass": 0, "indeterminate": 0},
            )
            bucket["total"] += 1

            if assessment.correctness == "correct":
                correct += 1
                fully_correct += 1
                bucket["fully_correct"] += 1
            elif assessment.correctness == "incorrect":
                incorrect += 1
                if assessment.tests_passed > 0:
                    partially_correct += 1
                    bucket["partially_correct"] += 1
                else:
                    zero_test_pass += 1
                    bucket["zero_test_pass"] += 1
            else:
                indeterminate += 1
                bucket["indeterminate"] += 1

        tied_candidates = sum(1 for r in rankings if r.tied)
        preference_eligible = sum(1 for r in rankings if r.eligible_for_preference)

        return cls(
            ranking_run_id=manifest.ranking_run_id,
            problems=len(problems),
            candidates=len(assessments),
            correct=correct,
            incorrect=incorrect,
            indeterminate=indeterminate,
            fully_correct=fully_correct,
            partially_correct=partially_correct,
            zero_test_pass=zero_test_pass,
            tied_candidates=tied_candidates,
            preference_eligible_candidates=preference_eligible,
            per_problem=per_problem,
            computed_at=computed_at or utc_now_iso(),
        )


__all__ = [
    "COMPARISON_RELATIONS",
    "CORRECTNESS_VALUES",
    "MANIFEST_VERSION",
    "RANKING_RUN_STATUSES",
    "RANKING_RUN_STATUS_TRANSITIONS",
    "STATISTICS_VERSION",
    "CandidateAssessment",
    "ComparisonResult",
    "RankingManifest",
    "RankingModelError",
    "RankingResult",
    "RankingStatistics",
    "compute_pass_rate",
    "utc_now_iso",
]
