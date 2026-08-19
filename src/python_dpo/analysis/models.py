"""The persisted Stage 11 schema (spec 11).

House style: frozen dataclasses validating in ``__post_init__``, explicit ``to_dict`` /
``from_dict`` rejecting unknown and missing fields -- matching
:mod:`python_dpo.model_evaluation.models` and :mod:`python_dpo.pipeline.manifest`.

:class:`Recommendation` is the load-bearing one. Sections 55 and 103 require every
recommendation to carry non-empty evidence and a real hypothesis; both are enforced in
``__post_init__``, so a recommendation without them cannot be constructed at all. That
makes the rule structural rather than something a future contributor has to remember.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .errors import AnalysisStoreError
from .taxonomy import ERROR_CATEGORIES

# Section 18's outcome set.
PROBLEM_OUTCOMES = (
    "complete_improvement",
    "partial_improvement",
    "unchanged",
    "partial_regression",
    "complete_regression",
)

SEVERITIES = ("none", "low", "medium", "high")

# Section 37's coverage verdicts. `not_in_benchmark` and `absent_from_both` exist because
# the ratio is undefined in those cases -- see CategoryGap.coverage_ratio.
COVERAGE_VERDICTS = (
    "underrepresented",
    "balanced",
    "overrepresented",
    "not_in_benchmark",
    "absent_from_both",
)

# Section 58's closed recommendation set.
RECOMMENDATION_CATEGORIES = (
    "add_data",
    "increase_problem_difficulty",
    "improve_problem_diversity",
    "refine_preference_pairs",
    "adjust_dpo_hyperparameters",
    "adjust_generation_parameters",
    "investigate_regression",
    "investigate_mode_collapse",
    "expand_benchmark",
    "no_action",
)

CONFIDENCE_LEVELS = ("low", "medium", "high")

# Section 90's five-value iteration decision.
ITERATION_DECISIONS = (
    "insufficient_evidence",
    "refine_data",
    "adjust_training",
    "expand_benchmark",
    "accept_model",
)

ANALYSIS_RUN_STATUSES = frozenset(
    {"created", "running", "completed", "failed", "interrupted", "cancelled"}
)
ANALYSIS_RUN_STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    "created": frozenset({"running", "cancelled"}),
    "running": frozenset({"completed", "failed", "interrupted", "cancelled"}),
    "interrupted": frozenset({"running", "cancelled"}),
    "failed": frozenset({"running"}),
    "completed": frozenset(),
    "cancelled": frozenset(),
}

MANIFEST_VERSION = "analysis_manifest_v1"
SUMMARY_VERSION = "analysis_summary_v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnalysisStoreError(f"{label} must be a non-empty string")
    return value


def _require_optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, label)


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisStoreError(f"{label} must be a number")
    return float(value)


def _require_nonneg_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AnalysisStoreError(f"{label} must be an integer of 0 or greater")
    return value


def _require_choice(value: Any, allowed: tuple[str, ...], label: str) -> str:
    if value not in allowed:
        raise AnalysisStoreError(f"{label} must be one of {', '.join(allowed)}, got {value!r}")
    return value


def _check_fields(data: Any, allowed: set[str], required: set[str], label: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise AnalysisStoreError(f"{label}: expected a JSON object")
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise AnalysisStoreError(f"{label}: unknown field(s): {', '.join(unknown)}")
    missing = sorted(required - set(data))
    if missing:
        raise AnalysisStoreError(f"{label}: missing required field(s): {', '.join(missing)}")
    return data


# ----------------------------------------------------------------------------- ErrorProfile


@dataclass(frozen=True)
class ErrorProfile:
    """Sections 14, 15: one variant's failure counts plus the hierarchical breakdown."""

    model_variant: str
    total_samples: int
    passed: int
    counts_by_category: dict[str, int] = field(default_factory=dict)
    counts_by_subcategory: dict[str, int] = field(default_factory=dict)
    infrastructure_errors: int = 0

    def __post_init__(self) -> None:
        _require_text(self.model_variant, "model_variant")
        _require_nonneg_int(self.total_samples, "total_samples")
        _require_nonneg_int(self.passed, "passed")
        for name in ("counts_by_category", "counts_by_subcategory"):
            if not isinstance(getattr(self, name), dict):
                raise AnalysisStoreError(f"{name} must be a mapping")

    @property
    def evaluable_samples(self) -> int:
        """Section 120: infrastructure errors are excluded from correctness rates."""
        return max(0, self.total_samples - self.infrastructure_errors)

    def rate_for(self, category: str) -> float:
        denominator = self.evaluable_samples
        if denominator == 0:
            return 0.0
        return self.counts_by_category.get(category, 0) / denominator

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_variant": self.model_variant,
            "total_samples": self.total_samples,
            "passed": self.passed,
            "counts_by_category": dict(self.counts_by_category),
            "counts_by_subcategory": dict(self.counts_by_subcategory),
            "infrastructure_errors": self.infrastructure_errors,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ErrorProfile:
        allowed = {
            "model_variant", "total_samples", "passed", "counts_by_category",
            "counts_by_subcategory", "infrastructure_errors",
        }
        _check_fields(data, allowed, {"model_variant", "total_samples"}, "error profile")
        return cls(
            model_variant=data["model_variant"],
            total_samples=data["total_samples"],
            passed=data.get("passed", 0),
            counts_by_category=data.get("counts_by_category") or {},
            counts_by_subcategory=data.get("counts_by_subcategory") or {},
            infrastructure_errors=data.get("infrastructure_errors", 0),
        )


@dataclass(frozen=True)
class ErrorRateComparison:
    """Section 16: per-category base-vs-DPO rates."""

    category: str
    base_rate: float
    dpo_rate: float

    def __post_init__(self) -> None:
        if self.category not in ERROR_CATEGORIES:
            raise AnalysisStoreError(f"unknown error category {self.category!r}")
        _require_number(self.base_rate, "base_rate")
        _require_number(self.dpo_rate, "dpo_rate")

    @property
    def delta(self) -> float:
        return self.dpo_rate - self.base_rate

    @property
    def relative_delta(self) -> float | None:
        """``None`` rather than infinity when the base rate is zero -- JSON cannot carry
        an infinity, and "infinitely worse than never" is not a meaningful figure."""
        if self.base_rate == 0:
            return None
        return (self.dpo_rate - self.base_rate) / self.base_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "base_rate": self.base_rate,
            "dpo_rate": self.dpo_rate,
            "delta": self.delta,
            "relative_delta": self.relative_delta,
        }


# --------------------------------------------------------------------------- ProblemOutcome


@dataclass(frozen=True)
class ProblemOutcome:
    """Sections 17-25, 50-52: one benchmark problem's base-vs-DPO outcome."""

    problem_id: str
    outcome: str
    base_best_score: float
    dpo_best_score: float
    base_solved: bool
    dpo_solved: bool
    severity: str = "none"
    category: str | None = None
    difficulty: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.problem_id, "problem_id")
        _require_choice(self.outcome, PROBLEM_OUTCOMES, "outcome")
        _require_choice(self.severity, SEVERITIES, "severity")
        _require_number(self.base_best_score, "base_best_score")
        _require_number(self.dpo_best_score, "dpo_best_score")

    @property
    def delta(self) -> float:
        return self.dpo_best_score - self.base_best_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "outcome": self.outcome,
            "base_best_score": self.base_best_score,
            "dpo_best_score": self.dpo_best_score,
            "delta": self.delta,
            "base_solved": self.base_solved,
            "dpo_solved": self.dpo_solved,
            "severity": self.severity,
            "category": self.category,
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ProblemOutcome:
        allowed = {
            "problem_id", "outcome", "base_best_score", "dpo_best_score", "delta",
            "base_solved", "dpo_solved", "severity", "category", "difficulty",
        }
        _check_fields(data, allowed, {"problem_id", "outcome"}, "problem outcome")
        return cls(
            problem_id=data["problem_id"],
            outcome=data["outcome"],
            base_best_score=data.get("base_best_score", 0.0),
            dpo_best_score=data.get("dpo_best_score", 0.0),
            base_solved=data.get("base_solved", False),
            dpo_solved=data.get("dpo_solved", False),
            severity=data.get("severity", "none"),
            category=data.get("category"),
            difficulty=data.get("difficulty"),
        )


# -------------------------------------------------------------------------- TestFailureStat


@dataclass(frozen=True)
class TestFailureStat:
    """Sections 45-49: one test case's per-variant failure frequency."""

    problem_id: str
    test_case_id: str
    base_failures: int
    base_runs: int
    dpo_failures: int
    dpo_runs: int
    hard_test: bool = False
    dpo_specific: bool = False
    base_specific: bool = False

    def __post_init__(self) -> None:
        _require_text(self.problem_id, "problem_id")
        _require_text(self.test_case_id, "test_case_id")

    @property
    def base_failure_rate(self) -> float:
        return self.base_failures / self.base_runs if self.base_runs else 0.0

    @property
    def dpo_failure_rate(self) -> float:
        return self.dpo_failures / self.dpo_runs if self.dpo_runs else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "test_case_id": self.test_case_id,
            "base_failures": self.base_failures,
            "base_runs": self.base_runs,
            "base_failure_rate": self.base_failure_rate,
            "dpo_failures": self.dpo_failures,
            "dpo_runs": self.dpo_runs,
            "dpo_failure_rate": self.dpo_failure_rate,
            "hard_test": self.hard_test,
            "dpo_specific": self.dpo_specific,
            "base_specific": self.base_specific,
        }


# --------------------------------------------------------------------------- DiversityReport


@dataclass(frozen=True)
class DiversityReport:
    """Sections 26-31, 88: unique-output ratios per variant."""

    base_unique: int
    base_total: int
    dpo_unique: int
    dpo_total: int
    per_problem: dict[str, dict[str, float]] = field(default_factory=dict)
    mode_collapse_warning: bool = False

    @property
    def base_diversity(self) -> float:
        return self.base_unique / self.base_total if self.base_total else 0.0

    @property
    def dpo_diversity(self) -> float:
        return self.dpo_unique / self.dpo_total if self.dpo_total else 0.0

    @property
    def relative_change(self) -> float | None:
        if self.base_diversity == 0:
            return None
        return (self.dpo_diversity - self.base_diversity) / self.base_diversity

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_unique": self.base_unique,
            "base_total": self.base_total,
            "base_diversity": self.base_diversity,
            "dpo_unique": self.dpo_unique,
            "dpo_total": self.dpo_total,
            "dpo_diversity": self.dpo_diversity,
            "relative_change": self.relative_change,
            "per_problem": self.per_problem,
            "mode_collapse_warning": self.mode_collapse_warning,
        }


# ------------------------------------------------------------------------------- gap models


@dataclass(frozen=True)
class CategoryGap:
    """Section 37: one category's training-vs-benchmark representation.

    ``coverage_ratio`` is ``float | None``. The real data produces two cases arithmetic
    cannot express and JSON cannot carry: a category present in training but absent from
    the benchmark divides by zero, and one absent from both is 0/0. Rather than writing
    ``Infinity`` or ``NaN`` into a file no downstream reader could parse, the ratio is
    ``None`` and the verdict enum carries the meaning.
    """

    name: str
    training_share: float
    benchmark_share: float
    coverage_ratio: float | None
    verdict: str

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_number(self.training_share, "training_share")
        _require_number(self.benchmark_share, "benchmark_share")
        _require_choice(self.verdict, COVERAGE_VERDICTS, "verdict")
        if self.coverage_ratio is not None:
            _require_number(self.coverage_ratio, "coverage_ratio")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "training_share": self.training_share,
            "benchmark_share": self.benchmark_share,
            "coverage_ratio": self.coverage_ratio,
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class Recommendation:
    """Sections 54-58, 102, 103.

    ``evidence`` and ``hypothesis`` are validated non-empty: section 55 requires every
    recommendation to cite evidence and section 103 requires a real hypothesis rather than
    a bare "change X". Enforcing both here means an unsupported recommendation cannot be
    constructed, let alone written to a file.
    """

    category: str
    hypothesis: str
    evidence: dict[str, Any]
    confidence: str
    expected_impact: float
    evidence_strength: float
    implementation_cost: float
    recommendation_score: float = 0.0

    def __post_init__(self) -> None:
        _require_choice(self.category, RECOMMENDATION_CATEGORIES, "category")
        _require_choice(self.confidence, CONFIDENCE_LEVELS, "confidence")
        if not isinstance(self.hypothesis, str) or not self.hypothesis.strip():
            raise AnalysisStoreError(
                "recommendation.hypothesis must be a non-empty string (spec section 103): "
                "a recommendation without a stated hypothesis is not actionable"
            )
        if not isinstance(self.evidence, dict) or not self.evidence:
            raise AnalysisStoreError(
                "recommendation.evidence must be a non-empty mapping (spec section 55): "
                "a recommendation without evidence is an opinion"
            )
        for name in ("expected_impact", "evidence_strength", "implementation_cost"):
            value = _require_number(getattr(self, name), f"recommendation.{name}")
            if not 0.0 <= value <= 1.0:
                raise AnalysisStoreError(f"recommendation.{name} must be between 0 and 1")

    def scored(self, weights: dict[str, float]) -> Recommendation:
        """Section 57: implementation cost lowers the score, the other two raise it."""
        score = (
            weights["expected_impact"] * self.expected_impact
            + weights["evidence_strength"] * self.evidence_strength
            + weights["implementation_cost"] * (1.0 - self.implementation_cost)
        )
        return Recommendation(
            category=self.category, hypothesis=self.hypothesis, evidence=dict(self.evidence),
            confidence=self.confidence, expected_impact=self.expected_impact,
            evidence_strength=self.evidence_strength,
            implementation_cost=self.implementation_cost,
            recommendation_score=round(score, 6),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "hypothesis": self.hypothesis,
            "evidence": dict(self.evidence),
            "confidence": self.confidence,
            "expected_impact": self.expected_impact,
            "evidence_strength": self.evidence_strength,
            "implementation_cost": self.implementation_cost,
            "recommendation_score": self.recommendation_score,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Recommendation:
        allowed = {
            "category", "hypothesis", "evidence", "confidence", "expected_impact",
            "evidence_strength", "implementation_cost", "recommendation_score",
        }
        _check_fields(data, allowed, {"category", "hypothesis", "evidence"}, "recommendation")
        return cls(
            category=data["category"], hypothesis=data["hypothesis"],
            evidence=data["evidence"], confidence=data.get("confidence", "low"),
            expected_impact=data.get("expected_impact", 0.0),
            evidence_strength=data.get("evidence_strength", 0.0),
            implementation_cost=data.get("implementation_cost", 0.0),
            recommendation_score=data.get("recommendation_score", 0.0),
        )


# ------------------------------------------------------------------------ lineage, manifest


@dataclass(frozen=True)
class ExperimentLineage:
    """Section 7's mandatory chain, resolved by hopping manifests."""

    evaluation_run_id: str
    training_run_id: str
    preference_run_id: str
    ranking_run_id: str
    candidate_run_id: str
    sandbox_evaluation_run_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "evaluation_run_id", "training_run_id", "preference_run_id",
            "ranking_run_id", "candidate_run_id",
        ):
            _require_text(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_run_id": self.evaluation_run_id,
            "training_run_id": self.training_run_id,
            "preference_run_id": self.preference_run_id,
            "ranking_run_id": self.ranking_run_id,
            "candidate_run_id": self.candidate_run_id,
            "sandbox_evaluation_run_id": self.sandbox_evaluation_run_id,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ExperimentLineage:
        allowed = {
            "evaluation_run_id", "training_run_id", "preference_run_id", "ranking_run_id",
            "candidate_run_id", "sandbox_evaluation_run_id",
        }
        required = {
            "evaluation_run_id", "training_run_id", "preference_run_id",
            "ranking_run_id", "candidate_run_id",
        }
        _check_fields(data, allowed, required, "lineage")
        return cls(
            evaluation_run_id=data["evaluation_run_id"],
            training_run_id=data["training_run_id"],
            preference_run_id=data["preference_run_id"],
            ranking_run_id=data["ranking_run_id"],
            candidate_run_id=data["candidate_run_id"],
            sandbox_evaluation_run_id=data.get("sandbox_evaluation_run_id"),
        )


@dataclass(frozen=True)
class AnalysisManifest:
    """The analysis run's own manifest (spec section 8)."""

    analysis_run_id: str
    status: str
    created_at: str
    lineage: ExperimentLineage
    benchmark_version: str | None = None
    manifest_version: str = MANIFEST_VERSION
    started_at: str | None = None
    completed_at: str | None = None
    error: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_text(self.analysis_run_id, "analysis_run_id")
        if self.status not in ANALYSIS_RUN_STATUSES:
            raise AnalysisStoreError(
                f"status must be one of {', '.join(sorted(ANALYSIS_RUN_STATUSES))}"
            )
        _require_text(self.created_at, "created_at")
        _require_optional_text(self.started_at, "started_at")
        _require_optional_text(self.completed_at, "completed_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "analysis_run_id": self.analysis_run_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "lineage": self.lineage.to_dict(),
            "benchmark_version": self.benchmark_version,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Any) -> AnalysisManifest:
        allowed = {
            "manifest_version", "analysis_run_id", "status", "created_at", "started_at",
            "completed_at", "lineage", "benchmark_version", "error",
        }
        _check_fields(data, allowed, {"analysis_run_id", "status", "created_at", "lineage"},
                      "analysis manifest")
        return cls(
            manifest_version=data.get("manifest_version", MANIFEST_VERSION),
            analysis_run_id=data["analysis_run_id"],
            status=data["status"],
            created_at=data["created_at"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            lineage=ExperimentLineage.from_dict(data["lineage"]),
            benchmark_version=data.get("benchmark_version"),
            error=data.get("error"),
        )

    def with_status(self, status: str, **changes: Any) -> AnalysisManifest:
        if status not in ANALYSIS_RUN_STATUSES:
            raise AnalysisStoreError(f"unknown status {status!r}")
        if status not in ANALYSIS_RUN_STATUS_TRANSITIONS[self.status]:
            raise AnalysisStoreError(
                f"illegal analysis status transition: {self.status!r} -> {status!r}"
            )
        data = self.to_dict()
        data["status"] = status
        data.update({k: v for k, v in changes.items() if v is not None})
        return AnalysisManifest.from_dict(data)


__all__ = [
    "ANALYSIS_RUN_STATUSES",
    "ANALYSIS_RUN_STATUS_TRANSITIONS",
    "CONFIDENCE_LEVELS",
    "COVERAGE_VERDICTS",
    "ITERATION_DECISIONS",
    "MANIFEST_VERSION",
    "PROBLEM_OUTCOMES",
    "RECOMMENDATION_CATEGORIES",
    "SEVERITIES",
    "SUMMARY_VERSION",
    "AnalysisManifest",
    "CategoryGap",
    "DiversityReport",
    "ErrorProfile",
    "ErrorRateComparison",
    "ExperimentLineage",
    "ProblemOutcome",
    "Recommendation",
    "TestFailureStat",
    "utc_now_iso",
]
