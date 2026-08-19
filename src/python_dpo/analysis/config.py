"""Analysis thresholds, loaded from ``configs/analysis/python_analysis.yaml`` (spec 11).

A standalone file rather than a ``analysis:`` section in the root ``config.yaml``, matching
Stage 9's ``configs/training/dpo_qlora.yaml`` and Stage 10's
``configs/evaluation/python_eval.yaml``: these are experiment tunables, not project paths.

Every threshold the spec calls configurable lives here, so a verdict can never depend on a
number hard-coded in a comparison somewhere in the analysis code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import AnalysisConfigError

DEFAULT_CONFIG_PATH = Path("configs/analysis/python_analysis.yaml")


def _require_ratio(value: Any, label: str, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisConfigError(f"{label} must be a number")
    value = float(value)
    if not minimum <= value <= maximum:
        raise AnalysisConfigError(f"{label} must be between {minimum} and {maximum}, got {value}")
    return value


def _require_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise AnalysisConfigError(f"{label} must be a positive number")
    return float(value)


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AnalysisConfigError(f"{label} must be an integer of 1 or greater")
    return value


@dataclass(frozen=True)
class Thresholds:
    """Section 37, 47, 48, 49, 52, 88 -- every comparison constant in the stage."""

    regression_threshold: float = 0.2
    coverage_underrepresented: float = 0.5
    coverage_overrepresented: float = 2.0
    mode_collapse_reduction: float = 0.2
    hard_test_failure_rate: float = 0.5
    variant_specific_test_delta: float = 0.2

    def __post_init__(self) -> None:
        _require_ratio(self.regression_threshold, "thresholds.regression_threshold")
        _require_positive(self.coverage_underrepresented, "thresholds.coverage_underrepresented")
        _require_positive(self.coverage_overrepresented, "thresholds.coverage_overrepresented")
        _require_ratio(self.mode_collapse_reduction, "thresholds.mode_collapse_reduction")
        _require_ratio(self.hard_test_failure_rate, "thresholds.hard_test_failure_rate")
        _require_ratio(self.variant_specific_test_delta, "thresholds.variant_specific_test_delta")
        if self.coverage_underrepresented >= self.coverage_overrepresented:
            raise AnalysisConfigError(
                "thresholds.coverage_underrepresented must be less than "
                "thresholds.coverage_overrepresented"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "regression_threshold": self.regression_threshold,
            "coverage_underrepresented": self.coverage_underrepresented,
            "coverage_overrepresented": self.coverage_overrepresented,
            "mode_collapse_reduction": self.mode_collapse_reduction,
            "hard_test_failure_rate": self.hard_test_failure_rate,
            "variant_specific_test_delta": self.variant_specific_test_delta,
        }

    @classmethod
    def from_mapping(cls, data: Any) -> Thresholds:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise AnalysisConfigError("thresholds must be a mapping")
        unknown = sorted(set(data) - set(cls().to_dict()))
        if unknown:
            raise AnalysisConfigError(f"thresholds: unknown key(s): {', '.join(unknown)}")
        return cls(**{**cls().to_dict(), **data})


@dataclass(frozen=True)
class MinimumEvidence:
    """Section 95's gates. Below either of these, no outcome-level conclusion is reported
    regardless of what the other analyses found."""

    benchmark_problems: int = 30
    max_ci_width: float = 0.15

    def __post_init__(self) -> None:
        _require_positive_int(self.benchmark_problems, "minimum_evidence.benchmark_problems")
        _require_ratio(self.max_ci_width, "minimum_evidence.max_ci_width")

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_problems": self.benchmark_problems,
            "max_ci_width": self.max_ci_width,
        }

    @classmethod
    def from_mapping(cls, data: Any) -> MinimumEvidence:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise AnalysisConfigError("minimum_evidence must be a mapping")
        unknown = sorted(set(data) - set(cls().to_dict()))
        if unknown:
            raise AnalysisConfigError(f"minimum_evidence: unknown key(s): {', '.join(unknown)}")
        return cls(**{**cls().to_dict(), **data})


@dataclass(frozen=True)
class RecommendationSettings:
    """Section 57 -- how recommendations are scored and how many survive."""

    max_recommendations: int = 10
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "expected_impact": 0.5,
            "evidence_strength": 0.3,
            "implementation_cost": 0.2,
        }
    )

    _WEIGHT_KEYS = ("expected_impact", "evidence_strength", "implementation_cost")

    def __post_init__(self) -> None:
        _require_positive_int(self.max_recommendations, "recommendations.max_recommendations")
        if not isinstance(self.weights, dict):
            raise AnalysisConfigError("recommendations.weights must be a mapping")
        missing = sorted(set(self._WEIGHT_KEYS) - set(self.weights))
        if missing:
            raise AnalysisConfigError(
                f"recommendations.weights: missing key(s): {', '.join(missing)}"
            )
        unknown = sorted(set(self.weights) - set(self._WEIGHT_KEYS))
        if unknown:
            raise AnalysisConfigError(
                f"recommendations.weights: unknown key(s): {', '.join(unknown)}"
            )
        for key in self._WEIGHT_KEYS:
            _require_ratio(self.weights[key], f"recommendations.weights.{key}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_recommendations": self.max_recommendations,
            "weights": dict(self.weights),
        }

    @classmethod
    def from_mapping(cls, data: Any) -> RecommendationSettings:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise AnalysisConfigError("recommendations must be a mapping")
        unknown = sorted(set(data) - {"max_recommendations", "weights"})
        if unknown:
            raise AnalysisConfigError(f"recommendations: unknown key(s): {', '.join(unknown)}")
        defaults = cls()
        return cls(
            max_recommendations=data.get("max_recommendations", defaults.max_recommendations),
            weights=data.get("weights", dict(defaults.weights)),
        )


@dataclass(frozen=True)
class RefinementSettings:
    """Section 71 -- which Stage 8 pairs survive into the refined dataset."""

    minimum_score_margin: float = 0.2
    drop_duplicate_code: bool = True
    drop_infrastructure_errors: bool = True

    def __post_init__(self) -> None:
        _require_ratio(self.minimum_score_margin, "refinement.minimum_score_margin")
        for name in ("drop_duplicate_code", "drop_infrastructure_errors"):
            if not isinstance(getattr(self, name), bool):
                raise AnalysisConfigError(f"refinement.{name} must be true or false")

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_score_margin": self.minimum_score_margin,
            "drop_duplicate_code": self.drop_duplicate_code,
            "drop_infrastructure_errors": self.drop_infrastructure_errors,
        }

    @classmethod
    def from_mapping(cls, data: Any) -> RefinementSettings:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise AnalysisConfigError("refinement must be a mapping")
        unknown = sorted(set(data) - set(cls().to_dict()))
        if unknown:
            raise AnalysisConfigError(f"refinement: unknown key(s): {', '.join(unknown)}")
        return cls(**{**cls().to_dict(), **data})


@dataclass(frozen=True)
class AnalysisConfig:
    """The fully resolved Stage 11 configuration."""

    thresholds: Thresholds = field(default_factory=Thresholds)
    minimum_evidence: MinimumEvidence = field(default_factory=MinimumEvidence)
    recommendations: RecommendationSettings = field(default_factory=RecommendationSettings)
    refinement: RefinementSettings = field(default_factory=RefinementSettings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds": self.thresholds.to_dict(),
            "minimum_evidence": self.minimum_evidence.to_dict(),
            "recommendations": self.recommendations.to_dict(),
            "refinement": self.refinement.to_dict(),
        }

    @classmethod
    def from_mapping(cls, data: Any) -> AnalysisConfig:
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise AnalysisConfigError("analysis config: root must be a mapping")
        unknown = sorted(
            set(data) - {"thresholds", "minimum_evidence", "recommendations", "refinement"}
        )
        if unknown:
            raise AnalysisConfigError(
                f"analysis config: unknown top-level key(s): {', '.join(unknown)}"
            )
        return cls(
            thresholds=Thresholds.from_mapping(data.get("thresholds")),
            minimum_evidence=MinimumEvidence.from_mapping(data.get("minimum_evidence")),
            recommendations=RecommendationSettings.from_mapping(data.get("recommendations")),
            refinement=RefinementSettings.from_mapping(data.get("refinement")),
        )

    @classmethod
    def load(cls, path: Path | None = None) -> AnalysisConfig:
        """Load from ``path``; fall back to built-in defaults when the file is absent.

        Absence is not an error: every threshold has a documented default, so an analysis
        can run on a checkout that has not customised anything.
        """
        path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if not path.is_file():
            if path == DEFAULT_CONFIG_PATH:
                return cls()
            raise AnalysisConfigError(f"analysis config not found at {path}")
        with path.open("r", encoding="utf-8") as handle:
            try:
                raw = yaml.safe_load(handle)
            except yaml.YAMLError as exc:
                raise AnalysisConfigError(f"{path}: invalid YAML: {exc}") from exc
        return cls.from_mapping(raw)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "AnalysisConfig",
    "MinimumEvidence",
    "RecommendationSettings",
    "RefinementSettings",
    "Thresholds",
]
