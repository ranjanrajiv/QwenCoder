"""Recommendations and the iteration decision (spec 11 sections 53-58, 89-95).

Every rule here is explicit and testable: an observation maps onto one of section 58's ten
categories, carrying the evidence that produced it and a hypothesis stating what is
expected to change. :class:`~python_dpo.analysis.models.Recommendation` refuses to be
constructed without both, so an unsupported recommendation cannot reach a file.

Two gates matter more than the rules themselves.

**Section 56 gates ``adjust_dpo_hyperparameters``.** Data-shaped evidence -- coverage
holes, missing categories, thin preference pairs -- may never produce a hyperparameter
recommendation. Tuning beta because the training data did not cover the benchmark is
cargo-cult optimisation, and the resulting number would be unattributable.

**Section 95 evaluates ``insufficient_evidence`` first.** Below the configured benchmark
size or above the configured CI width, no other decision may be reported as the headline
however suggestive it looks. On this project's real run both gates fire at once.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .config import AnalysisConfig
from .models import ITERATION_DECISIONS, Recommendation


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_recommendations(
    *,
    config: AnalysisConfig,
    outcomes: Sequence[Any],
    category_gaps: Sequence[Any],
    difficulty_gaps: Sequence[Any],
    diversity: Any,
    error_comparisons: Sequence[Any],
    training_curve: dict[str, Any],
    coverage: dict[str, Any],
) -> list[Recommendation]:
    """Sections 53-58. Returns recommendations ordered by score, highest first."""
    recommendations: list[Recommendation] = []

    improvements = [o for o in outcomes if o.outcome.endswith("improvement")]
    regressions = [o for o in outcomes if o.outcome.endswith("regression")]

    # Section 116: a category the benchmark measures but training barely covered, with
    # observed failures there, is the canonical add_data case.
    starved = [g for g in category_gaps if g.verdict == "underrepresented"]
    if starved:
        unsolved = {o.category for o in outcomes if not o.dpo_solved and o.category}
        overlap = sorted({g.name for g in starved} & unsolved)
        recommendations.append(
            Recommendation(
                category="add_data",
                hypothesis=(
                    "Adding preference pairs for "
                    f"{', '.join(overlap or [g.name for g in starved][:3])} should raise "
                    "held-out pass@1 in those categories, which the current training split "
                    "does not cover"
                ),
                evidence={
                    "underrepresented_categories": [g.name for g in starved],
                    "benchmark_categories_unsolved_by_dpo": overlap,
                    "coverage_ratios": {g.name: g.coverage_ratio for g in starved},
                },
                confidence="medium" if overlap else "low",
                expected_impact=_clamp(0.3 + 0.1 * len(overlap)),
                evidence_strength=0.7 if overlap else 0.4,
                implementation_cost=0.6,
            )
        )

    # Sections 34-37: trained on categories the benchmark never measures. Structural, and
    # the strongest finding available on a small dataset.
    unmeasured = [g for g in category_gaps if g.verdict == "not_in_benchmark"]
    if unmeasured:
        recommendations.append(
            Recommendation(
                category="expand_benchmark",
                hypothesis=(
                    "Extending the benchmark to cover "
                    f"{', '.join(g.name for g in unmeasured)} would make the categories DPO "
                    "actually trained on measurable; at present nothing that was trained is "
                    "evaluated"
                ),
                evidence={
                    "trained_categories_absent_from_benchmark": [g.name for g in unmeasured],
                    "training_shares": {g.name: g.training_share for g in unmeasured},
                },
                confidence="high",
                expected_impact=0.6,
                evidence_strength=0.9,
                implementation_cost=0.5,
            )
        )

    # Section 89: more losses than wins warrants investigation before anything else.
    if len(regressions) > len(improvements):
        recommendations.append(
            Recommendation(
                category="investigate_regression",
                hypothesis=(
                    f"{len(regressions)} problem(s) regressed against {len(improvements)} "
                    "improved; identifying what the regressed problems share should explain "
                    "whether the preference signal is mis-specified"
                ),
                evidence={
                    "regressions": [o.problem_id for o in regressions],
                    "improvements": [o.problem_id for o in improvements],
                    "regression_warning": True,
                },
                confidence="high",
                expected_impact=0.7,
                evidence_strength=0.8,
                implementation_cost=0.3,
            )
        )

    if getattr(diversity, "mode_collapse_warning", False):
        recommendations.append(
            Recommendation(
                category="investigate_mode_collapse",
                hypothesis=(
                    "DPO output diversity fell materially against base; if alignment is "
                    "collapsing the sampling distribution, lowering beta or shortening "
                    "training should recover it"
                ),
                evidence={
                    "base_diversity": diversity.base_diversity,
                    "dpo_diversity": diversity.dpo_diversity,
                    "relative_change": diversity.relative_change,
                },
                confidence="medium",
                expected_impact=0.5,
                evidence_strength=0.6,
                implementation_cost=0.4,
            )
        )

    # Skew shows up two ways, and the second is the more direct evidence: either easy
    # problems dominate training, or the harder difficulties the benchmark actually tests
    # are missing from it. Checking only the former misses a training split that is 100%
    # easy against a benchmark that is 30% hard, because "all easy" reads as balanced
    # whenever the benchmark is mostly easy too.
    skewed = [g for g in difficulty_gaps if g.verdict == "overrepresented" and g.name == "easy"]
    starved_difficulty = [
        g for g in difficulty_gaps
        if g.verdict == "underrepresented" and g.name in ("medium", "hard")
    ]
    if skewed or starved_difficulty:
        recommendations.append(
            Recommendation(
                category="increase_problem_difficulty",
                hypothesis=(
                    "The training split is skewed toward easy problems; adding medium and "
                    "hard problems should produce preference pairs that discriminate on "
                    "cases the benchmark actually tests"
                ),
                evidence={
                    "difficulty_shares": {
                        g.name: {"training": g.training_share, "benchmark": g.benchmark_share}
                        for g in difficulty_gaps
                    },
                },
                confidence="medium",
                expected_impact=0.4,
                evidence_strength=0.6,
                implementation_cost=0.7,
            )
        )

    trained_pairs = coverage.get("trained_pairs", 0)
    if trained_pairs and trained_pairs < 50:
        recommendations.append(
            Recommendation(
                category="refine_preference_pairs",
                hypothesis=(
                    f"Only {trained_pairs} pair(s) reached training; increasing the pair "
                    "count should give the optimiser enough signal to move the policy at all"
                ),
                evidence={
                    "trained_pairs": trained_pairs,
                    "total_pairs": coverage.get("total_pairs"),
                    "problems_without_pairs": coverage.get("problems_without_pairs", []),
                },
                confidence="high",
                expected_impact=0.6,
                evidence_strength=0.8,
                implementation_cost=0.5,
            )
        )

    # Section 56: only an optimisation-shaped observation may produce a hyperparameter
    # recommendation. A data gap never may.
    if training_curve.get("verdict") in ("undertrained", "overtrained"):
        recommendations.append(
            Recommendation(
                category="adjust_dpo_hyperparameters",
                hypothesis=(
                    f"The training curve reads {training_curve['verdict']}; adjusting the "
                    "learning rate or step count should move the loss trend"
                ),
                evidence={
                    "training_curve_verdict": training_curve["verdict"],
                    "reason": training_curve.get("reason"),
                    "first_train_loss": training_curve.get("first_train_loss"),
                    "final_train_loss": training_curve.get("final_train_loss"),
                },
                confidence="medium",
                expected_impact=0.5,
                evidence_strength=0.5,
                implementation_cost=0.2,
            )
        )

    if not recommendations:
        recommendations.append(
            Recommendation(
                category="no_action",
                hypothesis=(
                    "No observation in this analysis crossed a configured threshold; the "
                    "evidence does not support a specific change"
                ),
                evidence={"outcomes": len(outcomes), "improvements": len(improvements)},
                confidence="low",
                expected_impact=0.0,
                evidence_strength=0.3,
                implementation_cost=0.0,
            )
        )

    weights = config.recommendations.weights
    scored = [r.scored(weights) for r in recommendations]
    scored.sort(key=lambda r: (-r.recommendation_score, r.category))
    return scored[: config.recommendations.max_recommendations]


def decide_iteration(
    *,
    config: AnalysisConfig,
    benchmark_problem_count: int,
    paired_ci_width: float | None,
    outcomes: Sequence[Any],
    category_gaps: Sequence[Any],
    diversity: Any,
) -> dict[str, Any]:
    """Section 90's decision, with section 95's evidence gates evaluated **first**."""
    reasons: list[str] = []

    if benchmark_problem_count < config.minimum_evidence.benchmark_problems:
        reasons.append(
            f"benchmark has {benchmark_problem_count} problem(s), below the configured "
            f"minimum of {config.minimum_evidence.benchmark_problems}"
        )
    if paired_ci_width is not None and paired_ci_width > config.minimum_evidence.max_ci_width:
        reasons.append(
            f"paired pass@1 confidence interval is {paired_ci_width:.3f} wide, above the "
            f"configured maximum of {config.minimum_evidence.max_ci_width}"
        )

    if reasons:
        # Section 95: nothing else may be reported as the headline decision.
        return {
            "decision": "insufficient_evidence",
            "reasons": reasons,
            "gated": True,
        }

    regressions = [o for o in outcomes if o.outcome.endswith("regression")]
    improvements = [o for o in outcomes if o.outcome.endswith("improvement")]

    if getattr(diversity, "mode_collapse_warning", False) or len(regressions) > len(improvements):
        decision = "adjust_training"
        reason = "regressions outnumber improvements, or output diversity collapsed"
    elif any(g.verdict == "underrepresented" for g in category_gaps):
        decision = "refine_data"
        reason = "benchmark categories are underrepresented in the training split"
    elif any(g.verdict == "not_in_benchmark" for g in category_gaps):
        decision = "expand_benchmark"
        reason = "trained categories are not measured by the benchmark"
    else:
        decision = "accept_model"
        reason = "no threshold was crossed and the evidence gates passed"

    return {"decision": decision, "reasons": [reason], "gated": False}


__all__ = ["ITERATION_DECISIONS", "build_recommendations", "decide_iteration"]
