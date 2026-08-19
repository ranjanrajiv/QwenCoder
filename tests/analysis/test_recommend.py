"""Tests for recommendations and the iteration decision (spec 11 sections 53-58, 89-95, 116)."""

from __future__ import annotations

import pytest

from python_dpo.analysis.config import AnalysisConfig
from python_dpo.analysis.errors import AnalysisStoreError
from python_dpo.analysis.models import CategoryGap, DiversityReport, ProblemOutcome, Recommendation
from python_dpo.analysis.recommend import build_recommendations, decide_iteration

CONFIG = AnalysisConfig()


def gap(name, training, benchmark, ratio, verdict):
    return CategoryGap(name=name, training_share=training, benchmark_share=benchmark,
                       coverage_ratio=ratio, verdict=verdict)


def outcome(problem_id, kind, *, category=None, dpo_solved=True):
    return ProblemOutcome(
        problem_id=problem_id, outcome=kind, base_best_score=0.5, dpo_best_score=0.5,
        base_solved=True, dpo_solved=dpo_solved, category=category,
    )


def recommend(**overrides):
    kwargs = dict(
        config=CONFIG, outcomes=[], category_gaps=[], difficulty_gaps=[],
        diversity=DiversityReport(0, 0, 0, 0), error_comparisons=[],
        training_curve={"verdict": "insufficient_data"}, coverage={},
    )
    kwargs.update(overrides)
    return build_recommendations(**kwargs)


# ------------------------------------------------ sections 55, 103: structural validation


def test_a_recommendation_without_evidence_cannot_be_constructed():
    with pytest.raises(AnalysisStoreError, match="evidence"):
        Recommendation(
            category="add_data", hypothesis="do a thing", evidence={}, confidence="low",
            expected_impact=0.5, evidence_strength=0.5, implementation_cost=0.5,
        )


def test_a_recommendation_without_a_hypothesis_cannot_be_constructed():
    with pytest.raises(AnalysisStoreError, match="hypothesis"):
        Recommendation(
            category="add_data", hypothesis="   ", evidence={"a": 1}, confidence="low",
            expected_impact=0.5, evidence_strength=0.5, implementation_cost=0.5,
        )


# ------------------------------------------------------------- section 116's canonical case


def test_high_error_rate_plus_low_coverage_yields_add_data():
    recommendations = recommend(
        outcomes=[outcome("p001", "unchanged", category="recursion", dpo_solved=False)],
        category_gaps=[gap("recursion", 0.0, 0.25, 0.0, "underrepresented")],
    )
    categories = [r.category for r in recommendations]
    assert "add_data" in categories
    add_data = next(r for r in recommendations if r.category == "add_data")
    assert "recursion" in add_data.evidence["benchmark_categories_unsolved_by_dpo"]


# ------------------------------------------------ section 56: the hyperparameter gate


def test_data_shaped_evidence_never_produces_a_hyperparameter_recommendation():
    """Tuning beta because the training data did not cover the benchmark would be
    cargo-cult optimisation, and the resulting number unattributable."""
    recommendations = recommend(
        outcomes=[outcome("p001", "unchanged", category="recursion", dpo_solved=False)],
        category_gaps=[gap("recursion", 0.0, 0.25, 0.0, "underrepresented")],
        training_curve={"verdict": "insufficient_data"},
    )
    assert "adjust_dpo_hyperparameters" not in [r.category for r in recommendations]


def test_an_optimisation_shaped_curve_does_produce_one():
    recommendations = recommend(
        training_curve={"verdict": "undertrained", "reason": "loss rose"},
    )
    assert "adjust_dpo_hyperparameters" in [r.category for r in recommendations]


# ---------------------------------------------------------------- sections 57, 89, 31


def test_recommendations_are_ordered_by_score_descending():
    recommendations = recommend(
        outcomes=[outcome("p001", "complete_regression", category="lists", dpo_solved=False)],
        category_gaps=[
            gap("lists", 0.0, 0.5, 0.0, "underrepresented"),
            gap("exceptions", 0.5, 0.0, None, "not_in_benchmark"),
        ],
    )
    scores = [r.recommendation_score for r in recommendations]
    assert scores == sorted(scores, reverse=True)


def test_more_regressions_than_improvements_flags_investigate_regression():
    recommendations = recommend(
        outcomes=[
            outcome("p001", "complete_regression", dpo_solved=False),
            outcome("p002", "partial_regression", dpo_solved=False),
            outcome("p003", "partial_improvement"),
        ],
    )
    regression = next(r for r in recommendations if r.category == "investigate_regression")
    assert regression.evidence["regression_warning"] is True


def test_mode_collapse_produces_its_own_recommendation():
    recommendations = recommend(
        diversity=DiversityReport(10, 10, 5, 10, mode_collapse_warning=True),
    )
    assert "investigate_mode_collapse" in [r.category for r in recommendations]


def test_max_recommendations_is_honoured():
    config = AnalysisConfig.from_mapping({"recommendations": {"max_recommendations": 2}})
    recommendations = build_recommendations(
        config=config,
        outcomes=[outcome("p001", "complete_regression", category="lists", dpo_solved=False)],
        category_gaps=[
            gap("lists", 0.0, 0.5, 0.0, "underrepresented"),
            gap("exceptions", 0.5, 0.0, None, "not_in_benchmark"),
        ],
        difficulty_gaps=[gap("easy", 1.0, 0.4, 2.5, "overrepresented")],
        diversity=DiversityReport(10, 10, 5, 10, mode_collapse_warning=True),
        error_comparisons=[], training_curve={"verdict": "undertrained"},
        coverage={"trained_pairs": 3},
    )
    assert len(recommendations) == 2


def test_no_observation_yields_no_action():
    assert [r.category for r in recommend()] == ["no_action"]


# --------------------------------------------------- section 95: the decision precedence


def test_insufficient_evidence_wins_over_refine_data():
    """Section 95: below the evidence floor no other decision may be the headline,
    however suggestive the coverage analysis looks."""
    decision = decide_iteration(
        config=CONFIG, benchmark_problem_count=7, paired_ci_width=0.04,
        outcomes=[], category_gaps=[gap("lists", 0.0, 0.5, 0.0, "underrepresented")],
        diversity=DiversityReport(0, 0, 0, 0),
    )
    assert decision["decision"] == "insufficient_evidence"
    assert decision["gated"] is True


def test_a_wide_confidence_interval_also_gates():
    decision = decide_iteration(
        config=CONFIG, benchmark_problem_count=100, paired_ci_width=0.9,
        outcomes=[], category_gaps=[], diversity=DiversityReport(0, 0, 0, 0),
    )
    assert decision["decision"] == "insufficient_evidence"
    assert any("confidence interval" in r for r in decision["reasons"])


def test_refine_data_is_reported_once_the_gates_pass():
    decision = decide_iteration(
        config=CONFIG, benchmark_problem_count=100, paired_ci_width=0.02,
        outcomes=[], category_gaps=[gap("lists", 0.0, 0.5, 0.0, "underrepresented")],
        diversity=DiversityReport(0, 0, 0, 0),
    )
    assert decision["decision"] == "refine_data"
    assert decision["gated"] is False


def test_regressions_outweigh_coverage_and_yield_adjust_training():
    decision = decide_iteration(
        config=CONFIG, benchmark_problem_count=100, paired_ci_width=0.02,
        outcomes=[outcome("p001", "complete_regression", dpo_solved=False)],
        category_gaps=[gap("lists", 0.0, 0.5, 0.0, "underrepresented")],
        diversity=DiversityReport(0, 0, 0, 0),
    )
    assert decision["decision"] == "adjust_training"


def test_accept_model_when_nothing_is_wrong():
    decision = decide_iteration(
        config=CONFIG, benchmark_problem_count=100, paired_ci_width=0.02,
        outcomes=[outcome("p001", "partial_improvement")],
        category_gaps=[gap("lists", 0.5, 0.5, 1.0, "balanced")],
        diversity=DiversityReport(10, 10, 10, 10),
    )
    assert decision["decision"] == "accept_model"
