"""Tests for per-problem outcome classification (spec 11 sections 17-25, 50-52, 113)."""

from __future__ import annotations

from python_dpo.analysis.outcomes import build_problem_outcomes, partition

from .conftest import FakeEvaluation, evaluations_for, make_problem

PROBLEMS = {f"p{i:03d}": make_problem(f"p{i:03d}") for i in range(1, 6)}


def outcome_for(base_passed: int, dpo_passed: int) -> str:
    evaluations = evaluations_for({"p001": (base_passed, dpo_passed)})
    return build_problem_outcomes(evaluations, PROBLEMS)[0].outcome


# ---------------------------------------------------------- section 113's four named cases


def test_zero_to_all_is_complete_improvement():
    assert outcome_for(0, 10) == "complete_improvement"


def test_all_to_zero_is_complete_regression():
    assert outcome_for(10, 0) == "complete_regression"


def test_partial_gain_is_partial_improvement():
    """Section 113's 3/10 -> 7/10 case: the *test-pass rate within a sample* rises, with
    neither side reaching a full solve. (All-or-nothing samples would score 1.0 on both
    sides and read as unchanged, which is a property of the fixture, not the model.)"""
    evaluations = {
        "base": [FakeEvaluation("p001", "base", 0, tests_passed=3, status="failed",
                                error_type="assertion_failure")],
        "dpo": [FakeEvaluation("p001", "dpo", 0, tests_passed=7, status="failed",
                               error_type="assertion_failure")],
    }
    [outcome] = build_problem_outcomes(evaluations, PROBLEMS)
    assert outcome.outcome == "partial_improvement"
    assert outcome.base_best_score == 0.3
    assert outcome.dpo_best_score == 0.7


def test_equal_is_unchanged():
    assert outcome_for(5, 5) == "unchanged"


def test_partial_loss_is_partial_regression():
    """Section 20: base solved it, DPO still passes something but fewer."""
    evaluations = {
        "base": [FakeEvaluation("p001", "base", 0, tests_passed=10, status="passed")],
        "dpo": [FakeEvaluation("p001", "dpo", 0, tests_passed=4, status="failed",
                               error_type="assertion_failure")],
    }
    assert build_problem_outcomes(evaluations, PROBLEMS)[0].outcome == "partial_regression"


# -------------------------------------------------------------- section 25: best, not mean


def test_score_is_the_maximum_across_samples_not_the_mean():
    """A model that solves a problem once in ten attempts must not score below one that
    never solves it. Constructed so the mean would give the opposite answer."""
    evaluations = {
        # base: one perfect sample, nine zeros -> best 1.0, mean 0.1
        "base": [
            FakeEvaluation("p001", "base", 0, tests_passed=10, status="passed"),
            *[
                FakeEvaluation("p001", "base", i, tests_passed=0, status="failed",
                               error_type="assertion_failure")
                for i in range(1, 10)
            ],
        ],
        # dpo: ten samples at 5/10 -> best 0.5, mean 0.5
        "dpo": [
            FakeEvaluation("p001", "dpo", i, tests_passed=5, status="failed",
                           error_type="assertion_failure")
            for i in range(10)
        ],
    }
    [outcome] = build_problem_outcomes(evaluations, PROBLEMS)
    assert outcome.base_best_score == 1.0
    assert outcome.dpo_best_score == 0.5
    assert outcome.base_solved is True
    assert outcome.dpo_solved is False
    # With the mean this would read as an improvement; with the max it is a regression.
    assert outcome.outcome == "partial_regression"


# ------------------------------------------------------------------- severity and grouping


def test_complete_cases_are_high_severity():
    evaluations = evaluations_for({"p001": (10, 0)})
    assert build_problem_outcomes(evaluations, PROBLEMS)[0].severity == "high"


def test_unchanged_has_no_severity():
    evaluations = evaluations_for({"p001": (5, 5)})
    assert build_problem_outcomes(evaluations, PROBLEMS)[0].severity == "none"


def test_small_change_is_low_severity_and_large_is_medium():
    small = build_problem_outcomes(
        {
            "base": [FakeEvaluation("p001", "base", 0, tests_passed=5, status="failed",
                                    error_type="assertion_failure")],
            "dpo": [FakeEvaluation("p001", "dpo", 0, tests_passed=6, status="failed",
                                   error_type="assertion_failure")],
        },
        PROBLEMS, regression_threshold=0.2,
    )[0]
    assert small.severity == "low"

    large = build_problem_outcomes(
        {
            "base": [FakeEvaluation("p001", "base", 0, tests_passed=2, status="failed",
                                    error_type="assertion_failure")],
            "dpo": [FakeEvaluation("p001", "dpo", 0, tests_passed=8, status="failed",
                                   error_type="assertion_failure")],
        },
        PROBLEMS, regression_threshold=0.2,
    )[0]
    assert large.severity == "medium"


def test_problems_evaluated_for_only_one_variant_are_skipped():
    """An outcome is a comparison; inventing a zero for the missing side would
    manufacture a result that was never measured."""
    evaluations = {
        "base": [FakeEvaluation("p001", "base", 0)],
        "dpo": [FakeEvaluation("p002", "dpo", 0)],
    }
    assert build_problem_outcomes(evaluations, PROBLEMS) == []


def test_infrastructure_errors_are_excluded_from_scoring():
    evaluations = {
        "base": [
            FakeEvaluation("p001", "base", 0, tests_passed=0, status="failed",
                           error_type="infrastructure_error"),
            FakeEvaluation("p001", "base", 1, tests_passed=10, status="passed"),
        ],
        "dpo": [FakeEvaluation("p001", "dpo", 0, tests_passed=10, status="passed")],
    }
    [outcome] = build_problem_outcomes(evaluations, PROBLEMS)
    assert outcome.base_best_score == 1.0
    assert outcome.outcome == "unchanged"


def test_partition_splits_into_three_buckets():
    evaluations = evaluations_for({"p001": (0, 10), "p002": (10, 0), "p003": (5, 5)})
    buckets = partition(build_problem_outcomes(evaluations, PROBLEMS))
    assert [o.problem_id for o in buckets["improvements"]] == ["p001"]
    assert [o.problem_id for o in buckets["regressions"]] == ["p002"]
    assert [o.problem_id for o in buckets["unchanged"]] == ["p003"]
