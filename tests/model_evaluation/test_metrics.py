"""Tests for pass@k and the other correctness metrics (spec sections 41-53, 111)."""

from __future__ import annotations

import pytest

from python_dpo.model_evaluation.metrics import (
    execution_success_rate,
    generation_failure_rate,
    mean_pass_at_k,
    mean_test_pass_rate,
    pass_at_k,
    solve_rate,
    syntax_success_rate,
    timeout_rate,
)
# Aliased on import: a bare `test_failure_distribution` name would make pytest collect
# the imported function itself as a test.
from python_dpo.model_evaluation.metrics import test_failure_distribution as compute_test_failure_distribution


# ------------------------------------------------------------------------------ pass@k


def test_pass_at_k_all_correct_is_one_for_every_k():
    """Spec section 111: n=10, c=10 -> pass@1 = pass@5 = pass@10 = 1."""
    for k in (1, 5, 10):
        assert pass_at_k(10, 10, k) == 1.0


def test_pass_at_k_none_correct_is_zero_for_every_k():
    """Spec section 111: n=10, c=0 -> pass@k = 0 for every k <= 10."""
    for k in (1, 5, 10):
        assert pass_at_k(10, 0, k) == 0.0


def test_pass_at_k_single_correct_sample():
    assert pass_at_k(10, 1, 1) == pytest.approx(0.1)
    assert pass_at_k(10, 1, 10) == 1.0


def test_pass_at_k_boundary_c_geq_n_minus_k_plus_1():
    # n=10, k=5, c=6 -> n - c = 4 < k=5, so pass@k must be exactly 1.
    assert pass_at_k(10, 6, 5) == 1.0


def test_pass_at_k_is_not_the_naive_ratio():
    """Spec section 44: c/n must never be reported as pass@k."""
    naive = 5 / 10
    assert pass_at_k(10, 5, 5) != naive
    assert pass_at_k(10, 5, 5) == pytest.approx(1 - __import__("math").comb(5, 5) / __import__("math").comb(10, 5))


def test_pass_at_k_rejects_k_greater_than_n():
    with pytest.raises(ValueError):
        pass_at_k(5, 3, 10)


def test_pass_at_k_rejects_invalid_c():
    with pytest.raises(ValueError):
        pass_at_k(5, 6, 1)
    with pytest.raises(ValueError):
        pass_at_k(5, -1, 1)


def test_mean_pass_at_k_averages_across_problems_not_samples():
    """Spec sections 45, 46, 110: problem-level pass@k, then averaged -- pooling every
    sample across problems into one (n, c) would give a different (wrong) answer here.

    Unequal per-problem sample counts are what makes this distinguishable: with equal
    counts, the mean of two fractions with the same denominator always equals the pooled
    fraction, so the two aggregations would coincide by arithmetic accident.
    """
    per_problem = [(5, 5), (15, 0)]  # one small fully-solved problem, one large unsolved one
    assert mean_pass_at_k(per_problem, 1) == pytest.approx(0.5)

    pooled_n, pooled_c = 20, 5
    assert pass_at_k(pooled_n, pooled_c, 1) == pytest.approx(0.25)
    assert mean_pass_at_k(per_problem, 1) != pass_at_k(pooled_n, pooled_c, 1)


def test_mean_pass_at_k_empty_is_zero():
    assert mean_pass_at_k([], 1) == 0.0


# --------------------------------------------------------------------------- other rates


def test_mean_test_pass_rate():
    assert mean_test_pass_rate([1.0, 0.5, 0.0]) == pytest.approx(0.5)
    assert mean_test_pass_rate([]) == 0.0


def test_solve_rate():
    assert solve_rate([True, True, False, False]) == pytest.approx(0.5)
    assert solve_rate([]) == 0.0


def test_syntax_success_rate():
    assert syntax_success_rate([True, True, True, False]) == pytest.approx(0.75)


def test_execution_success_rate():
    assert execution_success_rate([True, False]) == pytest.approx(0.5)


def test_timeout_rate():
    assert timeout_rate([False, False, True, False]) == pytest.approx(0.25)


def test_generation_failure_rate():
    assert generation_failure_rate([False, True, False, False]) == pytest.approx(0.25)


def test_test_failure_distribution_buckets_every_rate():
    distribution = compute_test_failure_distribution([0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 1.0])
    assert distribution == {
        "0%": 1,
        "1-20%": 1,
        "20-40%": 1,
        "40-60%": 1,
        "60-80%": 1,
        "80-99%": 1,
        "100%": 1,
    }


def test_test_failure_distribution_zero_fills_empty_buckets():
    distribution = compute_test_failure_distribution([0.0])
    assert distribution["100%"] == 0
    assert distribution["0%"] == 1
    assert set(distribution) == {
        "0%", "1-20%", "20-40%", "40-60%", "60-80%", "80-99%", "100%",
    }
