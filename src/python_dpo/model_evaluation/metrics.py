"""Correctness metrics (spec sections 41-53). Pure stdlib -- every function here takes
plain ints/floats/bools, never a model record, so it is testable against the spec's exact
hand-computed examples without constructing a single :class:`~python_dpo.model_evaluation.models.EvaluationRecord`.

The single most important function is :func:`pass_at_k`. Spec section 44 is explicit that
``c / n`` is **not** pass@k; the unbiased estimator below is the only correct one, and
section 111 gives the boundary cases this module is tested against.
"""

from __future__ import annotations

import math

# Spec section 53's seven buckets, in ascending order.
TEST_FAILURE_BUCKETS = (
    "0%",
    "1-20%",
    "20-40%",
    "40-60%",
    "60-80%",
    "80-99%",
    "100%",
)


def pass_at_k(n: int, c: int, k: int) -> float:
    """The unbiased pass@k estimator (spec section 42): ``1 - C(n-c, k) / C(n, k)``.

    ``c`` of ``n`` samples are correct; estimates the probability that at least one of
    ``k`` samples drawn (without replacement) from those ``n`` would be correct.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    if isinstance(c, bool) or not isinstance(c, int) or not 0 <= c <= n:
        raise ValueError("c must be an integer in the range [0, n]")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if k > n:
        raise ValueError(f"k ({k}) must not exceed n ({n}); not enough samples to estimate pass@{k}")

    if c == 0:
        return 0.0
    # Spec section 42: when c >= n - k + 1, every possible k-subset contains at least one
    # correct sample, so the estimator is exactly 1 without needing the binomial ratio.
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def mean_pass_at_k(per_problem: list[tuple[int, int]], k: int) -> float:
    """Aggregate pass@k across problems (spec sections 45, 46).

    ``per_problem`` is a list of ``(n, c)`` pairs, one per problem -- computed and averaged
    per-problem, never by pooling every candidate across problems into one ``(n, c)`` (spec
    section 110).
    """
    if not per_problem:
        return 0.0
    return sum(pass_at_k(n, c, k) for n, c in per_problem) / len(per_problem)


def mean_test_pass_rate(pass_rates: list[float]) -> float:
    """Spec section 47: the mean of ``tests_passed / tests_total`` across candidates."""
    if not pass_rates:
        return 0.0
    return sum(pass_rates) / len(pass_rates)


def solve_rate(solved_flags: list[bool]) -> float:
    """Spec section 48: fraction of problems with at least one fully correct candidate."""
    if not solved_flags:
        return 0.0
    return sum(1 for flag in solved_flags if flag) / len(solved_flags)


def syntax_success_rate(syntax_valid_flags: list[bool]) -> float:
    """Spec section 49: syntactically valid candidates / total generated candidates."""
    if not syntax_valid_flags:
        return 0.0
    return sum(1 for flag in syntax_valid_flags if flag) / len(syntax_valid_flags)


def execution_success_rate(executed_without_runtime_error_flags: list[bool]) -> float:
    """Spec section 50: candidates that ran without a runtime error / total candidates."""
    if not executed_without_runtime_error_flags:
        return 0.0
    return sum(1 for flag in executed_without_runtime_error_flags if flag) / len(
        executed_without_runtime_error_flags
    )


def timeout_rate(timeout_flags: list[bool]) -> float:
    """Spec section 51: timed-out candidates / total candidates."""
    if not timeout_flags:
        return 0.0
    return sum(1 for flag in timeout_flags if flag) / len(timeout_flags)


def generation_failure_rate(failure_flags: list[bool]) -> float:
    """Spec section 52: code-extraction/generation failures / total generation attempts."""
    if not failure_flags:
        return 0.0
    return sum(1 for flag in failure_flags if flag) / len(failure_flags)


def _bucket_for(pass_rate: float) -> str:
    percent = pass_rate * 100
    if percent <= 0:
        return "0%"
    if percent <= 20:
        return "1-20%"
    if percent <= 40:
        return "20-40%"
    if percent <= 60:
        return "40-60%"
    if percent <= 80:
        return "60-80%"
    if percent < 100:
        return "80-99%"
    return "100%"


def test_failure_distribution(pass_rates: list[float]) -> dict[str, int]:
    """Spec section 53: bucket candidates by their test pass rate.

    Every bucket key is always present (zero-filled), so a report can render a full
    histogram without special-casing an empty bucket.
    """
    buckets: dict[str, int] = {name: 0 for name in TEST_FAILURE_BUCKETS}
    for rate in pass_rates:
        buckets[_bucket_for(rate)] += 1
    return buckets


__all__ = [
    "TEST_FAILURE_BUCKETS",
    "execution_success_rate",
    "generation_failure_rate",
    "mean_pass_at_k",
    "mean_test_pass_rate",
    "pass_at_k",
    "solve_rate",
    "syntax_success_rate",
    "test_failure_distribution",
    "timeout_rate",
]
