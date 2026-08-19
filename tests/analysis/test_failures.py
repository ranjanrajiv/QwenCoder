"""Tests for test-level failure frequencies (spec 11 sections 45-49)."""

from __future__ import annotations

from python_dpo.analysis.failures import build_test_failure_stats, interesting

from .conftest import FakeTestResult


def rows(problem, test, variant_failures, runs):
    """``runs`` rows for one test, the first ``variant_failures`` of them failing."""
    return [
        FakeTestResult(
            problem, f"{problem}_c{i + 1:03d}", test,
            status="failed" if i < variant_failures else "passed",
            error_type="AssertionError" if i < variant_failures else None,
        )
        for i in range(runs)
    ]


def test_failure_counts_and_rates_are_computed_per_variant():
    stats = build_test_failure_stats(
        {"base": rows("p001", "t1", 3, 10), "dpo": rows("p001", "t1", 6, 10)}
    )
    [stat] = stats
    assert stat.base_failures == 3
    assert stat.base_failure_rate == 0.3
    assert stat.dpo_failures == 6
    assert stat.dpo_failure_rate == 0.6


def test_a_test_both_variants_mostly_fail_is_a_hard_test():
    """Section 47: a property of the problem, not of either model."""
    stats = build_test_failure_stats(
        {"base": rows("p001", "t1", 10, 10), "dpo": rows("p001", "t1", 10, 10)},
        hard_test_failure_rate=0.5,
    )
    assert stats[0].hard_test is True


def test_a_test_neither_variant_fails_often_is_not_hard():
    stats = build_test_failure_stats(
        {"base": rows("p001", "t1", 1, 10), "dpo": rows("p001", "t1", 1, 10)},
        hard_test_failure_rate=0.5,
    )
    assert stats[0].hard_test is False


def test_dpo_specific_difficulty_is_flagged():
    """Section 48: DPO fails it materially more than base -- a regression candidate."""
    stats = build_test_failure_stats(
        {"base": rows("p001", "t1", 1, 10), "dpo": rows("p001", "t1", 5, 10)},
        variant_specific_delta=0.2,
    )
    assert stats[0].dpo_specific is True
    assert stats[0].base_specific is False


def test_base_specific_difficulty_is_flagged():
    """Section 49: training helped something specific, even if aggregate pass@k did not move."""
    stats = build_test_failure_stats(
        {"base": rows("p001", "t1", 5, 10), "dpo": rows("p001", "t1", 1, 10)},
        variant_specific_delta=0.2,
    )
    assert stats[0].base_specific is True
    assert stats[0].dpo_specific is False


def test_a_small_gap_is_below_the_variant_specific_threshold():
    """A 10pp gap must not be flagged when the configured delta is 20pp -- the threshold
    is what decides, not the direction of the difference."""
    stats = build_test_failure_stats(
        {"base": rows("p001", "t1", 3, 10), "dpo": rows("p001", "t1", 4, 10)},
        variant_specific_delta=0.2,
    )
    assert stats[0].dpo_specific is False
    assert stats[0].base_specific is False


def test_error_status_counts_as_a_failure():
    base = [FakeTestResult("p001", "p001_c001", "t1", status="error", error_type="TypeError")]
    stats = build_test_failure_stats({"base": base, "dpo": []})
    assert stats[0].base_failures == 1


def test_interesting_filters_out_tests_nothing_ever_failed():
    all_pass = {"base": rows("p001", "t1", 0, 10), "dpo": rows("p001", "t1", 0, 10)}
    some_fail = {"base": rows("p002", "t1", 2, 10), "dpo": rows("p002", "t1", 2, 10)}
    stats = build_test_failure_stats(
        {v: all_pass[v] + some_fail[v] for v in ("base", "dpo")}
    )
    assert len(stats) == 2
    assert [s.problem_id for s in interesting(stats)] == ["p002"]
