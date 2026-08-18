"""Tests for problem-level and aggregate base-vs-DPO comparison (spec sections 54-58,
109, 112, 113, 121, 122, 127)."""

from __future__ import annotations

import pytest

from python_dpo.model_evaluation.comparison import compare
from python_dpo.model_evaluation.errors import IncompleteBenchmarkError
from python_dpo.model_evaluation.metrics import solve_rate
from python_dpo.model_evaluation.models import EvaluationRecord


def make_record(
    problem_id: str,
    variant: str,
    *,
    sample_index: int = 0,
    tests_total: int = 5,
    tests_passed: int = 0,
    status: str | None = None,
    error_type: str | None = None,
) -> EvaluationRecord:
    tests_failed = tests_total - tests_passed
    if status is None:
        status = "passed" if tests_total > 0 and tests_passed == tests_total else "failed"
    if status != "passed" and error_type is None and status != "infrastructure_error":
        error_type = "assertion_failure"
    if status == "infrastructure_error":
        error_type = "infrastructure_error"
    return EvaluationRecord(
        evaluation_run_id="eval_x",
        problem_id=problem_id,
        model_variant=variant,
        sample_index=sample_index,
        tests_total=tests_total,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        tests_error=0,
        tests_skipped=0,
        timeout=False,
        status=status,
        duration_ms=100,
        error_type=error_type,
    )


def solved(problem_id: str, variant: str, tests_total: int = 5) -> EvaluationRecord:
    return make_record(problem_id, variant, tests_total=tests_total, tests_passed=tests_total)


def failed(problem_id: str, variant: str, tests_total: int = 5, tests_passed: int = 0) -> EvaluationRecord:
    return make_record(problem_id, variant, tests_total=tests_total, tests_passed=tests_passed)


def test_model_comparison_base_4_dpo_6_solved():
    """Spec section 112."""
    problem_ids = [f"p{i:03d}" for i in range(1, 11)]
    base_solved_ids = set(problem_ids[:4])
    dpo_solved_ids = set(problem_ids[:4]) | set(problem_ids[4:6])  # 4 overlap + 2 new

    base_records = {
        pid: [solved(pid, "base") if pid in base_solved_ids else failed(pid, "base")]
        for pid in problem_ids
    }
    dpo_records = {
        pid: [solved(pid, "dpo") if pid in dpo_solved_ids else failed(pid, "dpo")]
        for pid in problem_ids
    }

    result = compare(problem_ids, base_records, dpo_records)

    base_rate = solve_rate([pc.base_pass for pc in result.per_problem])
    dpo_rate = solve_rate([pc.dpo_pass for pc in result.per_problem])
    assert base_rate == pytest.approx(0.4)
    assert dpo_rate == pytest.approx(0.6)
    assert (dpo_rate - base_rate) == pytest.approx(0.2)
    assert result.dpo_wins == 2
    assert result.ties == 8
    assert result.dpo_losses == 0


def test_regression_base_8_dpo_6_solved_is_identified():
    """Spec section 113: the report must clearly identify this as a regression."""
    problem_ids = [f"p{i:03d}" for i in range(1, 11)]
    base_solved_ids = set(problem_ids[:8])
    dpo_solved_ids = set(problem_ids[:6])

    base_records = {
        pid: [solved(pid, "base") if pid in base_solved_ids else failed(pid, "base")]
        for pid in problem_ids
    }
    dpo_records = {
        pid: [solved(pid, "dpo") if pid in dpo_solved_ids else failed(pid, "dpo")]
        for pid in problem_ids
    }

    result = compare(problem_ids, base_records, dpo_records)
    base_rate = solve_rate([pc.base_pass for pc in result.per_problem])
    dpo_rate = solve_rate([pc.dpo_pass for pc in result.per_problem])
    assert (dpo_rate - base_rate) == pytest.approx(-0.2)
    assert result.dpo_losses == 2
    assert result.dpo_wins == 0
    assert result.dpo_win_rate == 0.0


def test_win_rate_excludes_ties():
    """Spec section 56: ties must not be in the win-rate denominator."""
    problem_ids = ["p1", "p2", "p3", "p4"]
    base_records = {
        "p1": [solved("p1", "base")],
        "p2": [solved("p2", "base")],
        "p3": [failed("p3", "base")],
        "p4": [failed("p4", "base")],
    }
    dpo_records = {
        "p1": [solved("p1", "dpo")],
        "p2": [failed("p2", "dpo")],
        "p3": [solved("p3", "dpo")],
        "p4": [failed("p4", "dpo")],
    }
    result = compare(problem_ids, base_records, dpo_records)
    assert result.dpo_wins == 1
    assert result.dpo_losses == 1
    assert result.ties == 2
    assert result.dpo_win_rate == pytest.approx(0.5)


def test_test_pass_delta_reveals_partial_improvement():
    """Spec section 57: partial improvement is visible even when neither model solves
    the problem outright."""
    base_records = {"p1": [failed("p1", "base", tests_total=10, tests_passed=2)]}
    dpo_records = {"p1": [failed("p1", "dpo", tests_total=10, tests_passed=8)]}
    result = compare(["p1"], base_records, dpo_records)
    pc = result.per_problem[0]
    assert pc.base_test_pass_rate == pytest.approx(0.2)
    assert pc.dpo_test_pass_rate == pytest.approx(0.8)
    assert pc.test_pass_delta == pytest.approx(0.6)
    assert pc.improvement == 0


def test_incomplete_benchmark_raises_by_default():
    """Spec section 121: a mismatched set must be an error, not a silent narrowing."""
    base_records = {"p1": [solved("p1", "base")]}
    dpo_records: dict = {}
    with pytest.raises(IncompleteBenchmarkError):
        compare(["p1", "p2"], base_records, dpo_records)


def test_incomplete_benchmark_allows_explicit_paired_subset():
    """Spec section 122: the paired subset is usable when explicitly requested."""
    base_records = {"p1": [solved("p1", "base")], "p2": [solved("p2", "base")]}
    dpo_records = {"p1": [solved("p1", "dpo")]}
    result = compare(["p1", "p2"], base_records, dpo_records, allow_incomplete=True)
    assert result.paired_problems == 1
    assert result.benchmark_problems == 2
    assert result.base_successfully_evaluated == 2
    assert result.dpo_successfully_evaluated == 1


def test_infrastructure_error_excluded_from_correctness():
    """Spec section 120: an infrastructure failure is neither a pass nor a fail."""
    base_records = {
        "p1": [
            solved("p1", "base"),
            make_record("p1", "base", sample_index=1, status="infrastructure_error", tests_total=0),
        ]
    }
    dpo_records = {"p1": [solved("p1", "dpo")]}
    result = compare(["p1"], base_records, dpo_records)
    pc = result.per_problem[0]
    assert pc.base_n == 1
    assert pc.base_c == 1
