"""Tests for the evaluation schema (spec 06 sections 21-30, 45-58, 61-64)."""

from __future__ import annotations

from typing import Any

import pytest

from python_dpo.evaluation import (
    CANDIDATE_EVALUATION_STATUSES,
    EVALUATION_RESULT_STATUSES,
    TEST_CASE_STATUSES,
    EvaluationFailure,
    EvaluationManifest,
    EvaluationModelError,
    EvaluationResult,
    EvaluationStatistics,
    TestCaseResult,
)
from python_dpo.evaluation.models import compute_pass_rate, utc_now_iso


def make_test_result(**overrides: Any) -> TestCaseResult:
    fields: dict[str, Any] = {
        "evaluation_run_id": "eval_20260817_154500_a12f",
        "candidate_run_id": "run_20260817_055411",
        "candidate_id": "p001_c001",
        "problem_id": "p001",
        "test_case_id": "p001_t001",
        "status": "passed",
        "duration_ms": 3,
    }
    fields.update(overrides)
    return TestCaseResult(**fields)


def make_result(**overrides: Any) -> EvaluationResult:
    fields: dict[str, Any] = {
        "evaluation_run_id": "eval_20260817_154500_a12f",
        "candidate_run_id": "run_20260817_055411",
        "candidate_id": "p001_c001",
        "problem_id": "p001",
        "status": "passed",
        "tests_passed": 8,
        "tests_failed": 0,
        "tests_error": 0,
        "tests_skipped": 0,
        "duration_ms": 142,
    }
    fields.update(overrides)
    return EvaluationResult.create(**fields)


def make_manifest(**overrides: Any) -> EvaluationManifest:
    fields: dict[str, Any] = {
        "evaluation_run_id": "eval_20260817_154500_a12f",
        "candidate_run_id": "run_20260817_055411",
        "status": "created",
        "created_at": utc_now_iso(),
        "evaluator_version": "v1",
        "test_generator_version": "v1",
        "pytest_version": "8.3.4",
        "python_version": "3.12.7",
        "sandbox_config": {"image": "python-dpo-evaluator:1.0"},
        "requested_candidate_ids": ("p001_c001", "p001_c002"),
    }
    fields.update(overrides)
    return EvaluationManifest(**fields)


def make_failure(**overrides: Any) -> EvaluationFailure:
    fields: dict[str, Any] = {
        "evaluation_run_id": "eval_20260817_154500_a12f",
        "candidate_run_id": "run_20260817_055411",
        "candidate_id": "p001_c001",
        "problem_id": "p001",
        "error_type": "empty_test_suite",
        "error_message": "problem p001 has no test cases",
        "timestamp": utc_now_iso(),
    }
    fields.update(overrides)
    return EvaluationFailure(**fields)


# ---------------------------------------------------------------------- TestCaseResult


def test_status_sets_match_the_spec():
    assert TEST_CASE_STATUSES == {"passed", "failed", "error", "skipped"}
    assert EVALUATION_RESULT_STATUSES == {
        "passed",
        "failed",
        "timeout",
        "syntax_error",
        "infrastructure_error",
    }
    assert CANDIDATE_EVALUATION_STATUSES == {"passed", "failed", "timeout", "syntax_error"}


def test_test_case_result_round_trips():
    result = make_test_result()
    assert TestCaseResult.from_dict(result.to_dict()) == result


def test_test_case_result_rejects_unknown_status():
    with pytest.raises(EvaluationModelError, match="status"):
        make_test_result(status="broken")


def test_test_case_result_rejects_error_type_on_a_passed_test():
    with pytest.raises(EvaluationModelError, match="error_type must be null"):
        make_test_result(status="passed", error_type="ValueError")


def test_test_case_result_error_captures_type_and_message():
    result = make_test_result(
        status="error", error_type="ValueError", error_message="boom"
    )
    assert result.error_type == "ValueError"


def test_test_case_result_from_dict_rejects_unknown_and_missing_fields():
    payload = make_test_result().to_dict()
    with pytest.raises(EvaluationModelError, match="unknown field"):
        TestCaseResult.from_dict({**payload, "unexpected": 1})
    del payload["status"]
    with pytest.raises(EvaluationModelError, match="missing required field"):
        TestCaseResult.from_dict(payload)


# ----------------------------------------------------------------------- EvaluationResult


def test_compute_pass_rate():
    assert compute_pass_rate(5, 8) == pytest.approx(0.625)
    assert compute_pass_rate(0, 0) == 0.0


def test_create_computes_every_derived_field():
    result = make_result(status="passed", tests_passed=8, tests_failed=0, tests_error=0, tests_skipped=0)
    assert result.tests_total == 8
    assert result.pass_rate == pytest.approx(1.0)
    assert result.timeout is False
    assert result.syntax_error is False
    assert result.runtime_error is False
    assert result.infrastructure_error is False


def test_round_trips_through_dict():
    result = make_result()
    assert EvaluationResult.from_dict(result.to_dict()) == result


def test_evaluation_timestamp_is_stamped_when_absent():
    assert make_result().evaluation_timestamp


# ------------------------------------------------------------------ §24: passed is validated


def test_passed_requires_every_test_to_pass():
    with pytest.raises(EvaluationModelError, match="passed"):
        make_result(status="passed", tests_passed=6, tests_failed=2, tests_error=0, tests_skipped=0)


def test_passed_requires_at_least_one_test():
    # Exit code 0 alone must never produce "passed" — an empty suite is not a pass.
    with pytest.raises(EvaluationModelError, match="passed"):
        make_result(status="passed", tests_passed=0, tests_failed=0, tests_error=0, tests_skipped=0)


def test_passed_rejects_a_skipped_test():
    with pytest.raises(EvaluationModelError, match="passed"):
        make_result(status="passed", tests_passed=7, tests_failed=0, tests_error=0, tests_skipped=1)


def test_failed_status_with_a_wrong_answer():
    # §25: pytest ran fine, but a test's assertion failed.
    result = make_result(status="failed", tests_passed=6, tests_failed=2, tests_error=0, tests_skipped=0)
    assert result.status == "failed"
    assert result.tests_total == 8


# --------------------------------------------------------------------- §68 count invariant


def test_counts_must_partition_tests_total():
    with pytest.raises(EvaluationModelError, match="tests_total"):
        EvaluationResult(
            evaluation_run_id="e",
            candidate_run_id="r",
            candidate_id="c",
            problem_id="p",
            status="failed",
            tests_total=10,
            tests_passed=5,
            tests_failed=2,
            tests_error=0,
            tests_skipped=0,
            pass_rate=0.5,
            duration_ms=1,
        )


# ------------------------------------------------------------- derived-field cross-checks


def test_timeout_must_agree_with_status():
    with pytest.raises(EvaluationModelError, match="timeout"):
        EvaluationResult(
            evaluation_run_id="e",
            candidate_run_id="r",
            candidate_id="c",
            problem_id="p",
            status="failed",
            tests_total=1,
            tests_passed=0,
            tests_failed=1,
            tests_error=0,
            tests_skipped=0,
            pass_rate=0.0,
            duration_ms=1,
            timeout=True,
        )


def test_runtime_error_flag_reflects_tests_error_not_status():
    # §27: a candidate exception during a test is a candidate failure (status=failed) with
    # runtime_error=true, distinct from a wrong-answer-only failure.
    with_exception = make_result(status="failed", tests_passed=6, tests_failed=0, tests_error=2, tests_skipped=0)
    assert with_exception.runtime_error is True

    wrong_answer_only = make_result(status="failed", tests_passed=6, tests_failed=2, tests_error=0, tests_skipped=0)
    assert wrong_answer_only.runtime_error is False


def test_infrastructure_error_is_never_a_candidate_outcome():
    # §29, §81: infrastructure failures must not be conflated with candidate failures.
    infra = make_result(status="infrastructure_error", tests_passed=0, tests_failed=0, tests_error=0, tests_skipped=0)
    assert infra.infrastructure_error is True
    assert infra.is_candidate_outcome is False

    candidate_failure = make_result(status="failed", tests_passed=0, tests_failed=1, tests_error=0, tests_skipped=0)
    assert candidate_failure.is_candidate_outcome is True


def test_pass_rate_must_match_the_counts():
    with pytest.raises(EvaluationModelError, match="pass_rate"):
        EvaluationResult(
            evaluation_run_id="e",
            candidate_run_id="r",
            candidate_id="c",
            problem_id="p",
            status="failed",
            tests_total=8,
            tests_passed=5,
            tests_failed=3,
            tests_error=0,
            tests_skipped=0,
            pass_rate=0.99,
            duration_ms=1,
        )


# ------------------------------------------------------------------------- discrepancy


def test_discrepancy_requires_a_reason():
    with pytest.raises(EvaluationModelError, match="discrepancy_reason"):
        make_result(metadata_discrepancy=True)


def test_discrepancy_reason_without_the_flag_is_rejected():
    with pytest.raises(EvaluationModelError, match="discrepancy_reason"):
        make_result(discrepancy_reason="Stage 3 said syntax_valid=true")


def test_discrepancy_recorded_correctly():
    result = make_result(metadata_discrepancy=True, discrepancy_reason="Stage 3 said syntax_valid=true")
    assert result.metadata_discrepancy is True


# -------------------------------------------------------------------------- EvaluationFailure


def test_evaluation_failure_round_trips():
    failure = make_failure()
    assert EvaluationFailure.from_dict(failure.to_dict()) == failure


def test_evaluation_failure_rejects_unknown_error_type():
    with pytest.raises(EvaluationModelError, match="error_type"):
        make_failure(error_type="docker_down")


# ------------------------------------------------------------------------- EvaluationManifest


def test_manifest_round_trips():
    manifest = make_manifest()
    assert EvaluationManifest.from_dict(manifest.to_dict()) == manifest


def test_manifest_rejects_unknown_status():
    with pytest.raises(EvaluationModelError, match="status"):
        make_manifest(status="paused")


def test_manifest_requires_at_least_one_candidate():
    with pytest.raises(EvaluationModelError, match="requested_candidate_ids"):
        make_manifest(requested_candidate_ids=())


def test_manifest_rejects_duplicate_candidate_ids():
    with pytest.raises(EvaluationModelError, match="duplicate"):
        make_manifest(requested_candidate_ids=("p001_c001", "p001_c001"))


def test_manifest_requested_candidates_is_derived():
    assert make_manifest().requested_candidates == 2


def test_manifest_with_status_enforces_the_transition_graph():
    manifest = make_manifest(status="created")
    running = manifest.with_status("running", started_at=utc_now_iso())
    assert running.status == "running"

    completed = running.with_status("completed", completed_at=utc_now_iso())
    assert completed.status == "completed"

    with pytest.raises(EvaluationModelError, match="cannot transition"):
        completed.with_status("running")


# ----------------------------------------------------------------------- EvaluationStatistics


def test_statistics_round_trips():
    stats = EvaluationStatistics.from_records(make_manifest(), [make_result()], [])
    assert EvaluationStatistics.from_dict(stats.to_dict()) == stats


def test_statistics_matches_hand_counted_records():
    manifest = make_manifest(requested_candidate_ids=("p001_c001", "p001_c002"))
    results = [
        make_result(candidate_id="p001_c001", status="passed", tests_passed=8, tests_failed=0, tests_error=0, tests_skipped=0),
        make_result(candidate_id="p001_c002", status="failed", tests_passed=6, tests_failed=2, tests_error=0, tests_skipped=0),
    ]
    failures = [make_failure(candidate_id="p001_c003")]

    stats = EvaluationStatistics.from_records(manifest, results, failures)

    assert stats.candidates_requested == 2
    assert stats.candidates_evaluated == 2
    assert stats.evaluation_failures == 1
    assert stats.passed == 1
    assert stats.failed == 1
    assert stats.tests_total == 16
    assert stats.tests_passed == 14
    assert stats.tests_failed == 2


def test_statistics_distinguishes_timeout_and_syntax_error_from_plain_failed():
    manifest = make_manifest(requested_candidate_ids=("p001_c001", "p001_c002", "p001_c003"))
    results = [
        make_result(
            candidate_id="p001_c001", status="timeout",
            tests_passed=0, tests_failed=0, tests_error=0, tests_skipped=0,
        ),
        make_result(
            candidate_id="p001_c002", status="syntax_error",
            tests_passed=0, tests_failed=0, tests_error=0, tests_skipped=0,
        ),
        make_result(
            candidate_id="p001_c003", status="infrastructure_error",
            tests_passed=0, tests_failed=0, tests_error=0, tests_skipped=0,
        ),
    ]
    stats = EvaluationStatistics.from_records(manifest, results, [])
    assert stats.timeouts == 1
    assert stats.syntax_errors == 1
    assert stats.infrastructure_errors == 1
    # timeout/syntax_error still count toward "failed" as spec's summary counter, while
    # infrastructure_error does not (it says nothing about the candidate).
    assert stats.failed == 2
