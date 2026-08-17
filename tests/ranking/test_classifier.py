"""Tests for CorrectnessClassifier (spec 07 sections 9-14, 72)."""

from __future__ import annotations

from typing import Any

import pytest

from python_dpo.evaluation.models import EvaluationResult
from python_dpo.ranking.classifier import CorrectnessClassifier

EVAL_RUN_ID = "eval_20260817_154500_a12f"
CANDIDATE_RUN_ID = "run_20260817_055411"


def make_result(**overrides: Any) -> EvaluationResult:
    fields: dict[str, Any] = {
        "evaluation_run_id": EVAL_RUN_ID,
        "candidate_run_id": CANDIDATE_RUN_ID,
        "candidate_id": "p001_c001",
        "problem_id": "p001",
        "status": "passed",
        "tests_passed": 8,
        "tests_failed": 0,
        "tests_error": 0,
        "tests_skipped": 0,
        "duration_ms": 100,
    }
    fields.update(overrides)
    return EvaluationResult.create(**fields)


def test_all_tests_pass_is_correct():
    classifier = CorrectnessClassifier()
    correctness, reason = classifier.classify(make_result())
    assert correctness == "correct"
    assert reason is None


def test_one_test_fails_is_incorrect():
    classifier = CorrectnessClassifier()
    result = make_result(status="failed", tests_passed=7, tests_failed=1)
    correctness, reason = classifier.classify(result)
    assert correctness == "incorrect"
    assert reason is None


def test_all_tests_fail_is_incorrect():
    classifier = CorrectnessClassifier()
    result = make_result(status="failed", tests_passed=0, tests_failed=8)
    correctness, _ = classifier.classify(result)
    assert correctness == "incorrect"


def test_candidate_caused_timeout_is_incorrect_not_indeterminate():
    # Spec section 13: the defining rule of this stage.
    classifier = CorrectnessClassifier()
    result = make_result(status="timeout", tests_passed=0, tests_failed=8)
    correctness, reason = classifier.classify(result)
    assert correctness == "incorrect"
    assert reason is None


def test_infrastructure_error_is_indeterminate():
    # Spec section 14.
    classifier = CorrectnessClassifier()
    result = make_result(status="infrastructure_error", tests_passed=0, tests_failed=0)
    correctness, reason = classifier.classify(result)
    assert correctness == "indeterminate"
    assert reason == "infrastructure_error"


def test_skipped_tests_make_a_candidate_incorrect():
    # Spec section 10: correct requires tests_skipped == 0.
    classifier = CorrectnessClassifier()
    result = make_result(status="failed", tests_passed=7, tests_failed=0, tests_skipped=1)
    correctness, _ = classifier.classify(result)
    assert correctness == "incorrect"


def test_zero_tests_is_indeterminate():
    # Spec sections 10, 12, 72: no evidence either way.
    classifier = CorrectnessClassifier()
    result = make_result(
        status="infrastructure_error", tests_passed=0, tests_failed=0, tests_error=0, tests_skipped=0
    )
    correctness, reason = classifier.classify(result)
    assert correctness == "indeterminate"
    assert reason == "infrastructure_error"


def test_missing_evaluation_result_is_indeterminate():
    # Spec section 70: no result at all is never assumed bad.
    classifier = CorrectnessClassifier()
    correctness, reason = classifier.classify(None)
    assert correctness == "indeterminate"
    assert reason == "missing_evaluation_result"


def test_classify_missing_uses_the_failures_error_type():
    classifier = CorrectnessClassifier()
    correctness, reason = classifier.classify_missing("empty_test_suite")
    assert correctness == "indeterminate"
    assert reason == "empty_test_suite"


def test_candidate_runtime_exception_is_incorrect():
    # The real p008 shape: tests_error > 0 with runtime_error=true.
    classifier = CorrectnessClassifier()
    result = make_result(status="failed", tests_passed=8, tests_failed=0, tests_error=1)
    correctness, reason = classifier.classify(result)
    assert correctness == "incorrect"
    assert reason is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"status": "passed", "tests_passed": 5, "tests_failed": 0},
        {"status": "failed", "tests_passed": 4, "tests_failed": 1},
        {"status": "syntax_error", "tests_passed": 0, "tests_failed": 0, "tests_error": 5},
    ],
)
def test_classification_is_always_one_of_the_three_values(kwargs):
    classifier = CorrectnessClassifier()
    result = make_result(**kwargs)
    correctness, _ = classifier.classify(result)
    assert correctness in {"correct", "incorrect", "indeterminate"}
