"""Tests for EvaluationRepository (spec 06 section 52)."""

from __future__ import annotations

from typing import Any

import pytest

from python_dpo.evaluation import EvaluationFailure, EvaluationResult, EvaluationStoreError, TestCaseResult
from python_dpo.evaluation.repository import EvaluationRepository
from python_dpo.evaluation.models import utc_now_iso


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


def make_failure(**overrides: Any) -> EvaluationFailure:
    fields: dict[str, Any] = {
        "evaluation_run_id": "eval_20260817_154500_a12f",
        "candidate_run_id": "run_20260817_055411",
        "candidate_id": "p001_c002",
        "problem_id": "p001",
        "error_type": "empty_test_suite",
        "error_message": "problem p001 has no test cases",
        "timestamp": utc_now_iso(),
    }
    fields.update(overrides)
    return EvaluationFailure(**fields)


# ------------------------------------------------------------------------------- save/load


def test_save_and_load_round_trip(tmp_path):
    repo = EvaluationRepository(tmp_path)
    assert repo.load_all() == []

    first = make_result()
    second = make_result(candidate_id="p001_c002", status="failed", tests_passed=6, tests_failed=2, tests_error=0, tests_skipped=0)
    repo.save(first)
    repo.save(second)

    assert repo.load_all() == [first, second]
    assert repo.evaluations_path.name == "evaluations.jsonl"


def test_records_are_readable_before_a_run_finishes(tmp_path):
    repo = EvaluationRepository(tmp_path)
    repo.save(make_result())
    assert len(EvaluationRepository(tmp_path).load_all()) == 1


def test_failures_are_persisted_separately(tmp_path):
    repo = EvaluationRepository(tmp_path)
    failure = make_failure()
    repo.save_failure(failure)
    assert repo.load_failures() == [failure]
    assert repo.failures_path.name == "failures.jsonl"
    assert not repo.evaluations_path.exists()


def test_test_results_are_persisted_and_loadable(tmp_path):
    repo = EvaluationRepository(tmp_path)
    results = [
        make_test_result(test_case_id="p001_t001"),
        make_test_result(test_case_id="p001_t002", status="failed", error_type="AssertionError"),
    ]
    repo.append_test_results(results)
    assert repo.load_test_results() == results
    assert repo.test_results_path.name == "test_results.jsonl"


# ------------------------------------------------------------------------ spec section 52


def test_get_returns_the_matching_result(tmp_path):
    repo = EvaluationRepository(tmp_path)
    repo.save(make_result())
    assert repo.get("p001_c001").candidate_id == "p001_c001"
    assert repo.get("does-not-exist") is None


def test_list_and_count(tmp_path):
    repo = EvaluationRepository(tmp_path)
    repo.save(make_result())
    repo.save(make_result(candidate_id="p001_c002"))
    assert repo.count() == 2
    assert [r.candidate_id for r in repo.list()] == ["p001_c001", "p001_c002"]


def test_find_by_candidate(tmp_path):
    repo = EvaluationRepository(tmp_path)
    repo.save(make_result())
    assert repo.find_by_candidate("p001_c001").candidate_id == "p001_c001"


def test_find_by_problem(tmp_path):
    repo = EvaluationRepository(tmp_path)
    repo.save(make_result(problem_id="p001"))
    repo.save(make_result(candidate_id="p002_c001", problem_id="p002"))
    assert [r.candidate_id for r in repo.find_by_problem("p001")] == ["p001_c001"]


def test_test_results_for_one_candidate(tmp_path):
    repo = EvaluationRepository(tmp_path)
    repo.append_test_results([
        make_test_result(candidate_id="p001_c001", test_case_id="p001_t001"),
        make_test_result(candidate_id="p001_c002", test_case_id="p001_t001"),
    ])
    assert [t.candidate_id for t in repo.test_results_for("p001_c001")] == ["p001_c001"]


# ---------------------------------------------------------------------------- resume index


def test_evaluated_keys_covers_both_results_and_failures(tmp_path):
    repo = EvaluationRepository(tmp_path)
    repo.save(make_result(candidate_id="p001_c001"))
    repo.save_failure(make_failure(candidate_id="p001_c002"))
    assert repo.evaluated_keys() == {"p001_c001", "p001_c002"}


def test_evaluated_keys_is_empty_for_a_fresh_repository(tmp_path):
    assert EvaluationRepository(tmp_path).evaluated_keys() == set()


# --------------------------------------------------------------------------- malformed data


@pytest.mark.parametrize(
    "content, match",
    [
        ("not json\n", "invalid JSON"),
        ('{"candidate_id": "p001_c001"}\n', "missing required field"),
        ("[1, 2]\n", "expected a JSON object"),
    ],
)
def test_malformed_lines_are_rejected_with_a_line_number(tmp_path, content, match):
    repo = EvaluationRepository(tmp_path)
    repo.evaluations_path.parent.mkdir(parents=True, exist_ok=True)
    repo.evaluations_path.write_text(content, encoding="utf-8")

    with pytest.raises(EvaluationStoreError, match=match):
        repo.load_all()


def test_truncated_final_line_is_rejected(tmp_path):
    repo = EvaluationRepository(tmp_path)
    repo.save(make_result())
    with repo.evaluations_path.open("a", encoding="utf-8") as handle:
        handle.write('{"candidate_id": "p001_c002"')  # torn write, no trailing newline

    with pytest.raises(EvaluationStoreError, match="truncated final line"):
        repo.load_all()
