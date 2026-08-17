"""Tests for CandidateScorer (spec 07 sections 15-23, 72)."""

from __future__ import annotations

from typing import Any

from python_dpo.candidates.models import Candidate
from python_dpo.evaluation.models import EvaluationResult
from python_dpo.ranking.scorer import CandidateScorer

EVAL_RUN_ID = "eval_20260817_154500_a12f"
CANDIDATE_RUN_ID = "run_20260817_055411"
RANK_RUN_ID = "rank_20260817_180500_a91c"


def make_result(**overrides: Any) -> EvaluationResult:
    fields: dict[str, Any] = {
        "evaluation_run_id": EVAL_RUN_ID,
        "candidate_run_id": CANDIDATE_RUN_ID,
        "candidate_id": "p001_c001",
        "problem_id": "p001",
        "status": "passed",
        "tests_passed": 10,
        "tests_failed": 0,
        "tests_error": 0,
        "tests_skipped": 0,
        "duration_ms": 200,
    }
    fields.update(overrides)
    return EvaluationResult.create(**fields)


def make_candidate(**overrides: Any) -> Candidate:
    fields: dict[str, Any] = {
        "candidate_id": "p001_c001",
        "problem_id": "p001",
        "run_id": CANDIDATE_RUN_ID,
        "generation_index": 1,
        "strategy": "normal",
        "model": "mock/deterministic-coder",
        "provider": "mock",
        "prompt_version": "v1",
        "prompt": "Solve it.",
        "raw_output": "```python\ndef f():\n    return 1\n```",
        "code": "def f():\n    return 1\n",
        "extraction_format": "python_fence",
        "syntax_valid": True,
        "function_name_valid": True,
        "generation_config": {},
        "created_at": "2026-08-17T12:00:00Z",
    }
    fields.update(overrides)
    return Candidate.create(**fields)


def score_one(**kwargs: Any):
    scorer = CandidateScorer()
    base = {
        "ranking_run_id": RANK_RUN_ID,
        "evaluation_run_id": EVAL_RUN_ID,
        "candidate_run_id": CANDIDATE_RUN_ID,
        "candidate_id": "p001_c001",
        "problem_id": "p001",
    }
    base.update(kwargs)
    return scorer.score(**base)


def test_ten_of_ten_scores_one():
    a = score_one(result=make_result(tests_passed=10, tests_failed=0))
    assert a.score == 1.0
    assert a.pass_rate == 1.0
    assert a.all_tests_passed is True
    assert a.correctness == "correct"


def test_five_of_ten_scores_half():
    a = score_one(result=make_result(status="failed", tests_passed=5, tests_failed=5))
    assert a.score == 0.5
    assert a.pass_rate == 0.5
    assert a.all_tests_passed is False
    assert a.correctness == "incorrect"


def test_zero_of_ten_scores_zero():
    a = score_one(result=make_result(status="failed", tests_passed=0, tests_failed=10))
    assert a.score == 0.0
    assert a.correctness == "incorrect"


def test_fractional_rate_is_exact():
    a = score_one(result=make_result(status="failed", tests_passed=7, tests_failed=1, tests_error=0, tests_skipped=0, duration_ms=1))
    # tests_total must equal tests_passed+failed+error+skipped=8
    assert a.tests_total == 8
    assert a.pass_rate == 7 / 8
    assert a.score == a.pass_rate


def test_score_always_equals_pass_rate():
    a = score_one(result=make_result(status="failed", tests_passed=9, tests_failed=1))
    assert a.score == a.pass_rate


def test_all_tests_passed_is_distinct_from_pass_rate_at_high_but_not_perfect_rate():
    # Spec section 17: pass_rate=0.95, all_tests_passed=false must both be representable.
    a = score_one(
        result=make_result(status="failed", tests_passed=19, tests_failed=1, duration_ms=1)
    )
    assert abs(a.pass_rate - 0.95) < 1e-9
    assert a.all_tests_passed is False


def test_missing_evaluation_result_produces_an_indeterminate_zero_score():
    a = score_one(result=None, missing_error_type="empty_test_suite")
    assert a.correctness == "indeterminate"
    assert a.score == 0.0
    assert a.tests_total == 0
    assert a.indeterminate_reason == "empty_test_suite"


# --------------------------------------------------- secondary metadata never affects score


def test_duration_does_not_affect_score():
    fast = score_one(result=make_result(tests_passed=10, tests_failed=0, duration_ms=1))
    slow = score_one(result=make_result(tests_passed=10, tests_failed=0, duration_ms=99999))
    assert fast.score == slow.score == 1.0


def test_code_length_does_not_affect_score():
    short = score_one(
        result=make_result(tests_passed=10, tests_failed=0),
        candidate=make_candidate(code="def f():\n    return 1\n"),
    )
    long = score_one(
        result=make_result(tests_passed=10, tests_failed=0),
        candidate=make_candidate(code="def f():\n" + "    x = 1\n" * 200 + "    return 1\n"),
    )
    assert short.score == long.score == 1.0
    assert short.code_lines != long.code_lines  # metadata differs, score does not


def test_strategy_does_not_affect_score():
    a = score_one(
        result=make_result(tests_passed=10, tests_failed=0),
        candidate=make_candidate(strategy="normal", candidate_id="p001_c001"),
    )
    b = score_one(
        result=make_result(tests_passed=10, tests_failed=0),
        candidate=make_candidate(strategy="optimized", candidate_id="p001_c001"),
    )
    assert a.score == b.score == 1.0
    assert a.strategy != b.strategy


def test_syntax_validity_does_not_affect_score():
    # Spec section 23: correctness comes from sandbox execution, not the static check.
    a = score_one(
        result=make_result(tests_passed=10, tests_failed=0),
        candidate=make_candidate(syntax_valid=True),
    )
    assert a.score == 1.0
    assert a.syntax_valid is True


# ----------------------------------------------------------------- candidate metadata join


def test_candidate_metadata_is_recorded_when_present():
    candidate = make_candidate(
        code="def f():\n    return 1\n",
        duplicate_of="p001_c000",
        strategy="edge_case_focused",
    )
    a = score_one(result=make_result(tests_passed=10, tests_failed=0), candidate=candidate)
    assert a.duplicate_of == "p001_c000"
    assert a.strategy == "edge_case_focused"
    assert a.code_lines == 2
    assert a.code_chars == len("def f():\n    return 1\n")
    assert a.code_sha256 == candidate.code_sha256


def test_secondary_metadata_is_none_when_candidate_is_absent():
    a = score_one(result=make_result(tests_passed=10, tests_failed=0), candidate=None)
    assert a.code_sha256 is None
    assert a.duplicate_of is None
    assert a.code_lines is None
    assert a.code_chars is None
    assert a.strategy is None
    assert a.syntax_valid is None
