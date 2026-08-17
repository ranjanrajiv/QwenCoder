"""Tests for CandidateComparator (spec 07 sections 33-38, 66-68, 72, 74)."""

from __future__ import annotations

from typing import Any

import pytest

from python_dpo.ranking.comparator import CandidateComparator
from python_dpo.ranking.models import CandidateAssessment

RANK_RUN_ID = "rank_20260817_180500_a91c"


def make_assessment(candidate_id: str, passed: int, total: int = 10, **overrides: Any) -> CandidateAssessment:
    correctness = overrides.pop("correctness", None)
    if correctness is None:
        correctness = "correct" if passed == total and total > 0 else "incorrect"
    if correctness == "indeterminate":
        total = 0
        passed = 0
    fields: dict[str, Any] = {
        "ranking_run_id": RANK_RUN_ID,
        "evaluation_run_id": "eval_x",
        "candidate_run_id": "run_x",
        "candidate_id": candidate_id,
        "problem_id": "p001",
        "correctness": correctness,
        "all_tests_passed": total > 0 and passed == total,
        "pass_rate": passed / total if total > 0 else 0.0,
        "score": passed / total if total > 0 else 0.0,
        "tests_total": total,
        "tests_passed": passed,
        "tests_failed": total - passed if correctness != "indeterminate" else 0,
        "tests_error": 0,
        "tests_skipped": 0,
        "timeout": False,
        "infrastructure_error": correctness == "indeterminate",
        "indeterminate_reason": "infrastructure_error" if correctness == "indeterminate" else None,
    }
    fields.update(overrides)
    return CandidateAssessment(**fields)


def test_ten_vs_eight_a_better():
    comparator = CandidateComparator()
    result = comparator.compare(RANK_RUN_ID, make_assessment("A", 10), make_assessment("B", 8))
    assert result.relation == "A_BETTER"
    assert result.comparison_eligible is True
    assert abs(result.score_margin - 0.2) < 1e-9


def test_eight_vs_five_a_better():
    comparator = CandidateComparator()
    result = comparator.compare(RANK_RUN_ID, make_assessment("A", 8), make_assessment("B", 5))
    assert result.relation == "A_BETTER"
    assert result.comparison_eligible is True


def test_ten_vs_ten_tie():
    comparator = CandidateComparator()
    result = comparator.compare(RANK_RUN_ID, make_assessment("A", 10), make_assessment("B", 10))
    assert result.relation == "TIE"
    assert result.comparison_eligible is False
    assert result.score_margin == 0.0


def test_indeterminate_side_produces_indeterminate_relation():
    comparator = CandidateComparator()
    result = comparator.compare(
        RANK_RUN_ID, make_assessment("A", 0, correctness="indeterminate"), make_assessment("B", 8)
    )
    assert result.relation == "INDETERMINATE"
    assert result.comparison_eligible is False


def test_both_indeterminate_is_indeterminate():
    comparator = CandidateComparator()
    result = comparator.compare(
        RANK_RUN_ID,
        make_assessment("A", 0, correctness="indeterminate"),
        make_assessment("B", 0, correctness="indeterminate"),
    )
    assert result.relation == "INDETERMINATE"


def test_a_timed_out_candidate_is_still_incorrect_not_special_cased():
    # A candidate-caused timeout is incorrect (classifier's job), not indeterminate; the
    # comparator just compares scores like any other incorrect candidate.
    comparator = CandidateComparator()
    timed_out = make_assessment("A", 0, correctness="incorrect", timeout=True)
    result = comparator.compare(RANK_RUN_ID, timed_out, make_assessment("B", 5))
    assert result.relation == "B_BETTER"
    assert result.comparison_eligible is True


def test_score_margin_arithmetic():
    comparator = CandidateComparator()
    result = comparator.compare(RANK_RUN_ID, make_assessment("A", 10), make_assessment("B", 8))
    assert abs(result.score_margin - abs(result.score_a - result.score_b)) < 1e-9


def test_comparing_different_problems_is_rejected():
    comparator = CandidateComparator()
    a = make_assessment("A", 10)
    b = make_assessment("B", 8, problem_id="p002")
    with pytest.raises(ValueError, match="different problems"):
        comparator.compare(RANK_RUN_ID, a, b)


# ------------------------------------------------------------------- spec section 74 matrix


def test_matrix_matches_the_spec_section_74_example():
    a = make_assessment("A", 10)
    b = make_assessment("B", 8)
    c = make_assessment("C", 10)
    d = make_assessment("D", 5)
    e = make_assessment("E", 0, correctness="incorrect")

    comparator = CandidateComparator()
    matrix = comparator.build_matrix(RANK_RUN_ID, [a, b, c, d, e])
    by_pair = {(m.candidate_a, m.candidate_b): m.relation for m in matrix}

    # relation is expressed as slot a vs slot b, not candidate identity; expected results
    # are given here as (candidate_a, candidate_b) -> relation of a-vs-b.
    assert by_pair[("A", "B")] == "A_BETTER"  # A > B
    assert by_pair[("A", "C")] == "TIE"  # A = C
    assert by_pair[("A", "D")] == "A_BETTER"  # A > D
    assert by_pair[("A", "E")] == "A_BETTER"  # A > E
    assert by_pair[("B", "C")] == "B_BETTER"  # B < C
    assert by_pair[("B", "D")] == "A_BETTER"  # B > D
    assert by_pair[("B", "E")] == "A_BETTER"  # B > E
    assert by_pair[("C", "D")] == "A_BETTER"  # C > D
    assert by_pair[("C", "E")] == "A_BETTER"  # C > E
    assert by_pair[("D", "E")] == "A_BETTER"  # D > E


def test_matrix_size_is_n_choose_2():
    assessments = [make_assessment(f"c{i:03d}", passed) for i, passed in enumerate([10, 8, 10, 5, 0])]
    comparator = CandidateComparator()
    matrix = comparator.build_matrix(RANK_RUN_ID, assessments)
    assert len(matrix) == 5 * 4 // 2


def test_matrix_no_pair_involving_indeterminate_produces_a_winner():
    assessments = [
        make_assessment("A", 10),
        make_assessment("B", 0, correctness="indeterminate"),
        make_assessment("C", 5),
    ]
    comparator = CandidateComparator()
    matrix = comparator.build_matrix(RANK_RUN_ID, assessments)
    for comparison in matrix:
        if "B" in (comparison.candidate_a, comparison.candidate_b):
            assert comparison.relation == "INDETERMINATE"
            assert comparison.comparison_eligible is False
