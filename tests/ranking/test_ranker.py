"""Tests for CandidateRanker (spec 07 sections 7, 8, 28-31, 39, 46, 62-64, 72, 73)."""

from __future__ import annotations

from typing import Any

import pytest

from python_dpo.ranking.models import CandidateAssessment
from python_dpo.ranking.ranker import CandidateRanker

RANK_RUN_ID = "rank_20260817_180500_a91c"


def make_assessment(
    candidate_id: str, passed: int, total: int = 10, problem_id: str = "p001", **overrides: Any
) -> CandidateAssessment:
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
        "problem_id": problem_id,
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


def by_id(results):
    return {r.candidate_id: r for r in results}


# ----------------------------------------------------------------- spec section 73 example


def test_section_73_worked_example():
    assessments = [
        make_assessment("A", 10),
        make_assessment("B", 8),
        make_assessment("C", 10),
        make_assessment("D", 5),
        make_assessment("E", 0),
    ]
    ranker = CandidateRanker()
    results = by_id(ranker.rank_problem(RANK_RUN_ID, "p001", assessments))

    assert results["A"].rank == 1
    assert results["C"].rank == 1
    assert results["B"].rank == 3
    assert results["D"].rank == 4
    assert results["E"].rank == 5

    assert results["A"].tie_group == results["C"].tie_group
    assert results["A"].tied is True
    assert results["C"].tied is True
    assert results["B"].tied is False


def test_a_and_c_tied_neither_is_declared_better():
    assessments = [make_assessment("A", 10), make_assessment("C", 10)]
    ranker = CandidateRanker()
    results = ranker.rank_problem(RANK_RUN_ID, "p001", assessments)
    assert {r.rank for r in results} == {1}
    assert len({r.tie_group for r in results}) == 1


# --------------------------------------------------------------------------- unique scores


def test_unique_scores_produce_sequential_ranks():
    assessments = [
        make_assessment("A", 10),
        make_assessment("B", 8),
        make_assessment("C", 6),
    ]
    ranker = CandidateRanker()
    results = by_id(ranker.rank_problem(RANK_RUN_ID, "p001", assessments))
    assert results["A"].rank == 1
    assert results["B"].rank == 2
    assert results["C"].rank == 3
    assert all(not r.tied for r in results.values())


# --------------------------------------------------------------------------------- all-correct


def test_all_correct_produces_one_tie_group_and_zero_preference_relationships():
    # Spec section 62: 5 correct candidates, 0 preference relationships from correctness
    # alone.
    assessments = [make_assessment(f"c{i}", 10) for i in range(5)]
    ranker = CandidateRanker()
    results = ranker.rank_problem(RANK_RUN_ID, "p001", assessments)
    assert all(r.rank == 1 for r in results)
    assert len({r.tie_group for r in results}) == 1
    assert all(r.tied for r in results)
    assert all(r.eligible_for_preference for r in results)


# -------------------------------------------------------------------------------- all-incorrect


def test_all_incorrect_with_distinct_rates_preserves_order():
    # Spec section 63.
    assessments = [
        make_assessment("A", 9),
        make_assessment("B", 8),
        make_assessment("C", 6),
        make_assessment("D", 4),
        make_assessment("E", 1),
    ]
    ranker = CandidateRanker()
    results = by_id(ranker.rank_problem(RANK_RUN_ID, "p001", assessments))
    assert [results[c].rank for c in "ABCDE"] == [1, 2, 3, 4, 5]
    assert all(r.correctness == "incorrect" for r in results.values())


# ----------------------------------------------------------------------------------- mixed


def test_mixed_correct_and_incorrect():
    assessments = [
        make_assessment("A", 10),
        make_assessment("B", 8),
        make_assessment("C", 10),
        make_assessment("D", 5),
        make_assessment("E", 0),
    ]
    ranker = CandidateRanker()
    results = by_id(ranker.rank_problem(RANK_RUN_ID, "p001", assessments))
    assert results["A"].correctness == "correct"
    assert results["B"].correctness == "incorrect"
    # correct always outranks incorrect regardless of raw tests_passed (spec section 36).
    assert results["A"].rank < results["B"].rank


# ------------------------------------------------------------------------------ indeterminate


def test_all_indeterminate_yields_zero_preference_eligible():
    # Spec section 64.
    assessments = [make_assessment(f"c{i}", 0, correctness="indeterminate") for i in range(5)]
    ranker = CandidateRanker()
    results = ranker.rank_problem(RANK_RUN_ID, "p001", assessments)
    assert all(r.rank is None for r in results)
    assert all(r.tie_group is None for r in results)
    assert all(not r.eligible_for_preference for r in results)
    assert all(not r.tied for r in results)


def test_indeterminate_candidates_are_recorded_not_dropped():
    # Spec section 71: no silent data loss.
    assessments = [make_assessment("A", 10), make_assessment("B", 0, correctness="indeterminate")]
    ranker = CandidateRanker()
    results = by_id(ranker.rank_problem(RANK_RUN_ID, "p001", assessments))
    assert set(results) == {"A", "B"}
    assert results["B"].rank is None
    assert results["A"].rank == 1


def test_mixed_ranked_and_indeterminate_ranked_candidates_unaffected():
    assessments = [
        make_assessment("A", 10),
        make_assessment("B", 8),
        make_assessment("Z", 0, correctness="indeterminate"),
    ]
    ranker = CandidateRanker()
    results = by_id(ranker.rank_problem(RANK_RUN_ID, "p001", assessments))
    assert results["A"].rank == 1
    assert results["B"].rank == 2
    assert results["Z"].rank is None


# ------------------------------------------------------------------------------- transitivity


def test_transitivity_a_beats_b_b_beats_c_implies_a_beats_c():
    # Spec section 59.
    assessments = [make_assessment("A", 10), make_assessment("B", 8), make_assessment("C", 5)]
    ranker = CandidateRanker()
    results = by_id(ranker.rank_problem(RANK_RUN_ID, "p001", assessments))
    assert results["A"].rank < results["B"].rank < results["C"].rank


# --------------------------------------------------------------- candidate_id presentation only


def test_candidate_id_breaks_ties_only_for_presentation_never_the_tied_flag():
    # Spec section 31: candidate_id may order two equal-score candidates for display, but
    # both must still be marked tied — the ordering can never imply a preference.
    assessments = [make_assessment("zzz", 10), make_assessment("aaa", 10)]
    ranker = CandidateRanker()
    results = by_id(ranker.rank_problem(RANK_RUN_ID, "p001", assessments))
    assert results["zzz"].rank == results["aaa"].rank == 1
    assert results["zzz"].tied and results["aaa"].tied
    assert results["zzz"].tie_group == results["aaa"].tie_group


# ------------------------------------------------------------------------- competition ranking


def test_competition_rank_skips_by_tie_group_size():
    # 1, 1, 1, 4, 5 — a three-way tie for first, then rank jumps to 4.
    assessments = [
        make_assessment("A", 10),
        make_assessment("B", 10),
        make_assessment("C", 10),
        make_assessment("D", 5),
        make_assessment("E", 2),
    ]
    ranker = CandidateRanker()
    results = by_id(ranker.rank_problem(RANK_RUN_ID, "p001", assessments))
    assert [results[c].rank for c in "ABC"] == [1, 1, 1]
    assert results["D"].rank == 4
    assert results["E"].rank == 5


# ------------------------------------------------------------------------------ cross-problem


def test_rank_groups_strictly_by_problem_never_compares_across_problems():
    # Spec section 8.
    assessments = [
        make_assessment("p001_c001", 10, problem_id="p001"),
        make_assessment("p002_c001", 3, problem_id="p002"),
    ]
    ranker = CandidateRanker()
    results = by_id(ranker.rank(RANK_RUN_ID, assessments))
    # Each is alone in its own problem, so each is rank 1 regardless of score.
    assert results["p001_c001"].rank == 1
    assert results["p002_c001"].rank == 1
    assert results["p001_c001"].tie_group != results["p002_c001"].tie_group


def test_rank_problem_rejects_mismatched_problem_ids():
    ranker = CandidateRanker()
    with pytest.raises(ValueError, match="different problem"):
        ranker.rank_problem(
            RANK_RUN_ID,
            "p001",
            [make_assessment("A", 10, problem_id="p001"), make_assessment("B", 8, problem_id="p002")],
        )


def test_mismatched_tests_total_within_a_problem_is_rejected():
    # The ranker's own structural invariant: tests_total must be constant within a
    # problem, or integer tests_passed comparison would be meaningless.
    ranker = CandidateRanker()
    with pytest.raises(ValueError, match="tests_total"):
        ranker.rank_problem(
            RANK_RUN_ID,
            "p001",
            [make_assessment("A", 10, total=10), make_assessment("B", 8, total=9)],
        )
