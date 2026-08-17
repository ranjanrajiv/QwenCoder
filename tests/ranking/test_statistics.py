"""Tests for RankingStatistics.from_records and the text formatters (spec 07 sections
40, 41, 49, 79).
"""

from __future__ import annotations

from typing import Any

from python_dpo.ranking.models import (
    CandidateAssessment,
    RankingManifest,
    RankingResult,
    RankingStatistics,
)
from python_dpo.ranking.statistics import format_ranking_statistics, format_ranking_table

RANK_RUN_ID = "rank_20260817_180500_a91c"


def make_manifest(**overrides: Any) -> RankingManifest:
    fields: dict[str, Any] = {
        "ranking_run_id": RANK_RUN_ID,
        "evaluation_run_id": "eval_x",
        "candidate_run_id": "run_x",
        "status": "completed",
        "created_at": "2026-08-17T18:05:00Z",
        "ranking_version": "v1",
        "scoring_version": "v1",
        "comparator_version": "v1",
        "requested_problem_ids": ("p001", "p002"),
    }
    fields.update(overrides)
    return RankingManifest(**fields)


def make_assessment(candidate_id: str, problem_id: str, passed: int, total: int, correctness: str, **overrides: Any) -> CandidateAssessment:
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


def make_ranking(candidate_id: str, problem_id: str, correctness: str, tied: bool, eligible: bool, **overrides: Any) -> RankingResult:
    fields: dict[str, Any] = {
        "ranking_run_id": RANK_RUN_ID,
        "evaluation_run_id": "eval_x",
        "problem_id": problem_id,
        "candidate_id": candidate_id,
        "score": 1.0,
        "correctness": correctness,
        "pass_rate": 1.0,
        "all_tests_passed": correctness == "correct",
        "eligible_for_preference": eligible,
        "rank": None if correctness == "indeterminate" else 1,
        "tie_group": None if correctness == "indeterminate" else f"{problem_id}_tg001",
        "tie_group_size": 0 if correctness == "indeterminate" else (2 if tied else 1),
        "tied": tied,
    }
    fields.update(overrides)
    return RankingResult(**fields)


def test_hand_counted_statistics():
    manifest = make_manifest()
    assessments = [
        make_assessment("p001_c001", "p001", 10, 10, "correct"),
        make_assessment("p001_c002", "p001", 7, 10, "incorrect"),
        make_assessment("p001_c003", "p001", 0, 10, "incorrect"),
        make_assessment("p002_c001", "p002", 0, 0, "indeterminate"),
    ]
    rankings = [
        make_ranking("p001_c001", "p001", "correct", tied=False, eligible=True),
        make_ranking("p001_c002", "p001", "incorrect", tied=False, eligible=True),
        make_ranking("p001_c003", "p001", "incorrect", tied=False, eligible=True),
        make_ranking("p002_c001", "p002", "indeterminate", tied=False, eligible=False),
    ]
    stats = RankingStatistics.from_records(manifest, assessments, rankings)

    assert stats.problems == 2
    assert stats.candidates == 4
    assert stats.correct == 1
    assert stats.incorrect == 2
    assert stats.indeterminate == 1
    assert stats.fully_correct == 1
    assert stats.partially_correct == 1  # 7/10
    assert stats.zero_test_pass == 1  # 0/10
    assert stats.tied_candidates == 0
    assert stats.preference_eligible_candidates == 3


def test_per_problem_distribution():
    manifest = make_manifest()
    assessments = [
        make_assessment("p001_c001", "p001", 10, 10, "correct"),
        make_assessment("p001_c002", "p001", 10, 10, "correct"),
        make_assessment("p002_c001", "p002", 5, 10, "incorrect"),
    ]
    rankings = [
        make_ranking("p001_c001", "p001", "correct", tied=True, eligible=True),
        make_ranking("p001_c002", "p001", "correct", tied=True, eligible=True),
        make_ranking("p002_c001", "p002", "incorrect", tied=False, eligible=True),
    ]
    stats = RankingStatistics.from_records(manifest, assessments, rankings)

    assert stats.per_problem["p001"] == {
        "total": 2,
        "fully_correct": 2,
        "partially_correct": 0,
        "zero_test_pass": 0,
        "indeterminate": 0,
    }
    assert stats.per_problem["p002"] == {
        "total": 1,
        "fully_correct": 0,
        "partially_correct": 1,
        "zero_test_pass": 0,
        "indeterminate": 0,
    }
    assert stats.tied_candidates == 2


def test_statistics_round_trips_through_dict():
    manifest = make_manifest()
    assessments = [make_assessment("p001_c001", "p001", 10, 10, "correct")]
    rankings = [make_ranking("p001_c001", "p001", "correct", tied=False, eligible=True)]
    stats = RankingStatistics.from_records(manifest, assessments, rankings)
    assert RankingStatistics.from_dict(stats.to_dict()) == stats


def test_format_ranking_statistics_includes_every_section_40_counter():
    manifest = make_manifest()
    assessments = [make_assessment("p001_c001", "p001", 10, 10, "correct")]
    rankings = [make_ranking("p001_c001", "p001", "correct", tied=False, eligible=True)]
    stats = RankingStatistics.from_records(manifest, assessments, rankings)
    text = format_ranking_statistics(stats)
    for label in (
        "Problems", "Candidates", "Correct", "Incorrect", "Indeterminate",
        "Fully correct", "Partially correct", "Zero test pass",
        "Tied candidates", "Preference eligible candidates",
    ):
        assert label in text


def test_format_ranking_table_matches_the_section_79_shape():
    a = make_assessment("p001_c001", "p001", 10, 10, "correct")
    b = make_assessment("p001_c003", "p001", 10, 10, "correct")
    c = make_assessment("p001_c002", "p001", 8, 10, "incorrect")
    d = make_assessment("p001_c004", "p001", 5, 10, "incorrect")
    e = make_assessment("p001_c005", "p001", 0, 10, "incorrect")

    ra = make_ranking("p001_c001", "p001", "correct", tied=True, eligible=True, rank=1, score=1.0)
    rb = make_ranking("p001_c003", "p001", "correct", tied=True, eligible=True, rank=1, score=1.0)
    rc = make_ranking("p001_c002", "p001", "incorrect", tied=False, eligible=True, rank=3, score=0.8)
    rd = make_ranking("p001_c004", "p001", "incorrect", tied=False, eligible=True, rank=4, score=0.5)
    re_ = make_ranking("p001_c005", "p001", "incorrect", tied=False, eligible=True, rank=5, score=0.0)

    table = format_ranking_table([(ra, a), (rb, b), (rc, c), (rd, d), (re_, e)])
    lines = table.strip().splitlines()
    assert lines[0].split()[0] == "RANK"
    # Rank 1 entries come before rank 3, which comes before rank 4 and 5.
    ranks_in_order = [line.split()[0] for line in lines[1:]]
    assert ranks_in_order == ["1", "1", "3", "4", "5"]


def test_format_ranking_table_lists_indeterminate_last():
    ranked = make_assessment("p001_c001", "p001", 10, 10, "correct")
    indet = make_assessment("p001_c002", "p001", 0, 0, "indeterminate")
    r1 = make_ranking("p001_c001", "p001", "correct", tied=False, eligible=True, rank=1)
    r2 = make_ranking("p001_c002", "p001", "indeterminate", tied=False, eligible=False)

    table = format_ranking_table([(r2, indet), (r1, ranked)])
    lines = table.strip().splitlines()[1:]
    assert lines[0].startswith("1")
    assert lines[1].startswith("-")
