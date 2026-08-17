"""Tests for RankingRepository (spec 07 section 53)."""

from __future__ import annotations

from typing import Any

import pytest

from python_dpo.ranking import (
    CandidateAssessment,
    ComparisonResult,
    RankingRepository,
    RankingResult,
    RankingStoreError,
)

RANK_RUN_ID = "rank_20260817_180500_a91c"


def make_assessment(**overrides: Any) -> CandidateAssessment:
    fields: dict[str, Any] = {
        "ranking_run_id": RANK_RUN_ID,
        "evaluation_run_id": "eval_x",
        "candidate_run_id": "run_x",
        "candidate_id": "p001_c001",
        "problem_id": "p001",
        "correctness": "correct",
        "all_tests_passed": True,
        "pass_rate": 1.0,
        "score": 1.0,
        "tests_total": 8,
        "tests_passed": 8,
        "tests_failed": 0,
        "tests_error": 0,
        "tests_skipped": 0,
        "timeout": False,
        "infrastructure_error": False,
    }
    fields.update(overrides)
    return CandidateAssessment(**fields)


def make_ranking(**overrides: Any) -> RankingResult:
    fields: dict[str, Any] = {
        "ranking_run_id": RANK_RUN_ID,
        "evaluation_run_id": "eval_x",
        "problem_id": "p001",
        "candidate_id": "p001_c001",
        "score": 1.0,
        "correctness": "correct",
        "pass_rate": 1.0,
        "all_tests_passed": True,
        "eligible_for_preference": True,
        "rank": 1,
        "tie_group": "p001_tg001",
        "tie_group_size": 1,
        "tied": False,
    }
    fields.update(overrides)
    return RankingResult(**fields)


def make_comparison(**overrides: Any) -> ComparisonResult:
    fields: dict[str, Any] = {
        "ranking_run_id": RANK_RUN_ID,
        "problem_id": "p001",
        "candidate_a": "p001_c001",
        "candidate_b": "p001_c002",
        "relation": "A_BETTER",
        "score_a": 1.0,
        "score_b": 0.5,
        "score_margin": 0.5,
        "correctness_a": "correct",
        "correctness_b": "incorrect",
        "comparison_eligible": True,
    }
    fields.update(overrides)
    return ComparisonResult(**fields)


# -------------------------------------------------------------------------- save/load


def test_save_and_load_assessments_round_trip(tmp_path):
    repo = RankingRepository(tmp_path)
    assert repo.load_assessments() == []

    first = make_assessment()
    second = make_assessment(candidate_id="p001_c002")
    repo.save_assessment(first)
    repo.save_assessment(second)

    assert repo.load_assessments() == [first, second]
    assert repo.assessments_path.name == "assessments.jsonl"


def test_save_and_load_rankings_round_trip(tmp_path):
    repo = RankingRepository(tmp_path)
    ranking = make_ranking()
    repo.save_ranking(ranking)
    assert repo.load_rankings() == [ranking]
    assert repo.rankings_path.name == "rankings.jsonl"


def test_save_and_load_comparisons_round_trip(tmp_path):
    repo = RankingRepository(tmp_path)
    comparison = make_comparison()
    repo.save_comparison(comparison)
    assert repo.load_comparisons() == [comparison]
    assert repo.comparisons_path.name == "comparisons.jsonl"


def test_save_many_helpers(tmp_path):
    repo = RankingRepository(tmp_path)
    repo.save_assessments([make_assessment(), make_assessment(candidate_id="p001_c002")])
    repo.save_rankings([make_ranking(), make_ranking(candidate_id="p001_c002")])
    repo.save_comparisons([make_comparison()])
    assert repo.count() == 2
    assert len(repo.load_rankings()) == 2
    assert len(repo.load_comparisons()) == 1


# ------------------------------------------------------------------ spec section 53 API


def test_get_assessment(tmp_path):
    repo = RankingRepository(tmp_path)
    repo.save_assessment(make_assessment())
    assert repo.get_assessment("p001_c001").candidate_id == "p001_c001"
    assert repo.get_assessment("does-not-exist") is None


def test_get_ranking(tmp_path):
    repo = RankingRepository(tmp_path)
    repo.save_ranking(make_ranking())
    assert repo.get_ranking("p001_c001").candidate_id == "p001_c001"
    assert repo.get_ranking("does-not-exist") is None


def test_list_problem_rankings(tmp_path):
    repo = RankingRepository(tmp_path)
    repo.save_ranking(make_ranking(problem_id="p001"))
    repo.save_ranking(make_ranking(candidate_id="p002_c001", problem_id="p002"))
    assert [r.candidate_id for r in repo.list_problem_rankings("p001")] == ["p001_c001"]


def test_list_all_rankings(tmp_path):
    repo = RankingRepository(tmp_path)
    repo.save_ranking(make_ranking())
    repo.save_ranking(make_ranking(candidate_id="p001_c002"))
    assert len(repo.list_all_rankings()) == 2


def test_count_counts_assessments(tmp_path):
    repo = RankingRepository(tmp_path)
    assert repo.count() == 0
    repo.save_assessment(make_assessment())
    assert repo.count() == 1


# ---------------------------------------------------------------------------- resume index


def test_ranked_problem_ids(tmp_path):
    repo = RankingRepository(tmp_path)
    repo.save_ranking(make_ranking(problem_id="p001"))
    repo.save_ranking(make_ranking(candidate_id="p002_c001", problem_id="p002"))
    assert repo.ranked_problem_ids() == {"p001", "p002"}


def test_ranked_problem_ids_empty_for_fresh_repository(tmp_path):
    assert RankingRepository(tmp_path).ranked_problem_ids() == set()


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
    repo = RankingRepository(tmp_path)
    repo.assessments_path.parent.mkdir(parents=True, exist_ok=True)
    repo.assessments_path.write_text(content, encoding="utf-8")

    with pytest.raises(RankingStoreError, match=match):
        repo.load_assessments()


def test_truncated_final_line_is_rejected(tmp_path):
    repo = RankingRepository(tmp_path)
    repo.save_assessment(make_assessment())
    with repo.assessments_path.open("a", encoding="utf-8") as handle:
        handle.write('{"candidate_id": "p001_c002"')  # torn write, no trailing newline

    with pytest.raises(RankingStoreError, match="truncated final line"):
        repo.load_assessments()
