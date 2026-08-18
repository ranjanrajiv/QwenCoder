"""Tests for PreferenceStatistics.from_records / build_quality_report / formatters
(spec 08 sections 54, 75, 76, 80, 81).
"""

from __future__ import annotations

from python_dpo.preferences.models import build_quality_report, derive_candidates_considered
from python_dpo.preferences.statistics import (
    format_pair_detail,
    format_pair_table,
    format_preference_statistics,
    format_quality_report,
)

from .test_models import make_manifest, make_pair, make_rejection


def test_every_section_54_counter_matches_a_hand_count():
    manifest = make_manifest()
    pairs = [
        make_pair(preference_id="pref_p001_c001__p001_c002"),  # strong
        make_pair(
            preference_id="pref_p002_c001__p002_c002",
            problem_id="p002",
            chosen_candidate_id="p002_c001",
            rejected_candidate_id="p002_c002",
            chosen_correctness="incorrect",
            rejected_correctness="incorrect",
            chosen_score=0.8,
            rejected_score=0.5,
            score_margin=0.3,
            chosen_pass_rate=0.8,
            rejected_pass_rate=0.5,
            chosen_tests_passed=8,
            rejected_tests_passed=5,
            preference_strength="medium",
        ),
    ]
    rejections = [
        make_rejection(problem_id="p003", candidate_a="p003_c001", candidate_b="p003_c002", reason="tie"),
        make_rejection(
            problem_id="p003",
            candidate_a="p003_c003",
            candidate_b="p003_c004",
            reason="tie",
        ),
        make_rejection(
            problem_id="p004",
            candidate_a="p004_c001",
            candidate_b="p004_c002",
            reason="indeterminate",
            relation="INDETERMINATE",
        ),
        make_rejection(
            problem_id="p005",
            candidate_a="p005_c001",
            candidate_b="p005_c002",
            reason="identical_code",
        ),
        make_rejection(
            problem_id="p006",
            candidate_a="p006_c001",
            candidate_b="p006_c002",
            reason="invalid_prompt_match",
        ),
        make_rejection(
            problem_id="p007",
            candidate_a="p007_c001",
            candidate_b="p007_c002",
            reason="integrity_failure",
        ),
    ]
    stats = build_stats(manifest, pairs, rejections)

    assert stats.problems_processed == 7  # p001..p007
    assert stats.candidates_considered == derive_candidates_considered(pairs, rejections)
    assert stats.candidate_pairs_considered == len(pairs) + len(rejections)
    assert stats.pairs_generated == 2
    assert stats.pairs_rejected == 6
    assert stats.ties == 2
    assert stats.duplicates == 1
    assert stats.indeterminate == 1
    assert stats.prompt_mismatches == 1
    assert stats.integrity_failures == 1
    assert stats.strong_pairs == 1
    assert stats.medium_pairs == 1
    assert stats.training_records == 2


def build_stats(manifest, pairs, rejections):
    from python_dpo.preferences.models import PreferenceStatistics

    return PreferenceStatistics.from_records(
        manifest, pairs, rejections, candidates_considered=derive_candidates_considered(pairs, rejections)
    )


def test_per_problem_distribution():
    manifest = make_manifest()
    pairs = [make_pair(preference_id="pref_p001_c001__p001_c002", problem_id="p001")]
    rejections = [
        make_rejection(problem_id="p001", candidate_a="p001_c003", candidate_b="p001_c004", reason="tie")
    ]
    stats = build_stats(manifest, pairs, rejections)
    assert stats.per_problem["p001"]["pairs_generated"] == 1
    assert stats.per_problem["p001"]["pairs_rejected"] == 1
    assert stats.per_problem["p001"]["training_records"] == 1


# ---------------------------------------------------------------------------- quality report


def test_quality_report_problems_without_pairs_reasons():
    manifest = make_manifest()
    pairs = [make_pair(preference_id="pref_p001_c001__p001_c002", problem_id="p001")]
    problem_correctness = {
        "p001": ["correct", "incorrect"],
        "p002": ["correct", "correct", "correct"],  # all_candidates_correct
        "p003": ["incorrect", "incorrect"],  # all_candidates_tied (same non-indet value)
        "p004": ["indeterminate", "indeterminate"],  # all_candidates_indeterminate
        "p005": ["correct"],  # insufficient_candidates
    }
    report = build_quality_report(manifest, pairs, problem_correctness)
    assert report.problems_with_pairs == ["p001"]
    assert report.problems_without_pairs == {
        "p002": "all_candidates_correct",
        "p003": "all_candidates_tied",
        "p004": "all_candidates_indeterminate",
        "p005": "insufficient_candidates",
    }
    assert report.total_pairs == 1


def test_quality_report_strategy_and_margin_distributions():
    manifest = make_manifest()
    pairs = [
        make_pair(
            preference_id="pref_p001_c001__p001_c002",
            problem_id="p001",
            chosen_strategy="normal",
            rejected_strategy="alternative",
            score_margin=0.5,
        ),
        make_pair(
            preference_id="pref_p001_c003__p001_c004",
            problem_id="p001",
            chosen_candidate_id="p001_c003",
            rejected_candidate_id="p001_c004",
            chosen_strategy="normal",
            rejected_strategy="optimized",
            chosen="def f():\n    return 3",
            rejected="def f():\n    return 4",
            chosen_code_sha256="f" * 64,
            rejected_code_sha256="g" * 64,
            score_margin=0.5,
        ),
    ]
    report = build_quality_report(manifest, pairs, {"p001": ["correct", "incorrect"]})
    assert report.strategy_distribution["chosen"] == {"normal": 2}
    assert report.strategy_distribution["rejected"] == {"alternative": 1, "optimized": 1}
    assert report.score_margin_distribution == {"0.5": 2}


# --------------------------------------------------------------------------------- formatters


def test_format_preference_statistics_contains_the_headline_counters():
    manifest = make_manifest()
    pairs = [make_pair(preference_id="pref_p001_c001__p001_c002")]
    rejections = [make_rejection()]
    stats = build_stats(manifest, pairs, rejections)
    text = format_preference_statistics(stats)
    assert "Pairs generated: 1" in text
    assert "Ties: 1" in text


def test_format_quality_report_lists_problems_without_pairs():
    manifest = make_manifest()
    report = build_quality_report(manifest, [], {"p001": ["correct", "correct"]})
    text = format_quality_report(report)
    assert "p001: all_candidates_correct" in text


def test_format_pair_table_and_detail():
    pair = make_pair()
    table = format_pair_table([pair])
    assert pair.preference_id in table
    detail = format_pair_detail(pair, show_code=True)
    assert "chosen_candidate_id: p001_c001" in detail
    assert pair.chosen in detail
    detail_no_code = format_pair_detail(pair, show_code=False)
    assert pair.chosen not in detail_no_code
