"""Tests for the strict/margin/all_better selection policies (spec 08 sections 24-28, 88).

``chosen``/``rejected`` here are bare ``CandidateAssessment`` stand-ins built with just
the fields a policy actually reads (``score``, ``correctness``) — the builder is
responsible for everything upstream of "is this admitted".
"""

from __future__ import annotations

from typing import Any

import pytest

from python_dpo.preferences.errors import PreferencePolicyError
from python_dpo.preferences.policies import (
    AllBetterPolicy,
    MarginPolicy,
    StrictPolicy,
    make_policy,
)
from python_dpo.ranking.models import CandidateAssessment

RANKING_RUN_ID = "rank_20260817_161726_a84d"
EVAL_RUN_ID = "eval_20260817_115154_dcd4"
CANDIDATE_RUN_ID = "run_20260817_055411"


def make_assessment(candidate_id: str, passed: int, total: int = 10, **overrides: Any) -> CandidateAssessment:
    correctness = overrides.pop("correctness", None)
    if correctness is None:
        correctness = "correct" if passed == total else "incorrect"
    fields: dict[str, Any] = {
        "ranking_run_id": RANKING_RUN_ID,
        "evaluation_run_id": EVAL_RUN_ID,
        "candidate_run_id": CANDIDATE_RUN_ID,
        "candidate_id": candidate_id,
        "problem_id": "p001",
        "correctness": correctness,
        "all_tests_passed": passed == total,
        "pass_rate": passed / total,
        "score": passed / total,
        "tests_total": total,
        "tests_passed": passed,
        "tests_failed": total - passed,
        "tests_error": 0,
        "tests_skipped": 0,
        "timeout": False,
        "infrastructure_error": False,
    }
    fields.update(overrides)
    return CandidateAssessment(**fields)


# ------------------------------------------------------------------------------- strict


def test_strict_admits_correct_vs_incorrect():
    policy = StrictPolicy()
    chosen = make_assessment("p001_c001", 10, 10)
    rejected = make_assessment("p001_c002", 8, 10)
    admitted, reason = policy.admits(chosen, rejected, minimum_score_margin=0.2)
    assert admitted
    assert reason is None


def test_strict_excludes_correct_vs_correct():
    policy = StrictPolicy()
    chosen = make_assessment("p001_c001", 10, 10)
    rejected = make_assessment("p001_c002", 10, 10)
    admitted, reason = policy.admits(chosen, rejected, minimum_score_margin=0.2)
    assert not admitted
    assert reason == "not_correct_vs_incorrect"


def test_strict_excludes_incorrect_vs_incorrect():
    policy = StrictPolicy()
    chosen = make_assessment("p001_c001", 8, 10)
    rejected = make_assessment("p001_c002", 5, 10)
    admitted, reason = policy.admits(chosen, rejected, minimum_score_margin=0.2)
    assert not admitted
    assert reason == "not_correct_vs_incorrect"


def test_strict_admits_correct_vs_incorrect_with_small_margin():
    # Decision 2: strict ignores minimum_score_margin entirely. 9/9 vs 8/9 has margin
    # 0.111, well under 0.2, but the correctness gap alone is enough.
    policy = StrictPolicy()
    chosen = make_assessment("p008_c003", 9, 9)
    rejected = make_assessment("p008_c001", 8, 9)
    admitted, reason = policy.admits(chosen, rejected, minimum_score_margin=0.2)
    assert admitted
    assert reason is None


# ------------------------------------------------------------------------------- margin


def test_margin_includes_at_exactly_the_threshold():
    policy = MarginPolicy()
    chosen = make_assessment("p001_c001", 10, 10)
    rejected = make_assessment("p001_c002", 8, 10)  # margin 0.2
    admitted, reason = policy.admits(chosen, rejected, minimum_score_margin=0.2)
    assert admitted
    assert reason is None


def test_margin_excludes_below_the_threshold():
    policy = MarginPolicy()
    chosen = make_assessment("p001_c001", 8, 10)
    rejected = make_assessment("p001_c002", 7, 10)  # margin 0.1
    admitted, reason = policy.admits(chosen, rejected, minimum_score_margin=0.2)
    assert not admitted
    assert reason == "insufficient_margin"


def test_margin_includes_partial_vs_partial_above_threshold():
    policy = MarginPolicy()
    chosen = make_assessment("p001_c001", 9, 10)
    rejected = make_assessment("p001_c002", 7, 10)  # margin 0.2
    admitted, _ = policy.admits(chosen, rejected, minimum_score_margin=0.2)
    assert admitted


def test_margin_includes_correct_vs_partial_above_threshold():
    policy = MarginPolicy()
    chosen = make_assessment("p001_c001", 8, 10)
    rejected = make_assessment("p001_c002", 6, 10)  # margin 0.2
    admitted, _ = policy.admits(chosen, rejected, minimum_score_margin=0.2)
    assert admitted


# ---------------------------------------------------------------------------- all_better


def test_all_better_admits_any_decisive_margin():
    policy = AllBetterPolicy()
    chosen = make_assessment("p001_c001", 9, 10)
    rejected = make_assessment("p001_c002", 8, 10)  # margin 0.1, would fail margin@0.2
    admitted, reason = policy.admits(chosen, rejected, minimum_score_margin=0.2)
    assert admitted
    assert reason is None


# ------------------------------------------------------------------------------- factory


def test_make_policy_returns_the_named_policy():
    assert make_policy("strict").name == "strict"
    assert make_policy("margin").version == "margin_v1"
    assert make_policy("all_better").name == "all_better"


def test_make_policy_rejects_unknown_name():
    with pytest.raises(PreferencePolicyError):
        make_policy("nonexistent")
