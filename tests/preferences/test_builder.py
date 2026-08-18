"""Tests for PreferencePairBuilder (spec 08 sections 29-45, 88, 90)."""

from __future__ import annotations

from typing import Any

import pytest

from python_dpo.candidates.models import Candidate
from python_dpo.generation.prompt_builder import PROMPT_VERSION, build_canonical_prompt, build_prompt
from python_dpo.preferences.builder import PreferencePairBuilder
from python_dpo.preferences.policies import AllBetterPolicy, MarginPolicy, StrictPolicy
from python_dpo.problems.models import Problem, TestCase
from python_dpo.ranking.models import CandidateAssessment

RANKING_RUN_ID = "rank_20260817_161726_a84d"
EVAL_RUN_ID = "eval_20260817_115154_dcd4"
CANDIDATE_RUN_ID = "run_20260817_055411"

PROBLEM = Problem(
    id="p001",
    prompt="Write a function that returns the sum of all even integers in a list.",
    signature="def sum_even(numbers):",
    entry_point="sum_even",
    category="lists",
    difficulty="easy",
    reference_solution="def sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)",
    tests=(TestCase(id="t1", input={"numbers": [1, 2, 3, 4]}, expected=6),),
)


def make_assessment(candidate_id: str, passed: int, total: int = 10, **overrides: Any) -> CandidateAssessment:
    correctness = overrides.pop("correctness", None)
    if correctness is None:
        correctness = "correct" if passed == total else "incorrect"
    indeterminate_reason = overrides.pop(
        "indeterminate_reason", "infrastructure_error" if correctness == "indeterminate" else None
    )
    if correctness == "indeterminate":
        passed, total = 0, 0
    fields: dict[str, Any] = {
        "ranking_run_id": RANKING_RUN_ID,
        "evaluation_run_id": EVAL_RUN_ID,
        "candidate_run_id": CANDIDATE_RUN_ID,
        "candidate_id": candidate_id,
        "problem_id": PROBLEM.id,
        "correctness": correctness,
        "all_tests_passed": total > 0 and passed == total,
        "pass_rate": (passed / total) if total else 0.0,
        "score": (passed / total) if total else 0.0,
        "tests_total": total,
        "tests_passed": passed,
        "tests_failed": total - passed,
        "tests_error": 0,
        "tests_skipped": 0,
        "timeout": False,
        "infrastructure_error": correctness == "indeterminate",
        "indeterminate_reason": indeterminate_reason,
    }
    fields.update(overrides)
    return CandidateAssessment(**fields)


def make_candidate(candidate_id: str, code: str, *, strategy: str = "normal", **overrides: Any) -> Candidate:
    prompt = overrides.pop("prompt", None) or build_prompt(PROBLEM, strategy)
    fields: dict[str, Any] = {
        "candidate_id": candidate_id,
        "problem_id": PROBLEM.id,
        "run_id": CANDIDATE_RUN_ID,
        "generation_index": int(candidate_id.rsplit("c", 1)[1]),
        "strategy": strategy,
        "model": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "provider": "transformers",
        "prompt_version": PROMPT_VERSION,
        "prompt": prompt,
        "raw_output": f"```python\n{code}\n```",
        "code": code,
        "extraction_format": "python_fence",
        "syntax_valid": True,
        "function_name_valid": True,
        "generation_config": {},
        "created_at": "2026-08-17T05:54:18Z",
    }
    fields.update(overrides)
    return Candidate.create(**fields)


STRATEGIES = ("normal", "straightforward", "edge_case_focused", "alternative", "optimized")


def build_candidates(specs: dict[str, tuple[int, int]]) -> tuple[dict[str, Candidate], dict[str, CandidateAssessment]]:
    """``specs`` maps candidate_id -> (passed, total); each gets distinct code and a
    strategy drawn round-robin from the five real strategies, matching the real dataset's
    shape (one strategy per candidate index).
    """
    candidates: dict[str, Candidate] = {}
    assessments: dict[str, CandidateAssessment] = {}
    for index, (candidate_id, (passed, total)) in enumerate(sorted(specs.items())):
        strategy = STRATEGIES[index % len(STRATEGIES)]
        code = f"def sum_even(numbers):\n    return {index}  # variant {candidate_id}"
        candidates[candidate_id] = make_candidate(candidate_id, code, strategy=strategy)
        assessments[candidate_id] = make_assessment(candidate_id, passed, total)
    return candidates, assessments


def run_builder(policy, specs, **builder_kwargs):
    candidates_by_id, assessments_by_id = build_candidates(specs)
    builder = PreferencePairBuilder(policy, **builder_kwargs)
    result = builder.build_problem(
        ranking_run_id=RANKING_RUN_ID,
        evaluation_run_id=EVAL_RUN_ID,
        candidate_run_id=CANDIDATE_RUN_ID,
        problem=PROBLEM,
        assessments=list(assessments_by_id.values()),
        candidates_by_id=candidates_by_id,
    )
    return result, candidates_by_id, assessments_by_id


# --------------------------------------------------------------------- section 90's matrix


def test_section_90_strict_matrix():
    # A=10/10, B=8/10, C=10/10, D=5/10, E=0/10.
    specs = {
        "p001_c001": (10, 10),  # A
        "p001_c002": (8, 10),  # B
        "p001_c003": (10, 10),  # C
        "p001_c004": (5, 10),  # D
        "p001_c005": (0, 10),  # E
    }
    result, _, _ = run_builder(StrictPolicy(), specs, minimum_score_margin=0.2)
    directions = {(p.chosen_candidate_id, p.rejected_candidate_id) for p in result.pairs}
    assert directions == {
        ("p001_c001", "p001_c002"),  # A>B
        ("p001_c001", "p001_c004"),  # A>D
        ("p001_c001", "p001_c005"),  # A>E
        ("p001_c003", "p001_c002"),  # C>B
        ("p001_c003", "p001_c004"),  # C>D
        ("p001_c003", "p001_c005"),  # C>E
    }
    # No A/C pair: both are correct, so their comparison is a tie.
    assert ("p001_c001", "p001_c003") not in directions
    assert ("p001_c003", "p001_c001") not in directions
    # B vs D and B vs E must not appear under strict (both incorrect).
    assert ("p001_c002", "p001_c004") not in directions
    assert ("p001_c002", "p001_c005") not in directions


def test_candidate_pairs_considered_equals_generated_plus_rejected():
    # The mechanical form of "never silently discard" (CLAUDE.md).
    specs = {
        "p001_c001": (10, 10),
        "p001_c002": (8, 10),
        "p001_c003": (10, 10),
        "p001_c004": (5, 10),
        "p001_c005": (0, 10),
    }
    result, _, _ = run_builder(StrictPolicy(), specs, minimum_score_margin=0.2)
    considered = len(specs) * (len(specs) - 1) // 2
    assert len(result.pairs) + len(result.rejections) == considered


# --------------------------------------------------------------------------------- exclusions


def test_ties_are_excluded():
    specs = {"p001_c001": (10, 10), "p001_c002": (10, 10)}
    result, _, _ = run_builder(AllBetterPolicy(), specs)
    assert result.pairs == []
    assert [r.reason for r in result.rejections] == ["tie"]


def test_indeterminate_vs_correct_is_excluded():
    candidates_by_id = {
        "p001_c001": make_candidate("p001_c001", "def sum_even(numbers):\n    return 1", strategy="normal"),
        "p001_c002": make_candidate(
            "p001_c002", "def sum_even(numbers):\n    return 2", strategy="straightforward"
        ),
    }
    assessments = [
        make_assessment("p001_c001", 10, 10),
        make_assessment("p001_c002", 0, 0, correctness="indeterminate"),
    ]
    builder = PreferencePairBuilder(AllBetterPolicy())
    result = builder.build_problem(
        ranking_run_id=RANKING_RUN_ID,
        evaluation_run_id=EVAL_RUN_ID,
        candidate_run_id=CANDIDATE_RUN_ID,
        problem=PROBLEM,
        assessments=assessments,
        candidates_by_id=candidates_by_id,
    )
    assert result.pairs == []
    assert [r.reason for r in result.rejections] == ["indeterminate"]


def test_indeterminate_vs_incorrect_is_excluded():
    candidates_by_id = {
        "p001_c001": make_candidate("p001_c001", "def sum_even(numbers):\n    return 1", strategy="normal"),
        "p001_c002": make_candidate(
            "p001_c002", "def sum_even(numbers):\n    return 2", strategy="straightforward"
        ),
    }
    assessments = [
        make_assessment("p001_c001", 5, 10),
        make_assessment("p001_c002", 0, 0, correctness="indeterminate"),
    ]
    builder = PreferencePairBuilder(AllBetterPolicy())
    result = builder.build_problem(
        ranking_run_id=RANKING_RUN_ID,
        evaluation_run_id=EVAL_RUN_ID,
        candidate_run_id=CANDIDATE_RUN_ID,
        problem=PROBLEM,
        assessments=assessments,
        candidates_by_id=candidates_by_id,
    )
    assert result.pairs == []
    assert [r.reason for r in result.rejections] == ["indeterminate"]


def test_identical_code_with_equal_scores_is_recorded_as_a_tie_not_identical_code():
    # On real, deterministic evaluation, identical code always produces an identical
    # score, so the comparator's tie check intercepts before code identity is ever
    # consulted (the "duplicate code -> TIE" finding the Stage 8 plan measured from the
    # real ranking run). identical_code is therefore structurally defensive, not
    # load-bearing on ordinary data.
    candidates_by_id: dict[str, Candidate] = {}
    assessments: list[CandidateAssessment] = []
    shared_code = "def sum_even(numbers):\n    return 1"
    for cid, strategy in (("p001_c001", "normal"), ("p001_c002", "straightforward")):
        candidates_by_id[cid] = make_candidate(cid, shared_code, strategy=strategy)
        assessments.append(make_assessment(cid, 10, 10))
    candidates_by_id["p001_c003"] = make_candidate(
        "p001_c003", "def sum_even(numbers):\n    return 2", strategy="edge_case_focused"
    )
    assessments.append(make_assessment("p001_c003", 5, 10))

    builder = PreferencePairBuilder(AllBetterPolicy())
    result = builder.build_problem(
        ranking_run_id=RANKING_RUN_ID,
        evaluation_run_id=EVAL_RUN_ID,
        candidate_run_id=CANDIDATE_RUN_ID,
        problem=PROBLEM,
        assessments=assessments,
        candidates_by_id=candidates_by_id,
    )
    directions = {(p.chosen_candidate_id, p.rejected_candidate_id) for p in result.pairs}
    assert directions == {("p001_c001", "p001_c003"), ("p001_c002", "p001_c003")}
    by_pair = {(r.candidate_a, r.candidate_b): r.reason for r in result.rejections}
    assert by_pair[("p001_c001", "p001_c002")] == "tie"


def test_identical_code_reason_fires_when_scores_nonetheless_differ():
    # A defensive scenario a deterministic pipeline should never produce on its own
    # (spec sections 33, 34): two candidates sharing code_sha256 but carrying different
    # assessment scores (e.g. a flaky evaluation). The identical_code guard must still
    # catch it rather than emitting a pair built from an uninformative comparison.
    shared_code = "def sum_even(numbers):\n    return 1"
    candidates_by_id = {
        "p001_c001": make_candidate("p001_c001", shared_code, strategy="normal"),
        "p001_c002": make_candidate("p001_c002", shared_code, strategy="straightforward"),
    }
    assessments = [
        make_assessment("p001_c001", 10, 10),
        make_assessment("p001_c002", 5, 10),  # decisive relative to c001 despite same code
    ]
    builder = PreferencePairBuilder(AllBetterPolicy())
    result = builder.build_problem(
        ranking_run_id=RANKING_RUN_ID,
        evaluation_run_id=EVAL_RUN_ID,
        candidate_run_id=CANDIDATE_RUN_ID,
        problem=PROBLEM,
        assessments=assessments,
        candidates_by_id=candidates_by_id,
    )
    assert result.pairs == []
    assert result.rejections[0].reason == "identical_code"


def test_cross_problem_pairs_are_never_generated():
    other_problem = Problem(
        id="p002",
        prompt="Reverse a string.",
        signature="def reverse_str(value):",
        entry_point="reverse_str",
        category="strings",
        difficulty="easy",
        reference_solution="def reverse_str(value):\n    return value[::-1]",
        tests=(TestCase(id="t1", input={"value": "ab"}, expected="ba"),),
    )
    with pytest.raises(ValueError):
        PreferencePairBuilder(AllBetterPolicy()).build_problem(
            ranking_run_id=RANKING_RUN_ID,
            evaluation_run_id=EVAL_RUN_ID,
            candidate_run_id=CANDIDATE_RUN_ID,
            problem=PROBLEM,
            assessments=[make_assessment("p002_c001", 10, 10, problem_id=other_problem.id)],
            candidates_by_id={},
        )


def test_prompt_lineage_failure_rejects_every_pair_as_integrity_failure():
    candidates_by_id, assessments_by_id = build_candidates(
        {"p001_c001": (10, 10), "p001_c002": (5, 10)}
    )
    # Corrupt one candidate's prompt_version so lineage verification fails.
    stale = candidates_by_id["p001_c001"]
    import dataclasses

    candidates_by_id["p001_c001"] = dataclasses.replace(stale, prompt_version="v0")

    builder = PreferencePairBuilder(AllBetterPolicy())
    result = builder.build_problem(
        ranking_run_id=RANKING_RUN_ID,
        evaluation_run_id=EVAL_RUN_ID,
        candidate_run_id=CANDIDATE_RUN_ID,
        problem=PROBLEM,
        assessments=list(assessments_by_id.values()),
        candidates_by_id=candidates_by_id,
    )
    assert result.pairs == []
    assert len(result.rejections) == 1
    assert result.rejections[0].reason == "integrity_failure"


def test_missing_candidate_join_is_an_integrity_failure_not_a_crash():
    candidates_by_id, assessments_by_id = build_candidates(
        {"p001_c001": (10, 10), "p001_c002": (5, 10)}
    )
    del candidates_by_id["p001_c002"]  # simulate a join miss

    builder = PreferencePairBuilder(AllBetterPolicy())
    result = builder.build_problem(
        ranking_run_id=RANKING_RUN_ID,
        evaluation_run_id=EVAL_RUN_ID,
        candidate_run_id=CANDIDATE_RUN_ID,
        problem=PROBLEM,
        assessments=list(assessments_by_id.values()),
        candidates_by_id=candidates_by_id,
    )
    assert result.pairs == []
    assert result.rejections[0].reason == "integrity_failure"


# ------------------------------------------------------------------------------- direction


def test_only_one_direction_is_ever_emitted():
    specs = {"p001_c001": (10, 10), "p001_c002": (5, 10)}
    result, _, _ = run_builder(AllBetterPolicy(), specs)
    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert (pair.chosen_candidate_id, pair.rejected_candidate_id) == ("p001_c001", "p001_c002")


def test_preference_id_is_deterministic():
    specs = {"p001_c001": (10, 10), "p001_c002": (5, 10)}
    result_a, _, _ = run_builder(AllBetterPolicy(), specs)
    result_b, _, _ = run_builder(AllBetterPolicy(), specs)
    assert result_a.pairs[0].preference_id == result_b.pairs[0].preference_id
    assert result_a.pairs[0].preference_id == "pref_p001_c001__p001_c002"


# ------------------------------------------------------------------------ max_pairs_per_problem


def test_max_pairs_per_problem_truncation_is_deterministic_and_margin_ordered():
    specs = {
        "p001_c001": (10, 10),
        "p001_c002": (8, 10),
        "p001_c003": (5, 10),
        "p001_c004": (0, 10),
    }
    result, _, _ = run_builder(AllBetterPolicy(), specs, max_pairs_per_problem=2)
    assert len(result.pairs) == 2
    margins = sorted((p.score_margin for p in result.pairs), reverse=True)
    assert margins == sorted(margins, reverse=True)
    truncated_reasons = {r.reason for r in result.rejections}
    assert "max_pairs_per_problem" in truncated_reasons
    # Repeating the run gives the identical kept set — no RNG anywhere.
    result_2, _, _ = run_builder(AllBetterPolicy(), specs, max_pairs_per_problem=2)
    assert {p.preference_id for p in result.pairs} == {p.preference_id for p in result_2.pairs}


def test_max_pairs_per_problem_keeps_the_largest_margins():
    specs = {
        "p001_c001": (10, 10),
        "p001_c002": (8, 10),  # margin 0.2 vs A
        "p001_c003": (0, 10),  # margin 1.0 vs A -- must survive over c002
    }
    result, _, _ = run_builder(AllBetterPolicy(), specs, max_pairs_per_problem=1)
    assert len(result.pairs) == 1
    assert result.pairs[0].rejected_candidate_id == "p001_c003"


# ---------------------------------------------------------------------------- strength


def test_strong_vs_medium_strength():
    specs = {
        "p001_c001": (10, 10),  # correct
        "p001_c002": (5, 10),  # incorrect
        "p001_c003": (2, 10),  # incorrect, lower
    }
    result, _, _ = run_builder(AllBetterPolicy(), specs)
    by_pair = {(p.chosen_candidate_id, p.rejected_candidate_id): p for p in result.pairs}
    assert by_pair[("p001_c001", "p001_c002")].preference_strength == "strong"
    assert by_pair[("p001_c001", "p001_c003")].preference_strength == "strong"
    assert by_pair[("p001_c002", "p001_c003")].preference_strength == "medium"


# ---------------------------------------------------------------------------- build_run


def test_build_run_groups_strictly_by_problem():
    other_problem = Problem(
        id="p002",
        prompt="Reverse a string.",
        signature="def reverse_str(value):",
        entry_point="reverse_str",
        category="strings",
        difficulty="easy",
        reference_solution="def reverse_str(value):\n    return value[::-1]",
        tests=(TestCase(id="t1", input={"value": "ab"}, expected="ba"),),
    )
    candidates_by_id, assessments_by_id_p1 = build_candidates(
        {"p001_c001": (10, 10), "p001_c002": (5, 10)}
    )
    p2_candidate = make_candidate(
        "p002_c001",
        "def reverse_str(value):\n    return value[::-1]",
        strategy="normal",
        problem_id=other_problem.id,
        prompt=build_prompt(other_problem, "normal"),
    )
    p2_candidate_2 = make_candidate(
        "p002_c002",
        "def reverse_str(value):\n    return ''",
        strategy="straightforward",
        problem_id=other_problem.id,
        prompt=build_prompt(other_problem, "straightforward"),
    )
    candidates_by_id[p2_candidate.candidate_id] = p2_candidate
    candidates_by_id[p2_candidate_2.candidate_id] = p2_candidate_2

    assessments_by_problem = {
        PROBLEM.id: list(assessments_by_id_p1.values()),
        other_problem.id: [
            make_assessment("p002_c001", 5, 5, problem_id=other_problem.id),
            make_assessment("p002_c002", 0, 5, problem_id=other_problem.id),
        ],
    }
    builder = PreferencePairBuilder(AllBetterPolicy())
    result = builder.build_run(
        ranking_run_id=RANKING_RUN_ID,
        evaluation_run_id=EVAL_RUN_ID,
        candidate_run_id=CANDIDATE_RUN_ID,
        problems_by_id={PROBLEM.id: PROBLEM, other_problem.id: other_problem},
        assessments_by_problem=assessments_by_problem,
        candidates_by_id=candidates_by_id,
    )
    assert {p.problem_id for p in result.pairs} == {PROBLEM.id, other_problem.id}
    # No pair ever mixes candidate ids across the two problems.
    for pair in result.pairs:
        assert pair.chosen_candidate_id.startswith(pair.problem_id)
        assert pair.rejected_candidate_id.startswith(pair.problem_id)
    assert result.candidates_considered == 4


def test_margin_policy_via_builder_matches_spec_example():
    # Spec section 56: A=1.0 B=0.8 C=1.0 D=0.5 E=0.0, margin 0.2.
    specs = {
        "p001_c001": (10, 10),  # A
        "p001_c002": (8, 10),  # B
        "p001_c003": (10, 10),  # C
        "p001_c004": (5, 10),  # D
        "p001_c005": (0, 10),  # E
    }
    result, _, _ = run_builder(MarginPolicy(), specs, minimum_score_margin=0.2)
    directions = {(p.chosen_candidate_id, p.rejected_candidate_id) for p in result.pairs}
    # A vs C is still a tie (excluded); every other margin>=0.2 pair is included.
    assert ("p001_c001", "p001_c003") not in directions
    assert ("p001_c003", "p001_c001") not in directions
    for margin_pair in [
        ("p001_c001", "p001_c002"),
        ("p001_c001", "p001_c004"),
        ("p001_c001", "p001_c005"),
        ("p001_c003", "p001_c002"),
        ("p001_c003", "p001_c004"),
        ("p001_c003", "p001_c005"),
        ("p001_c002", "p001_c004"),
        ("p001_c002", "p001_c005"),
        ("p001_c004", "p001_c005"),
    ]:
        assert margin_pair in directions, margin_pair
