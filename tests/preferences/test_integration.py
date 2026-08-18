"""End-to-end preference-pair tests (spec 08 sections 90, 91).

Pure computation — no Docker, no model, no filesystem beyond ``tmp_path`` — so this runs
in the default offline suite, exactly like ``tests/ranking/test_integration.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from python_dpo.candidates.models import Candidate
from python_dpo.candidates.repository import CandidateRepository
from python_dpo.generation.prompt_builder import PROMPT_VERSION, build_prompt
from python_dpo.preferences.builder import BUILDER_VERSION, PreferencePairBuilder
from python_dpo.preferences.models import (
    PreferenceStatistics,
    build_quality_report,
    derive_candidates_considered,
)
from python_dpo.preferences.policies import MarginPolicy, PreferencePolicy, StrictPolicy
from python_dpo.preferences.run_repository import PreferenceRunRepository
from python_dpo.preferences.splitter import ProblemSplitter
from python_dpo.preferences.validation import validate_preference_run
from python_dpo.problems.models import Problem, TestCase
from python_dpo.ranking.models import CandidateAssessment
from python_dpo.ranking.repository import RankingRepository

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

# Second problem, kept intentionally pairless (all candidates tied) so the pipeline also
# exercises the problem-level split pool being a strict subset of every processed problem.
OTHER_PROBLEM = Problem(
    id="p002",
    prompt="Write a function that reverses a string.",
    signature="def reverse_str(value):",
    entry_point="reverse_str",
    category="strings",
    difficulty="easy",
    reference_solution="def reverse_str(value):\n    return value[::-1]",
    tests=(TestCase(id="t1", input={"value": "ab"}, expected="ba"),),
)


def _make_candidate(problem: Problem, candidate_id: str, code: str, strategy: str) -> Candidate:
    return Candidate.create(
        candidate_id=candidate_id,
        problem_id=problem.id,
        run_id=CANDIDATE_RUN_ID,
        generation_index=int(candidate_id.rsplit("c", 1)[1]),
        strategy=strategy,
        model="Qwen/Qwen2.5-Coder-3B-Instruct",
        provider="transformers",
        prompt_version=PROMPT_VERSION,
        prompt=build_prompt(problem, strategy),
        raw_output=f"```python\n{code}\n```",
        code=code,
        extraction_format="python_fence",
        syntax_valid=True,
        function_name_valid=True,
        generation_config={},
        created_at="2026-08-17T05:54:18Z",
    )


def _make_assessment(problem_id: str, candidate_id: str, passed: int, total: int = 7) -> CandidateAssessment:
    correctness = "correct" if passed == total else "incorrect"
    return CandidateAssessment(
        ranking_run_id=RANKING_RUN_ID,
        evaluation_run_id=EVAL_RUN_ID,
        candidate_run_id=CANDIDATE_RUN_ID,
        candidate_id=candidate_id,
        problem_id=problem_id,
        correctness=correctness,
        all_tests_passed=passed == total,
        pass_rate=passed / total,
        score=passed / total,
        tests_total=total,
        tests_passed=passed,
        tests_failed=total - passed,
        tests_error=0,
        tests_skipped=0,
        timeout=False,
        infrastructure_error=False,
    )


def build_upstream_runs(tmp_path: Path) -> tuple[Path, Path, dict[str, list[CandidateAssessment]], dict[str, Candidate]]:
    """The spec section 90 fixture (A=10/10, B=8/10, C=10/10, D=5/10, E=0/10, at
    tests_total=10) for ``p001``, plus a fully-tied ``p002`` that produces zero pairs.
    Returns ``(ranking_run_dir, candidate_run_dir, assessments_by_problem, candidates_by_id)``.
    """
    candidate_run_dir = tmp_path / "candidates"
    candidate_repo = CandidateRepository(candidate_run_dir)
    candidates_by_id: dict[str, Candidate] = {}

    p1_specs = {
        "p001_c001": 10,  # A
        "p001_c002": 8,  # B
        "p001_c003": 10,  # C (tied with A)
        "p001_c004": 5,  # D
        "p001_c005": 0,  # E
    }
    strategies = ("normal", "straightforward", "edge_case_focused", "alternative", "optimized")
    p1_assessments = []
    for index, (candidate_id, passed) in enumerate(sorted(p1_specs.items())):
        strategy = strategies[index % len(strategies)]
        code = f"def sum_even(numbers):\n    return {index}  # {candidate_id}"
        candidate = _make_candidate(PROBLEM, candidate_id, code, strategy)
        candidate_repo.save(candidate)
        candidates_by_id[candidate_id] = candidate
        p1_assessments.append(_make_assessment(PROBLEM.id, candidate_id, passed, total=10))

    p2_assessments = []
    for index, candidate_id in enumerate(("p002_c001", "p002_c002")):
        strategy = strategies[index % len(strategies)]
        code = f"def reverse_str(value):\n    return value[::-1]  # {candidate_id}"
        candidate = _make_candidate(OTHER_PROBLEM, candidate_id, code, strategy)
        candidate_repo.save(candidate)
        candidates_by_id[candidate_id] = candidate
        p2_assessments.append(_make_assessment(OTHER_PROBLEM.id, candidate_id, 5, total=5))  # all correct, tied

    ranking_run_dir = tmp_path / "ranking"
    ranking_repo = RankingRepository(ranking_run_dir)
    ranking_repo.save_assessments(p1_assessments + p2_assessments)

    assessments_by_problem = {PROBLEM.id: p1_assessments, OTHER_PROBLEM.id: p2_assessments}
    return ranking_run_dir, candidate_run_dir, assessments_by_problem, candidates_by_id


def run_full_preferences_pipeline(
    preferences_root: Path,
    problems_by_id: dict[str, Problem],
    assessments_by_problem: dict[str, list[CandidateAssessment]],
    candidates_by_id: dict[str, Candidate],
    *,
    policy: PreferencePolicy = StrictPolicy(),
    minimum_score_margin: float = 0.2,
) -> tuple[Path, str]:
    """The full pipeline a CLI invocation drives: build -> persist -> split -> statistics
    -> quality report -> complete. Returns ``(run_dir, preference_run_id)``.
    """
    run_repo = PreferenceRunRepository(preferences_root)
    manifest = run_repo.create_run(
        ranking_run_id=RANKING_RUN_ID,
        evaluation_run_id=EVAL_RUN_ID,
        candidate_run_id=CANDIDATE_RUN_ID,
        preference_version="v1",
        selection_policy=policy.name,
        selection_policy_version=policy.version,
        minimum_score_margin=minimum_score_margin,
        split_ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
        split_seed=42,
        builder_version=BUILDER_VERSION,
    )
    manifest = run_repo.start_run(manifest.preference_run_id)
    repository = run_repo.results(manifest.preference_run_id)

    builder = PreferencePairBuilder(policy, minimum_score_margin=minimum_score_margin)
    result = builder.build_run(
        ranking_run_id=RANKING_RUN_ID,
        evaluation_run_id=EVAL_RUN_ID,
        candidate_run_id=CANDIDATE_RUN_ID,
        problems_by_id=problems_by_id,
        assessments_by_problem=assessments_by_problem,
        candidates_by_id=candidates_by_id,
    )
    repository.save_pairs(result.pairs)
    repository.save_rejections(result.rejections)

    training_ids = {p.problem_id for p in result.pairs if not p.duplicate_training_record}
    split_manifest = ProblemSplitter(ratios=manifest.split_ratios, seed=manifest.split_seed).split(
        training_ids
    )
    run_repo.write_split_manifest(manifest.preference_run_id, split_manifest)
    repository.write_dataset(split_manifest)

    stats = PreferenceStatistics.from_records(
        manifest,
        result.pairs,
        result.rejections,
        candidates_considered=derive_candidates_considered(result.pairs, result.rejections),
    )
    run_repo.write_statistics(stats)

    problem_correctness = {
        problem_id: [a.correctness for a in assessments]
        for problem_id, assessments in assessments_by_problem.items()
    }
    quality_report = build_quality_report(manifest, result.pairs, problem_correctness)
    run_repo.write_quality_report(quality_report)

    run_repo.complete_run(manifest.preference_run_id)
    return run_repo.run_dir(manifest.preference_run_id), manifest.preference_run_id


# --------------------------------------------------------------------------------- section 90


def test_strict_matrix_end_to_end(tmp_path):
    _, _, assessments_by_problem, candidates_by_id = build_upstream_runs(tmp_path)
    problems_by_id = {PROBLEM.id: PROBLEM, OTHER_PROBLEM.id: OTHER_PROBLEM}

    run_dir, preference_run_id = run_full_preferences_pipeline(
        tmp_path / "preferences", problems_by_id, assessments_by_problem, candidates_by_id
    )
    repo = PreferenceRunRepository(tmp_path / "preferences").results(preference_run_id)
    pairs = repo.load_pairs()

    directions = {(p.chosen_candidate_id, p.rejected_candidate_id) for p in pairs}
    assert directions == {
        ("p001_c001", "p001_c002"),
        ("p001_c001", "p001_c004"),
        ("p001_c001", "p001_c005"),
        ("p001_c003", "p001_c002"),
        ("p001_c003", "p001_c004"),
        ("p001_c003", "p001_c005"),
    }
    assert ("p001_c001", "p001_c003") not in directions  # both correct: no A/C pair
    # p002 is entirely tied: zero pairs, but the problem was processed (not silently
    # dropped) — every one of its C(2,2)=1 candidate pair is a recorded rejection.
    assert all(p.problem_id != OTHER_PROBLEM.id for p in pairs)

    report = validate_preference_run(run_dir)
    assert report.valid, [i.message for i in report.issues]


def test_margin_adds_qualifying_partial_vs_partial_pairs(tmp_path):
    _, _, assessments_by_problem, candidates_by_id = build_upstream_runs(tmp_path)
    problems_by_id = {PROBLEM.id: PROBLEM, OTHER_PROBLEM.id: OTHER_PROBLEM}

    run_dir, preference_run_id = run_full_preferences_pipeline(
        tmp_path / "preferences_margin",
        problems_by_id,
        assessments_by_problem,
        candidates_by_id,
        policy=MarginPolicy(),
        minimum_score_margin=0.2,
    )
    repo = PreferenceRunRepository(tmp_path / "preferences_margin").results(preference_run_id)
    directions = {(p.chosen_candidate_id, p.rejected_candidate_id) for p in repo.load_pairs()}

    # Everything strict admits is still admitted, plus B>D and B>E (both incorrect, but
    # margin 0.3 and 0.8 respectively) and D>E (margin 0.5).
    assert ("p001_c002", "p001_c004") in directions
    assert ("p001_c002", "p001_c005") in directions
    assert ("p001_c004", "p001_c005") in directions
    assert ("p001_c001", "p001_c003") not in directions  # tie, still excluded


# --------------------------------------------------------------------------------- section 91


def test_end_to_end_provenance_and_split(tmp_path):
    _, candidate_run_dir, assessments_by_problem, candidates_by_id = build_upstream_runs(tmp_path)
    ranking_run_dir = tmp_path / "ranking"
    problems_by_id = {PROBLEM.id: PROBLEM, OTHER_PROBLEM.id: OTHER_PROBLEM}

    run_dir, preference_run_id = run_full_preferences_pipeline(
        tmp_path / "preferences", problems_by_id, assessments_by_problem, candidates_by_id
    )
    repo = PreferenceRunRepository(tmp_path / "preferences").results(preference_run_id)
    pairs = repo.load_pairs()
    assert pairs  # the strict policy on this fixture must produce at least one pair

    # Chosen really scores higher, rejected really scores lower.
    for pair in pairs:
        assert pair.chosen_score > pair.rejected_score
        assert pair.chosen_tests_passed > pair.rejected_tests_passed or (
            pair.chosen_correctness == "correct" and pair.rejected_correctness == "incorrect"
        )

    # No ties, no reverse pairs, no duplicate metadata rows, correct provenance —
    # everything the validator checks, run for real against real upstream artifacts.
    report = validate_preference_run(run_dir, ranking_run_dir, candidate_run_dir)
    assert report.valid, [i.message for i in report.issues]

    # The split is problem-level: p001 (the only pair-bearing problem) lands in exactly
    # one split, and p002 (pairless) is in none of them.
    split_manifest = PreferenceRunRepository(tmp_path / "preferences").read_split_manifest(
        preference_run_id
    )
    all_split_ids = (
        split_manifest.train_problem_ids
        + split_manifest.validation_problem_ids
        + split_manifest.test_problem_ids
    )
    assert all_split_ids == (PROBLEM.id,)


# ------------------------------------------------------------------------------ reproducibility


def test_reproducibility_two_runs_over_identical_input_agree(tmp_path):
    _, _, assessments_by_problem, candidates_by_id = build_upstream_runs(tmp_path)
    problems_by_id = {PROBLEM.id: PROBLEM, OTHER_PROBLEM.id: OTHER_PROBLEM}

    _, id_a = run_full_preferences_pipeline(
        tmp_path / "preferences_a", problems_by_id, assessments_by_problem, candidates_by_id
    )
    _, id_b = run_full_preferences_pipeline(
        tmp_path / "preferences_b", problems_by_id, assessments_by_problem, candidates_by_id
    )
    repo_a = PreferenceRunRepository(tmp_path / "preferences_a").results(id_a)
    repo_b = PreferenceRunRepository(tmp_path / "preferences_b").results(id_b)

    def strip(record: dict[str, Any]) -> dict[str, Any]:
        stripped = dict(record)
        for key in ("preference_run_id", "created_at"):
            stripped.pop(key, None)
        return stripped

    pairs_a = sorted(
        (strip(p.to_dict()) for p in repo_a.load_pairs()), key=lambda d: d["preference_id"]
    )
    pairs_b = sorted(
        (strip(p.to_dict()) for p in repo_b.load_pairs()), key=lambda d: d["preference_id"]
    )
    assert pairs_a == pairs_b

    training_a = repo_a.preferences_path.read_text(encoding="utf-8")
    training_b = repo_b.preferences_path.read_text(encoding="utf-8")
    assert training_a == training_b
