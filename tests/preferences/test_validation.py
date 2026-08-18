"""Tests for validate_preference_run (spec 08 sections 69-73), each built by mutating a
real, valid preference run produced by the actual builder — mirroring
``tests/ranking/test_validation.py``'s approach for Stage 7.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from python_dpo.candidates.models import Candidate
from python_dpo.candidates.repository import CandidateRepository
from python_dpo.generation.prompt_builder import PROMPT_VERSION, build_prompt
from python_dpo.preferences.builder import PreferencePairBuilder
from python_dpo.preferences.models import (
    PreferenceStatistics,
    build_quality_report,
    derive_candidates_considered,
)
from python_dpo.preferences.policies import StrictPolicy
from python_dpo.preferences.repository import PreferenceRepository
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


def _make_candidate(candidate_id: str, code: str, strategy: str) -> Candidate:
    return Candidate.create(
        candidate_id=candidate_id,
        problem_id=PROBLEM.id,
        run_id=CANDIDATE_RUN_ID,
        generation_index=int(candidate_id.rsplit("c", 1)[1]),
        strategy=strategy,
        model="Qwen/Qwen2.5-Coder-3B-Instruct",
        provider="transformers",
        prompt_version=PROMPT_VERSION,
        prompt=build_prompt(PROBLEM, strategy),
        raw_output=f"```python\n{code}\n```",
        code=code,
        extraction_format="python_fence",
        syntax_valid=True,
        function_name_valid=True,
        generation_config={},
        created_at="2026-08-17T05:54:18Z",
    )


def _make_assessment(candidate_id: str, passed: int, total: int = 7) -> CandidateAssessment:
    correctness = "correct" if passed == total else "incorrect"
    return CandidateAssessment(
        ranking_run_id=RANKING_RUN_ID,
        evaluation_run_id=EVAL_RUN_ID,
        candidate_run_id=CANDIDATE_RUN_ID,
        candidate_id=candidate_id,
        problem_id=PROBLEM.id,
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


def build_valid_run(tmp_path: Path) -> tuple[Path, Path, Path, PreferenceRepository]:
    """Build a small, real, valid preference run (one strong pair) plus its source
    candidate and ranking runs, and return
    ``(preference_run_dir, ranking_run_dir, candidate_run_dir, repo)``.
    """
    candidate_run_dir = tmp_path / "candidates"
    candidate_repo = CandidateRepository(candidate_run_dir)
    c1 = _make_candidate("p001_c001", "def sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)", "normal")
    c2 = _make_candidate("p001_c002", "def sum_even(numbers):\n    return 0", "alternative")
    candidate_repo.save(c1)
    candidate_repo.save(c2)
    candidates_by_id = {c1.candidate_id: c1, c2.candidate_id: c2}

    ranking_run_dir = tmp_path / "ranking"
    ranking_repo = RankingRepository(ranking_run_dir)
    a1 = _make_assessment("p001_c001", 7, 7)
    a2 = _make_assessment("p001_c002", 3, 7)
    ranking_repo.save_assessments([a1, a2])

    preferences_root = tmp_path / "preferences"
    run_repo = PreferenceRunRepository(preferences_root)
    manifest = run_repo.create_run(
        ranking_run_id=RANKING_RUN_ID,
        evaluation_run_id=EVAL_RUN_ID,
        candidate_run_id=CANDIDATE_RUN_ID,
        preference_version="v1",
        selection_policy="strict",
        selection_policy_version="strict_v1",
        minimum_score_margin=0.2,
        split_ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
        split_seed=42,
        builder_version="v1",
    )
    run_repo.start_run(manifest.preference_run_id)
    repo = run_repo.results(manifest.preference_run_id)

    builder = PreferencePairBuilder(StrictPolicy(), minimum_score_margin=0.2)
    result = builder.build_problem(
        ranking_run_id=RANKING_RUN_ID,
        evaluation_run_id=EVAL_RUN_ID,
        candidate_run_id=CANDIDATE_RUN_ID,
        problem=PROBLEM,
        assessments=[a1, a2],
        candidates_by_id=candidates_by_id,
    )
    repo.save_pairs(result.pairs)
    repo.save_rejections(result.rejections)

    training_ids = {p.problem_id for p in result.pairs if not p.duplicate_training_record}
    split_manifest = ProblemSplitter(ratios=manifest.split_ratios, seed=manifest.split_seed).split(
        training_ids
    )
    run_repo.write_split_manifest(manifest.preference_run_id, split_manifest)
    repo.write_dataset(split_manifest)

    stats = PreferenceStatistics.from_records(
        manifest,
        result.pairs,
        result.rejections,
        candidates_considered=derive_candidates_considered(result.pairs, result.rejections),
    )
    run_repo.write_statistics(stats)

    quality_report = build_quality_report(
        manifest, result.pairs, {PROBLEM.id: [a1.correctness, a2.correctness]}
    )
    run_repo.write_quality_report(quality_report)

    run_repo.complete_run(manifest.preference_run_id)

    return run_repo.run_dir(manifest.preference_run_id), ranking_run_dir, candidate_run_dir, repo


def _rewrite_line(path: Path, match_key: str, match_value: str, **updates: Any) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    for line in lines:
        record = json.loads(line)
        if record.get(match_key) == match_value:
            record.update(updates)
        new_lines.append(json.dumps(record))
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _append_line(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------------- baseline


def test_a_valid_run_passes(tmp_path):
    run_dir, ranking_run_dir, candidate_run_dir, _ = build_valid_run(tmp_path)
    report = validate_preference_run(run_dir, ranking_run_dir, candidate_run_dir)
    assert report.valid, [i.message for i in report.issues]


def test_valid_run_passes_without_either_cross_check(tmp_path):
    run_dir, _, _, _ = build_valid_run(tmp_path)
    report = validate_preference_run(run_dir)
    assert report.valid


# ---------------------------------------------------------------------------- spec section 71


def test_a_reversed_pair_is_caught(tmp_path):
    run_dir, ranking_run_dir, candidate_run_dir, repo = build_valid_run(tmp_path)
    from .test_models import make_pair

    reversed_pair = make_pair(
        preference_id="pref_p001_c002__p001_c001",
        problem_id="p001",
        chosen_candidate_id="p001_c002",
        rejected_candidate_id="p001_c001",
        chosen_score=0.9,
        rejected_score=0.1,
        score_margin=0.8,
        chosen_pass_rate=0.9,
        rejected_pass_rate=0.1,
        chosen_tests_passed=9,
        rejected_tests_passed=1,
        chosen_tests_total=10,
        rejected_tests_total=10,
    )
    _append_line(repo.metadata_path, reversed_pair.to_dict())

    report = validate_preference_run(run_dir)
    assert not report.valid
    assert any(issue.check == "reverse_pair" for issue in report.issues)


# ---------------------------------------------------------------------------- spec section 72


def test_duplicate_metadata_row_is_caught(tmp_path):
    run_dir, _, _, repo = build_valid_run(tmp_path)
    first_line = repo.metadata_path.read_text(encoding="utf-8").splitlines()[0]
    _append_line(repo.metadata_path, json.loads(first_line))

    report = validate_preference_run(run_dir)
    assert not report.valid
    assert any(issue.check == "duplicate_metadata" for issue in report.issues)


def test_duplicate_training_record_in_preferences_file_is_caught(tmp_path):
    run_dir, _, _, repo = build_valid_run(tmp_path)
    first_line = repo.preferences_path.read_text(encoding="utf-8").splitlines()[0]
    _append_line(repo.preferences_path, json.loads(first_line))

    report = validate_preference_run(run_dir)
    assert not report.valid
    assert any(issue.check == "duplicate_training_record" for issue in report.issues)


# -------------------------------------------------------------------- spec sections 69, 70


def test_a_flipped_score_fails_the_cross_check(tmp_path):
    run_dir, ranking_run_dir, _, repo = build_valid_run(tmp_path)
    _rewrite_line(
        repo.metadata_path,
        "chosen_candidate_id",
        "p001_c001",
        chosen_score=0.9,
        score_margin=0.9 - (3 / 7),
    )
    report = validate_preference_run(run_dir, ranking_run_dir)
    assert not report.valid
    assert any(issue.check == "score_recompute" for issue in report.issues)


def test_chosen_code_no_longer_matching_the_candidate_is_caught(tmp_path):
    run_dir, _, candidate_run_dir, repo = build_valid_run(tmp_path)
    _rewrite_line(
        repo.metadata_path,
        "chosen_candidate_id",
        "p001_c001",
        chosen="def sum_even(numbers):\n    return 999",
    )
    report = validate_preference_run(run_dir, candidate_run_dir=candidate_run_dir)
    assert not report.valid
    assert any(issue.check == "code_provenance" for issue in report.issues)


def test_unknown_candidate_in_metadata_is_caught(tmp_path):
    run_dir, _, candidate_run_dir, repo = build_valid_run(tmp_path)
    _rewrite_line(
        repo.metadata_path,
        "chosen_candidate_id",
        "p001_c001",
        chosen_candidate_id="p001_c999",
    )
    report = validate_preference_run(run_dir, candidate_run_dir=candidate_run_dir)
    assert not report.valid
    assert any(issue.check == "unknown_candidate" for issue in report.issues)


# ------------------------------------------------------------------------------ splits


def test_a_problem_in_two_splits_is_caught(tmp_path):
    run_dir, _, _, _ = build_valid_run(tmp_path)
    split_path = run_dir / "split_manifest.json"
    on_disk = json.loads(split_path.read_text(encoding="utf-8"))
    on_disk["validation_problem_ids"] = list(on_disk["train_problem_ids"])
    split_path.write_text(json.dumps(on_disk), encoding="utf-8")

    report = validate_preference_run(run_dir)
    assert not report.valid
    assert any(issue.check == "split_manifest" for issue in report.issues)


# --------------------------------------------------------------------------------- statistics


def test_drifted_statistics_are_caught(tmp_path):
    run_dir, _, _, _ = build_valid_run(tmp_path)
    stats_path = run_dir / "statistics.json"
    on_disk = json.loads(stats_path.read_text(encoding="utf-8"))
    on_disk["pairs_generated"] = on_disk["pairs_generated"] + 1
    on_disk["candidate_pairs_considered"] = on_disk["candidate_pairs_considered"] + 1
    stats_path.write_text(json.dumps(on_disk), encoding="utf-8")

    report = validate_preference_run(run_dir)
    assert not report.valid
    assert any(issue.check == "statistics" for issue in report.issues)


def test_missing_statistics_file_is_caught(tmp_path):
    run_dir, _, _, _ = build_valid_run(tmp_path)
    (run_dir / "statistics.json").unlink()

    report = validate_preference_run(run_dir)
    assert not report.valid
    assert any(issue.check == "statistics" for issue in report.issues)


# ----------------------------------------------------------------------------------- manifest


def test_missing_manifest_is_caught(tmp_path):
    run_dir, _, _, _ = build_valid_run(tmp_path)
    (run_dir / "manifest.json").unlink()

    report = validate_preference_run(run_dir)
    assert not report.valid
    assert any(issue.check == "manifest" for issue in report.issues)
