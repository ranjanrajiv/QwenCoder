"""Tests for validate_ranking_run (spec 07 sections 50-52), each built by mutating a
real, valid ranking run produced by the actual scorer/ranker/comparator — mirroring
``tests/test_run_validation.py``'s approach for Stage 4.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from python_dpo.evaluation.models import EvaluationManifest, EvaluationResult
from python_dpo.evaluation.repository import EvaluationRepository
from python_dpo.ranking.comparator import CandidateComparator
from python_dpo.ranking.ranker import CandidateRanker
from python_dpo.ranking.repository import RankingRepository
from python_dpo.ranking.run_repository import RankingRunRepository
from python_dpo.ranking.scorer import CandidateScorer
from python_dpo.ranking.validation import validate_ranking_run

EVAL_RUN_ID = "eval_20260817_154500_a12f"
CANDIDATE_RUN_ID = "run_20260817_055411"


def _make_result(candidate_id: str, passed: int, total: int = 10) -> EvaluationResult:
    status = "passed" if passed == total else "failed"
    return EvaluationResult.create(
        evaluation_run_id=EVAL_RUN_ID,
        candidate_run_id=CANDIDATE_RUN_ID,
        candidate_id=candidate_id,
        problem_id="p001",
        status=status,
        tests_passed=passed,
        tests_failed=total - passed,
        tests_error=0,
        tests_skipped=0,
        duration_ms=100,
    )


def build_valid_run(tmp_path: Path) -> tuple[Path, Path, RankingRepository]:
    """Build a small, real, valid ranking run (3 candidates, one tied pair) plus its
    source evaluation run, and return ``(ranking_run_dir, evaluation_run_dir, repo)``.
    """
    evaluation_run_dir = tmp_path / "eval_run"
    eval_repo = EvaluationRepository(evaluation_run_dir)
    results = [
        _make_result("p001_c001", 10),
        _make_result("p001_c002", 10),  # tied with c001
        _make_result("p001_c003", 7),
    ]
    for result in results:
        eval_repo.save(result)

    manifest = EvaluationManifest(
        evaluation_run_id=EVAL_RUN_ID,
        candidate_run_id=CANDIDATE_RUN_ID,
        status="completed",
        created_at="2026-08-17T11:51:54Z",
        evaluator_version="v1",
        test_generator_version="v1",
        pytest_version="8.3.4",
        python_version="3.12.14",
        sandbox_config={"image": "python-dpo-evaluator:1.0"},
        requested_candidate_ids=tuple(r.candidate_id for r in results),
    )
    (evaluation_run_dir).mkdir(parents=True, exist_ok=True)
    (evaluation_run_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict()), encoding="utf-8"
    )

    rankings_root = tmp_path / "rankings"
    run_repo = RankingRunRepository(rankings_root)
    ranking_manifest = run_repo.create_run(
        evaluation_run_id=EVAL_RUN_ID,
        candidate_run_id=CANDIDATE_RUN_ID,
        ranking_version="v1",
        scoring_version="v1",
        comparator_version="v1",
        requested_problem_ids=["p001"],
    )
    run_repo.start_run(ranking_manifest.ranking_run_id)
    repo = run_repo.results(ranking_manifest.ranking_run_id)

    scorer = CandidateScorer()
    assessments = [
        scorer.score(
            ranking_run_id=ranking_manifest.ranking_run_id,
            evaluation_run_id=EVAL_RUN_ID,
            candidate_run_id=CANDIDATE_RUN_ID,
            candidate_id=result.candidate_id,
            problem_id=result.problem_id,
            result=result,
        )
        for result in results
    ]
    repo.save_assessments(assessments)

    ranker = CandidateRanker()
    rankings = ranker.rank(ranking_manifest.ranking_run_id, assessments)
    repo.save_rankings(rankings)

    comparator = CandidateComparator()
    comparisons = comparator.build_matrix(ranking_manifest.ranking_run_id, assessments)
    repo.save_comparisons(comparisons)

    ranking_manifest = run_repo.complete_run(ranking_manifest.ranking_run_id)
    from python_dpo.ranking.models import RankingStatistics

    stats = RankingStatistics.from_records(ranking_manifest, assessments, rankings)
    run_repo.write_statistics(stats)

    return run_repo.run_dir(ranking_manifest.ranking_run_id), evaluation_run_dir, repo


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
    run_dir, evaluation_run_dir, _ = build_valid_run(tmp_path)
    report = validate_ranking_run(run_dir, evaluation_run_dir)
    assert report.valid, [i.message for i in report.issues]


def test_valid_run_passes_without_the_evaluation_run_cross_check(tmp_path):
    # evaluation_run_dir is optional; omitting it must not itself be an error.
    run_dir, _, _ = build_valid_run(tmp_path)
    report = validate_ranking_run(run_dir)
    assert report.valid


# ------------------------------------------------------------------- spec sections 51, 52


def test_a_tampered_but_self_consistent_assessment_fails_the_cross_check(tmp_path):
    # A record that is internally coherent (CandidateAssessment.__post_init__ accepts it)
    # but no longer reflects the source evaluation evidence: the real p001_c003 result is
    # 7/10 incorrect; tampered to claim a full 10/10 pass.
    run_dir, evaluation_run_dir, repo = build_valid_run(tmp_path)
    _rewrite_line(
        repo.assessments_path,
        "candidate_id",
        "p001_c003",
        correctness="correct",
        all_tests_passed=True,
        pass_rate=1.0,
        score=1.0,
        tests_total=10,
        tests_passed=10,
        tests_failed=0,
    )
    report = validate_ranking_run(run_dir, evaluation_run_dir)
    assert not report.valid
    checks = {issue.check for issue in report.issues}
    assert "correctness_recompute" in checks
    assert "pass_rate_recompute" in checks
    assert "counts_recompute" in checks


def test_missing_assessment_for_an_evaluated_candidate_is_caught(tmp_path):
    # Spec section 70: every evaluated candidate must have an assessment.
    run_dir, evaluation_run_dir, repo = build_valid_run(tmp_path)
    lines = repo.assessments_path.read_text(encoding="utf-8").splitlines()
    kept = [l for l in lines if json.loads(l)["candidate_id"] != "p001_c003"]
    repo.assessments_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    report = validate_ranking_run(run_dir, evaluation_run_dir)
    assert not report.valid
    assert any(issue.check == "missing_assessment" for issue in report.issues)


# ------------------------------------------------------------------------- spec section 43


def test_duplicate_assessment_is_caught(tmp_path):
    run_dir, evaluation_run_dir, repo = build_valid_run(tmp_path)
    lines = repo.assessments_path.read_text(encoding="utf-8").splitlines()
    first_record = json.loads(lines[0])
    _append_line(repo.assessments_path, first_record)

    report = validate_ranking_run(run_dir, evaluation_run_dir)
    assert not report.valid
    assert any(issue.check == "duplicate_assessment" for issue in report.issues)


# ------------------------------------------------------- spec sections 29, 31, 35 (tied pair)


def test_a_tied_pair_given_a_winner_is_caught(tmp_path):
    run_dir, evaluation_run_dir, repo = build_valid_run(tmp_path)
    # c001 and c002 are tied (both 10/10); rewrite their comparison to declare a winner.
    _rewrite_line(
        repo.comparisons_path,
        "candidate_a",
        "p001_c001",
        relation="A_BETTER",
        comparison_eligible=True,
        score_margin=0.0,  # keep the model's own invariant (score_margin == |a-b|) satisfied
    )
    report = validate_ranking_run(run_dir, evaluation_run_dir)
    assert not report.valid
    assert any(issue.check == "artificial_preference" for issue in report.issues)


# --------------------------------------------------------------------- unconstructible states


def test_an_indeterminate_result_marked_preference_eligible_cannot_even_be_loaded(tmp_path):
    # RankingResult.__post_init__ itself forbids this combination (spec section 32), so a
    # tampered raw record fails to construct on load and surfaces as a schema issue —
    # the model is the first line of defense, the validator's load step is the second.
    run_dir, evaluation_run_dir, repo = build_valid_run(tmp_path)
    _append_line(
        repo.rankings_path,
        {
            "ranking_run_id": json.loads(repo.rankings_path.read_text().splitlines()[0])["ranking_run_id"],
            "evaluation_run_id": EVAL_RUN_ID,
            "problem_id": "p001",
            "candidate_id": "p001_c999",
            "rank": None,
            "score": 0.0,
            "correctness": "indeterminate",
            "pass_rate": 0.0,
            "all_tests_passed": False,
            "tie_group": None,
            "tie_group_size": 0,
            "tied": False,
            "eligible_for_preference": True,  # invalid: indeterminate must be false
            "created_at": "2026-08-17T18:05:00Z",
        },
    )
    report = validate_ranking_run(run_dir, evaluation_run_dir)
    assert not report.valid
    assert any(issue.check == "schema" for issue in report.issues)


# --------------------------------------------------------------------------------- statistics


def test_drifted_statistics_are_caught(tmp_path):
    run_dir, evaluation_run_dir, _ = build_valid_run(tmp_path)
    stats_path = run_dir / "statistics.json"
    on_disk = json.loads(stats_path.read_text(encoding="utf-8"))
    on_disk["candidates"] = on_disk["candidates"] + 1
    stats_path.write_text(json.dumps(on_disk), encoding="utf-8")

    report = validate_ranking_run(run_dir, evaluation_run_dir)
    assert not report.valid
    assert any(issue.check == "statistics" for issue in report.issues)


def test_missing_statistics_file_is_caught(tmp_path):
    run_dir, evaluation_run_dir, _ = build_valid_run(tmp_path)
    (run_dir / "statistics.json").unlink()

    report = validate_ranking_run(run_dir, evaluation_run_dir)
    assert not report.valid
    assert any(issue.check == "statistics" for issue in report.issues)


# --------------------------------------------------------------------------------- manifest


def test_missing_manifest_is_caught(tmp_path):
    run_dir, evaluation_run_dir, _ = build_valid_run(tmp_path)
    (run_dir / "manifest.json").unlink()

    report = validate_ranking_run(run_dir, evaluation_run_dir)
    assert not report.valid
    assert any(issue.check == "manifest" for issue in report.issues)
