"""End-to-end ranking tests (spec 07 sections 73, 75, 76).

Pure computation — no Docker, no model, no filesystem beyond ``tmp_path`` — so this runs
in the default offline suite, unlike the Stage 5/6 integration suites.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from python_dpo.evaluation.models import EvaluationManifest, EvaluationResult
from python_dpo.evaluation.repository import EvaluationRepository
from python_dpo.ranking.comparator import COMPARATOR_VERSION, CandidateComparator
from python_dpo.ranking.models import RankingStatistics
from python_dpo.ranking.ranker import RANKING_VERSION, CandidateRanker
from python_dpo.ranking.run_repository import RankingRunRepository
from python_dpo.ranking.scorer import SCORING_VERSION, CandidateScorer
from python_dpo.ranking.validation import validate_ranking_run

EVAL_RUN_ID = "eval_20260817_154500_a12f"
CANDIDATE_RUN_ID = "run_20260817_055411"


def _make_result(candidate_id: str, passed: int, total: int = 10) -> EvaluationResult:
    return EvaluationResult.create(
        evaluation_run_id=EVAL_RUN_ID,
        candidate_run_id=CANDIDATE_RUN_ID,
        candidate_id=candidate_id,
        problem_id="p001",
        status="passed" if passed == total else "failed",
        tests_passed=passed,
        tests_failed=total - passed,
        tests_error=0,
        tests_skipped=0,
        duration_ms=100,
    )


def build_evaluation_run(evaluation_run_dir: Path) -> list[EvaluationResult]:
    """The spec section 73/74 fixture: A=10/10, B=8/10, C=10/10, D=5/10, E=0/10."""
    eval_repo = EvaluationRepository(evaluation_run_dir)
    results = [
        _make_result("p001_A", 10),
        _make_result("p001_B", 8),
        _make_result("p001_C", 10),
        _make_result("p001_D", 5),
        _make_result("p001_E", 0),
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
    evaluation_run_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_run_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict()), encoding="utf-8"
    )
    return results


def run_full_ranking_pipeline(
    rankings_root: Path,
    results: list[EvaluationResult],
    *,
    ranking_version: str = RANKING_VERSION,
    scoring_version: str = SCORING_VERSION,
    comparator_version: str = COMPARATOR_VERSION,
) -> tuple[Path, str]:
    """The full pipeline a CLI invocation drives: score -> rank -> compare -> persist ->
    complete -> statistics. Returns ``(run_dir, ranking_run_id)``.
    """
    run_repo = RankingRunRepository(rankings_root)
    manifest = run_repo.create_run(
        evaluation_run_id=EVAL_RUN_ID,
        candidate_run_id=CANDIDATE_RUN_ID,
        ranking_version=ranking_version,
        scoring_version=scoring_version,
        comparator_version=comparator_version,
        requested_problem_ids=["p001"],
    )
    manifest = run_repo.start_run(manifest.ranking_run_id)
    repository = run_repo.results(manifest.ranking_run_id)

    scorer = CandidateScorer()
    assessments = [
        scorer.score(
            ranking_run_id=manifest.ranking_run_id,
            evaluation_run_id=r.evaluation_run_id,
            candidate_run_id=r.candidate_run_id,
            candidate_id=r.candidate_id,
            problem_id=r.problem_id,
            result=r,
        )
        for r in results
    ]
    repository.save_assessments(assessments)

    ranker = CandidateRanker()
    rankings = ranker.rank(manifest.ranking_run_id, assessments)
    repository.save_rankings(rankings)

    comparator = CandidateComparator()
    comparisons = comparator.build_matrix(manifest.ranking_run_id, assessments)
    repository.save_comparisons(comparisons)

    manifest = run_repo.complete_run(manifest.ranking_run_id)
    stats = RankingStatistics.from_records(manifest, assessments, rankings)
    run_repo.write_statistics(stats)

    return run_repo.run_dir(manifest.ranking_run_id), manifest.ranking_run_id


# ------------------------------------------------------------------------- section 73/74


def test_end_to_end_flow_matches_the_worked_example(tmp_path):
    evaluation_run_dir = tmp_path / "eval_run"
    results = build_evaluation_run(evaluation_run_dir)

    run_dir, ranking_run_id = run_full_ranking_pipeline(tmp_path / "rankings", results)
    repository = RankingRunRepository(tmp_path / "rankings").results(ranking_run_id)

    by_id = {r.candidate_id: r for r in repository.load_rankings()}
    assert by_id["p001_A"].rank == 1
    assert by_id["p001_C"].rank == 1
    assert by_id["p001_A"].tie_group == by_id["p001_C"].tie_group
    assert by_id["p001_B"].rank == 3
    assert by_id["p001_D"].rank == 4
    assert by_id["p001_E"].rank == 5

    report = validate_ranking_run(run_dir, evaluation_run_dir)
    assert report.valid, [i.message for i in report.issues]


# ------------------------------------------------------------------------------ section 75


def test_reproducibility_two_runs_over_identical_input_agree(tmp_path):
    evaluation_run_dir = tmp_path / "eval_run"
    results = build_evaluation_run(evaluation_run_dir)

    run_dir_a, ranking_run_id_a = run_full_ranking_pipeline(tmp_path / "rankings_a", results)
    run_dir_b, ranking_run_id_b = run_full_ranking_pipeline(tmp_path / "rankings_b", results)

    repo_a = RankingRunRepository(tmp_path / "rankings_a").results(ranking_run_id_a)
    repo_b = RankingRunRepository(tmp_path / "rankings_b").results(ranking_run_id_b)

    def strip_ids(record: dict[str, Any]) -> dict[str, Any]:
        stripped = dict(record)
        for key in ("ranking_run_id", "created_at"):
            stripped.pop(key, None)
        return stripped

    assessments_a = sorted(
        (strip_ids(a.to_dict()) for a in repo_a.load_assessments()),
        key=lambda d: d["candidate_id"],
    )
    assessments_b = sorted(
        (strip_ids(a.to_dict()) for a in repo_b.load_assessments()),
        key=lambda d: d["candidate_id"],
    )
    assert assessments_a == assessments_b

    rankings_a = sorted(
        (strip_ids(r.to_dict()) for r in repo_a.load_rankings()), key=lambda d: d["candidate_id"]
    )
    rankings_b = sorted(
        (strip_ids(r.to_dict()) for r in repo_b.load_rankings()), key=lambda d: d["candidate_id"]
    )
    # tie_group ids are also run-scoped in shape (problem-scoped, not globally unique
    # across separate runs) but their *values* are still deterministic given the same
    # input, so no extra stripping is needed beyond the run id.
    assert rankings_a == rankings_b

    def strip_comparison(record: dict[str, Any]) -> dict[str, Any]:
        stripped = dict(record)
        for key in ("ranking_run_id", "created_at"):
            stripped.pop(key, None)
        return stripped

    comparisons_a = sorted(
        (strip_comparison(c.to_dict()) for c in repo_a.load_comparisons()),
        key=lambda d: (d["candidate_a"], d["candidate_b"]),
    )
    comparisons_b = sorted(
        (strip_comparison(c.to_dict()) for c in repo_b.load_comparisons()),
        key=lambda d: (d["candidate_a"], d["candidate_b"]),
    )
    assert comparisons_a == comparisons_b


# ------------------------------------------------------------------------------ section 76


def test_changing_scoring_version_creates_a_new_run_leaving_the_old_untouched(tmp_path):
    evaluation_run_dir = tmp_path / "eval_run"
    results = build_evaluation_run(evaluation_run_dir)
    rankings_root = tmp_path / "rankings"

    run_dir_v1, ranking_run_id_v1 = run_full_ranking_pipeline(
        rankings_root, results, scoring_version="v1"
    )
    before = (run_dir_v1 / "rankings.jsonl").read_text(encoding="utf-8")

    run_dir_v2, ranking_run_id_v2 = run_full_ranking_pipeline(
        rankings_root, results, scoring_version="v2"
    )

    assert ranking_run_id_v1 != ranking_run_id_v2
    assert run_dir_v1 != run_dir_v2
    # The original artifact is byte-for-byte unchanged (spec section 45).
    after = (run_dir_v1 / "rankings.jsonl").read_text(encoding="utf-8")
    assert before == after

    run_repo = RankingRunRepository(rankings_root)
    assert run_repo.get_run(ranking_run_id_v1).scoring_version == "v1"
    assert run_repo.get_run(ranking_run_id_v2).scoring_version == "v2"
