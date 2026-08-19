"""Stage 7 as a pipeline stage: candidate evaluation / ranking (spec 12 section 5, item 4).

Named ``candidate_evaluation`` -- this is the repo's ``rank`` command: it classifies each
candidate correct/incorrect/indeterminate, scores it, and ranks candidates per problem.
The Docker execution that produced the evidence it judges already happened in
``candidate_execution`` (Stage 6); this stage calls no sandbox and no model.

``_rank_problem_group``, ``_write_ranking_statistics`` and ``_rank_selected_problem_ids``
are moved here from ``python_dpo.cli`` -- all three were already argparse-independent.
"""

from __future__ import annotations

import logging
from typing import Any

from ...evaluation import EvaluationRunRepository
from ...ranking import (
    COMPARATOR_VERSION,
    RANKING_VERSION,
    SCORING_VERSION,
    CandidateComparator,
    CandidateRanker,
    CandidateScorer,
    RankingRunRepository,
    RankingStatistics,
)
from ...runs import RunRepository
from ..errors import StageFailedError
from ..hashing import sha256_tree
from ._context import StageContext, StageResult

logger = logging.getLogger("python_dpo.pipeline.stages.candidate_evaluation")


def _rank_selected_problem_ids(
    results: Any, failures: Any, problem_id: str | None, limit: int | None
) -> list[str]:
    all_ids = sorted({r.problem_id for r in results} | {f.problem_id for f in failures})
    if problem_id is not None:
        if problem_id not in all_ids:
            raise ValueError(f"no evaluated candidates for problem id {problem_id!r}")
        return [problem_id]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        return all_ids[:limit]
    return all_ids


def _rank_problem_group(
    ranking_run_id: str,
    problem_id: str,
    results: Any,
    failures: Any,
    candidates_by_id: dict,
    scorer: CandidateScorer,
    ranker: CandidateRanker,
    comparator: CandidateComparator,
    repository: Any,
) -> int:
    assessments = []
    for result in results:
        if result.problem_id != problem_id:
            continue
        assessments.append(
            scorer.score(
                ranking_run_id=ranking_run_id,
                evaluation_run_id=result.evaluation_run_id,
                candidate_run_id=result.candidate_run_id,
                candidate_id=result.candidate_id,
                problem_id=result.problem_id,
                result=result,
                candidate=candidates_by_id.get(result.candidate_id),
            )
        )
    for failure in failures:
        if failure.problem_id != problem_id:
            continue
        assessments.append(
            scorer.score(
                ranking_run_id=ranking_run_id,
                evaluation_run_id=failure.evaluation_run_id,
                candidate_run_id=failure.candidate_run_id,
                candidate_id=failure.candidate_id,
                problem_id=failure.problem_id,
                result=None,
                missing_error_type=failure.error_type,
                candidate=candidates_by_id.get(failure.candidate_id),
            )
        )

    repository.save_assessments(assessments)
    rankings = ranker.rank_problem(ranking_run_id, problem_id, assessments)
    repository.save_rankings(rankings)
    comparisons = comparator.build_matrix(ranking_run_id, assessments)
    repository.save_comparisons(comparisons)
    return len(assessments)


def _write_ranking_statistics(
    ranking_run_repo: RankingRunRepository, ranking_run_id: str
) -> RankingStatistics:
    manifest = ranking_run_repo.get_run(ranking_run_id)
    repository = ranking_run_repo.results(ranking_run_id)
    stats = RankingStatistics.from_records(
        manifest, repository.load_assessments(), repository.load_rankings()
    )
    ranking_run_repo.write_statistics(stats)
    return stats


def run(context: StageContext) -> StageResult:
    config = context.project_config
    settings = context.stage_config

    evaluation_run_id = context.upstream_run_id("candidate_execution")

    eval_run_repo = EvaluationRunRepository(config.paths.evaluations / "runs")
    evaluation_manifest = eval_run_repo.get_run(evaluation_run_id)
    eval_results_repo = eval_run_repo.results(evaluation_run_id)
    results = eval_results_repo.load_all()
    failures = eval_results_repo.load_failures()
    if not results and not failures:
        raise StageFailedError(f"evaluation run {evaluation_run_id!r} has no results to rank")

    candidate_repo = RunRepository(config.paths.candidates / "runs").candidates(
        evaluation_manifest.candidate_run_id
    )
    candidates_by_id = {c.candidate_id: c for c in candidate_repo.load_all()}

    try:
        selected_problem_ids = _rank_selected_problem_ids(
            results, failures, settings.get("problem_id"), settings.get("limit")
        )
    except ValueError as exc:
        raise StageFailedError(str(exc)) from exc

    ranking_run_repo = RankingRunRepository(config.paths.rankings / "runs")
    manifest = ranking_run_repo.create_run(
        evaluation_run_id=evaluation_run_id,
        candidate_run_id=evaluation_manifest.candidate_run_id,
        ranking_version=RANKING_VERSION,
        scoring_version=SCORING_VERSION,
        comparator_version=COMPARATOR_VERSION,
        requested_problem_ids=selected_problem_ids,
    )
    manifest = ranking_run_repo.start_run(manifest.ranking_run_id)
    logger.info(
        "Ranking run %s created | evaluation run %s | %d problem(s)",
        manifest.ranking_run_id,
        evaluation_run_id,
        len(selected_problem_ids),
    )

    repository = ranking_run_repo.results(manifest.ranking_run_id)
    scorer, ranker, comparator = CandidateScorer(), CandidateRanker(), CandidateComparator()

    for problem_id in manifest.requested_problem_ids:
        _rank_problem_group(
            manifest.ranking_run_id,
            problem_id,
            results,
            failures,
            candidates_by_id,
            scorer,
            ranker,
            comparator,
            repository,
        )

    stats = _write_ranking_statistics(ranking_run_repo, manifest.ranking_run_id)
    complete = set(manifest.requested_problem_ids) <= repository.ranked_problem_ids()

    if complete:
        ranking_run_repo.complete_run(manifest.ranking_run_id)
    else:
        ranking_run_repo.interrupt_run(manifest.ranking_run_id)
        raise StageFailedError(f"ranking run {manifest.ranking_run_id} did not complete")

    logger.info(
        "Ranking run %s completed | candidates=%d correct=%d incorrect=%d indeterminate=%d",
        manifest.ranking_run_id,
        stats.candidates,
        stats.correct,
        stats.incorrect,
        stats.indeterminate,
    )

    return StageResult(
        stage_run_id=manifest.ranking_run_id,
        output_artifacts={"rankings": sha256_tree(ranking_run_repo.run_dir(manifest.ranking_run_id))},
    )


__all__ = ["run"]
