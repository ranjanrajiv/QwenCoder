"""Stage 6 as a pipeline stage: candidate execution (spec 12 section 5, item 3).

Named ``candidate_execution`` -- the repo's ``evaluate run`` command both executes
candidates in the Docker sandbox *and* runs the generated pytest suite in one pass, which
this plan splits into two spec stages (execution produces evidence, in this module;
evaluation/judgement is ``candidate_evaluation``, Stage 7's ranking).

``_execute_evaluation``, ``_write_evaluation_statistics``, ``_ensure_evaluation_image`` and
``_select_candidates_for_evaluation`` are moved here from ``python_dpo.cli`` -- all four
were already argparse-independent. Every candidate still runs only inside
:class:`~python_dpo.sandbox.SandboxExecutor` via :class:`~python_dpo.evaluation.PytestRunner`;
this stage adds no second execution path (CLAUDE.md's Security rule, spec section 34).
"""

from __future__ import annotations

import logging
from typing import Any

from ...evaluation import (
    EVALUATOR_VERSION,
    TEST_GENERATOR_VERSION,
    CandidateEvaluator,
    EvaluationError,
    EvaluationRunRepository,
    EvaluationStatistics,
    PytestRunner,
    build_evaluation_sandbox_config,
    probe_versions,
)
from ...problems import DatasetError, dataset_path, load_problems
from ...runs import RunRepository
from ...sandbox import DockerContainerRuntime, SandboxError, SandboxExecutor
from ..errors import StageFailedError
from ..hashing import sha256_tree
from ._context import StageContext, StageResult

logger = logging.getLogger("python_dpo.pipeline.stages.candidate_execution")


def _ensure_evaluation_image(evaluation_config: Any, runtime: Any = None) -> str | None:
    runtime = runtime if runtime is not None else DockerContainerRuntime()
    try:
        runtime.check_available()
        if runtime.image_present(evaluation_config.image):
            return None
        if evaluation_config.auto_pull:
            runtime.pull(evaluation_config.image)
            return None
    except SandboxError as exc:
        return str(exc)
    return (
        f"{evaluation_config.image} is not present and evaluation.auto_pull is false; "
        f"run: docker build -t {evaluation_config.image} docker/evaluator/"
    )


def _select_candidates_for_evaluation(candidates: Any, problem_id: str | None, limit: int | None):
    selected = list(candidates)
    if problem_id is not None:
        selected = [c for c in selected if c.problem_id == problem_id]
        if not selected:
            raise ValueError(f"no candidates for problem id {problem_id!r}")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        selected = selected[:limit]
    return selected


def _write_evaluation_statistics(
    eval_run_repo: EvaluationRunRepository, evaluation_run_id: str
) -> EvaluationStatistics:
    manifest = eval_run_repo.get_run(evaluation_run_id)
    repository = eval_run_repo.results(evaluation_run_id)
    stats = EvaluationStatistics.from_records(
        manifest, repository.load_all(), repository.load_failures()
    )
    eval_run_repo.write_statistics(stats)
    return stats


def _execute_evaluation(
    eval_run_repo: EvaluationRunRepository,
    manifest: Any,
    candidates: Any,
    problems_by_id: dict,
    runner: PytestRunner,
) -> int:
    manifest = eval_run_repo.start_run(manifest.evaluation_run_id)
    repository = eval_run_repo.results(manifest.evaluation_run_id)
    evaluator = CandidateEvaluator(runner=runner, repository=repository)

    try:
        summary = evaluator.evaluate_many(
            list(candidates), problems_by_id, evaluation_run_id=manifest.evaluation_run_id
        )
    except KeyboardInterrupt:
        eval_run_repo.interrupt_run(manifest.evaluation_run_id)
        _write_evaluation_statistics(eval_run_repo, manifest.evaluation_run_id)
        logger.warning("Evaluation run %s interrupted.", manifest.evaluation_run_id)
        return 130

    stats = _write_evaluation_statistics(eval_run_repo, manifest.evaluation_run_id)
    complete = stats.candidates_evaluated + stats.evaluation_failures >= manifest.requested_candidates

    if complete:
        eval_run_repo.complete_run(manifest.evaluation_run_id)
        final_status = "completed"
    else:
        eval_run_repo.interrupt_run(manifest.evaluation_run_id)
        final_status = "interrupted"

    logger.info(
        "Evaluation run %s %s | evaluated=%d skipped=%d machinery_failed=%d | passed=%d failed=%d",
        manifest.evaluation_run_id,
        final_status,
        summary.evaluated,
        summary.skipped,
        summary.machinery_failed,
        stats.passed,
        stats.failed,
    )
    return 0 if final_status == "completed" else 1


def run(context: StageContext) -> StageResult:
    config = context.project_config
    settings = context.stage_config

    candidate_run_id = context.upstream_run_id("candidate_generation")

    try:
        problems = load_problems(dataset_path(config.paths.problems))
    except DatasetError as exc:
        raise StageFailedError(f"could not load problem dataset: {exc}") from exc
    problems_by_id = {p.id: p for p in problems}

    candidate_run_repo = RunRepository(config.paths.candidates / "runs")
    try:
        all_candidates = candidate_run_repo.candidates(candidate_run_id).load_all()
    except Exception as exc:  # noqa: BLE001 - reported uniformly as a stage failure
        raise StageFailedError(f"could not load candidates for run {candidate_run_id!r}: {exc}") from exc

    try:
        selected = _select_candidates_for_evaluation(
            all_candidates, settings.get("problem_id"), settings.get("limit")
        )
    except ValueError as exc:
        raise StageFailedError(str(exc)) from exc

    missing_problems = sorted({c.problem_id for c in selected} - set(problems_by_id))
    if missing_problems:
        raise StageFailedError(
            f"candidate(s) reference unknown problem id(s): {', '.join(missing_problems)}"
        )

    image_error = _ensure_evaluation_image(config.evaluation)
    if image_error:
        raise StageFailedError(image_error)

    eval_sandbox_config = build_evaluation_sandbox_config(config.sandbox, config.evaluation)

    try:
        python_version, pytest_version = probe_versions(eval_sandbox_config)
    except EvaluationError as exc:
        raise StageFailedError(str(exc)) from exc

    eval_run_repo = EvaluationRunRepository(config.paths.evaluations / "runs")
    manifest = eval_run_repo.create_run(
        candidate_run_id=candidate_run_id,
        evaluator_version=EVALUATOR_VERSION,
        test_generator_version=TEST_GENERATOR_VERSION,
        pytest_version=pytest_version,
        python_version=python_version,
        sandbox_config=eval_sandbox_config.to_dict(),
        requested_candidate_ids=[c.candidate_id for c in selected],
        requested_problem_id=settings.get("problem_id"),
    )
    logger.info(
        "Evaluation run %s created | candidate run %s | %d candidate(s)",
        manifest.evaluation_run_id,
        candidate_run_id,
        len(selected),
    )

    runner = PytestRunner(SandboxExecutor(config=eval_sandbox_config))
    exit_code = _execute_evaluation(eval_run_repo, manifest, selected, problems_by_id, runner)
    if exit_code != 0:
        raise StageFailedError(
            f"candidate execution run {manifest.evaluation_run_id} did not complete"
        )

    return StageResult(
        stage_run_id=manifest.evaluation_run_id,
        output_artifacts={
            "evaluations": sha256_tree(eval_run_repo.run_dir(manifest.evaluation_run_id))
        },
    )


__all__ = ["run"]
