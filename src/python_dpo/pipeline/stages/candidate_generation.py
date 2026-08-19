"""Stage 3/4 as a pipeline stage: candidate generation (spec 12 section 5, item 2).

``_execute_run`` and ``_write_statistics`` are moved here from ``python_dpo.cli`` --
they were already argparse-independent (only ``RunRepository``, a manifest, the selected
problems, and a model client), so this is a relocation, not a rewrite. ``cli.py``'s
``generate`` command imports them back, so the standalone CLI and the orchestrator share
exactly one implementation of "run generation to completion and settle the run's status".

Unlike the standalone ``generate`` CLI, this stage always mints a fresh candidate run per
invocation; the orchestrator's own cache (spec sections 17-19) is what decides whether to
call this adapter at all, so an in-stage resume/force flag would be redundant machinery.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from ...atomic_io import JsonlError
from ...candidates import CandidateStoreError
from ...generation import PROMPT_VERSION, CandidateGenerator, StrategyError, resolve_strategies
from ...models import PROVIDER_MOCK, MockModelClient, ModelError, ModelLoadError, QwenModelClient
from ...problems import DatasetError, Problem, dataset_path, load_problems
from ...runs import RunRepository, RunStatistics
from ..errors import StageFailedError
from ..hashing import sha256_tree
from ._context import StageContext, StageResult

logger = logging.getLogger("python_dpo.pipeline.stages.candidate_generation")


def _build_model_client(config: Any, use_mock: bool):
    """Pick the model client for this invocation (moved from ``cli.py``)."""
    if use_mock or config.model.provider == PROVIDER_MOCK:
        logger.info("Using the deterministic mock model; no weights will be loaded.")
        return MockModelClient()
    return QwenModelClient(config.model)


def _write_statistics(run_repo: RunRepository, run_id: str) -> RunStatistics:
    manifest = run_repo.get_run(run_id)
    repository = run_repo.candidates(run_id)
    stats = RunStatistics.from_records(manifest, repository.load_all(), repository.load_failures())
    run_repo.write_statistics(stats)
    return stats


def _execute_run(
    run_repo: RunRepository,
    manifest: Any,
    selected: Sequence[Problem],
    client: Any,
) -> int:
    """Run generation for an already-created manifest and settle its status."""
    manifest = run_repo.start_run(manifest.run_id)
    repository = run_repo.candidates(manifest.run_id)
    generator = CandidateGenerator(client=client, repository=repository)

    try:
        summary = generator.generate(selected, manifest)
    except KeyboardInterrupt:
        run_repo.interrupt_run(manifest.run_id)
        _write_statistics(run_repo, manifest.run_id)
        logger.warning("Run %s interrupted.", manifest.run_id)
        return 130
    except ModelLoadError as exc:
        failures = repository.load_failures()
        last = failures[-1] if failures else None
        run_repo.fail_run(
            manifest.run_id,
            error_type="model_load",
            error_message=str(exc),
            problem_id=last.problem_id if last else None,
            generation_index=last.generation_index if last else None,
        )
        _write_statistics(run_repo, manifest.run_id)
        logger.error("Aborting run %s: %s", manifest.run_id, exc)
        return 1
    except (CandidateStoreError, JsonlError, OSError) as exc:
        run_repo.fail_run(manifest.run_id, error_type="inference", error_message=str(exc))
        _write_statistics(run_repo, manifest.run_id)
        logger.error("%s", exc)
        return 1

    stats = _write_statistics(run_repo, manifest.run_id)
    complete = stats.problems_completed == stats.problems_requested

    if complete:
        run_repo.complete_run(manifest.run_id)
        final_status = "completed"
    else:
        run_repo.interrupt_run(manifest.run_id)
        final_status = "interrupted"

    logger.info(
        "Run %s %s | generated=%d skipped=%d failed=%d duplicates=%d retries=%d",
        manifest.run_id,
        final_status,
        summary.generated,
        summary.skipped,
        summary.failed,
        summary.duplicates,
        summary.retries,
    )
    return 0 if final_status == "completed" else 1


def run(context: StageContext) -> StageResult:
    config = context.project_config
    settings = context.stage_config

    try:
        problems = load_problems(dataset_path(config.paths.problems))
    except DatasetError as exc:
        raise StageFailedError(f"could not load problem dataset: {exc}") from exc

    problem_ids = settings.get("problem_ids")
    if problem_ids is not None:
        by_id = {p.id: p for p in problems}
        missing = [pid for pid in problem_ids if pid not in by_id]
        if missing:
            raise StageFailedError(f"unknown problem id(s): {', '.join(missing)}")
        selected = [by_id[pid] for pid in problem_ids]
    else:
        selected = list(problems)

    count = settings.get("candidates_per_problem", config.generation.candidates_per_problem)
    try:
        strategies = resolve_strategies(
            config.generation.strategies, count, override=settings.get("strategies")
        )
    except StrategyError as exc:
        raise StageFailedError(str(exc)) from exc

    try:
        client = _build_model_client(config, settings.get("mock_model", False))
    except ModelError as exc:
        raise StageFailedError(str(exc)) from exc

    run_repo = RunRepository(config.paths.candidates / "runs")
    manifest = run_repo.create_run(
        requested_problem_ids=[p.id for p in selected],
        requested_candidates_per_problem=count,
        strategies=strategies,
        model_config=config.model.to_dict(),
        generation_config=config.generation.config.to_dict(),
        prompt_version=PROMPT_VERSION,
        retry=config.generation.retry.to_dict(),
    )
    logger.info(
        "Run %s created | model=%s | %d problem(s) x %d candidate(s)",
        manifest.run_id,
        client.name,
        len(selected),
        count,
    )

    try:
        exit_code = _execute_run(run_repo, manifest, selected, client)
    finally:
        # Release GPU memory before the orchestrator moves on -- unlike the standalone
        # `generate` CLI command, which exits the process afterward, a pipeline run may
        # go straight into dpo_training or model_evaluation in the *same* process, and
        # those stages need the VRAM this client is still holding.
        unload = getattr(client, "unload", None)
        if unload is not None:
            unload()

    if exit_code != 0:
        raise StageFailedError(f"candidate generation run {manifest.run_id} did not complete")

    return StageResult(
        stage_run_id=manifest.run_id,
        output_artifacts={"candidates": sha256_tree(run_repo.run_dir(manifest.run_id))},
    )


__all__ = ["run"]
