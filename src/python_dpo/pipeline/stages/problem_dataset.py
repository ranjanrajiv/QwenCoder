"""Stage 2 as a pipeline stage: the curated problem dataset (spec 12 section 5, item 1).

Named ``problem_dataset`` rather than the spec's literal ``problem_generation`` -- Stage
2's catalog is ten hand-authored problems with reference solutions, validated by
executing those solutions in-process, not an LLM generator. The spec's own example
(``problem_count: 1000``) is unimplementable against that; ``problem_count`` here can
only *select a subset* of the curated catalog, which is exactly what the smoke test needs
(spec sections 24, 25).

Mirrors ``python_dpo.cli._cmd_problems_build``: build the catalog, validate every
reference solution in-process (never routing generated/untrusted code through this path --
see CLAUDE.md's Security rule), and persist. Nothing here is generative or stochastic, so
this stage's ``seed`` has no effect and is accepted only for interface uniformity.
"""

from __future__ import annotations

from ...pipeline.hashing import sha256_file
from ...problems import (
    DatasetError,
    InProcessReferenceExecutor,
    build_catalog,
    dataset_path,
    save_problems,
    validate_dataset,
)
from ..errors import StageFailedError
from ._context import StageContext, StageResult


def run(context: StageContext) -> StageResult:
    problems = build_catalog()

    # Validate the full catalog first, against its real expected count -- this is the
    # genuine integrity check (did build_catalog() really produce every curated problem,
    # intact?). Subsetting happens only after that guarantee holds, so a reduced
    # `problem_count` can never mask a corrupted catalog.
    report = validate_dataset(problems, InProcessReferenceExecutor())
    if not report.valid:
        messages = "; ".join(report.failure_messages())
        raise StageFailedError(f"problem dataset validation failed: {messages}")

    problem_count = context.stage_config.get("problem_count")
    if problem_count is not None:
        if not isinstance(problem_count, int) or isinstance(problem_count, bool) or problem_count < 1:
            raise StageFailedError("problem_dataset.problem_count must be an integer of 1 or greater")
        problems = problems[:problem_count]

    path = dataset_path(context.project_config.paths.problems)
    try:
        save_problems(problems, path)
    except DatasetError as exc:
        raise StageFailedError(f"could not write problem dataset: {exc}") from exc

    return StageResult(
        stage_run_id=f"{context.experiment_run_id}_problem_dataset",
        output_artifacts={"problem_dataset": sha256_file(path)},
    )


__all__ = ["run"]
