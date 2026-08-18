from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .atomic_io import JsonlError, repair_truncated_tail
from .candidates import (
    CANDIDATES_FILENAME as LEGACY_CANDIDATES_FILENAME,
    CandidateError,
    CandidateStoreError,
)
from .candidates.repository import FAILURES_FILENAME
from .config import Config, ConfigError
from .evaluation import (
    EVALUATOR_VERSION,
    TEST_GENERATOR_VERSION,
    CandidateEvaluator,
    EvaluationError,
    EvaluationRunError,
    EvaluationRunNotFoundError,
    EvaluationRunRepository,
    EvaluationStatistics,
    EvaluationStoreError,
    PytestRunner,
    build_evaluation_sandbox_config,
    probe_versions,
)
from .generation import (
    PROMPT_VERSION,
    STRATEGIES,
    CandidateGenerator,
    StrategyError,
    build_prompt,
    resolve_strategies,
)
from .logging_config import configure_logging
from .models import PROVIDER_MOCK, MockModelClient, ModelError, ModelLoadError, QwenModelClient
from .problems import (
    DatasetError,
    InProcessReferenceExecutor,
    Problem,
    build_catalog,
    dataset_path,
    format_report,
    load_problems,
    save_problems,
    validate_dataset,
)
from .preferences import (
    BUILDER_VERSION,
    POLICIES,
    PREFERENCE_VERSION,
    PreferencePairBuilder,
    PreferencePolicyError,
    PreferenceRunError,
    PreferenceRunNotFoundError,
    PreferenceRunRepository,
    PreferenceStatistics,
    ProblemSplitter,
    build_quality_report,
    derive_candidates_considered,
    format_pair_detail,
    format_preference_report,
    format_preference_statistics,
    make_policy,
    validate_preference_run,
)
from .preferences import RUN_MANIFEST_FILENAME as PREFERENCE_MANIFEST_FILENAME
from .training import (
    TRAINER_VERSION,
    DatasetManifest,
    DpoTrainingJob,
    ExperimentConfig,
    FinalReport,
    TrainingError,
    TrainingRunNotFoundError,
    TrainingRunRepository,
    check_hardware,
    format_dataset_statistics,
    format_exception,
    format_final_report,
    format_hardware_report,
    format_length_analysis,
    format_parameter_counts,
    format_run_table,
    load_training_dataset,
    run_inference,
    verify_adapter,
)
from .training.config import DEFAULT_CONFIG_PATH
from .training.run_repository import MANIFEST_FILENAME as TRAINING_MANIFEST_FILENAME
from .training.run_repository import run_log_file
from .training.versions import capture_environment
from .ranking import (
    CLASSIFIER_VERSION,
    COMPARATOR_VERSION,
    RANKING_VERSION,
    SCORING_VERSION,
    CandidateComparator,
    CandidateRanker,
    CandidateScorer,
    RankingRunError,
    RankingRunNotFoundError,
    RankingRunRepository,
    RankingStatistics,
    RankingStoreError,
    format_ranking_report,
    format_ranking_statistics,
    format_ranking_table,
    validate_ranking_run,
)
from .runs import (
    MigrationError,
    RunError,
    RunNotFoundError,
    RunRepository,
    RunStatistics,
    format_run_report,
    migrate_flat_file,
    validate_run,
)
from .runs.repository import MANIFEST_FILENAME
from .sandbox import (
    DockerContainerRuntime,
    SandboxError,
    SandboxExecutor,
    check_sandbox_health,
    format_health_report,
)

logger = logging.getLogger("python_dpo.cli")

_PLACEHOLDER_STAGES = {
    "run": "Full pipeline run",
}


def _make_placeholder_handler(name: str):
    stage = _PLACEHOLDER_STAGES[name]

    def _handler(args: argparse.Namespace, config: Config) -> int:
        logger.info("%s is not implemented yet.", stage)
        return 1

    return _handler


def _make_help_handler(parser: argparse.ArgumentParser):
    def _handler(args: argparse.Namespace, config: Config) -> int:
        parser.print_help()
        return 1

    return _handler


def _cmd_problems_build(args: argparse.Namespace, config: Config) -> int:
    """Build the curated catalog, validate it, and persist it as JSONL."""
    path = dataset_path(config.paths.problems)
    problems = build_catalog()
    report = validate_dataset(problems, InProcessReferenceExecutor())

    if not report.valid:
        for message in report.failure_messages():
            logger.error("%s", message)
        logger.error("Validation failed; %s was not written.", path)
        return 1

    save_problems(problems, path)
    logger.info(
        "Wrote %d problems (%d reference tests passing) to %s",
        report.total_problems,
        report.passed_tests,
        path,
    )
    return 0


def _cmd_problems_validate(args: argparse.Namespace, config: Config) -> int:
    """Load the persisted dataset and re-run every reference test. Read-only."""
    path = dataset_path(config.paths.problems)

    try:
        problems = load_problems(path)
    except DatasetError as exc:
        logger.error("%s", exc)
        return 1

    report = validate_dataset(problems, InProcessReferenceExecutor())

    # The summary is user-facing output rather than diagnostics, so it goes to stdout
    # instead of the log stream; see the Stage 2 plan.
    sys.stdout.write(format_report(report))

    return 0 if report.valid else 1


def _add_problems_parser(subparsers: argparse._SubParsersAction) -> None:
    problems_parser = subparsers.add_parser(
        "problems",
        help="Build and validate the problem dataset.",
        description="Build and validate the curated Python problem dataset.",
    )
    problems_parser.set_defaults(func=_make_help_handler(problems_parser))

    problems_subparsers = problems_parser.add_subparsers(dest="problems_command")

    build_parser_ = problems_subparsers.add_parser(
        "build",
        help="Validate the curated catalog and write problems.jsonl.",
    )
    build_parser_.set_defaults(func=_cmd_problems_build)

    validate_parser = problems_subparsers.add_parser(
        "validate",
        help="Re-validate the persisted dataset and report the result.",
    )
    validate_parser.set_defaults(func=_cmd_problems_validate)


def _select_problems(
    problems: Sequence[Problem],
    problem_id: str | None,
    limit: int | None,
) -> list[Problem]:
    """Narrow the dataset down to the problems this invocation should generate for."""
    if problem_id is not None and limit is not None:
        raise ValueError("use --problem-id or --limit, not both")

    if problem_id is not None:
        matches = [problem for problem in problems if problem.id == problem_id]
        if not matches:
            available = ", ".join(problem.id for problem in problems)
            raise ValueError(f"unknown problem id {problem_id!r}; dataset holds: {available}")
        return matches

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        return list(problems[:limit])

    return list(problems)


def _build_model_client(config: Config, use_mock: bool):
    """Pick the model client for this invocation.

    Constructing a QwenModelClient is free — weights load on the first generate() call,
    so `--dry-run` never reaches this function at all.
    """
    if use_mock or config.model.provider == PROVIDER_MOCK:
        logger.info("Using the deterministic mock model; no weights will be loaded.")
        return MockModelClient()
    return QwenModelClient(config.model)


def _write_dry_run(
    problems: Sequence[Problem], strategies: Sequence[str], count: int
) -> None:
    """Print the prompts that would be sent, so they can be inspected before inference."""
    for problem in problems:
        for index in range(1, count + 1):
            header = (
                f"=== {problem.id} candidate {index}/{count} | "
                f"strategy={strategies[index - 1]} | prompt_version={PROMPT_VERSION} ==="
            )
            sys.stdout.write(f"{header}\n{build_prompt(problem, strategies[index - 1])}\n\n")


def _run_repository(config: Config) -> RunRepository:
    return RunRepository(config.paths.candidates / "runs")


def _write_statistics(run_repo: RunRepository, run_id: str) -> RunStatistics:
    manifest = run_repo.get_run(run_id)
    repository = run_repo.candidates(run_id)
    stats = RunStatistics.from_records(manifest, repository.load_all(), repository.load_failures())
    run_repo.write_statistics(stats)
    return stats


def _execute_run(
    run_repo: RunRepository,
    manifest,
    selected: Sequence[Problem],
    client,
) -> int:
    """Run generation for an already-created (or resumed) manifest and settle its status."""
    manifest = run_repo.start_run(manifest.run_id)
    repository = run_repo.candidates(manifest.run_id)
    generator = CandidateGenerator(client=client, repository=repository)

    try:
        summary = generator.generate(selected, manifest)
    except KeyboardInterrupt:
        run_repo.interrupt_run(manifest.run_id)
        _write_statistics(run_repo, manifest.run_id)
        logger.warning(
            "Run %s interrupted; resume with: python -m python_dpo generate --resume %s",
            manifest.run_id,
            manifest.run_id,
        )
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
    if summary.failed:
        # Failures are data, not a broken run: they are recorded and observable
        # (spec 03 section 26; spec 04 section 27).
        logger.warning(
            "%d generation(s) produced no candidate; see %s",
            summary.failed,
            repository.failures_path,
        )

    return 0 if final_status == "completed" else 1


def _cmd_generate_fresh(
    args: argparse.Namespace, config: Config, problems: Sequence[Problem], run_repo: RunRepository
) -> int:
    try:
        selected = _select_problems(problems, args.problem_id, args.limit)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    count = (
        args.num_candidates
        if args.num_candidates is not None
        else config.generation.candidates_per_problem
    )

    try:
        strategies = resolve_strategies(
            config.generation.strategies, count, override=args.strategies
        )
    except StrategyError as exc:
        logger.error("%s", exc)
        return 1

    if args.dry_run:
        # Prompts are built before any client exists, so a dry run cannot load a model
        # even accidentally.
        logger.info(
            "Dry run: %d problem(s) x %d candidate(s); no model will be loaded.",
            len(selected),
            count,
        )
        _write_dry_run(selected, strategies, count)
        return 0

    try:
        client = _build_model_client(config, args.mock_model)
    except ModelError as exc:
        logger.error("%s", exc)
        return 1

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

    return _execute_run(run_repo, manifest, selected, client)


def _cmd_generate_resume(
    args: argparse.Namespace, config: Config, problems: Sequence[Problem], run_repo: RunRepository
) -> int:
    if args.dry_run:
        logger.error("--dry-run cannot be combined with --resume")
        return 1

    conflicting = [
        flag
        for flag, value in (
            ("--problem-id", args.problem_id),
            ("--limit", args.limit),
            ("--num-candidates", args.num_candidates),
            ("--strategy", args.strategies),
        )
        if value is not None
    ]
    if conflicting:
        logger.error(
            "%s cannot be combined with --resume; the run's manifest is authoritative",
            ", ".join(conflicting),
        )
        return 1

    try:
        source_manifest = run_repo.get_run(args.resume)
    except RunNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    by_id = {p.id: p for p in problems}
    missing = [pid for pid in source_manifest.requested_problem_ids if pid not in by_id]
    if missing:
        logger.error(
            "run %s references unknown problem id(s): %s", args.resume, ", ".join(missing)
        )
        return 1
    selected = [by_id[pid] for pid in source_manifest.requested_problem_ids]

    try:
        client = _build_model_client(config, args.mock_model)
    except ModelError as exc:
        logger.error("%s", exc)
        return 1

    if args.force:
        manifest = run_repo.create_run_from(source_manifest)
        logger.info(
            "Force: seeded new run %s from %s (original run left untouched)",
            manifest.run_id,
            args.resume,
        )
    else:
        try:
            manifest = run_repo.resume_run(args.resume)
        except RunError as exc:
            logger.error("%s", exc)
            return 1
        logger.info("Resuming run %s", manifest.run_id)

    return _execute_run(run_repo, manifest, selected, client)


def _cmd_generate(args: argparse.Namespace, config: Config) -> int:
    """Generate candidate solutions for the selected problems, inside a run."""
    path = dataset_path(config.paths.problems)

    try:
        problems = load_problems(path)
    except DatasetError as exc:
        logger.error("%s", exc)
        return 1

    run_repo = _run_repository(config)

    if args.resume:
        return _cmd_generate_resume(args, config, problems, run_repo)
    return _cmd_generate_fresh(args, config, problems, run_repo)


def _add_generate_parser(subparsers: argparse._SubParsersAction) -> None:
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate candidate solutions for problems in the dataset.",
        description=(
            "Generate candidate Python implementations with the configured model and "
            "persist them to a run directory under data/candidates/runs/. Candidates are "
            "not executed or evaluated at this stage."
        ),
    )
    generate_parser.add_argument(
        "--problem-id",
        default=None,
        help="Generate for a single problem, for example p001.",
    )
    generate_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate for the first N problems in the dataset.",
    )
    generate_parser.add_argument(
        "--num-candidates",
        type=int,
        default=None,
        help="Override generation.candidates_per_problem for this run only.",
    )
    generate_parser.add_argument(
        "--strategy",
        dest="strategies",
        action="append",
        choices=STRATEGIES,
        default=None,
        help="Use this strategy instead of the configured list. Repeatable.",
    )
    generate_parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        default=None,
        help="Resume an incomplete run instead of creating a new one.",
    )
    generate_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Never resume: with --resume RUN_ID, seed a brand-new run from that run's "
            "manifest and regenerate everything into it, leaving the original untouched. "
            "Without --resume this has no effect, since generate always starts a new run."
        ),
    )
    generate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prompts that would be sent, without loading a model or writing.",
    )
    generate_parser.add_argument(
        "--mock-model",
        action="store_true",
        help="Use the deterministic mock model instead of the configured one.",
    )
    generate_parser.set_defaults(func=_cmd_generate)


# --------------------------------------------------------------------------------- runs


def _cmd_runs_list(args: argparse.Namespace, config: Config) -> int:
    run_repo = _run_repository(config)
    runs = run_repo.list_runs()
    if not runs:
        sys.stdout.write("No runs found.\n")
        return 0

    rows = []
    for manifest in runs:
        stats = run_repo.read_statistics(manifest.run_id)
        candidates = stats.candidates_generated if stats else run_repo.candidates(manifest.run_id).count()
        failures = (
            stats.generation_failures
            if stats
            else len(run_repo.candidates(manifest.run_id).load_failures())
        )
        rows.append((manifest.run_id, manifest.status, candidates, failures))

    header = f"{'RUN ID':<32}{'STATUS':<13}{'CANDIDATES':<12}{'FAILURES'}"
    lines = [header]
    lines.extend(f"{run_id:<32}{status:<13}{candidates:<12}{failures}" for run_id, status, candidates, failures in rows)
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _cmd_runs_show(args: argparse.Namespace, config: Config) -> int:
    run_repo = _run_repository(config)
    try:
        manifest = run_repo.get_run(args.run_id)
    except RunNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    stats = run_repo.read_statistics(args.run_id)

    lines = [
        "Run:",
        f"  ID: {manifest.run_id}",
        f"  Status: {manifest.status}",
        f"  Source: {manifest.source}",
        f"  Model: {manifest.model.get('name')}",
        f"  Model revision: {manifest.model.get('revision')}",
        f"  Prompt version: {manifest.prompt_version}",
        f"  Problems: {manifest.requested_problems}",
        f"  Candidates per problem: {manifest.requested_candidates_per_problem}",
        f"  Strategies: {', '.join(manifest.strategies)}",
        f"  Created at: {manifest.created_at}",
        f"  Started at: {manifest.started_at}",
        f"  Completed at: {manifest.completed_at}",
        "",
        "Generation configuration:",
    ]
    lines.extend(f"  {key}: {value}" for key, value in sorted(manifest.generation_config.items()))

    if manifest.error is not None:
        lines.extend(
            [
                "",
                "Error:",
                f"  type: {manifest.error.error_type}",
                f"  message: {manifest.error.error_message}",
                f"  at: {manifest.error.timestamp}",
            ]
        )

    if stats is not None:
        lines.extend(
            [
                "",
                "Candidates:",
                f"  Generated: {stats.candidates_generated} / {stats.candidates_requested}",
                f"  Syntax valid: {stats.syntax_valid}",
                f"  Syntax invalid: {stats.syntax_invalid}",
                f"  Duplicates: {stats.duplicates}",
                f"  Generation failures: {stats.generation_failures}",
                f"  Retry attempts: {stats.retry_attempts}",
            ]
        )

    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _cmd_runs_validate(args: argparse.Namespace, config: Config) -> int:
    run_repo = _run_repository(config)
    run_dir = run_repo.run_dir(args.run_id)
    if not (run_dir / MANIFEST_FILENAME).is_file():
        logger.error("no run %r at %s", args.run_id, run_dir)
        return 1

    if args.repair:
        for filename in (LEGACY_CANDIDATES_FILENAME, FAILURES_FILENAME):
            removed = repair_truncated_tail(run_dir / filename)
            if removed:
                logger.warning(
                    "Repaired %s: removed %d torn byte(s) from the tail", filename, removed
                )

    known_problem_ids: set[str] | None
    try:
        known_problem_ids = {p.id for p in load_problems(dataset_path(config.paths.problems))}
    except DatasetError:
        known_problem_ids = None

    report = validate_run(run_dir, known_problem_ids)
    sys.stdout.write(format_run_report(report))
    return 0 if report.valid else 1


def _add_runs_parser(subparsers: argparse._SubParsersAction) -> None:
    runs_parser = subparsers.add_parser(
        "runs",
        help="Inspect generation runs.",
        description="List, show, and validate generation runs.",
    )
    runs_parser.set_defaults(func=_make_help_handler(runs_parser))

    runs_subparsers = runs_parser.add_subparsers(dest="runs_command")

    list_parser = runs_subparsers.add_parser("list", help="List all generation runs.")
    list_parser.set_defaults(func=_cmd_runs_list)

    show_parser = runs_subparsers.add_parser("show", help="Show one run's manifest and statistics.")
    show_parser.add_argument("run_id")
    show_parser.set_defaults(func=_cmd_runs_show)

    validate_parser = runs_subparsers.add_parser("validate", help="Validate one run's integrity.")
    validate_parser.add_argument("run_id")
    validate_parser.add_argument(
        "--repair",
        action="store_true",
        help="Truncate a torn JSONL tail before validating (never touches a mid-file error).",
    )
    validate_parser.set_defaults(func=_cmd_runs_validate)


# --------------------------------------------------------------------------- candidates


def _cmd_candidates_list(args: argparse.Namespace, config: Config) -> int:
    run_repo = _run_repository(config)
    run_dir = run_repo.run_dir(args.run_id)
    if not (run_dir / MANIFEST_FILENAME).is_file():
        logger.error("no run %r at %s", args.run_id, run_dir)
        return 1

    try:
        candidates = run_repo.candidates(args.run_id).load_all()
    except CandidateStoreError as exc:
        logger.error("%s", exc)
        return 1

    if args.problem_id:
        candidates = [c for c in candidates if c.problem_id == args.problem_id]
    if args.strategy:
        candidates = [c for c in candidates if c.strategy == args.strategy]

    if not candidates:
        sys.stdout.write("No candidates found.\n")
        return 0

    header = f"{'CANDIDATE_ID':<16}{'PROBLEM_ID':<13}{'STRATEGY':<20}{'SYNTAX'}"
    lines = [header]
    lines.extend(
        f"{c.candidate_id:<16}{c.problem_id:<13}{c.strategy:<20}{'valid' if c.syntax_valid else 'invalid'}"
        for c in candidates
    )
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _cmd_candidates_show(args: argparse.Namespace, config: Config) -> int:
    run_repo = _run_repository(config)
    run_dir = run_repo.run_dir(args.run_id)
    if not (run_dir / MANIFEST_FILENAME).is_file():
        logger.error("no run %r at %s", args.run_id, run_dir)
        return 1

    try:
        candidate = run_repo.candidates(args.run_id).get(args.candidate_id)
    except CandidateStoreError as exc:
        logger.error("%s", exc)
        return 1

    if candidate is None:
        logger.error("no candidate %r in run %r", args.candidate_id, args.run_id)
        return 1

    lines = [
        f"Candidate: {candidate.candidate_id}",
        f"  Problem: {candidate.problem_id}",
        f"  Strategy: {candidate.strategy}",
        f"  Model: {candidate.model}",
        f"  Model revision: {candidate.model_revision}",
        f"  Prompt version: {candidate.prompt_version}",
        f"  Syntax valid: {candidate.syntax_valid}",
        f"  Function name valid: {candidate.function_name_valid}",
        f"  Code hash: {candidate.code_sha256}",
        f"  Duplicate of: {candidate.duplicate_of}",
        f"  Attempt: {candidate.attempt}",
        f"  Created at: {candidate.created_at}",
    ]
    # Raw model output and extracted code are withheld by default (spec 04 section 40).
    if args.show_code:
        lines.extend(["", "Code:", candidate.code])
    if args.show_raw:
        lines.extend(["", "Raw output:", candidate.raw_output])

    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _cmd_candidates_stats(args: argparse.Namespace, config: Config) -> int:
    run_repo = _run_repository(config)
    try:
        manifest = run_repo.get_run(args.run_id)
    except RunNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    repository = run_repo.candidates(args.run_id)
    try:
        stats = RunStatistics.from_records(manifest, repository.load_all(), repository.load_failures())
    except CandidateStoreError as exc:
        logger.error("%s", exc)
        return 1

    lines = [
        f"Problems: {stats.problems_requested} requested, {stats.problems_completed} completed",
        f"Candidates requested: {stats.candidates_requested}",
        f"Candidates generated: {stats.candidates_generated}",
        f"Generation failures: {stats.generation_failures}",
        f"Syntax valid: {stats.syntax_valid}",
        f"Syntax invalid: {stats.syntax_invalid}",
        f"Duplicates: {stats.duplicates}",
        "",
        "By strategy:",
    ]
    lines.extend(f"  {strategy:<20}{count}" for strategy, count in sorted(stats.candidates_by_strategy.items()))

    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _cmd_candidates_migrate(args: argparse.Namespace, config: Config) -> int:
    source = Path(args.source) if args.source else (config.paths.candidates / LEGACY_CANDIDATES_FILENAME)
    if not source.is_file():
        logger.error("no legacy candidates file at %s", source)
        return 1

    run_repo = _run_repository(config)
    try:
        manifests = migrate_flat_file(source, run_repo, force=args.force)
    except (MigrationError, CandidateError, JsonlError) as exc:
        logger.error("%s", exc)
        return 1

    for manifest in manifests:
        logger.info(
            "Migrated run %s (%d problem(s), source: migrated_from_flat_file)",
            manifest.run_id,
            manifest.requested_problems,
        )
    logger.info("Migrated %d run(s) from %s; %s left unchanged", len(manifests), source, source)
    return 0


def _add_candidates_parser(subparsers: argparse._SubParsersAction) -> None:
    candidates_parser = subparsers.add_parser(
        "candidates",
        help="Inspect and migrate generated candidates.",
        description="List, show, summarize, and migrate generated candidates.",
    )
    candidates_parser.set_defaults(func=_make_help_handler(candidates_parser))

    candidates_subparsers = candidates_parser.add_subparsers(dest="candidates_command")

    list_parser = candidates_subparsers.add_parser("list", help="List candidates in a run.")
    list_parser.add_argument("run_id")
    list_parser.add_argument("--problem-id", default=None)
    list_parser.add_argument("--strategy", default=None, choices=STRATEGIES)
    list_parser.set_defaults(func=_cmd_candidates_list)

    show_parser = candidates_subparsers.add_parser("show", help="Show one candidate's metadata.")
    show_parser.add_argument("run_id")
    show_parser.add_argument("candidate_id")
    show_parser.add_argument(
        "--show-code", action="store_true", help="Also print the extracted code."
    )
    show_parser.add_argument(
        "--show-raw", action="store_true", help="Also print the raw model output."
    )
    show_parser.set_defaults(func=_cmd_candidates_show)

    stats_parser = candidates_subparsers.add_parser("stats", help="Show run statistics.")
    stats_parser.add_argument("run_id")
    stats_parser.set_defaults(func=_cmd_candidates_stats)

    migrate_parser = candidates_subparsers.add_parser(
        "migrate", help="Migrate the legacy flat candidates.jsonl into run directories."
    )
    migrate_parser.add_argument(
        "--source",
        default=None,
        help="Path to the legacy candidates.jsonl (default: data/candidates/candidates.jsonl).",
    )
    migrate_parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing run directory."
    )
    migrate_parser.set_defaults(func=_cmd_candidates_migrate)


# -------------------------------------------------------------------------------- evaluate


def _evaluation_run_repository(config: Config) -> EvaluationRunRepository:
    return EvaluationRunRepository(config.paths.evaluations / "runs")


def _ensure_evaluation_image(evaluation_config, runtime=None) -> str | None:
    """Verify the evaluation image is present before starting, so a missing image fails
    once with a clear message instead of producing N identical infrastructure errors.
    Returns an error message, or ``None`` if the image is ready to use.
    """
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


def _select_candidates_for_evaluation(candidates, problem_id, limit):
    """Narrow a generation run's candidates down to what this invocation should evaluate."""
    selected = list(candidates)
    if problem_id is not None:
        selected = [c for c in selected if c.problem_id == problem_id]
        if not selected:
            raise ValueError(f"no candidates for problem id {problem_id!r}")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
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
    manifest,
    candidates,
    problems_by_id: dict,
    runner: PytestRunner,
) -> int:
    """Run evaluation for an already-created (or resumed) manifest and settle its status."""
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
        logger.warning(
            "Evaluation run %s interrupted; resume with: "
            "python -m python_dpo evaluate run --run-id %s",
            manifest.evaluation_run_id,
            manifest.candidate_run_id,
        )
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
        "Evaluation run %s %s | evaluated=%d skipped=%d machinery_failed=%d | "
        "passed=%d failed=%d",
        manifest.evaluation_run_id,
        final_status,
        summary.evaluated,
        summary.skipped,
        summary.machinery_failed,
        stats.passed,
        stats.failed,
    )
    return 0 if final_status == "completed" else 1


def _run_evaluation(
    args: argparse.Namespace,
    config: Config,
    *,
    problem_id: str | None,
    candidate_id: str | None,
    limit: int | None,
) -> int:
    """Shared core of ``evaluate candidate`` and ``evaluate run``: resolve the target
    evaluation run (resuming by default, spec sections 56-57 -- the opposite default from
    ``generate``, since evaluation is idempotent and cheap to re-run where generation
    burns GPU time), then evaluate.
    """
    run_repo = _run_repository(config)
    run_dir = run_repo.run_dir(args.run_id)
    if not (run_dir / MANIFEST_FILENAME).is_file():
        logger.error("no run %r at %s", args.run_id, run_dir)
        return 1

    try:
        problems = load_problems(dataset_path(config.paths.problems))
    except DatasetError as exc:
        logger.error("%s", exc)
        return 1
    problems_by_id = {p.id: p for p in problems}

    try:
        all_candidates = run_repo.candidates(args.run_id).load_all()
    except CandidateStoreError as exc:
        logger.error("%s", exc)
        return 1

    if candidate_id is not None:
        selected = [c for c in all_candidates if c.candidate_id == candidate_id]
        if not selected:
            logger.error("no candidate %r in run %r", candidate_id, args.run_id)
            return 1
    else:
        try:
            selected = _select_candidates_for_evaluation(all_candidates, problem_id, limit)
        except ValueError as exc:
            logger.error("%s", exc)
            return 1

    missing_problems = sorted({c.problem_id for c in selected} - set(problems_by_id))
    if missing_problems:
        logger.error(
            "candidate(s) reference unknown problem id(s): %s", ", ".join(missing_problems)
        )
        return 1

    eval_run_repo = _evaluation_run_repository(config)
    selected_ids = [c.candidate_id for c in selected]

    manifest = None
    candidate_ids_to_run: set[str] = set(selected_ids)
    if not args.force:
        existing = eval_run_repo.latest_run_for_candidate_run(args.run_id)
        if existing is not None:
            requested = set(existing.requested_candidate_ids)
            if not set(selected_ids).issubset(requested):
                logger.error(
                    "evaluation run %r already covers %r with a different candidate "
                    "selection; use --force to start a new evaluation run",
                    existing.evaluation_run_id,
                    args.run_id,
                )
                return 1
            if existing.status == "completed":
                logger.info(
                    "Evaluation run %s already completed; nothing to do.",
                    existing.evaluation_run_id,
                )
                return 0
            manifest = existing
            candidate_ids_to_run = requested

    image_error = _ensure_evaluation_image(config.evaluation)
    if image_error:
        logger.error("%s", image_error)
        return 1

    eval_sandbox_config = build_evaluation_sandbox_config(config.sandbox, config.evaluation)

    if manifest is None:
        try:
            python_version, pytest_version = probe_versions(eval_sandbox_config)
        except EvaluationError as exc:
            logger.error("%s", exc)
            return 1
        manifest = eval_run_repo.create_run(
            candidate_run_id=args.run_id,
            evaluator_version=EVALUATOR_VERSION,
            test_generator_version=TEST_GENERATOR_VERSION,
            pytest_version=pytest_version,
            python_version=python_version,
            sandbox_config=eval_sandbox_config.to_dict(),
            requested_candidate_ids=selected_ids,
            requested_problem_id=problem_id,
        )
        logger.info(
            "Evaluation run %s created | candidate run %s | %d candidate(s)",
            manifest.evaluation_run_id,
            args.run_id,
            len(selected_ids),
        )
    else:
        logger.info("Resuming evaluation run %s", manifest.evaluation_run_id)

    candidates_to_run = [c for c in all_candidates if c.candidate_id in candidate_ids_to_run]
    runner = PytestRunner(SandboxExecutor(config=eval_sandbox_config))

    return _execute_evaluation(eval_run_repo, manifest, candidates_to_run, problems_by_id, runner)


def _cmd_evaluate_candidate(args: argparse.Namespace, config: Config) -> int:
    """Evaluate one candidate (spec section 53)."""
    return _run_evaluation(args, config, problem_id=None, candidate_id=args.candidate_id, limit=None)


def _cmd_evaluate_run(args: argparse.Namespace, config: Config) -> int:
    """Evaluate a generation run's candidates, resuming by default (spec sections 53-57)."""
    return _run_evaluation(args, config, problem_id=args.problem_id, candidate_id=None, limit=args.limit)


def _add_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate generated candidates against their problems' declared tests.",
        description=(
            "Run a candidate's persisted code against its problem's test suite inside the "
            "isolated Docker sandbox and record objective execution evidence (pass/fail "
            "counts, error types, durations). Never computes chosen/rejected, a reward, "
            "or a ranking; that belongs to a later stage."
        ),
    )
    evaluate_parser.set_defaults(func=_make_help_handler(evaluate_parser))

    evaluate_subparsers = evaluate_parser.add_subparsers(dest="evaluate_command")

    candidate_parser = evaluate_subparsers.add_parser("candidate", help="Evaluate one candidate.")
    candidate_parser.add_argument("--run-id", required=True, help="Generation run id.")
    candidate_parser.add_argument("--candidate-id", required=True)
    candidate_parser.add_argument(
        "--force",
        action="store_true",
        help="Never resume: start a new evaluation run even if one already covers this candidate.",
    )
    candidate_parser.set_defaults(func=_cmd_evaluate_candidate)

    run_parser = evaluate_subparsers.add_parser(
        "run", help="Evaluate a generation run's candidates, resuming by default."
    )
    run_parser.add_argument("--run-id", required=True, help="Generation run id.")
    run_parser.add_argument(
        "--problem-id", default=None, help="Evaluate only candidates for this problem."
    )
    run_parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate only the first N candidates."
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Never resume: start a new evaluation run even if one already exists for this generation run.",
    )
    run_parser.set_defaults(func=_cmd_evaluate_run)


# ------------------------------------------------------------------------------ evaluations


def _cmd_evaluations_list(args: argparse.Namespace, config: Config) -> int:
    eval_run_repo = _evaluation_run_repository(config)

    if args.eval_id is None:
        # No eval_id: list evaluation runs themselves, mirroring `runs list`.
        runs = eval_run_repo.list_runs()
        if not runs:
            sys.stdout.write("No evaluation runs found.\n")
            return 0
        header = f"{'EVAL ID':<32}{'STATUS':<13}{'CANDIDATE RUN'}"
        lines = [header]
        lines.extend(
            f"{m.evaluation_run_id:<32}{m.status:<13}{m.candidate_run_id}" for m in runs
        )
        sys.stdout.write("\n".join(lines) + "\n")
        return 0

    try:
        eval_run_repo.get_run(args.eval_id)
    except EvaluationRunNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    results = eval_run_repo.results(args.eval_id).load_all()
    if not results:
        sys.stdout.write("No evaluation results found.\n")
        return 0

    header = f"{'CANDIDATE_ID':<16}{'PROBLEM_ID':<13}{'STATUS':<14}{'TESTS'}"
    lines = [header]
    lines.extend(
        f"{r.candidate_id:<16}{r.problem_id:<13}{r.status:<14}{r.tests_passed}/{r.tests_total}"
        for r in results
    )
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _cmd_evaluations_show(args: argparse.Namespace, config: Config) -> int:
    eval_run_repo = _evaluation_run_repository(config)
    try:
        eval_run_repo.get_run(args.eval_id)
    except EvaluationRunNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    repository = eval_run_repo.results(args.eval_id)
    result = repository.get(args.candidate_id)
    if result is None:
        failure = next(
            (f for f in repository.load_failures() if f.candidate_id == args.candidate_id),
            None,
        )
        if failure is None:
            logger.error(
                "no evaluation result for candidate %r in %r", args.candidate_id, args.eval_id
            )
            return 1
        lines = [
            f"Candidate: {failure.candidate_id}",
            f"  Problem: {failure.problem_id}",
            "  No job was attempted (machinery failure):",
            f"  Error type: {failure.error_type}",
            f"  Error message: {failure.error_message}",
            f"  At: {failure.timestamp}",
        ]
        sys.stdout.write("\n".join(lines) + "\n")
        return 0

    lines = [
        f"Candidate: {result.candidate_id}",
        f"  Problem: {result.problem_id}",
        f"  Status: {result.status}",
        f"  Tests: {result.tests_passed}/{result.tests_total} passed "
        f"({result.tests_failed} failed, {result.tests_error} error, {result.tests_skipped} skipped)",
        f"  Pass rate: {result.pass_rate}",
        f"  Duration: {result.duration_ms} ms",
        f"  Exit code: {result.exit_code}",
        f"  Evaluated at: {result.evaluation_timestamp}",
    ]
    if result.metadata_discrepancy:
        lines.append(f"  Discrepancy: {result.discrepancy_reason}")

    failing = [t for t in repository.test_results_for(args.candidate_id) if t.status != "passed"]
    if failing:
        lines += ["", "Failing tests:"]
        lines.extend(
            f"  {t.test_case_id}: {t.status} ({t.error_type}) {t.error_message}" for t in failing
        )

    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _cmd_evaluations_stats(args: argparse.Namespace, config: Config) -> int:
    eval_run_repo = _evaluation_run_repository(config)
    try:
        manifest = eval_run_repo.get_run(args.eval_id)
    except EvaluationRunNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    repository = eval_run_repo.results(args.eval_id)
    stats = EvaluationStatistics.from_records(
        manifest, repository.load_all(), repository.load_failures()
    )

    lines = [
        f"Candidates requested: {stats.candidates_requested}",
        f"Candidates evaluated: {stats.candidates_evaluated}",
        f"Evaluation failures: {stats.evaluation_failures}",
        f"Passed: {stats.passed}",
        f"Failed: {stats.failed}",
        f"Timeouts: {stats.timeouts}",
        f"Syntax errors: {stats.syntax_errors}",
        f"Infrastructure errors: {stats.infrastructure_errors}",
        "",
        f"Tests: {stats.tests_passed}/{stats.tests_total} passed "
        f"({stats.tests_failed} failed, {stats.tests_errors} error, {stats.tests_skipped} skipped)",
    ]
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _add_evaluations_parser(subparsers: argparse._SubParsersAction) -> None:
    evaluations_parser = subparsers.add_parser(
        "evaluations",
        help="Inspect evaluation runs.",
        description="List, show, and summarize evaluation results.",
    )
    evaluations_parser.set_defaults(func=_make_help_handler(evaluations_parser))

    evaluations_subparsers = evaluations_parser.add_subparsers(dest="evaluations_command")

    list_parser = evaluations_subparsers.add_parser(
        "list", help="List evaluation runs, or results within one."
    )
    list_parser.add_argument(
        "eval_id", nargs="?", default=None, help="Omit to list all evaluation runs."
    )
    list_parser.set_defaults(func=_cmd_evaluations_list)

    show_parser = evaluations_subparsers.add_parser(
        "show", help="Show one candidate's evaluation result."
    )
    show_parser.add_argument("eval_id")
    show_parser.add_argument("candidate_id")
    show_parser.set_defaults(func=_cmd_evaluations_show)

    stats_parser = evaluations_subparsers.add_parser(
        "stats", help="Show evaluation run statistics."
    )
    stats_parser.add_argument("eval_id")
    stats_parser.set_defaults(func=_cmd_evaluations_stats)


# ---------------------------------------------------------------------------------- rank


def _ranking_run_repository(config: Config) -> RankingRunRepository:
    return RankingRunRepository(config.paths.rankings / "runs")


def _rank_selected_problem_ids(
    results, failures, problem_id: str | None, limit: int | None
) -> list[str]:
    """Narrow an evaluation run's covered problems down to what this invocation should
    rank. Only problems with at least one result or failure are ever selectable — a
    problem the evaluation run never touched cannot be ranked (spec section 70).
    """
    all_ids = sorted({r.problem_id for r in results} | {f.problem_id for f in failures})
    if problem_id is not None:
        if problem_id not in all_ids:
            raise ValueError(f"no evaluated candidates for problem id {problem_id!r}")
        return [problem_id]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        return all_ids[:limit]
    return all_ids


def _rank_problem_group(
    ranking_run_id: str,
    problem_id: str,
    results,
    failures,
    candidates_by_id: dict,
    scorer: CandidateScorer,
    ranker: CandidateRanker,
    comparator: CandidateComparator,
    repository,
) -> int:
    """Score, rank, and compare exactly one problem's candidates, persisting as it goes.
    Returns the number of candidates assessed.
    """
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


def _cmd_rank_run(args: argparse.Namespace, config: Config) -> int:
    """Rank an evaluation run's candidates (spec sections 47, 48, 53-55).

    Unlike ``evaluate run``'s resume-by-default, this follows the spec literally: a bare
    invocation always creates a **new** ranking run; ``--resume RANKING_RUN_ID`` is the
    only way to continue one. ``--force`` mints a new ranking run and never modifies an
    existing one (spec section 55) — like ``generate --force``, it has no effect unless
    combined with ``--resume``, since a bare invocation already always creates a new run.
    """
    eval_run_repo = _evaluation_run_repository(config)
    try:
        evaluation_manifest = eval_run_repo.get_run(args.evaluation_run_id)
    except EvaluationRunNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    eval_results_repo = eval_run_repo.results(args.evaluation_run_id)
    try:
        results = eval_results_repo.load_all()
        failures = eval_results_repo.load_failures()
    except EvaluationStoreError as exc:
        logger.error("%s", exc)
        return 1

    if not results and not failures:
        logger.error("evaluation run %r has no results to rank", args.evaluation_run_id)
        return 1

    try:
        candidate_repo = _run_repository(config).candidates(evaluation_manifest.candidate_run_id)
        candidates_by_id = {c.candidate_id: c for c in candidate_repo.load_all()}
    except CandidateStoreError as exc:
        logger.error("%s", exc)
        return 1

    ranking_run_repo = _ranking_run_repository(config)

    if args.resume:
        if args.problem_id is not None or args.limit is not None:
            logger.error(
                "--problem-id and --limit cannot be combined with --resume; the ranking "
                "run's manifest is authoritative"
            )
            return 1
        if args.force:
            logger.info(
                "--force overrides --resume %s; creating a new ranking run instead", args.resume
            )
            manifest = None
        else:
            try:
                manifest = ranking_run_repo.get_run(args.resume)
            except RankingRunNotFoundError as exc:
                logger.error("%s", exc)
                return 1
            if manifest.evaluation_run_id != args.evaluation_run_id:
                logger.error(
                    "ranking run %r covers evaluation run %r, not %r",
                    args.resume,
                    manifest.evaluation_run_id,
                    args.evaluation_run_id,
                )
                return 1
            try:
                manifest = ranking_run_repo.resume_run(args.resume)
            except RankingRunError as exc:
                logger.error("%s", exc)
                return 1
            logger.info("Resuming ranking run %s", manifest.ranking_run_id)
    else:
        manifest = None

    if manifest is None:
        try:
            selected_problem_ids = _rank_selected_problem_ids(
                results, failures, args.problem_id, args.limit
            )
        except ValueError as exc:
            logger.error("%s", exc)
            return 1
        manifest = ranking_run_repo.create_run(
            evaluation_run_id=args.evaluation_run_id,
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
            args.evaluation_run_id,
            len(selected_problem_ids),
        )

    repository = ranking_run_repo.results(manifest.ranking_run_id)
    already_ranked = repository.ranked_problem_ids()
    remaining = [pid for pid in manifest.requested_problem_ids if pid not in already_ranked]
    for pid in manifest.requested_problem_ids:
        if pid in already_ranked:
            logger.info("Skipping %s (already ranked)", pid)

    scorer = CandidateScorer()
    ranker = CandidateRanker()
    comparator = CandidateComparator()

    try:
        for problem_id in remaining:
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
    except KeyboardInterrupt:
        ranking_run_repo.interrupt_run(manifest.ranking_run_id)
        _write_ranking_statistics(ranking_run_repo, manifest.ranking_run_id)
        logger.warning(
            "Ranking run %s interrupted; resume with: python -m python_dpo rank run "
            "--evaluation-run-id %s --resume %s",
            manifest.ranking_run_id,
            args.evaluation_run_id,
            manifest.ranking_run_id,
        )
        return 130

    stats = _write_ranking_statistics(ranking_run_repo, manifest.ranking_run_id)
    complete = set(manifest.requested_problem_ids) <= repository.ranked_problem_ids()

    if complete:
        ranking_run_repo.complete_run(manifest.ranking_run_id)
        final_status = "completed"
    else:
        ranking_run_repo.interrupt_run(manifest.ranking_run_id)
        final_status = "interrupted"

    logger.info(
        "Ranking run %s %s | %d problem(s) ranked this call | candidates=%d correct=%d "
        "incorrect=%d indeterminate=%d",
        manifest.ranking_run_id,
        final_status,
        len(remaining),
        stats.candidates,
        stats.correct,
        stats.incorrect,
        stats.indeterminate,
    )
    return 0 if final_status == "completed" else 1


def _add_rank_parser(subparsers: argparse._SubParsersAction) -> None:
    rank_parser = subparsers.add_parser(
        "rank",
        help="Classify, score, and rank candidates from an evaluation run.",
        description=(
            "Convert Stage 6's objective execution evidence into a correctness "
            "classification, a score, and a deterministic per-problem ranking with "
            "explicit tie groups. Never computes chosen/rejected labels, a DPO pair, or "
            "calls an LLM judge; that belongs to a later stage."
        ),
    )
    rank_parser.set_defaults(func=_make_help_handler(rank_parser))

    rank_subparsers = rank_parser.add_subparsers(dest="rank_command")

    run_parser = rank_subparsers.add_parser(
        "run", help="Rank an evaluation run's candidates."
    )
    run_parser.add_argument("--evaluation-run-id", required=True, dest="evaluation_run_id")
    run_parser.add_argument(
        "--problem-id", default=None, help="Rank only candidates for this problem."
    )
    run_parser.add_argument(
        "--limit", type=int, default=None, help="Rank only the first N evaluated problems."
    )
    run_parser.add_argument(
        "--resume",
        metavar="RANKING_RUN_ID",
        default=None,
        help="Continue an existing ranking run instead of creating a new one.",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="With --resume, abandon it and create a new ranking run instead.",
    )
    run_parser.set_defaults(func=_cmd_rank_run)


# ----------------------------------------------------------------------------- rankings


def _cmd_rankings_list(args: argparse.Namespace, config: Config) -> int:
    ranking_run_repo = _ranking_run_repository(config)
    try:
        ranking_run_repo.get_run(args.ranking_run_id)
    except RankingRunNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    rankings = ranking_run_repo.results(args.ranking_run_id).list_all_rankings()
    if not rankings:
        sys.stdout.write("No rankings found.\n")
        return 0

    by_problem: dict[str, list] = {}
    for ranking in rankings:
        by_problem.setdefault(ranking.problem_id, []).append(ranking)

    header = f"{'PROBLEM':<10}{'CANDIDATES':<12}{'CORRECT':<9}{'INCORRECT':<11}{'INDET.':<8}{'TIE GROUPS'}"
    lines = [header]
    for problem_id in sorted(by_problem):
        group = by_problem[problem_id]
        correct = sum(1 for r in group if r.correctness == "correct")
        incorrect = sum(1 for r in group if r.correctness == "incorrect")
        indeterminate = sum(1 for r in group if r.correctness == "indeterminate")
        tie_groups = len({r.tie_group for r in group if r.tie_group is not None})
        lines.append(
            f"{problem_id:<10}{len(group):<12}{correct:<9}{incorrect:<11}{indeterminate:<8}{tie_groups}"
        )
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _cmd_rankings_show(args: argparse.Namespace, config: Config) -> int:
    ranking_run_repo = _ranking_run_repository(config)
    try:
        ranking_run_repo.get_run(args.ranking_run_id)
    except RankingRunNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    repository = ranking_run_repo.results(args.ranking_run_id)
    rankings = repository.list_problem_rankings(args.problem_id)
    if not rankings:
        logger.error(
            "no rankings for problem %r in %r", args.problem_id, args.ranking_run_id
        )
        return 1

    assessments_by_id = {
        a.candidate_id: a
        for a in repository.load_assessments()
        if a.problem_id == args.problem_id
    }
    rows = [
        (ranking, assessments_by_id[ranking.candidate_id])
        for ranking in rankings
        if ranking.candidate_id in assessments_by_id
    ]
    sys.stdout.write(format_ranking_table(rows))
    return 0


def _cmd_rankings_stats(args: argparse.Namespace, config: Config) -> int:
    ranking_run_repo = _ranking_run_repository(config)
    try:
        manifest = ranking_run_repo.get_run(args.ranking_run_id)
    except RankingRunNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    repository = ranking_run_repo.results(args.ranking_run_id)
    stats = RankingStatistics.from_records(
        manifest, repository.load_assessments(), repository.load_rankings()
    )
    sys.stdout.write(format_ranking_statistics(stats))
    return 0


def _cmd_rankings_validate(args: argparse.Namespace, config: Config) -> int:
    ranking_run_repo = _ranking_run_repository(config)
    run_dir = ranking_run_repo.run_dir(args.ranking_run_id)
    if not (run_dir / MANIFEST_FILENAME).is_file():
        logger.error("no ranking run %r at %s", args.ranking_run_id, run_dir)
        return 1

    evaluation_run_dir = None
    try:
        manifest = ranking_run_repo.get_run(args.ranking_run_id)
        evaluation_run_dir = _evaluation_run_repository(config).run_dir(manifest.evaluation_run_id)
    except RankingRunError:
        pass  # validate_ranking_run below will surface the manifest problem itself

    try:
        known_problem_ids = {p.id for p in load_problems(dataset_path(config.paths.problems))}
    except DatasetError:
        known_problem_ids = None

    report = validate_ranking_run(run_dir, evaluation_run_dir, known_problem_ids)
    sys.stdout.write(format_ranking_report(report))
    return 0 if report.valid else 1


def _add_rankings_parser(subparsers: argparse._SubParsersAction) -> None:
    rankings_parser = subparsers.add_parser(
        "rankings",
        help="Inspect ranking runs.",
        description="List, show, validate, and summarize ranking results.",
    )
    rankings_parser.set_defaults(func=_make_help_handler(rankings_parser))

    rankings_subparsers = rankings_parser.add_subparsers(dest="rankings_command")

    list_parser = rankings_subparsers.add_parser(
        "list", help="Per-problem summary of a ranking run."
    )
    list_parser.add_argument("ranking_run_id")
    list_parser.set_defaults(func=_cmd_rankings_list)

    show_parser = rankings_subparsers.add_parser(
        "show", help="Show the rank table for one problem."
    )
    show_parser.add_argument("ranking_run_id")
    show_parser.add_argument("problem_id")
    show_parser.set_defaults(func=_cmd_rankings_show)

    stats_parser = rankings_subparsers.add_parser(
        "stats", help="Show ranking run statistics."
    )
    stats_parser.add_argument("ranking_run_id")
    stats_parser.set_defaults(func=_cmd_rankings_stats)

    validate_parser = rankings_subparsers.add_parser(
        "validate", help="Validate one ranking run's integrity."
    )
    validate_parser.add_argument("ranking_run_id")
    validate_parser.set_defaults(func=_cmd_rankings_validate)


# ------------------------------------------------------------------------- preferences


def _parse_split_ratios(raw: str) -> dict[str, float]:
    """Parse ``--split-ratios TRAIN,VALIDATION,TEST`` into the splitter's mapping.

    Exists so the split can be varied per invocation like every other preference
    parameter. At small problem counts the ratios decide whether a validation split
    exists at all — ``floor(n x 0.1)`` is zero below ten pair-bearing problems — so
    requiring a config edit to change them would make the difference between a trainable
    and an untrainable dataset a file edit rather than a flag.
    """
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"--split-ratios must be three comma-separated numbers "
            f"(TRAIN,VALIDATION,TEST), got {raw!r}"
        )
    try:
        values = [float(part) for part in parts]
    except ValueError:
        raise ValueError(f"--split-ratios must be numbers, got {raw!r}") from None
    if any(value < 0 for value in values):
        raise ValueError("--split-ratios values must not be negative")
    total = sum(values)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"--split-ratios must sum to 1.0, got {total}")
    return {"train": values[0], "validation": values[1], "test": values[2]}


def _preference_run_repository(config: Config) -> PreferenceRunRepository:
    return PreferenceRunRepository(config.paths.preferences / "runs")


def _finalize_preference_run(
    preference_run_repo: PreferenceRunRepository,
    preference_run_id: str,
    assessments_by_problem: dict,
) -> PreferenceStatistics:
    """(Re)write every derived artifact — ``statistics.json``, ``split_manifest.json``,
    ``quality_report.json``, and the training JSONL files — from what is durably
    persisted in ``metadata.jsonl``/``rejections.jsonl`` so far (spec sections 54, 64, 75).
    Called both on normal completion and on interruption, mirroring
    ``_write_ranking_statistics``'s re-fetch-by-id pattern rather than trusting a
    possibly-stale in-memory manifest.
    """
    manifest = preference_run_repo.get_run(preference_run_id)
    repository = preference_run_repo.results(preference_run_id)
    pairs = repository.load_pairs()
    rejections = repository.load_rejections()

    stats = PreferenceStatistics.from_records(
        manifest,
        pairs,
        rejections,
        candidates_considered=derive_candidates_considered(pairs, rejections),
    )
    preference_run_repo.write_statistics(stats)

    training_problem_ids = {p.problem_id for p in pairs if not p.duplicate_training_record}
    split_manifest = ProblemSplitter(
        ratios=manifest.split_ratios, seed=manifest.split_seed
    ).split(training_problem_ids)
    preference_run_repo.write_split_manifest(preference_run_id, split_manifest)
    repository.write_dataset(split_manifest)

    problem_correctness = {
        problem_id: [a.correctness for a in assessments]
        for problem_id, assessments in assessments_by_problem.items()
    }
    quality_report = build_quality_report(manifest, pairs, problem_correctness)
    preference_run_repo.write_quality_report(quality_report)

    return stats


def _cmd_preferences_generate(args: argparse.Namespace, config: Config) -> int:
    """Build preference pairs from a ranking run (spec sections 78, 83, 84).

    Follows ``rank run``'s shape, not ``evaluate run``'s resume-by-default: a bare
    invocation always creates a **new** preference run; ``--resume PREFERENCE_RUN_ID`` is
    the only way to continue one, and ``--force`` mints a new run rather than modifying an
    existing one (spec section 84) — every policy/margin combination the same ranking run
    produces coexists (spec section 48).
    """
    ranking_run_repo = _ranking_run_repository(config)
    try:
        ranking_manifest = ranking_run_repo.get_run(args.ranking_run_id)
    except RankingRunNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    try:
        assessments = ranking_run_repo.results(args.ranking_run_id).load_assessments()
    except RankingStoreError as exc:
        logger.error("%s", exc)
        return 1
    if not assessments:
        logger.error(
            "ranking run %r has no assessments to build preferences from", args.ranking_run_id
        )
        return 1

    try:
        problems = {p.id: p for p in load_problems(dataset_path(config.paths.problems))}
    except DatasetError as exc:
        logger.error("%s", exc)
        return 1

    try:
        candidate_repo = _run_repository(config).candidates(ranking_manifest.candidate_run_id)
        candidates_by_id = {c.candidate_id: c for c in candidate_repo.load_all()}
    except CandidateStoreError as exc:
        logger.error("%s", exc)
        return 1

    assessments_by_problem: dict[str, list] = {}
    for assessment in assessments:
        assessments_by_problem.setdefault(assessment.problem_id, []).append(assessment)

    policy_name = args.policy or config.preferences.policy
    try:
        policy = make_policy(policy_name)
    except PreferencePolicyError as exc:
        logger.error("%s", exc)
        return 1

    minimum_score_margin = (
        args.margin if args.margin is not None else config.preferences.minimum_score_margin
    )
    max_pairs_per_problem = (
        args.max_pairs_per_problem
        if args.max_pairs_per_problem is not None
        else config.preferences.max_pairs_per_problem
    )
    split_seed = args.split_seed if args.split_seed is not None else config.preferences.split.seed
    if args.split_ratios is not None:
        try:
            split_ratios = _parse_split_ratios(args.split_ratios)
        except ValueError as exc:
            logger.error("%s", exc)
            return 1
    else:
        split_ratios = config.preferences.split.as_ratios()

    preference_run_repo = _preference_run_repository(config)

    if args.resume:
        if args.force:
            logger.info(
                "--force overrides --resume %s; creating a new preference run instead",
                args.resume,
            )
            manifest = None
        else:
            try:
                manifest = preference_run_repo.get_run(args.resume)
            except PreferenceRunNotFoundError as exc:
                logger.error("%s", exc)
                return 1
            if manifest.ranking_run_id != args.ranking_run_id:
                logger.error(
                    "preference run %r covers ranking run %r, not %r",
                    args.resume,
                    manifest.ranking_run_id,
                    args.ranking_run_id,
                )
                return 1
            try:
                manifest = preference_run_repo.resume_run(args.resume)
            except PreferenceRunError as exc:
                logger.error("%s", exc)
                return 1
            logger.info("Resuming preference run %s", manifest.preference_run_id)
    else:
        manifest = None

    if manifest is None:
        manifest = preference_run_repo.create_run(
            ranking_run_id=args.ranking_run_id,
            evaluation_run_id=ranking_manifest.evaluation_run_id,
            candidate_run_id=ranking_manifest.candidate_run_id,
            preference_version=PREFERENCE_VERSION,
            selection_policy=policy.name,
            selection_policy_version=policy.version,
            minimum_score_margin=minimum_score_margin,
            split_ratios=split_ratios,
            split_seed=split_seed,
            builder_version=BUILDER_VERSION,
            max_pairs_per_problem=max_pairs_per_problem,
        )
        manifest = preference_run_repo.start_run(manifest.preference_run_id)
        logger.info(
            "Preference run %s created | ranking run %s | policy=%s margin=%s",
            manifest.preference_run_id,
            args.ranking_run_id,
            policy.name,
            minimum_score_margin,
        )

    repository = preference_run_repo.results(manifest.preference_run_id)
    already_done = repository.paired_problem_ids()
    remaining = [pid for pid in sorted(assessments_by_problem) if pid not in already_done]
    for pid in sorted(assessments_by_problem):
        if pid in already_done:
            logger.info("Skipping %s (already processed)", pid)

    builder = PreferencePairBuilder(
        policy,
        minimum_score_margin=minimum_score_margin,
        max_pairs_per_problem=max_pairs_per_problem,
    )

    try:
        for problem_id in remaining:
            problem = problems.get(problem_id)
            if problem is None:
                logger.error(
                    "problem %r (referenced by ranking run %r) not found in the current "
                    "problem dataset",
                    problem_id,
                    args.ranking_run_id,
                )
                return 1
            result = builder.build_problem(
                ranking_run_id=args.ranking_run_id,
                evaluation_run_id=manifest.evaluation_run_id,
                candidate_run_id=manifest.candidate_run_id,
                problem=problem,
                assessments=assessments_by_problem[problem_id],
                candidates_by_id=candidates_by_id,
            )
            repository.save_pairs(result.pairs)
            repository.save_rejections(result.rejections)
    except KeyboardInterrupt:
        preference_run_repo.interrupt_run(manifest.preference_run_id)
        _finalize_preference_run(
            preference_run_repo, manifest.preference_run_id, assessments_by_problem
        )
        logger.warning(
            "Preference run %s interrupted; resume with: python -m python_dpo preferences "
            "generate --ranking-run-id %s --resume %s",
            manifest.preference_run_id,
            args.ranking_run_id,
            manifest.preference_run_id,
        )
        return 130

    stats = _finalize_preference_run(
        preference_run_repo, manifest.preference_run_id, assessments_by_problem
    )
    complete = set(assessments_by_problem) <= repository.paired_problem_ids()

    if complete:
        preference_run_repo.complete_run(manifest.preference_run_id)
        final_status = "completed"
    else:
        preference_run_repo.interrupt_run(manifest.preference_run_id)
        final_status = "interrupted"

    logger.info(
        "Preference run %s %s | %d problem(s) processed this call | pairs=%d rejected=%d "
        "training_records=%d",
        manifest.preference_run_id,
        final_status,
        len(remaining),
        stats.pairs_generated,
        stats.pairs_rejected,
        stats.training_records,
    )
    return 0 if final_status == "completed" else 1


def _cmd_preferences_list(args: argparse.Namespace, config: Config) -> int:
    preference_run_repo = _preference_run_repository(config)
    runs = preference_run_repo.list_runs()
    if not runs:
        sys.stdout.write("No preference runs found.\n")
        return 0

    header = f"{'PREFERENCE_RUN_ID':<28}{'RANKING_RUN_ID':<26}{'POLICY':<12}{'STATUS':<12}{'CREATED_AT'}"
    lines = [header]
    for m in runs:
        lines.append(
            f"{m.preference_run_id:<28}{m.ranking_run_id:<26}{m.selection_policy:<12}"
            f"{m.status:<12}{m.created_at}"
        )
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _cmd_preferences_show(args: argparse.Namespace, config: Config) -> int:
    preference_run_repo = _preference_run_repository(config)
    try:
        preference_run_repo.get_run(args.preference_run_id)
    except PreferenceRunNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    repository = preference_run_repo.results(args.preference_run_id)
    pair = repository.get(args.preference_id)
    if pair is None:
        logger.error("no preference %r in %r", args.preference_id, args.preference_run_id)
        return 1
    sys.stdout.write(format_pair_detail(pair, show_code=args.show_code))
    return 0


def _cmd_preferences_stats(args: argparse.Namespace, config: Config) -> int:
    preference_run_repo = _preference_run_repository(config)
    try:
        preference_run_repo.get_run(args.preference_run_id)
    except PreferenceRunNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    stats = preference_run_repo.read_statistics(args.preference_run_id)
    if stats is None:
        logger.error("no statistics recorded yet for %r", args.preference_run_id)
        return 1
    sys.stdout.write(format_preference_statistics(stats))

    split_manifest = preference_run_repo.read_split_manifest(args.preference_run_id)
    if split_manifest is not None:
        sys.stdout.write(
            f"\nSplit (seed={split_manifest.seed}): "
            f"train={len(split_manifest.train_problem_ids)} "
            f"validation={len(split_manifest.validation_problem_ids)} "
            f"test={len(split_manifest.test_problem_ids)}\n"
        )
    return 0


def _cmd_preferences_validate(args: argparse.Namespace, config: Config) -> int:
    preference_run_repo = _preference_run_repository(config)
    run_dir = preference_run_repo.run_dir(args.preference_run_id)
    if not (run_dir / PREFERENCE_MANIFEST_FILENAME).is_file():
        logger.error("no preference run %r at %s", args.preference_run_id, run_dir)
        return 1

    ranking_run_dir = None
    candidate_run_dir = None
    try:
        manifest = preference_run_repo.get_run(args.preference_run_id)
        ranking_run_dir = _ranking_run_repository(config).run_dir(manifest.ranking_run_id)
        candidate_run_dir = _run_repository(config).run_dir(manifest.candidate_run_id)
    except PreferenceRunError:
        pass  # validate_preference_run below will surface the manifest problem itself

    report = validate_preference_run(run_dir, ranking_run_dir, candidate_run_dir)
    sys.stdout.write(format_preference_report(report))
    return 0 if report.valid else 1


def _add_preferences_parser(subparsers: argparse._SubParsersAction) -> None:
    preferences_parser = subparsers.add_parser(
        "preferences",
        help="Generate and inspect DPO preference pairs.",
        description=(
            "Convert Stage 7's objective ranking into {prompt, chosen, rejected} DPO "
            "preference pairs. Calls no model of any kind; every label comes from "
            "execution evidence via the Stage 7 comparator."
        ),
    )
    preferences_parser.set_defaults(func=_make_help_handler(preferences_parser))

    preferences_subparsers = preferences_parser.add_subparsers(dest="preferences_command")

    generate_parser = preferences_subparsers.add_parser(
        "generate", help="Build preference pairs from a ranking run."
    )
    generate_parser.add_argument("--ranking-run-id", required=True, dest="ranking_run_id")
    generate_parser.add_argument(
        "--policy",
        choices=sorted(POLICIES),
        default=None,
        help="Selection policy (default: value from config.yaml).",
    )
    generate_parser.add_argument(
        "--margin",
        type=float,
        default=None,
        dest="margin",
        help="Minimum score margin for the margin policy (default: value from config.yaml).",
    )
    generate_parser.add_argument(
        "--max-pairs-per-problem",
        type=int,
        default=None,
        dest="max_pairs_per_problem",
        help="Cap pairs kept per problem, largest margin first (default: config.yaml value).",
    )
    generate_parser.add_argument(
        "--split-ratios",
        default=None,
        dest="split_ratios",
        metavar="TRAIN,VALIDATION,TEST",
        help="Train/validation/test split ratios, comma-separated and summing to 1.0 "
        "(default: values from config.yaml).",
    )
    generate_parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        dest="split_seed",
        help="Train/validation/test split seed (default: value from config.yaml).",
    )
    generate_parser.add_argument(
        "--resume",
        metavar="PREFERENCE_RUN_ID",
        default=None,
        help="Continue an existing preference run instead of creating a new one.",
    )
    generate_parser.add_argument(
        "--force",
        action="store_true",
        help="With --resume, abandon it and create a new preference run instead.",
    )
    generate_parser.set_defaults(func=_cmd_preferences_generate)

    list_parser = preferences_subparsers.add_parser("list", help="List preference runs.")
    list_parser.set_defaults(func=_cmd_preferences_list)

    show_parser = preferences_subparsers.add_parser("show", help="Show one preference pair.")
    show_parser.add_argument("--preference-run-id", required=True, dest="preference_run_id")
    show_parser.add_argument("--preference-id", required=True, dest="preference_id")
    show_parser.add_argument("--show-code", action="store_true", dest="show_code")
    show_parser.set_defaults(func=_cmd_preferences_show)

    stats_parser = preferences_subparsers.add_parser(
        "stats", help="Show preference run statistics."
    )
    stats_parser.add_argument("--preference-run-id", required=True, dest="preference_run_id")
    stats_parser.set_defaults(func=_cmd_preferences_stats)

    validate_parser = preferences_subparsers.add_parser(
        "validate", help="Validate one preference run's integrity."
    )
    validate_parser.add_argument("--preference-run-id", required=True, dest="preference_run_id")
    validate_parser.set_defaults(func=_cmd_preferences_validate)


# --------------------------------------------------------------------------------- train


def _training_run_repository(config: Config) -> TrainingRunRepository:
    return TrainingRunRepository(config.paths.training / "runs")


def _cmd_train_hardware_check(args: argparse.Namespace, config: Config) -> int:
    """Spec section 13. Writes to stdout and exits non-zero when unusable."""
    report = check_hardware()
    sys.stdout.write(format_hardware_report(report))
    return 0 if report.passed else 1


def _load_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    """Load the experiment YAML and apply CLI overrides (spec sections 62, 63)."""
    path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    experiment = ExperimentConfig.load(path)
    return experiment.with_overrides(
        preference_run_id=args.preference_run_id,
        experiment_name=getattr(args, "experiment_name", None),
        learning_rate=getattr(args, "learning_rate", None),
        beta=getattr(args, "beta", None),
        num_train_epochs=getattr(args, "epochs", None),
        max_steps=getattr(args, "max_steps", None),
        seed=getattr(args, "seed", None),
        lora_r=getattr(args, "lora_r", None),
    )


def _cmd_train_dpo(args: argparse.Namespace, config: Config) -> int:
    """Train a LoRA adapter with DPO (spec sections 64-67).

    ``--dry-run`` and ``--smoke-test`` take the same path as a real run and stop early,
    so what they validate is the code that actually trains.
    """
    try:
        experiment = _load_experiment_config(args)
    except TrainingError as exc:
        logger.error("%s", exc)
        return 1

    preference_run_id = experiment.dataset.preference_run_id
    if not preference_run_id:
        logger.error(
            "no preference run specified; pass --preference-run-id or set "
            "dataset.preference_run_id in the config"
        )
        return 1

    preference_run_dir = _preference_run_repository(config).run_dir(preference_run_id)
    try:
        dataset = load_training_dataset(
            preference_run_dir,
            allow_small_dataset=args.allow_small_dataset,
            min_training_pairs=experiment.training.min_training_pairs,
        )
    except TrainingError as exc:
        logger.error("%s", exc)
        return 1

    sys.stdout.write(format_dataset_statistics(dataset))

    mode = "train"
    if args.dry_run:
        mode = "dry_run"
    elif args.smoke_test:
        mode = "smoke_test"

    training_run_repo = _training_run_repository(config)
    hardware_report = check_hardware()
    environment = capture_environment()

    manifest = training_run_repo.create_run(
        experiment_name=experiment.experiment_name,
        model_name=experiment.model.name,
        model_revision=experiment.model.revision,
        tokenizer_revision=experiment.model.revision,
        preference_run_id=dataset.preference_run_id,
        ranking_run_id=dataset.provenance["ranking_run_id"],
        evaluation_run_id=dataset.provenance["evaluation_run_id"],
        candidate_run_id=dataset.provenance["candidate_run_id"],
        dataset_hashes=dataset.split_hashes,
        hardware=hardware_report.info.to_dict(),
        environment=environment,
        configuration=experiment.to_dict(),
        seed=experiment.training.seed,
        data_seed=experiment.training.data_seed,
        trainer_version=TRAINER_VERSION,
        mode=mode,
    )
    run_id = manifest.training_run_id

    if args.resume_from_checkpoint:
        try:
            training_run_repo.check_resume_compatibility(
                args.resume_from_checkpoint,
                dataset_hashes=dataset.split_hashes,
                configuration=experiment.to_dict(),
                force=args.force_resume,
            )
        except TrainingError as exc:
            logger.error("%s", exc)
            return 1

    training_run_repo.write_config(run_id, experiment.to_dict())
    training_run_repo.write_hardware(run_id, hardware_report.info.to_dict())
    training_run_repo.write_dataset_manifest(
        run_id,
        DatasetManifest(
            preference_run_id=dataset.preference_run_id,
            preference_version=dataset.provenance["preference_version"],
            selection_policy=dataset.provenance["selection_policy"],
            selection_policy_version=dataset.provenance["selection_policy_version"],
            dataset_schema_version=dataset.provenance["dataset_schema_version"],
            ranking_run_id=dataset.provenance["ranking_run_id"],
            evaluation_run_id=dataset.provenance["evaluation_run_id"],
            candidate_run_id=dataset.provenance["candidate_run_id"],
            split_hashes=dataset.split_hashes,
            split_counts=dataset.split_counts,
            split_problem_ids=dataset.split_problem_ids,
            statistics={
                "splits": dataset.statistics,
                "balance": dataset.balance.to_dict(),
            },
        ),
    )
    training_run_repo.start_run(run_id)
    logger.info(
        "Training run %s created | mode=%s | model=%s | dataset=%s",
        run_id,
        mode,
        experiment.model.name,
        dataset.preference_run_id,
    )

    job = DpoTrainingJob(
        experiment,
        dataset,
        training_run_repo,
        run_id,
        mode=mode,
        allow_small_dataset=args.allow_small_dataset,
        override_truncation=args.override_truncation,
    )

    try:
        with run_log_file(training_run_repo.log_path(run_id)):
            outcome = job.run()
    except KeyboardInterrupt:
        training_run_repo.interrupt_run(run_id)
        logger.warning("Training run %s interrupted", run_id)
        return 130
    except TrainingError as exc:
        error_type, message, tb = format_exception(exc)
        training_run_repo.fail_run(
            run_id, error_type=error_type, error_message=message, traceback_text=tb
        )
        logger.error("%s", message)
        return 1
    except Exception as exc:  # noqa: BLE001 - section 82: record, then re-report
        error_type, message, tb = format_exception(exc)
        training_run_repo.fail_run(
            run_id, error_type=error_type, error_message=message, traceback_text=tb
        )
        logger.error("training failed (%s): %s", error_type, message)
        return 1

    preflight = outcome.preflight
    if preflight.length_analysis is not None:
        sys.stdout.write(format_length_analysis(preflight.length_analysis))
    if preflight.parameter_counts is not None:
        sys.stdout.write(format_parameter_counts(preflight.parameter_counts))

    if mode == "dry_run":
        training_run_repo.complete_run(run_id)
        sys.stdout.write(
            f"Dry run complete for {run_id}. No training was performed.\n"
        )
        return 0

    counts = preflight.parameter_counts
    report = FinalReport(
        training_run_id=run_id,
        experiment_name=experiment.experiment_name,
        status="completed",
        model_name=experiment.model.name,
        model_revision=experiment.model.revision,
        preference_run_id=dataset.preference_run_id,
        number_of_examples={
            "train": dataset.split_counts["train"],
            "validation": dataset.split_counts["validation"],
        },
        epochs=outcome.epochs,
        steps=outcome.steps,
        final_train_loss=outcome.final_train_loss,
        final_eval_loss=outcome.final_eval_loss,
        reward_metrics=outcome.reward_metrics,
        peak_gpu_memory_bytes=outcome.peak_gpu_memory_bytes,
        trainable_parameters=counts.trainable if counts else 0,
        total_parameters=counts.total if counts else 0,
        effective_batch_size=experiment.effective_batch_size,
        optimizer=preflight.optimizer_name,
        compute_dtype=preflight.compute_dtype,
        adapter_path=outcome.adapter_path,
        checkpoint_path=outcome.checkpoint_path,
        adapter_reload_ok=bool(outcome.reload and outcome.reload.ok),
        training_duration_seconds=outcome.duration_seconds,
    )
    training_run_repo.write_final_report(run_id, report)
    training_run_repo.complete_run(run_id)

    sys.stdout.write(format_final_report(report))
    return 0


def _cmd_train_verify(args: argparse.Namespace, config: Config) -> int:
    """Spec section 74's mandatory adapter reload."""
    training_run_repo = _training_run_repository(config)
    try:
        manifest = training_run_repo.get_run(args.training_run_id)
    except TrainingRunNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    try:
        experiment = ExperimentConfig.from_mapping(manifest.configuration)
    except TrainingError as exc:
        logger.error("stored configuration is unusable: %s", exc)
        return 1

    adapter_dir = training_run_repo.adapter_dir(args.training_run_id)
    compute_dtype = manifest.configuration.get("quantization", {}).get(
        "compute_dtype", "bfloat16"
    )
    try:
        result = verify_adapter(adapter_dir, experiment, compute_dtype=compute_dtype)
    except TrainingError as exc:
        logger.error("%s", exc)
        return 1

    sys.stdout.write("Adapter reload successful.\n")
    sys.stdout.write(f"  adapter: {result.adapter_path}\n")
    sys.stdout.write(f"  prompt:  {result.prompt}\n")
    sys.stdout.write(f"  tokens:  {result.generated_tokens}\n")
    return 0


def _cmd_train_inference(args: argparse.Namespace, config: Config) -> int:
    """Spec section 75: base model + trained adapter."""
    training_run_repo = _training_run_repository(config)
    try:
        manifest = training_run_repo.get_run(args.training_run_id)
    except TrainingRunNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    try:
        experiment = ExperimentConfig.from_mapping(manifest.configuration)
    except TrainingError as exc:
        logger.error("stored configuration is unusable: %s", exc)
        return 1

    adapter_dir = training_run_repo.adapter_dir(args.training_run_id)
    compute_dtype = manifest.configuration.get("quantization", {}).get(
        "compute_dtype", "bfloat16"
    )
    try:
        response = run_inference(
            adapter_dir,
            experiment,
            args.prompt,
            compute_dtype=compute_dtype,
            max_new_tokens=args.max_new_tokens,
        )
    except TrainingError as exc:
        logger.error("%s", exc)
        return 1

    sys.stdout.write(response + "\n")
    return 0


def _cmd_train_list(args: argparse.Namespace, config: Config) -> int:
    manifests = _training_run_repository(config).list_runs()
    if not manifests:
        sys.stdout.write("No training runs found.\n")
        return 0
    sys.stdout.write(format_run_table(manifests))
    return 0


def _cmd_train_show(args: argparse.Namespace, config: Config) -> int:
    training_run_repo = _training_run_repository(config)
    try:
        training_run_repo.get_run(args.training_run_id)
    except TrainingRunNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    report = training_run_repo.read_final_report(args.training_run_id)
    if report is None:
        logger.error("no final report for %r yet", args.training_run_id)
        return 1
    sys.stdout.write(format_final_report(report))
    return 0


def _add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    train_parser = subparsers.add_parser(
        "train",
        help="Fine-tune a LoRA adapter on the preference dataset with DPO.",
        description=(
            "Train a LoRA adapter over a frozen, 4-bit quantized Qwen Coder base using "
            "TRL's DPOTrainer. Never executes candidate code, and makes no claim about "
            "Python programming performance -- that is Step 10's job."
        ),
    )
    train_parser.set_defaults(func=_make_help_handler(train_parser))
    train_subparsers = train_parser.add_subparsers(dest="train_command")

    hardware_parser = train_subparsers.add_parser(
        "hardware-check", help="Report GPU, CUDA, BF16 and 4-bit capability."
    )
    hardware_parser.set_defaults(func=_cmd_train_hardware_check)

    dpo_parser = train_subparsers.add_parser("dpo", help="Run DPO/QLoRA training.")
    dpo_parser.add_argument(
        "--config",
        default=None,
        help=f"Experiment config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    dpo_parser.add_argument(
        "--preference-run-id", default=None, dest="preference_run_id"
    )
    dpo_parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Validate hardware, dataset, tokenizer, model, quantization and LoRA, then "
        "stop without training.",
    )
    dpo_parser.add_argument(
        "--smoke-test",
        action="store_true",
        dest="smoke_test",
        help="Train a handful of examples for a couple of steps to prove the stack works.",
    )
    dpo_parser.add_argument(
        "--allow-small-dataset",
        action="store_true",
        dest="allow_small_dataset",
        help="Proceed when the validation split is empty (evaluation is disabled).",
    )
    dpo_parser.add_argument(
        "--override-truncation",
        action="store_true",
        dest="override_truncation",
        help="Continue even when the truncation rate exceeds the configured threshold.",
    )
    dpo_parser.add_argument(
        "--resume-from-checkpoint", default=None, dest="resume_from_checkpoint"
    )
    dpo_parser.add_argument(
        "--force-resume",
        action="store_true",
        dest="force_resume",
        help="Resume even when the dataset hashes differ from the original run.",
    )
    dpo_parser.add_argument("--experiment-name", default=None, dest="experiment_name")
    dpo_parser.add_argument("--learning-rate", type=float, default=None, dest="learning_rate")
    dpo_parser.add_argument("--beta", type=float, default=None, dest="beta")
    dpo_parser.add_argument("--epochs", type=int, default=None, dest="epochs")
    dpo_parser.add_argument("--max-steps", type=int, default=None, dest="max_steps")
    dpo_parser.add_argument("--seed", type=int, default=None, dest="seed")
    dpo_parser.add_argument("--lora-r", type=int, default=None, dest="lora_r")
    dpo_parser.set_defaults(func=_cmd_train_dpo)

    verify_parser = train_subparsers.add_parser(
        "verify", help="Reload the saved adapter and generate (mandatory check)."
    )
    verify_parser.add_argument(
        "--training-run-id", required=True, dest="training_run_id"
    )
    verify_parser.set_defaults(func=_cmd_train_verify)

    inference_parser = train_subparsers.add_parser(
        "inference", help="Generate with the base model plus the trained adapter."
    )
    inference_parser.add_argument(
        "--training-run-id", required=True, dest="training_run_id"
    )
    inference_parser.add_argument("--prompt", required=True)
    inference_parser.add_argument(
        "--max-new-tokens", type=int, default=256, dest="max_new_tokens"
    )
    inference_parser.set_defaults(func=_cmd_train_inference)

    list_parser = train_subparsers.add_parser("list", help="List training runs.")
    list_parser.set_defaults(func=_cmd_train_list)

    show_parser = train_subparsers.add_parser(
        "show", help="Show one training run's final report."
    )
    show_parser.add_argument("--training-run-id", required=True, dest="training_run_id")
    show_parser.set_defaults(func=_cmd_train_show)


# ------------------------------------------------------------------------------- sandbox


def _cmd_sandbox_health(args: argparse.Namespace, config: Config) -> int:
    """Verify the whole Docker execution path before anything depends on it (spec 05 §54)."""
    report = check_sandbox_health(config.sandbox)
    sys.stdout.write(format_health_report(report))
    return 0 if report.passed else 1


def _cmd_sandbox_run(args: argparse.Namespace, config: Config) -> int:
    """Execute a file inside the sandbox (spec 05 §56).

    The host file is **copied** into an isolated workspace and executed in a container; the
    path the user supplied is never mounted and never run on the host (spec 05 §57).
    """
    source = Path(args.file)
    if not source.is_file():
        logger.error("no such file: %s", source)
        return 1

    try:
        code = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.error("could not read %s: %s", source, exc)
        return 1

    executor = SandboxExecutor(config=config.sandbox)
    result = executor.execute(code, timeout_seconds=args.timeout)

    lines = [
        f"status: {result.status}",
        f"exit code: {result.exit_code}",
        f"duration: {result.duration_ms} ms",
    ]
    if result.timed_out:
        lines.append(f"timed out after {args.timeout or config.sandbox.timeout_seconds}s")
    if result.memory_limit_exceeded:
        lines.append("memory limit exceeded")
    if result.truncated:
        lines.append(
            "output truncated at "
            f"{config.sandbox.max_output_bytes} bytes "
            f"(stdout={result.stdout_truncated}, stderr={result.stderr_truncated})"
        )
    if result.signal is not None:
        lines.append(f"terminated by signal {result.signal}")
    if result.error_message:
        lines.append(f"error: {result.error_message}")

    if result.stdout:
        lines += ["", "--- stdout ---", result.stdout.rstrip("\n")]
    if result.stderr and (args.show_stderr or result.status != "success"):
        lines += ["", "--- stderr ---", result.stderr.rstrip("\n")]

    sys.stdout.write("\n".join(lines) + "\n")

    # A candidate that crashed still means the sandbox worked; only an infrastructure
    # failure is a failure of this command (spec 05 §81).
    return 1 if result.is_infrastructure_failure else 0


def _add_sandbox_parser(subparsers: argparse._SubParsersAction) -> None:
    sandbox_parser = subparsers.add_parser(
        "sandbox",
        help="Check and use the isolated Docker execution sandbox.",
        description=(
            "Verify the Docker sandbox and run Python inside it. Code executes only inside "
            "an isolated container with no network, no host filesystem access, a non-root "
            "user, and CPU/memory/PID/output/time limits. Nothing runs on the host."
        ),
    )
    sandbox_parser.set_defaults(func=_make_help_handler(sandbox_parser))

    sandbox_subparsers = sandbox_parser.add_subparsers(dest="sandbox_command")

    health_parser = sandbox_subparsers.add_parser(
        "health", help="Verify Docker, the sandbox image, and container execution."
    )
    health_parser.set_defaults(func=_cmd_sandbox_health)

    run_parser = sandbox_subparsers.add_parser(
        "run",
        help="Execute a Python file inside the sandbox (development aid).",
        description=(
            "Copy a Python file into an isolated temporary workspace and execute it in a "
            "sandboxed container. The supplied path is never mounted into the container "
            "and is never executed on the host."
        ),
    )
    run_parser.add_argument("--file", required=True, help="Path to the Python file to execute.")
    run_parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Override sandbox.timeout_seconds for this execution only.",
    )
    run_parser.add_argument(
        "--show-stderr",
        action="store_true",
        help="Print stderr even when the program exits successfully.",
    )
    run_parser.set_defaults(func=_cmd_sandbox_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python_dpo",
        description="Preference-data generation pipeline for DPO fine-tuning of a "
        "Qwen Coder model on Python tasks.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level (default: value from config.yaml).",
    )

    subparsers = parser.add_subparsers(dest="command")

    _add_problems_parser(subparsers)
    _add_generate_parser(subparsers)
    _add_runs_parser(subparsers)
    _add_candidates_parser(subparsers)
    _add_sandbox_parser(subparsers)
    _add_evaluate_parser(subparsers)
    _add_evaluations_parser(subparsers)
    _add_rank_parser(subparsers)
    _add_rankings_parser(subparsers)
    _add_preferences_parser(subparsers)
    _add_train_parser(subparsers)

    for name in _PLACEHOLDER_STAGES:
        sub = subparsers.add_parser(
            name, help=f"{_PLACEHOLDER_STAGES[name]} (not implemented yet)."
        )
        sub.set_defaults(func=_make_placeholder_handler(name))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    try:
        config = Config.load()
    except ConfigError as exc:
        configure_logging("INFO")
        logger.error(str(exc))
        return 2

    configure_logging(args.log_level or config.log_level)

    return args.func(args, config)
