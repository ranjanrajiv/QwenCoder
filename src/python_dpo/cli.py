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
    SandboxExecutor,
    check_sandbox_health,
    format_health_report,
)

logger = logging.getLogger("python_dpo.cli")

_PLACEHOLDER_STAGES = {
    "evaluate": "Candidate evaluation",
    "preferences": "Preference pair generation",
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
