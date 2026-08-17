from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from . import __version__
from .candidates import CandidateRepository, CandidateStoreError
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
from .models import PROVIDER_MOCK, MockModelClient, ModelError, QwenModelClient
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


def _cmd_generate(args: argparse.Namespace, config: Config) -> int:
    """Generate candidate solutions for the selected problems."""
    path = dataset_path(config.paths.problems)

    try:
        problems = load_problems(path)
    except DatasetError as exc:
        logger.error("%s", exc)
        return 1

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

    repository = CandidateRepository(config.paths.candidates)

    try:
        run_id = repository.new_run_id()
        client = _build_model_client(config, args.mock_model)
    except (CandidateStoreError, ModelError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "Run %s | model=%s | %d problem(s) x %d candidate(s) | force=%s",
        run_id,
        client.name,
        len(selected),
        count,
        args.force,
    )

    generator = CandidateGenerator(
        client=client,
        repository=repository,
        generation_config=config.generation.config,
    )

    try:
        summary = generator.generate(
            selected,
            count=count,
            strategies=strategies,
            run_id=run_id,
            force=args.force,
        )
    except ModelError as exc:
        logger.error("Aborting run %s: %s", run_id, exc)
        return 1
    except CandidateStoreError as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "Run %s complete | generated=%d skipped=%d failed=%d duplicates=%d",
        summary.run_id,
        summary.generated,
        summary.skipped,
        summary.failed,
        summary.duplicates,
    )
    if summary.failed:
        # Failures are data, not a broken run: they are recorded and observable, so the
        # command still succeeds (spec 03 section 26).
        logger.warning(
            "%d generation(s) produced no candidate; see %s",
            summary.failed,
            repository.failures_path,
        )
    return 0


def _add_generate_parser(subparsers: argparse._SubParsersAction) -> None:
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate candidate solutions for problems in the dataset.",
        description=(
            "Generate candidate Python implementations with the configured model and "
            "persist them to data/candidates/candidates.jsonl. Candidates are not "
            "executed or evaluated at this stage."
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
        "--force",
        action="store_true",
        help="Regenerate in a new run instead of resuming (existing records are kept).",
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
