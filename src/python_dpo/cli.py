from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from . import __version__
from .config import Config, ConfigError
from .logging_config import configure_logging
from .problems import (
    DatasetError,
    InProcessReferenceExecutor,
    build_catalog,
    dataset_path,
    format_report,
    load_problems,
    save_problems,
    validate_dataset,
)

logger = logging.getLogger("python_dpo.cli")

_PLACEHOLDER_STAGES = {
    "generate": "Candidate generation",
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
