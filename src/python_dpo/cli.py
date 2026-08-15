from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from . import __version__
from .config import Config, ConfigError
from .logging_config import configure_logging

logger = logging.getLogger("python_dpo.cli")

_STAGE_NAMES = {
    "problems": "Problem loading",
    "generate": "Candidate generation",
    "evaluate": "Candidate evaluation",
    "preferences": "Preference pair generation",
    "run": "Full pipeline run",
}


def _make_placeholder_handler(name: str):
    stage = _STAGE_NAMES[name]

    def _handler(args: argparse.Namespace) -> int:
        logger.info("%s is not implemented yet.", stage)
        return 1

    return _handler


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

    for name in _STAGE_NAMES:
        sub = subparsers.add_parser(name, help=f"{_STAGE_NAMES[name]} (not implemented yet).")
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

    return args.func(args)
