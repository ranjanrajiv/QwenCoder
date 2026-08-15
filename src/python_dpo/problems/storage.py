"""JSONL persistence for the problem dataset.

One JSON object per line, UTF-8, keys sorted so that rebuilding an unchanged catalog
produces a byte-identical file. Loading validates every record and never silently
skips a malformed one.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from .models import Problem, ProblemError

PROBLEMS_FILENAME = "problems.jsonl"


class DatasetError(Exception):
    """Raised when a persisted dataset is missing, malformed, or inconsistent."""


def save_problems(problems: Iterable[Problem], path: Path) -> None:
    """Write ``problems`` to ``path`` as JSONL, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        json.dumps(problem.to_dict(), sort_keys=True, ensure_ascii=False)
        for problem in problems
    ]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def load_problems(path: Path) -> list[Problem]:
    """Load and validate problems from a JSONL file, preserving file order."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"problem dataset not found at {path}")

    problems: list[Problem] = []
    first_seen: dict[str, int] = {}

    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                raise DatasetError(
                    f"{path}:{number}: blank line; JSONL must hold exactly one object per line"
                )

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{number}: invalid JSON: {exc}") from exc

            try:
                problem = Problem.from_dict(record)
            except ProblemError as exc:
                raise DatasetError(f"{path}:{number}: {exc}") from exc

            if problem.id in first_seen:
                raise DatasetError(
                    f"{path}:{number}: duplicate problem id {problem.id!r} "
                    f"(first seen on line {first_seen[problem.id]})"
                )
            first_seen[problem.id] = number
            problems.append(problem)

    if not problems:
        raise DatasetError(f"{path}: dataset is empty")

    return problems


def dataset_path(problems_dir: Path) -> Path:
    """Return the canonical dataset location inside a problems directory."""
    return Path(problems_dir) / PROBLEMS_FILENAME


__all__: Sequence[str] = [
    "DatasetError",
    "PROBLEMS_FILENAME",
    "dataset_path",
    "load_problems",
    "save_problems",
]
