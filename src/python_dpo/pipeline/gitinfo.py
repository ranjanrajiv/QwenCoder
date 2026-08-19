"""Git version capture (spec 12 section 29).

Three ``git`` subprocess calls with a fixed argv each, matching the house rule already
enforced for the sandbox: no shell interpolation of any kind, nothing that could turn a
branch name or commit message into a command.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .errors import PipelineError


class GitInfoError(PipelineError):
    """Raised when the working tree is dirty and configuration says to fail (section 29)."""


def _run_git(args: list[str], *, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def capture_git_info(project_root: Path) -> dict[str, Any]:
    """``{"sha", "branch", "dirty"}``, or ``None`` for fields git could not answer.

    ``dirty`` is ``None`` only when ``git status`` itself could not be run (no git
    binary, not a repository); an empty status output means ``dirty=False``.
    """
    sha = _run_git(["rev-parse", "HEAD"], cwd=project_root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=project_root)
    status = _run_git(["status", "--porcelain"], cwd=project_root)
    dirty = None if status is None else bool(status)
    return {"sha": sha, "branch": branch, "dirty": dirty}


def enforce_dirty_policy(git_info: dict[str, Any], *, on_dirty: str) -> None:
    """Apply the experiment config's ``on_dirty: warn | fail`` policy (section 29).

    ``on_dirty="fail"`` with an indeterminate ``dirty`` (git unavailable) is treated as
    dirty -- a preflight cannot certify reproducibility it could not actually check.
    """
    if on_dirty not in ("warn", "fail"):
        raise GitInfoError(f"on_dirty must be 'warn' or 'fail', got {on_dirty!r}")
    if git_info.get("dirty") is False:
        return
    message = (
        "the working tree is dirty (or its status could not be determined); "
        "an experiment started now would not be exactly reproducible from git alone"
    )
    if on_dirty == "fail":
        raise GitInfoError(message)


__all__ = ["GitInfoError", "capture_git_info", "enforce_dirty_policy"]
