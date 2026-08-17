"""The isolated per-execution job directory (spec 05 sections 15, 16, 41).

Each execution gets its own temporary directory holding exactly the files that execution
needs — for Stage 5, just ``candidate.py``. That directory is the *only* host path the
container ever sees, and it is mounted read-only (spec section 17).

**This module never executes anything** (spec section 41). It creates a directory, writes
text into it, and deletes it. Cleanup happens in ``__exit__``, so it runs whether the
candidate succeeded, crashed, timed out, or Docker itself blew up (spec section 16).
"""

from __future__ import annotations

import secrets
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

from .errors import WorkspaceError

CANDIDATE_FILENAME = "candidate.py"

# The mount point inside the container. Fixed, so the execution command is fixed too.
CONTAINER_WORKSPACE = "/workspace"

_WORKSPACE_PREFIX = "python-dpo-sandbox-"

# The container runs as a non-root UID that is not the directory's owner, so it needs
# world read+execute to traverse the directory and read the file. tempfile.mkdtemp creates
# 0o700, which that user cannot enter — every execution would fail with a confusing
# permission error. This is not a security loosening: the directory holds only the
# candidate's own source, lives under a private temporary root, is mounted read-only, and
# is destroyed after the run.
_DIR_MODE = 0o755
_FILE_MODE = 0o644


def new_job_id(now: datetime | None = None) -> str:
    """A sortable, debuggable job id: ``20260817_143700_a81f``.

    Spec section 37 rules out purely random names — the timestamp is what makes a stray
    container or directory traceable to when it ran. Same shape as Stage 4's run ids.
    """
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{secrets.token_hex(2)}"


class SandboxWorkspace:
    """A temporary job directory, usable as a context manager.

    ``with SandboxWorkspace(...) as workspace:`` guarantees the directory is removed on
    exit, including when the body raises.
    """

    def __init__(
        self,
        *,
        job_id: str | None = None,
        root: str | Path | None = None,
    ) -> None:
        self.job_id = job_id or new_job_id()
        self._root = Path(root) if root is not None else None
        self._path: Path | None = None

    @property
    def path(self) -> Path:
        if self._path is None:
            raise WorkspaceError("workspace has not been created; use it as a context manager")
        return self._path

    @property
    def candidate_path(self) -> Path:
        return self.path / CANDIDATE_FILENAME

    @property
    def created(self) -> bool:
        return self._path is not None

    def create(self) -> Path:
        if self._path is not None:
            raise WorkspaceError(f"workspace {self.job_id} already exists at {self._path}")
        try:
            if self._root is not None:
                self._root.mkdir(parents=True, exist_ok=True)
            path = Path(
                tempfile.mkdtemp(
                    prefix=f"{_WORKSPACE_PREFIX}{self.job_id}-",
                    dir=str(self._root) if self._root is not None else None,
                )
            )
            path.chmod(_DIR_MODE)
        except OSError as exc:
            raise WorkspaceError(f"could not create a sandbox workspace: {exc}") from exc
        self._path = path
        return path

    def write_candidate(self, code: str) -> Path:
        """Write ``code`` verbatim to ``candidate.py``.

        The source is written to a file and never interpolated into a command (spec
        sections 42, 43), which is what removes shell quoting and injection from the
        picture entirely.
        """
        if not isinstance(code, str):
            raise WorkspaceError("candidate code must be a string")
        target = self.candidate_path
        try:
            # Written as bytes rather than via write_text: text mode applies newline
            # translation, so a candidate containing \r\n would not reach the container
            # byte-for-byte as the model produced it.
            target.write_bytes(code.encode("utf-8"))
            target.chmod(_FILE_MODE)
        except OSError as exc:
            raise WorkspaceError(f"could not write {target}: {exc}") from exc
        return target

    def cleanup(self) -> None:
        """Remove the workspace. Safe to call more than once, and never raises."""
        if self._path is None:
            return
        shutil.rmtree(self._path, ignore_errors=True)
        self._path = None

    def __enter__(self) -> SandboxWorkspace:
        self.create()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Spec section 16: cleanup runs on every path out of the block.
        self.cleanup()


__all__ = [
    "CANDIDATE_FILENAME",
    "CONTAINER_WORKSPACE",
    "SandboxWorkspace",
    "new_job_id",
]
