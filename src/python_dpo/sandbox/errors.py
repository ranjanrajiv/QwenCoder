"""The sandbox exception hierarchy (spec 05 section 78).

Low-level failures — a missing ``docker`` binary, an unreachable daemon, a ``subprocess``
error, a filesystem error — are translated into these types at the boundary. Raw Docker or
``subprocess`` exceptions never propagate into the rest of the application, so callers can
handle "the sandbox could not run this" without importing anything Docker-specific.

The distinction these types encode is the one spec section 81 calls critical: an
infrastructure failure is *not* a statement about the candidate. A candidate must never be
judged badly because Docker was down.
"""

from __future__ import annotations


class SandboxError(Exception):
    """Base class for every sandbox failure."""


class SandboxConfigError(SandboxError):
    """Raised when the ``sandbox:`` configuration section is invalid.

    Deliberately *not* ``python_dpo.config.ConfigError``: this package must not import the
    configuration layer, or the dependency would run in both directions. ``config.py``
    catches this and re-raises as ``ConfigError``, exactly as it already does for
    ``ModelError`` from ``models/base.py``.
    """


class WorkspaceError(SandboxError):
    """Raised when a job workspace cannot be created, written, or removed."""


class DockerUnavailableError(SandboxError):
    """Raised when the Docker CLI is missing or the daemon is unreachable."""


class ImageUnavailableError(SandboxError):
    """Raised when the configured image is absent and cannot be pulled."""


class ContainerCreationError(SandboxError):
    """Raised when a container cannot be created or started (spec 05 section 80)."""


class ContainerExecutionError(SandboxError):
    """Raised when a running container fails for a host-side/infrastructure reason.

    Not used for a candidate that merely exits non-zero — that is a normal, recorded
    execution outcome, not an error.
    """


class SandboxTimeoutError(SandboxError):
    """Raised when a container cannot be stopped after exceeding its timeout.

    A candidate hitting its timeout is an ordinary result (``status="timeout"``), not an
    exception; this is for the case where termination itself fails.
    """


__all__ = [
    "ContainerCreationError",
    "ContainerExecutionError",
    "DockerUnavailableError",
    "ImageUnavailableError",
    "SandboxConfigError",
    "SandboxError",
    "SandboxTimeoutError",
    "WorkspaceError",
]
