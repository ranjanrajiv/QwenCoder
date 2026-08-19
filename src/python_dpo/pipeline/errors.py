"""The pipeline package's exception hierarchy (spec 12).

One base class, one subclass per failure mode the orchestrator must distinguish. Callers
catch :class:`PipelineError` when the distinction does not matter and a specific subclass
when it does — e.g. the CLI reports a :class:`DependencyError` differently from a
:class:`StageFailedError` (spec section 16 vs section 65).
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for every error raised by :mod:`python_dpo.pipeline`."""


class ExperimentConfigError(PipelineError):
    """Raised when an experiment configuration file is missing, malformed, or invalid."""


class DependencyError(PipelineError):
    """Raised when a stage's required input artifact does not exist (spec section 16).

    Never raised to trigger reconstruction — the stage that would produce the missing
    artifact is not silently re-run; the caller must run it explicitly first.
    """


class StageFailedError(PipelineError):
    """Raised when a stage's adapter raises during execution (spec section 65)."""


class StageNotImplementedError(PipelineError):
    """Raised when a registered-but-unimplemented stage is asked to run.

    Distinct from :class:`NotImplementedError` so callers can catch pipeline failures
    with one except clause without also swallowing genuine Python bugs.
    """


class PreflightError(PipelineError):
    """Raised when a preflight check fails and the caller asked for a hard failure."""


class ExperimentRunNotFoundError(PipelineError):
    """Raised when an experiment run id has no manifest on disk."""


class ArchiveError(PipelineError):
    """Raised when creating or inspecting an experiment archive fails."""


__all__ = [
    "ArchiveError",
    "DependencyError",
    "ExperimentConfigError",
    "ExperimentRunNotFoundError",
    "PipelineError",
    "PreflightError",
    "StageFailedError",
    "StageNotImplementedError",
]
