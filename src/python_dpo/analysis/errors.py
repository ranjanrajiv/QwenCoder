"""The analysis package's exception hierarchy (spec 11).

One base class, one subclass per failure mode a caller must distinguish -- matching the
house style of :mod:`python_dpo.pipeline.errors` and :mod:`python_dpo.packaging.errors`.
"""

from __future__ import annotations


class AnalysisError(Exception):
    """Base class for every error raised by :mod:`python_dpo.analysis`."""


class AnalysisConfigError(AnalysisError):
    """Raised when the analysis configuration is missing, malformed, or out of range."""


class AnalysisInputError(AnalysisError):
    """Raised when a required Stage 6/8/9/10 artifact is missing or unreadable."""


class AnalysisRunNotFoundError(AnalysisError):
    """Raised when an analysis run id has no manifest on disk."""


class AnalysisRunError(AnalysisError):
    """Raised when an analysis run's artifacts cannot be read or written."""


class AnalysisStoreError(AnalysisError):
    """Raised when a persisted analysis record is malformed."""


class LineageError(AnalysisError):
    """Raised when the evaluation -> training -> preference -> ranking -> candidate chain
    cannot be resolved (spec section 7).

    Never downgraded to a warning: section 7 makes the lineage a precondition, so a broken
    hop must stop the analysis rather than let it proceed over a partial chain.
    """


class RefinementLeakageError(AnalysisError):
    """Raised when a refined preference row would carry a held-out benchmark problem
    (spec sections 65, 66, 104, 117).

    The single most tempting mistake this stage could make is feeding the benchmark
    problems DPO failed back into training. That invalidates every future evaluation
    number, so it is a hard error rather than a filtered-out row.
    """


__all__ = [
    "AnalysisConfigError",
    "AnalysisError",
    "AnalysisInputError",
    "AnalysisRunError",
    "AnalysisRunNotFoundError",
    "AnalysisStoreError",
    "LineageError",
    "RefinementLeakageError",
]
