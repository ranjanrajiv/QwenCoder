"""End-to-end pipeline orchestration (spec 12).

Public surface for :mod:`python_dpo.pipeline`. Stage adapters (``pipeline.stages.*``) and
the orchestrator itself are not imported here at module scope -- several stage adapters
eventually call into ``training``/``model_evaluation`` code that imports torch lazily, and
this package must stay importable without that (``tests/test_no_heavy_imports.py``).
"""

from __future__ import annotations

from .artifacts import ArtifactError, ArtifactRef
from .cache import cache_key, invalidate, is_reusable
from .config import ExperimentConfig, StageConfig
from .errors import (
    ArchiveError,
    DependencyError,
    ExperimentConfigError,
    ExperimentRunNotFoundError,
    PipelineError,
    PreflightError,
    StageFailedError,
    StageNotImplementedError,
)
from .manifest import ExperimentManifest, StageError, StageManifest
from .repository import ExperimentRunRepository
from .stages import STAGE_NAMES, STAGES, StageSpec, dependents_of, topological_order
from .state import STAGE_STATES, validate_transition

__all__ = [
    "ArchiveError",
    "ArtifactError",
    "ArtifactRef",
    "DependencyError",
    "ExperimentConfig",
    "ExperimentConfigError",
    "ExperimentManifest",
    "ExperimentRunNotFoundError",
    "ExperimentRunRepository",
    "PipelineError",
    "PreflightError",
    "STAGES",
    "STAGE_NAMES",
    "STAGE_STATES",
    "StageConfig",
    "StageError",
    "StageFailedError",
    "StageManifest",
    "StageNotImplementedError",
    "StageSpec",
    "cache_key",
    "dependents_of",
    "invalidate",
    "is_reusable",
    "topological_order",
    "validate_transition",
]
