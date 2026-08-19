"""The stage adapter contract: ``StageContext`` in, ``StageResult`` out (spec 12 section 5).

Split out from this package's ``__init__.py`` to avoid a cycle: ``StageContext`` needs
``StageConfig`` from :mod:`python_dpo.pipeline.config`, and ``config.py`` needs
``STAGE_NAMES`` from this package's ``__init__.py`` -- so the registry (``__init__.py``)
must not depend on ``config.py``, and this module carries that dependency instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import StageConfig
from ..repository import ExperimentRunRepository

if TYPE_CHECKING:
    from ...config import Config


@dataclass(frozen=True)
class StageContext:
    """Everything one stage adapter invocation needs.

    ``upstream`` holds the completed :class:`~python_dpo.pipeline.manifest.StageManifest`
    for every stage this one directly requires, keyed by stage name -- a stage reaches an
    upstream identifier (a candidate run id, a preference run id, ...) via
    ``context.upstream_run_id("preference_generation")``, never by re-deriving it.
    """

    experiment_run_id: str
    stage_config: StageConfig
    project_config: "Config"
    experiment_repo: ExperimentRunRepository
    upstream: dict[str, Any] = field(default_factory=dict)
    seed: int = 42

    def upstream_run_id(self, stage_name: str) -> str:
        manifest = self.upstream.get(stage_name)
        if manifest is None:
            from ..errors import DependencyError

            raise DependencyError(
                f"stage {self.stage_config.name!r} requires {stage_name!r}'s output, "
                "but no completed upstream stage manifest was provided"
            )
        return manifest.stage_run_id

    def log_path(self) -> Path:
        return self.experiment_repo.log_path(self.experiment_run_id, self.stage_config.name)


@dataclass(frozen=True)
class StageResult:
    """What a stage adapter hands back to the orchestrator.

    ``stage_run_id`` becomes the resulting :class:`StageManifest`'s ``stage_run_id`` --
    the underlying repository's own run id when the stage has one (a training run's
    ``dpo_...`` id, a candidate run's ``run_...`` id), or a synthesized id otherwise.
    ``output_artifacts`` maps artifact names to SHA-256 hashes, feeding both the
    experiment-wide ``artifacts.json`` and the next stage's cache key input hashes.
    """

    stage_run_id: str
    output_artifacts: dict[str, str] = field(default_factory=dict)


__all__ = ["StageContext", "StageResult"]
