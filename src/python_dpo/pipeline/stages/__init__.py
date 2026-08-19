"""The nine-stage dependency graph and adapter contract (spec 12 sections 5, 15, 16).

The spec's stage names do not map 1:1 onto the repo's existing commands: Stage 6
(``evaluate run``) executes candidates in Docker *and* runs pytest in one pass, and
Stage 7 (``rank``) classifies, scores and orders them. This module keeps the spec's
intended meaning -- execution produces evidence, evaluation produces judgement -- by
naming the two stages ``candidate_execution`` (Stage 6) and ``candidate_evaluation``
(Stage 7), per the plan's decision.

``adapter`` is a dotted ``"module:function"`` string rather than an imported callable, so
this module never has to import any ``pipeline.stages.<stage>`` submodule (and,
transitively, torch/transformers) just to describe the graph. :func:`resolve_adapter`
performs the import lazily, on demand -- every stage submodule lives right here, as a
sibling of this file, and each exposes one ``run(context) -> StageResult`` function (the
contract in :mod:`python_dpo.pipeline.stages._context`).

``StageContext``/``StageResult`` themselves live in ``_context.py``, not here: that module
needs ``StageConfig`` from :mod:`python_dpo.pipeline.config`, and ``config.py`` needs
``STAGE_NAMES`` from this module -- so this registry must stay free of any dependency on
``config.py`` to avoid a cycle.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..errors import PipelineError


class StageGraphError(PipelineError):
    """Raised when the stage graph itself is invalid (unknown name, cycle, bad edge)."""


@dataclass(frozen=True)
class StageSpec:
    """One node in the pipeline's stage graph."""

    name: str
    requires: tuple[str, ...]
    adapter: str
    config_section: str


# Dependency edges are exactly spec section 15, with the Stage 6 / Stage 7 renaming above.
# `packaging` requires `dpo_training`, not `model_evaluation` -- the spec is explicit that
# packaging only needs a trained adapter, not a held-out evaluation of it.
STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        name="problem_dataset",
        requires=(),
        adapter="python_dpo.pipeline.stages.problem_dataset:run",
        config_section="problem_dataset",
    ),
    StageSpec(
        name="candidate_generation",
        requires=("problem_dataset",),
        adapter="python_dpo.pipeline.stages.candidate_generation:run",
        config_section="candidate_generation",
    ),
    StageSpec(
        name="candidate_execution",
        requires=("candidate_generation",),
        adapter="python_dpo.pipeline.stages.candidate_execution:run",
        config_section="candidate_execution",
    ),
    StageSpec(
        name="candidate_evaluation",
        requires=("candidate_execution",),
        adapter="python_dpo.pipeline.stages.candidate_evaluation:run",
        config_section="candidate_evaluation",
    ),
    StageSpec(
        name="preference_generation",
        requires=("candidate_evaluation",),
        adapter="python_dpo.pipeline.stages.preference_generation:run",
        config_section="preference_generation",
    ),
    StageSpec(
        name="dpo_training",
        requires=("preference_generation",),
        adapter="python_dpo.pipeline.stages.dpo_training:run",
        config_section="dpo_training",
    ),
    StageSpec(
        name="model_evaluation",
        requires=("dpo_training",),
        adapter="python_dpo.pipeline.stages.model_evaluation:run",
        config_section="model_evaluation",
    ),
    StageSpec(
        name="error_analysis",
        requires=("model_evaluation",),
        adapter="python_dpo.pipeline.stages.error_analysis:run",
        config_section="error_analysis",
    ),
    StageSpec(
        name="packaging",
        requires=("dpo_training",),
        adapter="python_dpo.pipeline.stages.packaging:run",
        config_section="packaging",
    ),
)

STAGE_NAMES: tuple[str, ...] = tuple(spec.name for spec in STAGES)
_STAGES_BY_NAME: dict[str, StageSpec] = {spec.name: spec for spec in STAGES}


def get_stage(name: str) -> StageSpec:
    try:
        return _STAGES_BY_NAME[name]
    except KeyError:
        raise StageGraphError(
            f"unknown stage {name!r}; must be one of {', '.join(STAGE_NAMES)}"
        ) from None


def validate_graph() -> None:
    """Raise :class:`StageGraphError` if the graph is malformed.

    Checked eagerly (rather than only surfacing as a topological-sort failure) so a typo
    in ``requires`` fails with a specific, actionable message.
    """
    seen: set[str] = set()
    for spec in STAGES:
        if spec.name in seen:
            raise StageGraphError(f"duplicate stage name {spec.name!r}")
        seen.add(spec.name)
    for spec in STAGES:
        for dep in spec.requires:
            if dep not in _STAGES_BY_NAME:
                raise StageGraphError(
                    f"stage {spec.name!r} requires unknown stage {dep!r}"
                )
    # A cycle would make topological_order() raise; running it here turns that into a
    # StageGraphError with graph-specific wording instead of a bare RuntimeError.
    topological_order()


def topological_order(names: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Kahn's algorithm over ``requires`` edges, restricted to ``names`` if given.

    Ties are broken by the graph's declared order (spec section 5's numbering), so the
    result is deterministic rather than dependent on set iteration order.
    """
    universe = STAGE_NAMES if names is None else names
    universe_set = set(universe)
    for name in universe:
        get_stage(name)  # raises StageGraphError on an unknown name

    remaining = {name: set(get_stage(name).requires) & universe_set for name in universe}
    ordered: list[str] = []
    while remaining:
        ready = [name for name in STAGE_NAMES if name in remaining and not remaining[name]]
        if not ready:
            raise StageGraphError(
                f"cycle detected among stages: {', '.join(sorted(remaining))}"
            )
        for name in ready:
            ordered.append(name)
            del remaining[name]
        for deps in remaining.values():
            deps.difference_update(ready)
    return tuple(ordered)


def dependents_of(name: str) -> tuple[str, ...]:
    """Every stage that transitively requires ``name``, in topological order.

    This is exactly the cascade `--force` must invalidate (spec section 22): forcing
    ``candidate_evaluation`` must also invalidate ``preference_generation``,
    ``dpo_training``, ``model_evaluation``, ``error_analysis`` and ``packaging``.
    """
    get_stage(name)  # raises on unknown name
    direct: dict[str, set[str]] = {spec.name: set(spec.requires) for spec in STAGES}
    dependents: set[str] = set()
    frontier = {name}
    while frontier:
        next_frontier: set[str] = set()
        for spec in STAGES:
            if direct[spec.name] & frontier and spec.name not in dependents:
                dependents.add(spec.name)
                next_frontier.add(spec.name)
        frontier = next_frontier
    return tuple(n for n in STAGE_NAMES if n in dependents)


def resolve_adapter(spec: StageSpec) -> Callable[..., Any]:
    """Import and return the callable named by ``spec.adapter``."""
    module_name, _, func_name = spec.adapter.partition(":")
    if not module_name or not func_name:
        raise StageGraphError(
            f"stage {spec.name!r}: adapter {spec.adapter!r} is not a valid "
            "'module:function' reference"
        )
    module = importlib.import_module(module_name)
    try:
        return getattr(module, func_name)
    except AttributeError:
        raise StageGraphError(
            f"stage {spec.name!r}: {module_name!r} has no attribute {func_name!r}"
        ) from None


__all__ = [
    "STAGES",
    "STAGE_NAMES",
    "StageGraphError",
    "StageSpec",
    "dependents_of",
    "get_stage",
    "resolve_adapter",
    "topological_order",
    "validate_graph",
]
