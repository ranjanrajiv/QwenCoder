"""Stage 12 as a pipeline stage: model packaging (spec 12 section 5, item 9).

Delegates to :mod:`python_dpo.packaging`, built in Phase 3 of the implementation plan
(``.claude/plans/12_pipeline_orchestration_and_productionization_plan.md``). Until then
this adapter fails loudly rather than silently -- exactly the same discipline applied to
``error_analysis`` for the still-unimplemented Stage 11.
"""

from __future__ import annotations

from ..errors import StageNotImplementedError
from ._context import StageContext, StageResult


def run(context: StageContext) -> StageResult:
    try:
        from ... import packaging as packaging_package  # noqa: F401
    except ImportError as exc:
        raise StageNotImplementedError(
            "packaging is implemented in Phase 3 of the Stage 12 plan and does not exist "
            "yet; set packaging.enabled: false in the experiment configuration."
        ) from exc

    from ...packaging.pipeline_stage import run as packaging_run

    return packaging_run(context)


__all__ = ["run"]
