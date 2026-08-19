"""Stage 11 as a pipeline stage: error analysis (spec 12 section 5, item 8).

``src/python_dpo/analysis/`` does not exist yet -- Stage 11 is specified and planned
(``.claude/specs/11_error_analysis_and_iteration.md``,
``.claude/plans/11_error_analysis_and_iteration_plan.md``) but unimplemented. Per the
plan's decision 1, this stage is registered with its real dependency edges so the graph,
cache and CLI all understand it, but it is shipped ``enabled: false`` in every experiment
config and fails loudly rather than silently if enabled -- the orchestrator must never
record a stage as complete, skipped-for-a-good-reason, or successful when the code behind
it simply is not there.
"""

from __future__ import annotations

from ..errors import StageNotImplementedError
from ._context import StageContext, StageResult


def run(context: StageContext) -> StageResult:
    raise StageNotImplementedError(
        "error_analysis is registered but not implemented (Stage 11 has no code yet; "
        "see .claude/plans/11_error_analysis_and_iteration_plan.md). Set "
        "error_analysis.enabled: false in the experiment configuration."
    )


__all__ = ["run"]
