"""Experiment cost accounting (spec 12 sections 52, 53).

GPU-hours are derived from each GPU-using stage's own recorded wall-clock duration --
never a separate timer, so there is nothing to keep in sync with the stage manifests.
``candidate_generation`` is counted because :class:`~python_dpo.models.qwen.QwenModelClient`
loads Qwen locally via Transformers (a GPU cost), not a remote API call.

The LLM-API cost schema is recorded with an explicit empty ``providers`` list rather than
omitted: this pipeline never calls an external LLM anywhere (spec section 53), so the
empty list is a deliberate, checked fact, not a gap the schema forgot to fill in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Stages that load a model onto the GPU, in the sense that matters for cost: local
# inference or training, never a network API call.
GPU_STAGES = frozenset({"candidate_generation", "dpo_training", "model_evaluation", "packaging"})


@dataclass(frozen=True)
class CostReport:
    gpu_hours: float
    gpu_hours_by_stage: dict[str, float] = field(default_factory=dict)
    llm_api_providers: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_hours": self.gpu_hours,
            "gpu_hours_by_stage": dict(self.gpu_hours_by_stage),
            "llm_api": {"providers": list(self.llm_api_providers)},
        }


def compute_cost(stage_durations: dict[str, float]) -> CostReport:
    """``stage_durations`` maps stage name -> wall-clock seconds
    (:func:`python_dpo.pipeline.resources.stage_durations`)."""
    gpu_hours_by_stage = {
        name: round(seconds / 3600, 4)
        for name, seconds in stage_durations.items()
        if name in GPU_STAGES
    }
    return CostReport(
        gpu_hours=round(sum(gpu_hours_by_stage.values()), 4),
        gpu_hours_by_stage=gpu_hours_by_stage,
    )


__all__ = ["GPU_STAGES", "CostReport", "compute_cost"]
