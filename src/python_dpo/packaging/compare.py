"""Comparing registered models by their recorded evaluation metrics (spec 12 section 49).

Reuses Stage 10's persisted ``summary``/``peak_gpu_memory`` metrics rather than
recomputing anything -- this module only joins :class:`~python_dpo.packaging.registry.RegistryEntry`
rows to the ``model_evaluation`` run each names, exactly the "assembly of existing parts"
the plan calls for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..model_evaluation.run_repository import ModelEvaluationRunRepository
from .registry import ModelRegistry, RegistryEntry

# The packaged model is always the "dpo" variant in its evaluation run (spec section 15's
# adapter runner has no other role).
_PACKAGED_VARIANT = "dpo"


@dataclass(frozen=True)
class ModelComparisonRow:
    """One registered model's headline evaluation numbers, for ``model compare``."""

    model_id: str
    status: str
    evaluation_run_id: str | None
    pass_at_1: float | None
    pass_at_5: float | None
    pass_at_10: float | None
    syntax_success_rate: float | None
    timeout_rate: float | None
    peak_gpu_memory_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "status": self.status,
            "evaluation_run_id": self.evaluation_run_id,
            "pass_at_1": self.pass_at_1,
            "pass_at_5": self.pass_at_5,
            "pass_at_10": self.pass_at_10,
            "syntax_success_rate": self.syntax_success_rate,
            "timeout_rate": self.timeout_rate,
            "peak_gpu_memory_bytes": self.peak_gpu_memory_bytes,
        }


def _row_for(entry: RegistryEntry, eval_repo: ModelEvaluationRunRepository) -> ModelComparisonRow:
    if not entry.evaluation_run_id:
        return ModelComparisonRow(
            model_id=entry.model_id, status=entry.status, evaluation_run_id=None,
            pass_at_1=None, pass_at_5=None, pass_at_10=None,
            syntax_success_rate=None, timeout_rate=None, peak_gpu_memory_bytes=None,
        )

    summary = eval_repo.read_metrics(entry.evaluation_run_id, "summary") or {}
    pass_at_k = (summary.get("pass_at_k") or {}).get(_PACKAGED_VARIANT, {})
    memory = eval_repo.read_metrics(entry.evaluation_run_id, "peak_gpu_memory") or {}

    return ModelComparisonRow(
        model_id=entry.model_id,
        status=entry.status,
        evaluation_run_id=entry.evaluation_run_id,
        pass_at_1=pass_at_k.get("1"),
        pass_at_5=pass_at_k.get("5"),
        pass_at_10=pass_at_k.get("10"),
        syntax_success_rate=(summary.get("syntax_success_rate") or {}).get(_PACKAGED_VARIANT),
        timeout_rate=(summary.get("timeout_rate") or {}).get(_PACKAGED_VARIANT),
        peak_gpu_memory_bytes=memory.get(_PACKAGED_VARIANT),
    )


def compare_models(
    registry: ModelRegistry, eval_repo: ModelEvaluationRunRepository
) -> list[ModelComparisonRow]:
    """One row per registered model, newest first (matches :meth:`ModelRegistry.list`)."""
    return [_row_for(entry, eval_repo) for entry in registry.list()]


__all__ = ["ModelComparisonRow", "compare_models"]
