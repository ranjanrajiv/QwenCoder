"""Artifact lineage (spec 12 section 27).

Built by hopping the completed stage manifests the orchestrator already holds in memory
at the end of a run -- no new data is read to build it, only reorganized into the chain
shape spec section 27 pictures: model adapter -> training run -> preference run ->
evaluation run -> problem dataset version.
"""

from __future__ import annotations

from typing import Any

from .manifest import StageManifest


def build_lineage(upstream: dict[str, StageManifest]) -> dict[str, Any]:
    def stage_run_id(name: str) -> str | None:
        manifest = upstream.get(name)
        return manifest.stage_run_id if manifest is not None else None

    return {
        "model_adapter": {
            "training_run_id": stage_run_id("dpo_training"),
            "preference_run_id": stage_run_id("preference_generation"),
            "ranking_run_id": stage_run_id("candidate_evaluation"),
            "evaluation_run_id": stage_run_id("candidate_execution"),
            "candidate_run_id": stage_run_id("candidate_generation"),
            "problem_dataset_run_id": stage_run_id("problem_dataset"),
        },
        "model_evaluation_run_id": stage_run_id("model_evaluation"),
        "packaging_run_id": stage_run_id("packaging"),
    }


__all__ = ["build_lineage"]
