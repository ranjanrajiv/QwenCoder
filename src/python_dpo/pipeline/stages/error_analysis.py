"""Stage 11 as a pipeline stage: error analysis (spec 12 section 5, item 8).

Delegates to :mod:`python_dpo.analysis`, which is pure computation over the artifacts the
upstream stages already persisted -- no model, no GPU, no Docker. That makes this the one
stage in the back half of the pipeline whose adapter can run anywhere the rest of the
package installs.

The stage consumes ``model_evaluation``'s run and produces an analysis run; per spec 11
sections 5 and 113 it emits ``next_experiment.yaml`` and stops, never retraining.
"""

from __future__ import annotations

from ...analysis import AnalysisError, run_analysis
from ..errors import StageFailedError
from ..hashing import sha256_tree
from ._context import StageContext, StageResult


def run(context: StageContext) -> StageResult:
    config = context.project_config
    settings = context.stage_config

    evaluation_run_id = context.upstream_run_id("model_evaluation")

    config_path = settings.get("config")
    analysis_config = None
    if config_path:
        from ...analysis.config import AnalysisConfig

        from pathlib import Path

        analysis_config = AnalysisConfig.load(Path(config_path))

    try:
        analysis_run_id, _summary = run_analysis(
            config, evaluation_run_id, analysis_config=analysis_config
        )
    except AnalysisError as exc:
        raise StageFailedError(f"error analysis failed: {exc}") from exc

    run_dir = (config.paths.analysis / "runs" / analysis_run_id)
    return StageResult(
        stage_run_id=analysis_run_id,
        output_artifacts={"error_analysis": sha256_tree(run_dir)},
    )


__all__ = ["run"]
