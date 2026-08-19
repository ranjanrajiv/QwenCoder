"""The experiment-wide report (spec 12 sections 50, 102): ``experiment_metrics.json``,
``experiment_summary.md``, ``model_comparison.md``, ``next_experiment.md``.

Written once, at the end of a successful :meth:`~python_dpo.pipeline.orchestrator.PipelineOrchestrator.run`
(the plan's "one command ... an experiment report" outcome) -- never regenerated
automatically otherwise, matching how Stage 10's own ``base_vs_dpo`` report is a one-shot
artifact of its run. ``model_comparison.md`` reuses :func:`python_dpo.packaging.compare.compare_models`
rather than re-deriving anything (the plan's "assembly of existing parts" theme); imported
lazily here so this module stays a thin consumer, not a new dependency edge baked into
:mod:`python_dpo.pipeline`'s own import graph.
"""

from __future__ import annotations

from typing import Any

from ..atomic_io import atomic_write_json
from .cost import CostReport, compute_cost
from .manifest import ExperimentManifest, StageManifest
from .repository import ExperimentRunRepository
from .resources import ResourceSnapshot, capture_resource_snapshot, stage_durations
from .stages import STAGE_NAMES


def build_experiment_metrics(
    manifest: ExperimentManifest,
    stage_manifests: dict[str, StageManifest],
    cost: CostReport,
    resources: ResourceSnapshot,
) -> dict[str, Any]:
    durations = stage_durations(stage_manifests)
    return {
        "experiment_run_id": manifest.experiment_run_id,
        "experiment_name": manifest.experiment_name,
        "status": manifest.status,
        "start_time": manifest.start_time,
        "end_time": manifest.end_time,
        "git_commit": manifest.git_commit,
        "stages": {
            name: {
                "status": sm.status,
                "reused": sm.reused,
                "stage_run_id": sm.stage_run_id,
                "duration_seconds": durations.get(name),
            }
            for name, sm in stage_manifests.items()
        },
        "cost": cost.to_dict(),
        "resources": resources.to_dict(),
    }


def render_experiment_summary_md(
    manifest: ExperimentManifest, stage_manifests: dict[str, StageManifest], cost: CostReport
) -> str:
    lines = [f"# Experiment {manifest.experiment_run_id}", ""]
    lines.append(f"**Name:** {manifest.experiment_name}  ")
    lines.append(f"**Status:** {manifest.status}  ")
    if manifest.git_commit:
        sha = manifest.git_commit.get("sha") or "unknown"
        dirty = "dirty" if manifest.git_commit.get("dirty") else "clean"
        lines.append(f"**Git commit:** {sha} ({dirty})  ")
    lines.append(f"**Started:** {manifest.start_time or '-'}  ")
    lines.append(f"**Ended:** {manifest.end_time or '-'}  ")
    lines.append("")

    lines.append("## Stages")
    lines.append("")
    lines.append("| Stage | Status | Reused | Stage run id |")
    lines.append("|---|---|---|---|")
    for name in STAGE_NAMES:
        sm = stage_manifests.get(name)
        if sm is None:
            lines.append(f"| {name} | - | - | - |")
        else:
            lines.append(f"| {name} | {sm.status} | {sm.reused} | {sm.stage_run_id} |")
    lines.append("")

    lines.append("## Cost")
    lines.append("")
    lines.append(f"GPU-hours: {cost.gpu_hours}")
    if cost.gpu_hours_by_stage:
        lines.append("")
        for name, hours in sorted(cost.gpu_hours_by_stage.items()):
            lines.append(f"- {name}: {hours}")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_model_comparison_md(rows: Any) -> str:
    lines = ["# Model Comparison", ""]
    if not rows:
        lines.append("No models are registered yet.")
        return "\n".join(lines) + "\n"

    lines.append("| Model | Status | pass@1 | pass@5 | pass@10 | Syntax OK | Timeout |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row.model_id} | {row.status} | {row.pass_at_1} | {row.pass_at_5} | "
            f"{row.pass_at_10} | {row.syntax_success_rate} | {row.timeout_rate} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def render_next_experiment_md(error_analysis_manifest: StageManifest | None) -> str:
    """Section 102: what to try next. Stage 11 (``src/python_dpo/analysis/``) is where
    that recommendation is actually computed (evidence-backed, no LLM judge) -- this
    module has no analysis logic of its own and never fabricates one. When Stage 11 has
    not run, that absence is reported plainly rather than papered over.
    """
    lines = ["# Next Experiment", ""]
    if error_analysis_manifest is None or error_analysis_manifest.status == "SKIPPED":
        lines.append(
            "No recommendation is available: the `error_analysis` stage did not run in "
            "this experiment (it is disabled by default until Stage 11 -- "
            "`src/python_dpo/analysis/` -- is implemented; see "
            "`.claude/plans/11_error_analysis_and_iteration_plan.md`)."
        )
    elif error_analysis_manifest.status != "COMPLETED":
        lines.append(
            f"No recommendation is available: the `error_analysis` stage ended in status "
            f"`{error_analysis_manifest.status}`."
        )
    else:
        lines.append(
            "The `error_analysis` stage completed "
            f"(run id `{error_analysis_manifest.stage_run_id}`). See its own "
            "`next_experiment.yaml` under `data/analysis/runs/` for the evidence-backed "
            "recommendation; this file does not duplicate Stage 11's own report."
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def write_experiment_reports(
    repo: ExperimentRunRepository,
    experiment_run_id: str,
    manifest: ExperimentManifest,
    stage_manifests: dict[str, StageManifest],
    project_config: Any,
) -> None:
    """Write all four report artifacts under ``reports/`` (spec section 50)."""
    from ..model_evaluation.run_repository import ModelEvaluationRunRepository
    from ..packaging.compare import compare_models
    from ..packaging.registry import ModelRegistry

    durations = stage_durations(stage_manifests)
    cost = compute_cost(durations)
    resources = capture_resource_snapshot()

    reports_dir = repo.reports_dir(experiment_run_id)
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics = build_experiment_metrics(manifest, stage_manifests, cost, resources)
    atomic_write_json(reports_dir / "experiment_metrics.json", metrics)

    (reports_dir / "experiment_summary.md").write_text(
        render_experiment_summary_md(manifest, stage_manifests, cost), encoding="utf-8"
    )

    registry = ModelRegistry(project_config.project_root / "models" / "registry.json")
    eval_repo = ModelEvaluationRunRepository(project_config.paths.model_evaluations / "runs")
    rows = compare_models(registry, eval_repo)
    (reports_dir / "model_comparison.md").write_text(
        render_model_comparison_md(rows), encoding="utf-8"
    )

    (reports_dir / "next_experiment.md").write_text(
        render_next_experiment_md(stage_manifests.get("error_analysis")), encoding="utf-8"
    )


__all__ = [
    "build_experiment_metrics",
    "render_experiment_summary_md",
    "render_model_comparison_md",
    "render_next_experiment_md",
    "write_experiment_reports",
]
