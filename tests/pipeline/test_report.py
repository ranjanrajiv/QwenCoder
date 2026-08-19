"""Tests for the experiment-wide report (spec 12 sections 50, 102)."""

from __future__ import annotations

from python_dpo.pipeline.cost import compute_cost
from python_dpo.pipeline.manifest import StageManifest
from python_dpo.pipeline.report import (
    build_experiment_metrics,
    render_experiment_summary_md,
    render_model_comparison_md,
    render_next_experiment_md,
    write_experiment_reports,
)
from python_dpo.pipeline.repository import ExperimentRunRepository
from python_dpo.pipeline.resources import capture_resource_snapshot


def make_stage(name: str, *, status: str = "COMPLETED", reused: bool = False) -> StageManifest:
    return StageManifest(
        stage_name=name, stage_run_id=f"{name}_run", status=status, code_version="0.12.0",
        start_time="2026-08-19T10:00:00Z", end_time="2026-08-19T10:01:00Z", reused=reused,
    )


def make_experiment_manifest(repo: ExperimentRunRepository, *, name: str = "test"):
    return repo.create_run(experiment_name=name, configuration_hash="h" * 64)


def test_build_experiment_metrics_includes_stage_status_and_cost(tmp_path):
    repo = ExperimentRunRepository(tmp_path / "experiments" / "runs")
    manifest = make_experiment_manifest(repo)
    stages = {"problem_dataset": make_stage("problem_dataset"), "dpo_training": make_stage("dpo_training")}
    cost = compute_cost({"dpo_training": 3600.0})
    resources = capture_resource_snapshot()

    metrics = build_experiment_metrics(manifest, stages, cost, resources)

    assert metrics["experiment_run_id"] == manifest.experiment_run_id
    assert metrics["stages"]["dpo_training"]["status"] == "COMPLETED"
    assert metrics["stages"]["dpo_training"]["duration_seconds"] == 60.0
    assert metrics["cost"]["gpu_hours"] == 1.0


def test_render_experiment_summary_md_lists_every_declared_stage(tmp_path):
    repo = ExperimentRunRepository(tmp_path / "experiments" / "runs")
    manifest = make_experiment_manifest(repo)
    stages = {"problem_dataset": make_stage("problem_dataset")}
    cost = compute_cost({})

    text = render_experiment_summary_md(manifest, stages, cost)

    assert manifest.experiment_run_id in text
    assert "problem_dataset" in text
    # Every stage in the graph is listed, including ones with no manifest yet.
    assert "packaging" in text


def test_render_model_comparison_md_with_no_models():
    text = render_model_comparison_md([])
    assert "No models are registered" in text


def test_render_model_comparison_md_with_rows():
    from python_dpo.packaging.compare import ModelComparisonRow

    row = ModelComparisonRow(
        model_id="exp_x", status="EXPERIMENTAL", evaluation_run_id="eval_x",
        pass_at_1=0.4, pass_at_5=0.6, pass_at_10=None,
        syntax_success_rate=0.95, timeout_rate=0.0, peak_gpu_memory_bytes=1024,
    )
    text = render_model_comparison_md([row])
    assert "exp_x" in text
    assert "0.4" in text


def test_render_next_experiment_md_when_error_analysis_is_disabled():
    text = render_next_experiment_md(None)
    assert "No recommendation is available" in text
    assert "error_analysis" in text


def test_render_next_experiment_md_when_error_analysis_completed():
    stage = make_stage("error_analysis", status="COMPLETED")
    text = render_next_experiment_md(stage)
    assert "completed" in text
    assert stage.stage_run_id in text


def test_write_experiment_reports_writes_all_four_files(tmp_path):
    import dataclasses

    from python_dpo.config import Config, Paths

    base = Config.load()
    paths = Paths(
        raw=tmp_path / "raw", problems=tmp_path / "problems", candidates=tmp_path / "candidates",
        evaluations=tmp_path / "evaluations", rankings=tmp_path / "rankings",
        preferences=tmp_path / "preferences", training=tmp_path / "training",
        model_evaluations=tmp_path / "model_evaluations", experiments=tmp_path / "experiments",
        reports=tmp_path / "reports",
    )
    paths.ensure_exists()
    project_config = dataclasses.replace(base, paths=paths, project_root=tmp_path)

    repo = ExperimentRunRepository(project_config.paths.experiments / "runs")
    manifest = make_experiment_manifest(repo)
    stages = {"problem_dataset": make_stage("problem_dataset")}

    write_experiment_reports(repo, manifest.experiment_run_id, manifest, stages, project_config)

    reports_dir = repo.reports_dir(manifest.experiment_run_id)
    assert (reports_dir / "experiment_metrics.json").is_file()
    assert (reports_dir / "experiment_summary.md").is_file()
    assert (reports_dir / "model_comparison.md").is_file()
    assert (reports_dir / "next_experiment.md").is_file()
