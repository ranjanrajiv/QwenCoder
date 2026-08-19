"""Tests for experiment reproduction (spec 12 sections 71, 72)."""

from __future__ import annotations

from pathlib import Path

from python_dpo.pipeline.environment import capture_environment
from python_dpo.pipeline.hashing import sha256_file
from python_dpo.pipeline.manifest import StageManifest
from python_dpo.pipeline.reproduce import (
    format_reproducibility_report,
    render_reproduce_commands,
    verify_reproducibility,
)
from python_dpo.pipeline.repository import ExperimentRunRepository


def test_render_reproduce_commands_points_at_the_resolved_config():
    text = render_reproduce_commands("exp_x", Path("data/experiments/runs/exp_x/resolved_config.yaml"))
    assert "exp_x" in text
    assert "experiment run --config" in text
    assert "resolved_config.yaml" in text


def test_verify_reproducibility_with_nothing_recorded_checks_nothing_but_environment(project_config, tmp_path):
    repo = ExperimentRunRepository(project_config.paths.experiments / "runs")
    manifest = repo.create_run(experiment_name="test", configuration_hash="h" * 64)

    report = verify_reproducibility(repo, manifest.experiment_run_id, project_config)

    assert report.config_hash_matches is None
    assert report.model_matches is None
    assert report.dataset_hash_matches is None


def test_verify_reproducibility_matches_when_the_recorded_environment_is_identical(project_config):
    repo = ExperimentRunRepository(project_config.paths.experiments / "runs")
    manifest = repo.create_run(experiment_name="test", configuration_hash="h" * 64)
    repo.write_environment(manifest.experiment_run_id, capture_environment())

    report = verify_reproducibility(repo, manifest.experiment_run_id, project_config)

    assert report.environment_diffs == {}
    assert report.reproducible is True


def test_verify_reproducibility_flags_a_python_version_drift(project_config):
    repo = ExperimentRunRepository(project_config.paths.experiments / "runs")
    manifest = repo.create_run(experiment_name="test", configuration_hash="h" * 64)
    recorded = dict(capture_environment())
    recorded["python_version"] = "3.9.0"
    repo.write_environment(manifest.experiment_run_id, recorded)

    report = verify_reproducibility(repo, manifest.experiment_run_id, project_config)

    assert "python_version" in report.environment_diffs
    assert report.environment_diffs["python_version"] == ("3.9.0", capture_environment()["python_version"])
    assert report.reproducible is False


def test_verify_reproducibility_checks_the_dataset_hash_against_the_recorded_stage_output(project_config):
    repo = ExperimentRunRepository(project_config.paths.experiments / "runs")
    manifest = repo.create_run(experiment_name="test", configuration_hash="h" * 64)

    from python_dpo.problems import dataset_path

    dataset_file = dataset_path(project_config.paths.problems)
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_file.write_text('{"id": "p001"}\n', encoding="utf-8")
    real_hash = sha256_file(dataset_file)

    repo.write_stage_manifest(
        manifest.experiment_run_id,
        StageManifest(
            stage_name="problem_dataset", stage_run_id="run_x", status="COMPLETED",
            code_version="0.12.0", output_artifacts={"problem_dataset": real_hash},
        ),
    )

    report = verify_reproducibility(repo, manifest.experiment_run_id, project_config)
    assert report.dataset_hash_matches is True

    dataset_file.write_text('{"id": "p001", "extra": true}\n', encoding="utf-8")
    report_after_change = verify_reproducibility(repo, manifest.experiment_run_id, project_config)
    assert report_after_change.dataset_hash_matches is False


def test_format_reproducibility_report_renders_mismatches(project_config):
    repo = ExperimentRunRepository(project_config.paths.experiments / "runs")
    manifest = repo.create_run(experiment_name="test", configuration_hash="h" * 64)
    recorded = dict(capture_environment())
    recorded["python_version"] = "3.9.0"
    repo.write_environment(manifest.experiment_run_id, recorded)

    report = verify_reproducibility(repo, manifest.experiment_run_id, project_config)
    text = format_reproducibility_report(report)

    assert "NOT REPRODUCIBLE" in text
    assert "python_version" in text
