"""Tests for preflight validation (spec 12 sections 60, 61)."""

from __future__ import annotations

from python_dpo.pipeline.config import ExperimentConfig
from python_dpo.pipeline.preflight import format_preflight_report, run_preflight
from python_dpo.problems import build_catalog, dataset_path, save_problems

from .conftest import full_experiment_mapping


def test_preflight_reports_all_eight_checks(project_config):
    experiment = ExperimentConfig.from_mapping(full_experiment_mapping())
    report = run_preflight(project_config, experiment)
    names = {check.name for check in report.checks}
    assert names == {
        "GPU",
        "CUDA",
        "Docker",
        "Model",
        "Dataset",
        "Benchmark",
        "Training configuration",
        "Evaluation configuration",
    }


def test_dataset_check_fails_when_no_dataset_exists(project_config):
    experiment = ExperimentConfig.from_mapping(full_experiment_mapping())
    report = run_preflight(project_config, experiment)
    dataset_check = next(c for c in report.checks if c.name == "Dataset")
    assert dataset_check.passed is False


def test_dataset_check_passes_once_the_dataset_is_built(project_config):
    problems = build_catalog()
    save_problems(problems, dataset_path(project_config.paths.problems))
    experiment = ExperimentConfig.from_mapping(full_experiment_mapping())
    report = run_preflight(project_config, experiment)
    dataset_check = next(c for c in report.checks if c.name == "Dataset")
    assert dataset_check.passed is True
    assert str(len(problems)) in dataset_check.detail


def test_model_check_passes_when_model_name_is_set(project_config):
    experiment = ExperimentConfig.from_mapping(full_experiment_mapping())
    report = run_preflight(project_config, experiment)
    model_check = next(c for c in report.checks if c.name == "Model")
    assert model_check.passed is True
    assert model_check.detail == project_config.model.name


def test_training_configuration_check_skips_when_stage_disabled(project_config):
    experiment = ExperimentConfig.from_mapping(
        full_experiment_mapping(enabled={"dpo_training": False})
    )
    report = run_preflight(project_config, experiment)
    check = next(c for c in report.checks if c.name == "Training configuration")
    assert check.passed is True


def test_training_configuration_check_fails_for_a_missing_config_path(project_config):
    experiment = ExperimentConfig.from_mapping(
        full_experiment_mapping(
            enabled={"dpo_training": True}
        )
    )
    experiment.stage("dpo_training").settings["config"] = "configs/training/does_not_exist.yaml"
    report = run_preflight(project_config, experiment)
    check = next(c for c in report.checks if c.name == "Training configuration")
    assert check.passed is False


def test_benchmark_check_skips_when_model_evaluation_disabled(project_config):
    experiment = ExperimentConfig.from_mapping(
        full_experiment_mapping(enabled={"model_evaluation": False})
    )
    report = run_preflight(project_config, experiment)
    check = next(c for c in report.checks if c.name == "Benchmark")
    assert check.passed is True


def test_format_preflight_report_uses_pass_fail_brackets(project_config):
    experiment = ExperimentConfig.from_mapping(
        full_experiment_mapping(enabled={n: False for n in ("dpo_training", "model_evaluation")})
    )
    text = format_preflight_report(run_preflight(project_config, experiment))
    assert "[PASS]" in text or "[FAIL]" in text
    assert "Dataset" in text


def test_report_passed_property_reflects_every_check():
    from python_dpo.pipeline.preflight import PreflightCheck, PreflightReport

    all_pass = PreflightReport((PreflightCheck("A", True, "ok"), PreflightCheck("B", True, "ok")))
    assert all_pass.passed is True
    assert all_pass.failures == ()

    one_fail = PreflightReport((PreflightCheck("A", True, "ok"), PreflightCheck("B", False, "no")))
    assert one_fail.passed is False
    assert one_fail.failures == (PreflightCheck("B", False, "no"),)
