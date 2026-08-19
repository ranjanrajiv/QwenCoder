"""Tests for the experiment and stage manifest schemas (spec 12 sections 14, 28, 65, 83)."""

from __future__ import annotations

import pytest

from python_dpo.pipeline.manifest import (
    ExperimentManifest,
    ManifestError,
    StageError,
    StageManifest,
    StageRunSummary,
)


# --------------------------------------------------------------------------- StageError


def test_stage_error_round_trip():
    error = StageError(
        stage="dpo_training",
        error_type="training_failure",
        message="CUDA out of memory",
        timestamp="2026-08-19T10:00:00Z",
        stack_trace="Traceback ...",
        input_artifacts={"preference_dataset": "a" * 64},
    )
    assert StageError.from_dict(error.to_dict()) == error


def test_stage_error_rejects_unknown_field():
    with pytest.raises(ManifestError, match="unknown field"):
        StageError.from_dict(
            {
                "stage": "x",
                "error_type": "y",
                "message": "z",
                "timestamp": "t",
                "bogus": 1,
            }
        )


def test_stage_error_rejects_missing_required_field():
    with pytest.raises(ManifestError, match="missing required field"):
        StageError.from_dict({"stage": "x", "error_type": "y", "message": "z"})


# ------------------------------------------------------------------------- StageManifest


def make_stage_manifest(**overrides):
    fields = dict(
        stage_name="problem_dataset",
        stage_run_id="exp_x_problem_dataset",
        status="PENDING",
        code_version="0.12.0",
    )
    fields.update(overrides)
    return StageManifest(**fields)


def test_stage_manifest_round_trip():
    manifest = make_stage_manifest(
        status="COMPLETED",
        start_time="t0",
        end_time="t1",
        input_artifacts={"problems": "a" * 64},
        output_artifacts={"problems": "b" * 64},
        configuration_hash="c" * 64,
        cache_key="d" * 64,
        reused=True,
    )
    assert StageManifest.from_dict(manifest.to_dict()) == manifest


def test_stage_manifest_rejects_unknown_stage_name():
    with pytest.raises(ManifestError, match="stage_name"):
        make_stage_manifest(stage_name="not_a_real_stage")


def test_stage_manifest_rejects_unknown_status():
    with pytest.raises(ManifestError, match="status"):
        make_stage_manifest(status="NOT_A_STATE")


def test_stage_manifest_rejects_non_string_artifact_hash():
    with pytest.raises(ManifestError):
        make_stage_manifest(output_artifacts={"problems": 123})


def test_stage_manifest_with_status_validates_the_transition():
    manifest = make_stage_manifest(status="PENDING")
    running = manifest.with_status("RUNNING", start_time="t0")
    assert running.status == "RUNNING"
    assert running.start_time == "t0"

    from python_dpo.pipeline.state import StateError

    with pytest.raises(StateError):
        manifest.with_status("COMPLETED")  # PENDING -> COMPLETED is illegal


def test_stage_manifest_with_status_preserves_unset_fields():
    manifest = make_stage_manifest(status="PENDING", configuration_hash="abc")
    running = manifest.with_status("RUNNING", start_time="t0")
    assert running.configuration_hash == "abc"
    assert running.stage_run_id == manifest.stage_run_id


def test_stage_manifest_from_dict_rejects_unknown_field():
    data = make_stage_manifest().to_dict()
    data["bogus"] = 1
    with pytest.raises(ManifestError, match="unknown field"):
        StageManifest.from_dict(data)


def test_stage_manifest_from_dict_rejects_missing_required_field():
    data = make_stage_manifest().to_dict()
    del data["code_version"]
    with pytest.raises(ManifestError, match="missing required field"):
        StageManifest.from_dict(data)


def test_stage_manifest_with_embedded_error_round_trips():
    error = StageError(
        stage="dpo_training",
        error_type="training_failure",
        message="boom",
        timestamp="t",
    )
    manifest = make_stage_manifest(status="RUNNING").with_status(
        "FAILED", end_time="t1", error=error
    )
    assert StageManifest.from_dict(manifest.to_dict()).error == error


# --------------------------------------------------------------------- StageRunSummary


def test_stage_run_summary_round_trip():
    summary = StageRunSummary(status="COMPLETED", stage_run_id="run_x", reused=True)
    assert StageRunSummary.from_dict(summary.to_dict()) == summary


def test_stage_run_summary_rejects_unknown_status():
    with pytest.raises(ManifestError):
        StageRunSummary(status="NOT_A_STATE")


# ----------------------------------------------------------------------- ExperimentManifest


def make_experiment_manifest(**overrides):
    fields = dict(
        experiment_run_id="exp_20260819_120000_a1b2",
        experiment_name="qwen-python-dpo-v1",
        status="created",
        configuration_hash="e" * 64,
    )
    fields.update(overrides)
    return ExperimentManifest(**fields)


def test_experiment_manifest_round_trip():
    manifest = make_experiment_manifest(
        status="running",
        start_time="t0",
        git_commit={"sha": "abc123", "branch": "main", "dirty": False},
        dataset_versions={"problem_dataset": "a" * 64},
        model_versions={"base_model": "Qwen/Qwen2.5-Coder-3B-Instruct@main"},
        stage_runs={
            "problem_dataset": StageRunSummary(status="COMPLETED", stage_run_id="x", reused=False)
        },
        final_model={"path": "model/", "sha256": "f" * 64},
        final_evaluation={"pass_at_1": 0.5},
        recommendation="Generate more DP-focused preference data.",
    )
    assert ExperimentManifest.from_dict(manifest.to_dict()) == manifest


def test_experiment_manifest_rejects_unknown_status():
    with pytest.raises(ManifestError, match="status"):
        make_experiment_manifest(status="not_a_status")


def test_experiment_manifest_rejects_non_string_dataset_version():
    with pytest.raises(ManifestError):
        make_experiment_manifest(dataset_versions={"problem_dataset": 123})


def test_experiment_manifest_from_dict_rejects_unknown_field():
    data = make_experiment_manifest().to_dict()
    data["bogus"] = 1
    with pytest.raises(ManifestError, match="unknown field"):
        ExperimentManifest.from_dict(data)


def test_experiment_manifest_from_dict_rejects_missing_required_field():
    data = make_experiment_manifest().to_dict()
    del data["configuration_hash"]
    with pytest.raises(ManifestError, match="missing required field"):
        ExperimentManifest.from_dict(data)


def test_experiment_manifest_with_status_legal_transition():
    manifest = make_experiment_manifest(status="created")
    running = manifest.with_status("running", start_time="t0")
    assert running.status == "running"
    completed = running.with_status("completed", end_time="t1")
    assert completed.status == "completed"


def test_experiment_manifest_with_status_illegal_transition_raises():
    manifest = make_experiment_manifest(status="created")
    with pytest.raises(ManifestError, match="illegal"):
        manifest.with_status("completed")  # created -> completed skips running
