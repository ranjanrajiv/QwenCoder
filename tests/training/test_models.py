"""Tests for the training run schema (spec 09 sections 25, 26, 70, 94, 102)."""

from __future__ import annotations

from typing import Any

import pytest

from python_dpo.training.models import (
    DatasetManifest,
    FinalReport,
    TrainingManifest,
    TrainingModelError,
)


def make_dataset_manifest(**overrides: Any) -> DatasetManifest:
    fields: dict[str, Any] = {
        "preference_run_id": "pref_x",
        "preference_version": "v1",
        "selection_policy": "all_better",
        "selection_policy_version": "all_better_v1",
        "dataset_schema_version": "dpo_preference_v1",
        "ranking_run_id": "rank_x",
        "evaluation_run_id": "eval_x",
        "candidate_run_id": "run_x",
        "split_hashes": {"train": "a", "validation": "b", "test": "c"},
        "split_counts": {"train": 3, "validation": 2, "test": 2},
        "split_problem_ids": {
            "train": ["p007", "p008"],
            "validation": ["p010"],
            "test": ["p004"],
        },
    }
    fields.update(overrides)
    return DatasetManifest(**fields)


def make_manifest(**overrides: Any) -> TrainingManifest:
    fields: dict[str, Any] = {
        "training_run_id": "dpo_20260818_101500_a91f",
        "experiment_name": "qwen-python-dpo",
        "status": "created",
        "created_at": "2026-08-18T10:15:00Z",
        "model_name": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "preference_run_id": "pref_x",
        "ranking_run_id": "rank_x",
        "evaluation_run_id": "eval_x",
        "candidate_run_id": "run_x",
        "dataset_hashes": {"train": "a", "validation": "b", "test": "c"},
        "hardware": {"cuda_available": True},
        "environment": {"packages": {"torch": "2.13.0"}},
        "configuration": {"model": {"name": "Qwen/Qwen2.5-Coder-3B-Instruct"}},
        "seed": 42,
        "data_seed": 42,
        "trainer_version": "v1",
    }
    fields.update(overrides)
    return TrainingManifest(**fields)


def make_report(**overrides: Any) -> FinalReport:
    fields: dict[str, Any] = {
        "training_run_id": "dpo_20260818_101500_a91f",
        "experiment_name": "qwen-python-dpo",
        "status": "completed",
        "model_name": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "preference_run_id": "pref_x",
        "number_of_examples": {"train": 3, "validation": 2},
        "epochs": 1.0,
        "steps": 1,
        "trainable_parameters": 7_372_800,
        "total_parameters": 1_706_045_440,
        "effective_batch_size": 8,
        "optimizer": "paged_adamw_8bit",
        "compute_dtype": "bfloat16",
    }
    fields.update(overrides)
    return FinalReport(**fields)


# ------------------------------------------------------------------ DatasetManifest


def test_dataset_manifest_round_trips():
    manifest = make_dataset_manifest()
    assert DatasetManifest.from_dict(manifest.to_dict()) == manifest


def test_dataset_manifest_requires_all_three_split_hashes():
    with pytest.raises(TrainingModelError, match="missing split"):
        make_dataset_manifest(split_hashes={"train": "a"})


def test_dataset_manifest_rejects_overlapping_problem_ids():
    """Spec section 102, enforced in the record as well as at load time."""
    with pytest.raises(TrainingModelError, match="problem-disjoint"):
        make_dataset_manifest(
            split_problem_ids={
                "train": ["p001"],
                "validation": ["p001"],
                "test": ["p004"],
            }
        )


def test_dataset_manifest_rejects_unknown_field():
    data = make_dataset_manifest().to_dict()
    data["bogus"] = 1
    with pytest.raises(TrainingModelError, match="unknown field"):
        DatasetManifest.from_dict(data)


# ----------------------------------------------------------------- TrainingManifest


def test_manifest_round_trips():
    manifest = make_manifest()
    assert TrainingManifest.from_dict(manifest.to_dict()) == manifest


def test_manifest_rejects_unknown_status():
    with pytest.raises(TrainingModelError, match="status must be"):
        make_manifest(status="nonsense")


def test_manifest_rejects_unknown_mode():
    with pytest.raises(TrainingModelError, match="mode must be"):
        make_manifest(mode="nonsense")


@pytest.mark.parametrize("mode", ["dry_run", "smoke_test", "train"])
def test_every_mode_is_accepted(mode):
    assert make_manifest(mode=mode).mode == mode


def test_manifest_status_transitions():
    manifest = make_manifest()
    running = manifest.with_status("running", started_at="2026-08-18T10:15:01Z")
    assert running.status == "running"
    completed = running.with_status("completed", completed_at="2026-08-18T10:20:00Z")
    with pytest.raises(TrainingModelError, match="cannot transition"):
        completed.with_status("running")


def test_manifest_error_requires_the_core_fields():
    with pytest.raises(TrainingModelError, match="missing required field"):
        make_manifest(error={"error_type": "Boom"})


def test_manifest_error_accepts_traceback_and_last_step():
    manifest = make_manifest(
        error={
            "error_type": "OutOfMemoryError",
            "error_message": "CUDA OOM",
            "timestamp": "2026-08-18T10:16:00Z",
            "traceback": "Traceback...",
            "last_step": 3,
        }
    )
    assert manifest.error["last_step"] == 3


def test_manifest_rejects_missing_required_field():
    data = make_manifest().to_dict()
    del data["model_name"]
    with pytest.raises(TrainingModelError, match="missing required field"):
        TrainingManifest.from_dict(data)


# --------------------------------------------------------------------- FinalReport


def test_final_report_round_trips():
    report = make_report()
    assert FinalReport.from_dict(report.to_dict()) == report


def test_trainable_percentage_is_derived_not_stored():
    report = make_report()
    assert report.trainable_percentage == pytest.approx(0.4322, rel=1e-3)
    # Serialized for readers, but recomputed on load rather than trusted.
    assert "trainable_percentage" in report.to_dict()
    assert FinalReport.from_dict(report.to_dict()) == report


def test_trainable_cannot_exceed_total():
    with pytest.raises(TrainingModelError, match="cannot exceed"):
        make_report(trainable_parameters=10, total_parameters=5)


def test_reward_metrics_default_to_empty():
    assert make_report().reward_metrics == {}


def test_adapter_reload_defaults_to_not_verified():
    # Spec section 82: an adapter is not final until it has been reloaded.
    assert make_report().adapter_reload_ok is False
