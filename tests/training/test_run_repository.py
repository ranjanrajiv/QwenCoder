"""Tests for TrainingRunRepository (spec 09 sections 68-70, 90-92)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from python_dpo.training.errors import (
    CheckpointCompatibilityError,
    TrainingRunError,
    TrainingRunNotFoundError,
)
from python_dpo.training.run_repository import TrainingRunRepository, run_log_file

CONFIG: dict[str, Any] = {
    "model": {"name": "Qwen/Qwen2.5-Coder-3B-Instruct", "revision": None},
    "lora": {"r": 16, "alpha": 32, "target_modules": ["q_proj", "k_proj"]},
    "quantization": {"bits": 4, "quant_type": "nf4"},
}
HASHES = {"train": "a" * 64, "validation": "b" * 64, "test": "c" * 64}


def make_run_kwargs(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "experiment_name": "qwen-python-dpo",
        "model_name": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "model_revision": None,
        "tokenizer_revision": None,
        "preference_run_id": "pref_x",
        "ranking_run_id": "rank_x",
        "evaluation_run_id": "eval_x",
        "candidate_run_id": "run_x",
        "dataset_hashes": dict(HASHES),
        "hardware": {"cuda_available": True},
        "environment": {"packages": {"torch": "2.13.0"}},
        "configuration": CONFIG,
        "seed": 42,
        "data_seed": 42,
        "trainer_version": "v1",
    }
    fields.update(overrides)
    return fields


# -------------------------------------------------------------------------- run ids


def test_run_id_format(tmp_path):
    repo = TrainingRunRepository(tmp_path)
    run_id = repo.new_run_id(now=datetime(2026, 8, 18, 10, 15, 0, tzinfo=timezone.utc))
    assert run_id.startswith("dpo_20260818_101500_")
    assert len(run_id) == len("dpo_20260818_101500_") + 4


def test_run_ids_do_not_collide(tmp_path):
    repo = TrainingRunRepository(tmp_path)
    first = repo.create_run(**make_run_kwargs())
    now = datetime(2026, 8, 18, 10, 15, 0, tzinfo=timezone.utc)
    assert repo.new_run_id(now=now) != first.training_run_id


# ---------------------------------------------------------------- create / read


def test_create_and_get(tmp_path):
    repo = TrainingRunRepository(tmp_path)
    manifest = repo.create_run(**make_run_kwargs())
    assert repo.get_run(manifest.training_run_id) == manifest
    assert manifest.status == "created"


def test_get_missing_raises(tmp_path):
    with pytest.raises(TrainingRunNotFoundError):
        TrainingRunRepository(tmp_path).get_run("dpo_nope")


def test_list_runs_is_newest_first(tmp_path):
    repo = TrainingRunRepository(tmp_path)
    a = repo.create_run(**make_run_kwargs(training_run_id="dpo_20260818_100000_0001"))
    b = repo.create_run(**make_run_kwargs(training_run_id="dpo_20260818_110000_0002"))
    assert [m.training_run_id for m in repo.list_runs()] == [
        b.training_run_id,
        a.training_run_id,
    ]


def test_latest_checkpoint_picks_the_highest_number(tmp_path):
    repo = TrainingRunRepository(tmp_path)
    manifest = repo.create_run(**make_run_kwargs())
    checkpoints = repo.checkpoints_dir(manifest.training_run_id)
    for step in (1, 2, 10):
        (checkpoints / f"checkpoint-{step}").mkdir(parents=True)
    assert repo.latest_checkpoint(manifest.training_run_id).name == "checkpoint-10"


def test_latest_checkpoint_is_none_when_absent(tmp_path):
    repo = TrainingRunRepository(tmp_path)
    manifest = repo.create_run(**make_run_kwargs())
    assert repo.latest_checkpoint(manifest.training_run_id) is None


# -------------------------------------------------------------------- lifecycle


def test_status_lifecycle(tmp_path):
    repo = TrainingRunRepository(tmp_path)
    run_id = repo.create_run(**make_run_kwargs()).training_run_id
    assert repo.start_run(run_id).status == "running"
    assert repo.complete_run(run_id).status == "completed"
    with pytest.raises(TrainingRunError):
        repo.start_run(run_id)


def test_fail_run_records_the_diagnosis(tmp_path):
    """Spec section 82: error type, message, traceback and last step."""
    repo = TrainingRunRepository(tmp_path)
    run_id = repo.create_run(**make_run_kwargs()).training_run_id
    repo.start_run(run_id)
    manifest = repo.fail_run(
        run_id,
        error_type="OutOfMemoryError",
        error_message="CUDA OOM",
        traceback_text="Traceback...",
        last_step=7,
    )
    assert manifest.status == "failed"
    assert manifest.error["error_type"] == "OutOfMemoryError"
    assert manifest.error["last_step"] == 7
    assert "Traceback" in manifest.error["traceback"]


def test_interrupt_then_resume(tmp_path):
    repo = TrainingRunRepository(tmp_path)
    run_id = repo.create_run(**make_run_kwargs()).training_run_id
    repo.start_run(run_id)
    repo.interrupt_run(run_id)
    assert repo.start_run(run_id).status == "running"


# ----------------------------------------------------------- resume (90, 91, 92)


def test_identical_inputs_resume_cleanly(tmp_path):
    repo = TrainingRunRepository(tmp_path)
    run_id = repo.create_run(**make_run_kwargs()).training_run_id
    manifest = repo.check_resume_compatibility(
        run_id, dataset_hashes=dict(HASHES), configuration=CONFIG
    )
    assert manifest.training_run_id == run_id


def test_changed_dataset_refuses_resume(tmp_path):
    repo = TrainingRunRepository(tmp_path)
    run_id = repo.create_run(**make_run_kwargs()).training_run_id
    changed = dict(HASHES, train="z" * 64)
    with pytest.raises(CheckpointCompatibilityError, match="--force-resume"):
        repo.check_resume_compatibility(
            run_id, dataset_hashes=changed, configuration=CONFIG
        )


def test_force_resume_overrides_a_dataset_change_and_records_it(tmp_path):
    repo = TrainingRunRepository(tmp_path)
    run_id = repo.create_run(**make_run_kwargs()).training_run_id
    changed = dict(HASHES, train="z" * 64)
    repo.check_resume_compatibility(
        run_id, dataset_hashes=changed, configuration=CONFIG, force=True
    )
    override = repo.run_dir(run_id) / "resume_dataset_override.json"
    assert override.is_file()


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("model", "name", "Qwen/Other-Model"),
        ("lora", "r", 8),
        ("lora", "target_modules", ["q_proj"]),
        ("quantization", "bits", 8),
        ("quantization", "quant_type", "fp4"),
    ],
)
def test_critical_config_change_is_never_resumable(tmp_path, section, key, value):
    """Spec section 92: unlike a dataset change, these are not overridable — the saved
    state does not describe the same training problem."""
    repo = TrainingRunRepository(tmp_path)
    run_id = repo.create_run(**make_run_kwargs()).training_run_id
    changed = {**CONFIG, section: {**CONFIG[section], key: value}}
    with pytest.raises(CheckpointCompatibilityError, match="Critical configuration"):
        repo.check_resume_compatibility(
            run_id, dataset_hashes=dict(HASHES), configuration=changed, force=True
        )


def test_target_module_reordering_is_not_a_change(tmp_path):
    # YAML round-tripping can reorder a list; that must not read as an incompatibility.
    repo = TrainingRunRepository(tmp_path)
    run_id = repo.create_run(**make_run_kwargs()).training_run_id
    reordered = {**CONFIG, "lora": {**CONFIG["lora"], "target_modules": ["k_proj", "q_proj"]}}
    repo.check_resume_compatibility(
        run_id, dataset_hashes=dict(HASHES), configuration=reordered
    )


# ------------------------------------------------------------------------ artifacts


def test_write_config_is_yaml(tmp_path):
    repo = TrainingRunRepository(tmp_path)
    run_id = repo.create_run(**make_run_kwargs()).training_run_id
    repo.write_config(run_id, CONFIG)
    import yaml

    written = yaml.safe_load((repo.run_dir(run_id) / "config.yaml").read_text())
    assert written["model"]["name"] == CONFIG["model"]["name"]


def test_run_log_file_captures_and_detaches(tmp_path):
    import logging

    repo = TrainingRunRepository(tmp_path)
    run_id = repo.create_run(**make_run_kwargs()).training_run_id
    path = repo.log_path(run_id)
    logger = logging.getLogger("python_dpo.training.test")
    before = len(logging.getLogger("python_dpo").handlers)

    with run_log_file(path):
        logger.warning("inside the run")
    logger.warning("after the run")

    text = path.read_text(encoding="utf-8")
    assert "inside the run" in text
    assert "after the run" not in text
    # The handler must not accumulate across invocations.
    assert len(logging.getLogger("python_dpo").handlers) == before
