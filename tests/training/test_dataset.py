"""Tests for preference dataset loading and validation (spec 09 sections 21-27, 101-103).

Spec section 24's nine checks get one test each, driven from fixtures built in ``tmp_path``
rather than from the committed data, so each failure mode is exercised in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from python_dpo.training.dataset import load_training_dataset
from python_dpo.training.errors import DatasetValidationError

RECORD = {"prompt": "Write a function.", "chosen": "def f(): return 1", "rejected": "def f(): return 2"}


def build_run(
    tmp_path: Path,
    *,
    train: list[dict[str, Any]] | None = None,
    validation: list[dict[str, Any]] | None = None,
    test: list[dict[str, Any]] | None = None,
    split_manifest: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    omit: str | None = None,
    metadata: list[dict[str, Any]] | None = None,
) -> Path:
    run_dir = tmp_path / "pref_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": [dict(RECORD)] if train is None else train,
        "validation": [dict(RECORD, prompt="Another.")] if validation is None else validation,
        "test": [dict(RECORD, prompt="Third.")] if test is None else test,
    }
    for name, rows in splits.items():
        if omit == name:
            continue
        path = run_dir / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(r) + "\n" for r in rows) if rows else "", encoding="utf-8"
        )

    (run_dir / "manifest.json").write_text(
        json.dumps(
            manifest
            or {
                "preference_run_id": "pref_x",
                "preference_version": "v1",
                "selection_policy": "all_better",
                "selection_policy_version": "all_better_v1",
                "dataset_schema_version": "dpo_preference_v1",
                "ranking_run_id": "rank_x",
                "evaluation_run_id": "eval_x",
                "candidate_run_id": "run_x",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "split_manifest.json").write_text(
        json.dumps(
            split_manifest
            or {
                "train_problem_ids": ["p001"],
                "validation_problem_ids": ["p002"],
                "test_problem_ids": ["p003"],
            }
        ),
        encoding="utf-8",
    )
    if metadata is not None:
        (run_dir / "metadata.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in metadata), encoding="utf-8"
        )
    return run_dir


# ------------------------------------------------------------------------- happy path


def test_a_valid_dataset_loads(tmp_path):
    dataset = load_training_dataset(build_run(tmp_path), min_training_pairs=0)
    assert dataset.preference_run_id == "pref_x"
    assert len(dataset.train) == 1
    assert len(dataset.validation) == 1
    assert dataset.test_count == 1
    assert dataset.has_validation


def test_the_test_split_is_not_exposed_as_records(tmp_path):
    """Spec sections 21, 22: there must be no attribute a caller could hand the trainer."""
    dataset = load_training_dataset(build_run(tmp_path), min_training_pairs=0)
    assert not hasattr(dataset, "test")
    assert dataset.test_count == 1
    assert "test" in dataset.split_hashes  # hashed for reproducibility, not loaded


def test_hashes_cover_all_three_splits_and_are_stable(tmp_path):
    run_dir = build_run(tmp_path)
    first = load_training_dataset(run_dir, min_training_pairs=0)
    second = load_training_dataset(run_dir, min_training_pairs=0)
    assert set(first.split_hashes) == {"train", "validation", "test"}
    assert first.split_hashes == second.split_hashes


def test_provenance_is_read_from_the_preference_manifest(tmp_path):
    dataset = load_training_dataset(build_run(tmp_path), min_training_pairs=0)
    assert dataset.provenance["selection_policy"] == "all_better"
    assert dataset.provenance["ranking_run_id"] == "rank_x"


# --------------------------------------------------------------- section 24's checks


def test_missing_run_directory_raises(tmp_path):
    with pytest.raises(DatasetValidationError, match="not found"):
        load_training_dataset(tmp_path / "nope")


def test_missing_split_file_raises(tmp_path):
    with pytest.raises(DatasetValidationError, match="not found"):
        load_training_dataset(build_run(tmp_path, omit="validation"), min_training_pairs=0)


def test_invalid_jsonl_raises(tmp_path):
    run_dir = build_run(tmp_path)
    (run_dir / "train.jsonl").write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="invalid JSON"):
        load_training_dataset(run_dir, min_training_pairs=0)


def test_missing_required_field_raises(tmp_path):
    run_dir = build_run(tmp_path, train=[{"prompt": "a", "chosen": "b"}])
    with pytest.raises(DatasetValidationError, match="missing required field"):
        load_training_dataset(run_dir, min_training_pairs=0)


@pytest.mark.parametrize("field", ["prompt", "chosen", "rejected"])
def test_empty_field_raises(tmp_path, field):
    run_dir = build_run(tmp_path, train=[dict(RECORD, **{field: "   "})])
    with pytest.raises(DatasetValidationError, match="must not be empty"):
        load_training_dataset(run_dir, min_training_pairs=0)


def test_identical_chosen_and_rejected_raises(tmp_path):
    run_dir = build_run(tmp_path, train=[dict(RECORD, rejected=RECORD["chosen"])])
    with pytest.raises(DatasetValidationError, match="identical"):
        load_training_dataset(run_dir, min_training_pairs=0)


def test_empty_train_split_is_always_fatal(tmp_path):
    run_dir = build_run(tmp_path, train=[])
    with pytest.raises(DatasetValidationError, match="training split is empty"):
        load_training_dataset(run_dir, allow_small_dataset=True, min_training_pairs=0)


def test_empty_validation_split_raises_by_default(tmp_path):
    run_dir = build_run(tmp_path, validation=[], split_manifest={
        "train_problem_ids": ["p001"], "validation_problem_ids": [], "test_problem_ids": ["p003"]
    })
    with pytest.raises(DatasetValidationError, match="validation split is empty"):
        load_training_dataset(run_dir, min_training_pairs=0)


def test_empty_validation_split_is_allowed_with_the_flag(tmp_path):
    run_dir = build_run(tmp_path, validation=[], split_manifest={
        "train_problem_ids": ["p001"], "validation_problem_ids": [], "test_problem_ids": ["p003"]
    })
    dataset = load_training_dataset(
        run_dir, allow_small_dataset=True, min_training_pairs=0
    )
    assert not dataset.has_validation


# ------------------------------------------------------------- section 101 (security)


@pytest.mark.parametrize("value", [123, ["def f(): pass"], {"code": "x"}, None])
def test_non_string_response_is_rejected(tmp_path, value):
    """Spec section 101: chosen/rejected must be ordinary text, never a structure the
    training loop might be tempted to interpret."""
    run_dir = build_run(tmp_path, train=[dict(RECORD, chosen=value)])
    with pytest.raises(DatasetValidationError, match="must be a string"):
        load_training_dataset(run_dir, min_training_pairs=0)


# ------------------------------------------------------------ section 102 (leakage)


def test_a_problem_in_two_splits_aborts(tmp_path):
    run_dir = build_run(
        tmp_path,
        split_manifest={
            "train_problem_ids": ["p001"],
            "validation_problem_ids": ["p001"],  # leaked
            "test_problem_ids": ["p003"],
        },
    )
    with pytest.raises(DatasetValidationError, match="problem-disjoint"):
        load_training_dataset(run_dir, min_training_pairs=0)


def test_train_test_overlap_aborts(tmp_path):
    run_dir = build_run(
        tmp_path,
        split_manifest={
            "train_problem_ids": ["p001"],
            "validation_problem_ids": ["p002"],
            "test_problem_ids": ["p001"],  # leaked into test
        },
    )
    with pytest.raises(DatasetValidationError, match="problem-disjoint"):
        load_training_dataset(run_dir, min_training_pairs=0)


# ----------------------------------------------------------- section 103 (small data)


def test_small_dataset_warns_but_loads(tmp_path, caplog):
    dataset = load_training_dataset(build_run(tmp_path), min_training_pairs=500)
    assert len(dataset.train) == 1
    assert any("suitable for pipeline validation" in r.message for r in caplog.records)


# --------------------------------------------------------- sections 105, 106 (balance)


def test_balance_is_read_from_stage_8_metadata(tmp_path):
    run_dir = build_run(
        tmp_path,
        metadata=[
            {
                "chosen_score": 1.0, "rejected_score": 0.5, "score_margin": 0.5,
                "chosen_pass_rate": 1.0, "rejected_pass_rate": 0.5,
                "preference_strength": "strong",
            },
            {
                "chosen_score": 0.8, "rejected_score": 0.6, "score_margin": 0.2,
                "chosen_pass_rate": 0.8, "rejected_pass_rate": 0.6,
                "preference_strength": "medium",
            },
        ],
    )
    dataset = load_training_dataset(run_dir, min_training_pairs=0)
    assert dataset.balance.pairs == 2
    assert dataset.balance.mean_score_margin == pytest.approx(0.35)
    assert dataset.balance.strong_pair_percentage == pytest.approx(50.0)


def test_absent_metadata_is_not_fatal(tmp_path):
    dataset = load_training_dataset(build_run(tmp_path), min_training_pairs=0)
    assert dataset.balance.pairs == 0


def test_incomplete_provenance_raises(tmp_path):
    run_dir = build_run(tmp_path, manifest={"preference_run_id": "pref_x"})
    with pytest.raises(DatasetValidationError, match="missing provenance"):
        load_training_dataset(run_dir, min_training_pairs=0)
