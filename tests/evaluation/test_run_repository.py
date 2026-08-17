"""Tests for EvaluationRunRepository (spec 06 sections 48-51, 56-57)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from python_dpo.evaluation import (
    EvaluationRepository,
    EvaluationResult,
    EvaluationRunError,
    EvaluationRunNotFoundError,
    EvaluationRunRepository,
    EvaluationStatistics,
)

SANDBOX_CONFIG = {"image": "python-dpo-evaluator:1.0", "network_mode": "none"}


def make_manifest_kwargs(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "candidate_run_id": "run_20260817_055411",
        "evaluator_version": "v1",
        "test_generator_version": "v1",
        "pytest_version": "8.3.4",
        "python_version": "3.12.7",
        "sandbox_config": SANDBOX_CONFIG,
        "requested_candidate_ids": ["p001_c001"],
    }
    fields.update(overrides)
    return fields


# ----------------------------------------------------------------------------- run ids


def test_run_id_format(tmp_path):
    repo = EvaluationRunRepository(Path(tmp_path))
    moment = datetime(2026, 8, 17, 13, 37, 0, tzinfo=timezone.utc)
    run_id = repo.new_run_id(moment)
    assert run_id.startswith("eval_20260817_133700_")
    suffix = run_id.removeprefix("eval_20260817_133700_")
    assert len(suffix) == 4
    int(suffix, 16)  # hex


# ---------------------------------------------------------------------- create/get/list


def test_create_get_and_list_runs(tmp_path):
    repo = EvaluationRunRepository(Path(tmp_path))
    manifest = repo.create_run(**make_manifest_kwargs())

    assert manifest.status == "created"
    assert repo.get_run(manifest.evaluation_run_id) == manifest
    assert [m.evaluation_run_id for m in repo.list_runs()] == [manifest.evaluation_run_id]


def test_get_run_raises_for_unknown_id(tmp_path):
    repo = EvaluationRunRepository(Path(tmp_path))
    with pytest.raises(EvaluationRunNotFoundError):
        repo.get_run("eval_does_not_exist")


def test_list_runs_newest_first(tmp_path):
    repo = EvaluationRunRepository(Path(tmp_path))
    first = repo.create_run(**make_manifest_kwargs(evaluation_run_id="eval_20260817_100000_0001"))
    second = repo.create_run(**make_manifest_kwargs(evaluation_run_id="eval_20260817_110000_0002"))
    assert [m.evaluation_run_id for m in repo.list_runs()] == [
        second.evaluation_run_id,
        first.evaluation_run_id,
    ]


def test_latest_run_for_candidate_run_returns_the_most_recent_match(tmp_path):
    repo = EvaluationRunRepository(Path(tmp_path))
    repo.create_run(
        **make_manifest_kwargs(
            candidate_run_id="run_A", evaluation_run_id="eval_20260817_100000_0001"
        )
    )
    latest = repo.create_run(
        **make_manifest_kwargs(
            candidate_run_id="run_A", evaluation_run_id="eval_20260817_110000_0002"
        )
    )
    repo.create_run(
        **make_manifest_kwargs(
            candidate_run_id="run_B", evaluation_run_id="eval_20260817_120000_0003"
        )
    )

    found = repo.latest_run_for_candidate_run("run_A")
    assert found.evaluation_run_id == latest.evaluation_run_id


def test_latest_run_for_candidate_run_is_none_when_never_evaluated(tmp_path):
    repo = EvaluationRunRepository(Path(tmp_path))
    assert repo.latest_run_for_candidate_run("run_never_evaluated") is None


# -------------------------------------------------------------------------- lifecycle


def test_status_lifecycle_created_running_completed(tmp_path):
    repo = EvaluationRunRepository(Path(tmp_path))
    manifest = repo.create_run(**make_manifest_kwargs())

    running = repo.start_run(manifest.evaluation_run_id)
    assert running.status == "running"
    assert running.started_at is not None

    completed = repo.complete_run(manifest.evaluation_run_id)
    assert completed.status == "completed"
    assert completed.completed_at is not None


def test_interrupted_run_can_be_resumed(tmp_path):
    repo = EvaluationRunRepository(Path(tmp_path))
    manifest = repo.create_run(**make_manifest_kwargs())
    repo.start_run(manifest.evaluation_run_id)
    repo.interrupt_run(manifest.evaluation_run_id)

    resumed = repo.resume_run(manifest.evaluation_run_id)
    assert resumed.status == "running"


def test_resume_refuses_a_completed_run(tmp_path):
    repo = EvaluationRunRepository(Path(tmp_path))
    manifest = repo.create_run(**make_manifest_kwargs())
    repo.start_run(manifest.evaluation_run_id)
    repo.complete_run(manifest.evaluation_run_id)

    with pytest.raises(EvaluationRunError, match="already completed"):
        repo.resume_run(manifest.evaluation_run_id)


def test_fail_run_records_the_error(tmp_path):
    repo = EvaluationRunRepository(Path(tmp_path))
    manifest = repo.create_run(**make_manifest_kwargs())
    repo.start_run(manifest.evaluation_run_id)

    failed = repo.fail_run(
        manifest.evaluation_run_id, error_type="infrastructure", error_message="docker died"
    )
    assert failed.status == "failed"
    assert failed.error["error_type"] == "infrastructure"


def test_cancel_run(tmp_path):
    repo = EvaluationRunRepository(Path(tmp_path))
    manifest = repo.create_run(**make_manifest_kwargs())
    cancelled = repo.cancel_run(manifest.evaluation_run_id)
    assert cancelled.status == "cancelled"


# --------------------------------------------------------------------------- statistics


def test_write_and_read_statistics(tmp_path):
    repo = EvaluationRunRepository(Path(tmp_path))
    manifest = repo.create_run(**make_manifest_kwargs())

    stats = EvaluationStatistics.from_records(manifest, [], [])
    repo.write_statistics(stats)

    assert repo.read_statistics(manifest.evaluation_run_id) == stats
    assert repo.read_statistics("eval_missing") is None


# --------------------------------------------------------------------------- results()


def test_results_returns_a_repository_scoped_to_the_run_dir(tmp_path):
    repo = EvaluationRunRepository(Path(tmp_path))
    manifest = repo.create_run(**make_manifest_kwargs())

    results_repo = repo.results(manifest.evaluation_run_id)
    assert isinstance(results_repo, EvaluationRepository)
    assert results_repo.directory == repo.run_dir(manifest.evaluation_run_id)


def test_a_result_saved_via_results_is_visible_after_reload(tmp_path):
    repo = EvaluationRunRepository(Path(tmp_path))
    manifest = repo.create_run(**make_manifest_kwargs())

    result = EvaluationResult.create(
        evaluation_run_id=manifest.evaluation_run_id,
        candidate_run_id=manifest.candidate_run_id,
        candidate_id="p001_c001",
        problem_id="p001",
        status="passed",
        tests_passed=3,
        tests_failed=0,
        tests_error=0,
        tests_skipped=0,
        duration_ms=42,
    )
    repo.results(manifest.evaluation_run_id).save(result)

    reloaded = EvaluationRunRepository(Path(tmp_path)).results(manifest.evaluation_run_id)
    assert reloaded.load_all() == [result]
