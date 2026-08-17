"""Tests for run manifests, statistics, and the run repository (spec 04)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from python_dpo.candidates import Candidate, GenerationFailure, sha256_text, utc_now_iso
from python_dpo.runs import (
    RUN_STATUSES,
    RunError,
    RunFailure,
    RunManifest,
    RunNotFoundError,
    RunRepository,
    RunStatistics,
    capture_environment,
)

CODE = "def sum_even(numbers):\n    return 0"
PROMPT = "Solve the problem."
RAW_OUTPUT = f"```python\n{CODE}\n```"


def make_manifest(**overrides: Any) -> RunManifest:
    fields: dict[str, Any] = {
        "run_id": "run_20260817_120000_ab12",
        "status": "created",
        "created_at": utc_now_iso(),
        "candidate_schema_version": "2.0",
        "prompt_version": "v1",
        "model": {"provider": "mock", "name": "mock/deterministic-coder"},
        "generation_config": {"temperature": 0.8, "seed": 42},
        "strategies": ("normal", "optimized"),
        "requested_problem_ids": ("p001", "p002"),
        "requested_candidates_per_problem": 2,
        "retry": {"max_attempts": 2},
        "environment": {"python_version": "3.12.0"},
    }
    fields.update(overrides)
    return RunManifest(**fields)


def make_candidate(**overrides: Any) -> Candidate:
    fields: dict[str, Any] = {
        "candidate_id": "p001_c001",
        "problem_id": "p001",
        "run_id": "run_20260817_120000_ab12",
        "generation_index": 1,
        "strategy": "normal",
        "model": "mock/deterministic-coder",
        "provider": "mock",
        "prompt_version": "v1",
        "prompt": PROMPT,
        "raw_output": RAW_OUTPUT,
        "code": CODE,
        "extraction_format": "python_fence",
        "syntax_valid": True,
        "function_name_valid": True,
        "generation_config": {},
        "created_at": utc_now_iso(),
    }
    fields.update(overrides)
    return Candidate.create(**fields)


def make_failure(**overrides: Any) -> GenerationFailure:
    fields: dict[str, Any] = {
        "run_id": "run_20260817_120000_ab12",
        "problem_id": "p001",
        "generation_index": 2,
        "strategy": "optimized",
        "error_type": "inference",
        "error_message": "boom",
        "timestamp": utc_now_iso(),
        "prompt_sha256": sha256_text(PROMPT),
    }
    fields.update(overrides)
    return GenerationFailure(**fields)


# ------------------------------------------------------------------------------ manifest


def test_run_statuses_match_the_spec_closed_set():
    assert RUN_STATUSES == {
        "created",
        "running",
        "completed",
        "failed",
        "interrupted",
        "cancelled",
    }


def test_manifest_round_trips_through_dict():
    manifest = make_manifest()
    assert RunManifest.from_dict(manifest.to_dict()) == manifest


def test_manifest_rejects_unknown_status():
    with pytest.raises(RunError, match="status"):
        make_manifest(status="paused")


def test_manifest_requested_problems_is_derived():
    manifest = make_manifest()
    assert manifest.requested_problems == 2


def test_manifest_rejects_strategies_count_mismatch():
    with pytest.raises(RunError, match="strategies"):
        make_manifest(strategies=("normal",))


def test_manifest_rejects_duplicate_problem_ids():
    with pytest.raises(RunError, match="duplicate"):
        make_manifest(requested_problem_ids=("p001", "p001"))


def test_manifest_rejects_missing_retry_max_attempts():
    with pytest.raises(RunError, match="max_attempts"):
        make_manifest(retry={})


def test_manifest_with_status_enforces_allowed_transitions():
    manifest = make_manifest(status="created")
    running = manifest.with_status("running", started_at=utc_now_iso())
    assert running.status == "running"

    completed = running.with_status("completed", completed_at=utc_now_iso())
    assert completed.status == "completed"

    with pytest.raises(RunError, match="cannot transition"):
        completed.with_status("running")


def test_with_status_carries_an_error():
    manifest = make_manifest(status="running")
    error = RunFailure(error_type="model_load", error_message="no weights", timestamp=utc_now_iso())
    failed = manifest.with_status("failed", completed_at=utc_now_iso(), error=error)
    assert failed.error == error
    assert RunManifest.from_dict(failed.to_dict()).error == error


# ----------------------------------------------------------------------------- statistics


def test_statistics_round_trips():
    stats = RunStatistics.from_records(make_manifest(), [make_candidate()], [])
    assert RunStatistics.from_dict(stats.to_dict()) == stats


def test_statistics_matches_hand_counted_records():
    manifest = make_manifest(requested_problem_ids=("p001",), requested_candidates_per_problem=2)
    candidates = [
        make_candidate(candidate_id="p001_c001", generation_index=1, strategy="normal"),
    ]
    failures = [make_failure(problem_id="p001", generation_index=2, error_type="empty_output")]

    stats = RunStatistics.from_records(manifest, candidates, failures)

    assert stats.problems_requested == 1
    assert stats.candidates_requested == 2
    assert stats.candidates_generated == 1
    assert stats.generation_failures == 1
    assert stats.problems_completed == 1  # both indices have an outcome
    assert stats.syntax_valid == 1
    assert stats.syntax_invalid == 0
    assert stats.candidates_by_strategy == {"normal": 1}
    assert stats.failures_by_error_type == {"empty_output": 1}


def test_incomplete_problem_is_not_counted_as_completed():
    manifest = make_manifest(requested_problem_ids=("p001",), requested_candidates_per_problem=2)
    candidates = [make_candidate(candidate_id="p001_c001", generation_index=1)]
    stats = RunStatistics.from_records(manifest, candidates, [])
    assert stats.problems_completed == 0  # index 2 has neither a candidate nor a failure


def test_retry_attempts_count_infrastructure_failures_only():
    manifest = make_manifest(
        requested_problem_ids=("p001",),
        requested_candidates_per_problem=1,
        strategies=("normal",),
    )
    candidate = make_candidate(candidate_id="p001_c001", generation_index=1, attempt=2)
    retried_failure = make_failure(
        problem_id="p001", generation_index=1, error_type="inference", attempt=1
    )
    stats = RunStatistics.from_records(manifest, [candidate], [retried_failure])
    assert stats.retry_attempts == 1
    assert stats.candidates_generated == 1
    assert stats.generation_failures == 0  # the index eventually succeeded


def test_duplicates_are_counted():
    manifest = make_manifest(requested_problem_ids=("p001",), requested_candidates_per_problem=2)
    a = make_candidate(candidate_id="p001_c001", generation_index=1)
    b = make_candidate(candidate_id="p001_c002", generation_index=2, duplicate_of="p001_c001")
    stats = RunStatistics.from_records(manifest, [a, b], [])
    assert stats.duplicates == 1


def test_environment_never_records_username_home_or_token():
    env = capture_environment()
    dump = str(env).lower()
    for forbidden in ("home", "user", "token", "/root", "hf_token"):
        assert forbidden not in dump


# --------------------------------------------------------------------------- repository


def test_run_id_format(tmp_path):
    repo = RunRepository(Path(tmp_path))
    moment = datetime(2026, 8, 17, 13, 37, 0, tzinfo=timezone.utc)
    run_id = repo.new_run_id(moment)
    assert run_id.startswith("run_20260817_133700_")
    suffix = run_id.removeprefix("run_20260817_133700_")
    assert len(suffix) == 4
    int(suffix, 16)  # hex


def test_create_get_and_list_runs(tmp_path):
    repo = RunRepository(Path(tmp_path))
    manifest = repo.create_run(
        requested_problem_ids=["p001"],
        requested_candidates_per_problem=1,
        strategies=["normal"],
        model_config={"provider": "mock", "name": "mock/deterministic-coder"},
        generation_config={"temperature": 0.8},
        prompt_version="v1",
        retry={"max_attempts": 2},
    )
    assert manifest.status == "created"
    assert repo.get_run(manifest.run_id) == manifest
    assert [r.run_id for r in repo.list_runs()] == [manifest.run_id]


def test_get_run_raises_for_unknown_id(tmp_path):
    repo = RunRepository(Path(tmp_path))
    with pytest.raises(RunNotFoundError):
        repo.get_run("run_does_not_exist")


def test_list_runs_newest_first(tmp_path):
    repo = RunRepository(Path(tmp_path))
    first = repo.create_run(
        requested_problem_ids=["p001"],
        requested_candidates_per_problem=1,
        strategies=["normal"],
        model_config={"provider": "mock", "name": "x"},
        generation_config={"temperature": 0.8},
        prompt_version="v1",
        retry={"max_attempts": 1},
        run_id="run_20260817_100000_0001",
    )
    second = repo.create_run(
        requested_problem_ids=["p002"],
        requested_candidates_per_problem=1,
        strategies=["normal"],
        model_config={"provider": "mock", "name": "x"},
        generation_config={"temperature": 0.8},
        prompt_version="v1",
        retry={"max_attempts": 1},
        run_id="run_20260817_110000_0002",
    )
    assert [r.run_id for r in repo.list_runs()] == [second.run_id, first.run_id]


def test_status_lifecycle_created_running_completed(tmp_path):
    repo = RunRepository(Path(tmp_path))
    manifest = repo.create_run(
        requested_problem_ids=["p001"],
        requested_candidates_per_problem=1,
        strategies=["normal"],
        model_config={"provider": "mock", "name": "x"},
        generation_config={"temperature": 0.8},
        prompt_version="v1",
        retry={"max_attempts": 1},
    )
    running = repo.start_run(manifest.run_id)
    assert running.status == "running"
    assert running.started_at is not None

    completed = repo.complete_run(manifest.run_id)
    assert completed.status == "completed"
    assert completed.completed_at is not None


def test_interrupted_run_can_be_resumed(tmp_path):
    repo = RunRepository(Path(tmp_path))
    manifest = repo.create_run(
        requested_problem_ids=["p001"],
        requested_candidates_per_problem=1,
        strategies=["normal"],
        model_config={"provider": "mock", "name": "x"},
        generation_config={"temperature": 0.8},
        prompt_version="v1",
        retry={"max_attempts": 1},
    )
    repo.start_run(manifest.run_id)
    repo.interrupt_run(manifest.run_id)

    resumed = repo.resume_run(manifest.run_id)
    assert resumed.status == "running"


def test_resume_refuses_a_completed_run(tmp_path):
    repo = RunRepository(Path(tmp_path))
    manifest = repo.create_run(
        requested_problem_ids=["p001"],
        requested_candidates_per_problem=1,
        strategies=["normal"],
        model_config={"provider": "mock", "name": "x"},
        generation_config={"temperature": 0.8},
        prompt_version="v1",
        retry={"max_attempts": 1},
    )
    repo.start_run(manifest.run_id)
    repo.complete_run(manifest.run_id)

    with pytest.raises(RunError, match="already completed"):
        repo.resume_run(manifest.run_id)


def test_fail_run_records_the_error(tmp_path):
    repo = RunRepository(Path(tmp_path))
    manifest = repo.create_run(
        requested_problem_ids=["p001"],
        requested_candidates_per_problem=1,
        strategies=["normal"],
        model_config={"provider": "mock", "name": "x"},
        generation_config={"temperature": 0.8},
        prompt_version="v1",
        retry={"max_attempts": 1},
    )
    repo.start_run(manifest.run_id)
    failed = repo.fail_run(
        manifest.run_id,
        error_type="model_load",
        error_message="no weights",
        problem_id="p001",
        generation_index=1,
    )
    assert failed.status == "failed"
    assert failed.error.error_type == "model_load"
    assert failed.error.problem_id == "p001"


def test_create_run_from_seeds_a_new_run_with_a_fresh_id(tmp_path):
    repo = RunRepository(Path(tmp_path))
    original = repo.create_run(
        requested_problem_ids=["p001", "p002"],
        requested_candidates_per_problem=1,
        strategies=["normal"],
        model_config={"provider": "mock", "name": "x"},
        generation_config={"temperature": 0.8},
        prompt_version="v1",
        retry={"max_attempts": 1},
    )
    seeded = repo.create_run_from(original)
    assert seeded.run_id != original.run_id
    assert seeded.status == "created"
    assert seeded.requested_problem_ids == original.requested_problem_ids
    assert seeded.generation_config == original.generation_config


def test_write_and_read_statistics(tmp_path):
    repo = RunRepository(Path(tmp_path))
    manifest = repo.create_run(
        requested_problem_ids=["p001"],
        requested_candidates_per_problem=1,
        strategies=["normal"],
        model_config={"provider": "mock", "name": "x"},
        generation_config={"temperature": 0.8},
        prompt_version="v1",
        retry={"max_attempts": 1},
    )
    stats = RunStatistics.from_records(manifest, [], [])
    repo.write_statistics(stats)
    assert repo.read_statistics(manifest.run_id) == stats
    assert repo.read_statistics("run_missing") is None


def test_candidates_returns_a_repository_scoped_to_the_run_dir(tmp_path):
    repo = RunRepository(Path(tmp_path))
    manifest = repo.create_run(
        requested_problem_ids=["p001"],
        requested_candidates_per_problem=1,
        strategies=["normal"],
        model_config={"provider": "mock", "name": "x"},
        generation_config={"temperature": 0.8},
        prompt_version="v1",
        retry={"max_attempts": 1},
    )
    candidates_repo = repo.candidates(manifest.run_id)
    candidates_repo.save(make_candidate(run_id=manifest.run_id))
    assert candidates_repo.directory == repo.run_dir(manifest.run_id)
    assert repo.candidates(manifest.run_id).count() == 1
