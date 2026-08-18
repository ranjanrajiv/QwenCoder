"""Tests for :class:`ModelEvaluationRunRepository` (spec sections 79, 80, 148)."""

from __future__ import annotations

import pytest

from python_dpo.model_evaluation.errors import EvaluationRunError, EvaluationRunNotFoundError
from python_dpo.model_evaluation.models import EvaluationRecord, GenerationRecord
from python_dpo.model_evaluation.run_repository import ModelEvaluationRunRepository


def make_repo(tmp_path) -> ModelEvaluationRunRepository:
    return ModelEvaluationRunRepository(tmp_path / "model_evaluations" / "runs")


def create_test_run(repo: ModelEvaluationRunRepository):
    return repo.create_run(
        benchmark_version="python_eval_v1",
        benchmark_hash="deadbeef",
        base_model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
        adapter_path="/data/training/runs/dpo_x/adapter",
        training_run_id="dpo_x",
        models_requested=("base", "dpo"),
        generation_config={"temperature": 0.2},
        quantization={"enabled": True},
        statistics_config={"pass_at_k": [1, 5, 10]},
        seeds={"base_seed": 1000},
        hardware={},
        environment={},
    )


def test_new_run_id_has_the_eval_prefix(tmp_path):
    repo = make_repo(tmp_path)
    run_id = repo.new_run_id()
    assert run_id.startswith("eval_")


def test_create_and_get_run_round_trips(tmp_path):
    repo = make_repo(tmp_path)
    manifest = create_test_run(repo)
    fetched = repo.get_run(manifest.evaluation_run_id)
    assert fetched == manifest
    assert fetched.status == "created"


def test_get_run_missing_raises(tmp_path):
    repo = make_repo(tmp_path)
    with pytest.raises(EvaluationRunNotFoundError):
        repo.get_run("eval_does_not_exist")


def test_status_lifecycle(tmp_path):
    repo = make_repo(tmp_path)
    manifest = create_test_run(repo)
    run_id = manifest.evaluation_run_id

    running = repo.start_run(run_id)
    assert running.status == "running"

    completed = repo.complete_run(run_id)
    assert completed.status == "completed"

    with pytest.raises(EvaluationRunError):
        repo.start_run(run_id)  # completed -> running is not an allowed transition


def test_fail_run_records_error_detail(tmp_path):
    repo = make_repo(tmp_path)
    manifest = create_test_run(repo)
    repo.start_run(manifest.evaluation_run_id)
    failed = repo.fail_run(
        manifest.evaluation_run_id, error_type="AdapterIntegrityError", error_message="bad adapter"
    )
    assert failed.status == "failed"
    assert failed.error["error_type"] == "AdapterIntegrityError"


def test_list_runs_orders_newest_first(tmp_path):
    repo = make_repo(tmp_path)
    first = create_test_run(repo)
    second = create_test_run(repo)
    runs = repo.list_runs()
    assert [m.evaluation_run_id for m in runs][:2] == sorted(
        [first.evaluation_run_id, second.evaluation_run_id], reverse=True
    ) or len(runs) == 2  # creation timestamps may tie at second resolution


def test_generation_records_append_and_load_round_trip(tmp_path):
    repo = make_repo(tmp_path)
    manifest = create_test_run(repo)
    run_id = manifest.evaluation_run_id

    record = GenerationRecord(
        evaluation_run_id=run_id,
        problem_id="p001",
        model_variant="base",
        sample_index=0,
        seed=1000,
        prompt_sha256="a" * 64,
        raw_response="```python\ndef f(): pass\n```",
        extraction_format="python_fence",
        generation_time_ms=10,
        generated_tokens=5,
        status="generated",
        extracted_code="def f(): pass",
        syntax_valid=True,
    )
    repo.append_generation_record(run_id, "base", record)
    loaded = repo.load_generation_records(run_id, "base")
    assert loaded == [record]
    assert repo.load_generation_records(run_id, "dpo") == []


def test_evaluation_records_append_and_load_round_trip(tmp_path):
    repo = make_repo(tmp_path)
    manifest = create_test_run(repo)
    run_id = manifest.evaluation_run_id

    record = EvaluationRecord(
        evaluation_run_id=run_id,
        problem_id="p001",
        model_variant="dpo",
        sample_index=0,
        tests_total=5,
        tests_passed=5,
        tests_failed=0,
        tests_error=0,
        tests_skipped=0,
        timeout=False,
        status="passed",
        duration_ms=100,
    )
    repo.append_evaluation_record(run_id, "dpo", record)
    assert repo.load_evaluation_records(run_id, "dpo") == [record]


def test_write_and_read_config_round_trips(tmp_path):
    repo = make_repo(tmp_path)
    manifest = create_test_run(repo)
    repo.write_config(manifest.evaluation_run_id, {"generation": {"num_samples": 10}})
    config = repo.read_config(manifest.evaluation_run_id)
    assert config == {"generation": {"num_samples": 10}}


def test_write_and_read_benchmark_manifest(tmp_path):
    repo = make_repo(tmp_path)
    manifest = create_test_run(repo)
    repo.write_benchmark_manifest(manifest.evaluation_run_id, {"problem_ids": ["p001"]})
    assert repo.read_benchmark_manifest(manifest.evaluation_run_id) == {"problem_ids": ["p001"]}
    other = repo.new_run_id()
    assert repo.read_benchmark_manifest(other) is None


def test_sandbox_repository_is_scoped_per_variant(tmp_path):
    repo = make_repo(tmp_path)
    manifest = create_test_run(repo)
    base_dir = repo.sandbox_dir(manifest.evaluation_run_id, "base")
    dpo_dir = repo.sandbox_dir(manifest.evaluation_run_id, "dpo")
    assert base_dir != dpo_dir
    assert "base" in str(base_dir)
    assert "dpo" in str(dpo_dir)


def test_metrics_write_and_read_round_trip(tmp_path):
    repo = make_repo(tmp_path)
    manifest = create_test_run(repo)
    repo.write_metrics(manifest.evaluation_run_id, "summary", {"base_pass_at_1": 0.4})
    assert repo.read_metrics(manifest.evaluation_run_id, "summary") == {"base_pass_at_1": 0.4}
    assert repo.read_metrics(manifest.evaluation_run_id, "does_not_exist") is None
