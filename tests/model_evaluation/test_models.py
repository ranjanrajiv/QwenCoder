"""Schema round-trip tests for Stage 10 records (spec sections 81-83)."""

from __future__ import annotations

import pytest

from python_dpo.model_evaluation.models import (
    EvaluationRecord,
    GenerationRecord,
    MetricsSummary,
    ModelEvaluationManifest,
    ModelEvaluationModelError,
)


def make_generation_record(**overrides) -> GenerationRecord:
    fields = dict(
        evaluation_run_id="eval_x",
        problem_id="p001",
        model_variant="base",
        sample_index=0,
        seed=1000,
        prompt_sha256="a" * 64,
        raw_response="```python\ndef solve(x):\n    return x\n```",
        extraction_format="python_fence",
        generation_time_ms=100,
        generated_tokens=20,
        status="generated",
        extracted_code="def solve(x):\n    return x",
        syntax_valid=True,
    )
    fields.update(overrides)
    return GenerationRecord(**fields)


def test_generation_record_round_trips():
    record = make_generation_record()
    assert GenerationRecord.from_dict(record.to_dict()) == record


def test_generation_record_candidate_id():
    assert make_generation_record(sample_index=4).candidate_id == "p001_c005"


def test_generation_error_record_requires_no_code():
    record = GenerationRecord(
        evaluation_run_id="eval_x",
        problem_id="p001",
        model_variant="dpo",
        sample_index=0,
        seed=1000,
        prompt_sha256="a" * 64,
        raw_response="no code here",
        extraction_format="unknown",
        generation_time_ms=50,
        generated_tokens=10,
        status="generation_error",
        error="No Python code detected",
    )
    assert record.extracted_code is None
    assert record.syntax_valid is None


def test_generated_status_requires_extracted_code():
    with pytest.raises(ModelEvaluationModelError):
        make_generation_record(extracted_code=None)


def test_generation_error_status_rejects_extracted_code():
    with pytest.raises(ModelEvaluationModelError):
        GenerationRecord(
            evaluation_run_id="eval_x",
            problem_id="p001",
            model_variant="base",
            sample_index=0,
            seed=1000,
            prompt_sha256="a" * 64,
            raw_response="text",
            extraction_format="unknown",
            generation_time_ms=1,
            generated_tokens=1,
            status="generation_error",
            extracted_code="def f(): pass",
            error="oops",
        )


def test_generation_record_rejects_unknown_variant():
    with pytest.raises(ModelEvaluationModelError):
        make_generation_record(model_variant="strategy_x")


def make_evaluation_record(**overrides) -> EvaluationRecord:
    fields = dict(
        evaluation_run_id="eval_x",
        problem_id="p001",
        model_variant="base",
        sample_index=0,
        tests_total=5,
        tests_passed=5,
        tests_failed=0,
        tests_error=0,
        tests_skipped=0,
        timeout=False,
        status="passed",
        duration_ms=200,
    )
    fields.update(overrides)
    return EvaluationRecord(**fields)


def test_evaluation_record_round_trips():
    record = make_evaluation_record()
    assert EvaluationRecord.from_dict(record.to_dict()) == record


def test_evaluation_record_correct_requires_exact_pass():
    nine_of_ten = make_evaluation_record(
        tests_total=10, tests_passed=9, tests_failed=1, status="failed", error_type="assertion_failure"
    )
    assert nine_of_ten.correct is False

    ten_of_ten = make_evaluation_record(tests_total=10, tests_passed=10)
    assert ten_of_ten.correct is True


def test_evaluation_record_infrastructure_error_is_excluded_flag():
    record = make_evaluation_record(
        tests_total=0, tests_passed=0, status="infrastructure_error", error_type="infrastructure_error"
    )
    assert record.is_infrastructure_error is True
    assert record.correct is False


def test_evaluation_record_counts_must_sum_to_total():
    with pytest.raises(ModelEvaluationModelError):
        make_evaluation_record(tests_total=5, tests_passed=3, tests_failed=3)


def test_evaluation_record_passed_requires_null_error_type():
    with pytest.raises(ModelEvaluationModelError):
        make_evaluation_record(status="passed", error_type="assertion_failure")


def test_evaluation_record_failed_requires_error_type():
    with pytest.raises(ModelEvaluationModelError):
        make_evaluation_record(
            status="failed", tests_passed=3, tests_failed=2, error_type=None
        )


def test_manifest_round_trips():
    manifest = ModelEvaluationManifest(
        evaluation_run_id="eval_20260818_140000_a1b2",
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
        hardware={"gpu_name": "RTX 3060"},
        environment={"python_version": "3.12.3"},
        created_at="2026-08-18T14:00:00Z",
    )
    assert ModelEvaluationManifest.from_dict(manifest.to_dict()) == manifest


def test_manifest_status_lifecycle():
    manifest = ModelEvaluationManifest(
        evaluation_run_id="eval_x",
        benchmark_version="python_eval_v1",
        benchmark_hash="hash",
        base_model_name="model",
        adapter_path="/adapter",
        training_run_id="dpo_x",
        models_requested=("base",),
        generation_config={},
        quantization={},
        statistics_config={},
        seeds={},
        hardware={},
        environment={},
        created_at="2026-08-18T14:00:00Z",
    )
    running = manifest.with_status("running", started_at="2026-08-18T14:00:01Z")
    assert running.status == "running"
    completed = running.with_status("completed", completed_at="2026-08-18T14:05:00Z")
    assert completed.status == "completed"

    with pytest.raises(ModelEvaluationModelError):
        completed.with_status("running")


def test_metrics_summary_flattens_pass_at_k_into_named_keys():
    """Spec section 83's exact field names: base_pass_at_1, dpo_pass_at_1, ..."""
    summary = MetricsSummary(
        evaluation_run_id="eval_x",
        pass_at_k={"base": {"1": 0.4, "5": 0.6}, "dpo": {"1": 0.5, "5": 0.7}},
        test_pass_rate={"base": 0.5, "dpo": 0.6},
        solve_rate={"base": 0.4, "dpo": 0.5},
        syntax_success_rate={"base": 0.9, "dpo": 0.95},
        execution_success_rate={"base": 0.8, "dpo": 0.85},
        timeout_rate={"base": 0.02, "dpo": 0.01},
        generation_failure_rate={"base": 0.0, "dpo": 0.0},
        test_failure_distribution={"base": {}, "dpo": {}},
        computed_at="2026-08-18T14:00:00Z",
    )
    flat = summary.to_dict()
    assert flat["base_pass_at_1"] == 0.4
    assert flat["dpo_pass_at_1"] == 0.5
    assert flat["base_test_pass_rate"] == 0.5
    assert flat["dpo_timeout_rate"] == 0.01
    assert flat["base_syntax_success"] == 0.9
