"""Tests for GPU-hour cost accounting (spec 12 sections 52, 53)."""

from __future__ import annotations

from python_dpo.pipeline.cost import GPU_STAGES, compute_cost


def test_compute_cost_sums_only_gpu_stages():
    durations = {
        "problem_dataset": 5.0,
        "candidate_execution": 100.0,  # Docker, not GPU
        "dpo_training": 3600.0,
        "model_evaluation": 1800.0,
    }
    cost = compute_cost(durations)

    assert cost.gpu_hours == 1.5
    assert cost.gpu_hours_by_stage == {"dpo_training": 1.0, "model_evaluation": 0.5}


def test_compute_cost_with_no_gpu_stages_is_zero():
    cost = compute_cost({"problem_dataset": 5.0, "candidate_execution": 10.0})
    assert cost.gpu_hours == 0.0
    assert cost.gpu_hours_by_stage == {}


def test_compute_cost_records_an_explicit_empty_llm_api_provider_list():
    cost = compute_cost({})
    assert cost.to_dict()["llm_api"] == {"providers": []}


def test_gpu_stages_excludes_pure_computation_and_docker_stages():
    assert "problem_dataset" not in GPU_STAGES
    assert "candidate_execution" not in GPU_STAGES
    assert "candidate_evaluation" not in GPU_STAGES
    assert "preference_generation" not in GPU_STAGES
    assert "error_analysis" not in GPU_STAGES
