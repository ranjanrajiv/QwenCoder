"""Tests for joining registered models to their recorded evaluation metrics (spec 12
section 49)."""

from __future__ import annotations

from python_dpo.model_evaluation.run_repository import ModelEvaluationRunRepository
from python_dpo.packaging.compare import compare_models
from python_dpo.packaging.registry import ModelRegistry, RegistryEntry


def make_entry(**overrides) -> RegistryEntry:
    fields = {
        "model_id": "exp_x",
        "status": "EXPERIMENTAL",
        "package_path": "models/packages/exp_x",
        "base_model_name": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "training_run_id": "dpo_x",
        "created_at": "2026-08-19T10:00:00Z",
        "verification": {},
    }
    fields.update(overrides)
    return RegistryEntry(**fields)


def test_compare_models_returns_one_row_per_entry_newest_first(tmp_path):
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    registry.register(make_entry(model_id="a", created_at="2026-08-19T09:00:00Z"))
    registry.register(make_entry(model_id="b", created_at="2026-08-19T11:00:00Z"))
    eval_repo = ModelEvaluationRunRepository(tmp_path / "model_evaluations" / "runs")

    rows = compare_models(registry, eval_repo)

    assert [row.model_id for row in rows] == ["b", "a"]


def test_compare_models_with_no_evaluation_run_has_null_metrics(tmp_path):
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    registry.register(make_entry())
    eval_repo = ModelEvaluationRunRepository(tmp_path / "model_evaluations" / "runs")

    [row] = compare_models(registry, eval_repo)

    assert row.evaluation_run_id is None
    assert row.pass_at_1 is None
    assert row.syntax_success_rate is None


def test_compare_models_pulls_pass_at_k_and_memory_from_the_recorded_evaluation(tmp_path):
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    registry.register(make_entry(evaluation_run_id="eval_x"))
    eval_repo = ModelEvaluationRunRepository(tmp_path / "model_evaluations" / "runs")
    eval_repo.write_metrics(
        "eval_x", "summary",
        {
            "pass_at_k": {"dpo": {"1": 0.4, "5": 0.6}},
            "syntax_success_rate": {"dpo": 0.95},
            "timeout_rate": {"dpo": 0.0},
        },
    )
    eval_repo.write_metrics("eval_x", "peak_gpu_memory", {"dpo": 12345})

    [row] = compare_models(registry, eval_repo)

    assert row.pass_at_1 == 0.4
    assert row.pass_at_5 == 0.6
    assert row.pass_at_10 is None
    assert row.syntax_success_rate == 0.95
    assert row.peak_gpu_memory_bytes == 12345
