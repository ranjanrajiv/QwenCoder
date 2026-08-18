"""Tests for the Stage 10 evaluation configuration (spec section 140)."""

from __future__ import annotations

import pytest

from python_dpo.model_evaluation.config import (
    EvaluationExperimentConfig,
    GenerationSettings,
    ModelEvaluationConfigError,
    StatisticsSettings,
)


def test_defaults_load_without_a_file():
    config = EvaluationExperimentConfig()
    assert config.benchmark.name == "python_eval_v1"
    assert config.generation.num_samples == 10
    assert config.statistics.pass_at_k == (1, 5, 10)


def test_pass_at_10_requires_at_least_10_samples():
    """Spec section 43: requesting pass@10 with num_samples < 10 is a config error."""
    with pytest.raises(ModelEvaluationConfigError, match="pass@10"):
        EvaluationExperimentConfig(
            generation=GenerationSettings(num_samples=5),
            statistics=StatisticsSettings(pass_at_k=(1, 5, 10)),
        )


def test_pass_at_k_within_num_samples_is_fine():
    config = EvaluationExperimentConfig(
        generation=GenerationSettings(num_samples=5),
        statistics=StatisticsSettings(pass_at_k=(1, 5)),
    )
    assert config.statistics.pass_at_k == (1, 5)


def test_real_config_file_loads(tmp_path):
    path = tmp_path / "python_eval.yaml"
    path.write_text(
        "benchmark:\n  name: python_eval_v1\n"
        "generation:\n  temperature: 0.2\n  num_samples: 10\n"
        "statistics:\n  pass_at_k: [1, 5, 10]\n",
        encoding="utf-8",
    )
    config = EvaluationExperimentConfig.load(path)
    assert config.generation.temperature == 0.2
    assert config.generation.num_samples == 10


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(ModelEvaluationConfigError, match="not found"):
        EvaluationExperimentConfig.load(tmp_path / "missing.yaml")


def test_unknown_top_level_key_rejected():
    with pytest.raises(ModelEvaluationConfigError, match="unknown"):
        EvaluationExperimentConfig.from_mapping({"not_a_real_section": {}})


def test_generation_do_sample_requires_positive_temperature():
    with pytest.raises(ModelEvaluationConfigError):
        GenerationSettings(do_sample=True, temperature=0.0)


def test_generation_config_round_trips_through_to_dict():
    config = EvaluationExperimentConfig()
    rebuilt = EvaluationExperimentConfig.from_mapping(config.to_dict())
    assert rebuilt == config


def test_with_overrides_applies_num_samples():
    config = EvaluationExperimentConfig()
    overridden = config.with_overrides(num_samples=15)
    assert overridden.generation.num_samples == 15
    assert config.generation.num_samples == 10  # original untouched


def test_with_overrides_rejects_unknown_key():
    config = EvaluationExperimentConfig()
    with pytest.raises(ModelEvaluationConfigError):
        config.with_overrides(not_a_real_override=1)


def test_success_criteria_defaults():
    config = EvaluationExperimentConfig()
    assert config.success_criteria.minimum_pass_at_1_improvement == 0.02
    assert config.success_criteria.maximum_allowed_regression == 0.02
