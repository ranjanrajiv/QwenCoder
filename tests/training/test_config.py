"""Tests for the training experiment configuration (spec 09 sections 62, 63, 87)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from python_dpo.training.config import DEFAULT_CONFIG_PATH, ExperimentConfig
from python_dpo.training.errors import TrainingConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MINIMAL: dict[str, Any] = {"model": {"name": "Qwen/Qwen2.5-Coder-3B-Instruct"}}


def write_config(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------- defaults


def test_minimal_config_supplies_every_default():
    config = ExperimentConfig.from_mapping(MINIMAL)
    assert config.model.name == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert config.quantization.bits == 4
    assert config.quantization.quant_type == "nf4"
    assert config.lora.r == 16
    assert config.lora.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert config.dpo.beta == 0.1
    assert config.training.max_length == 1024
    assert config.optimizer.name == "paged_adamw_8bit"
    assert config.distributed.enabled is False


def test_effective_batch_size_is_batch_times_accumulation():
    config = ExperimentConfig.from_mapping(MINIMAL)
    assert config.effective_batch_size == 1 * 8


def test_model_section_is_required():
    with pytest.raises(TrainingConfigError, match="model"):
        ExperimentConfig.from_mapping({})


def test_the_shipped_config_loads():
    """The committed configs/training/dpo_qlora.yaml must always be valid."""
    config = ExperimentConfig.load(PROJECT_ROOT / DEFAULT_CONFIG_PATH)
    assert config.experiment_name
    assert config.lora.r > 0


# -------------------------------------------------------------------------- validation


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(TrainingConfigError, match="unknown key"):
        ExperimentConfig.from_mapping({**MINIMAL, "nonsense": 1})


def test_unknown_nested_key_is_rejected():
    with pytest.raises(TrainingConfigError, match="unknown key"):
        ExperimentConfig.from_mapping({**MINIMAL, "lora": {"r": 8, "bogus": 1}})


def test_distributed_enabled_is_rejected():
    # Spec sections 86, 87: the key may exist, but Step 9 must not train distributed.
    with pytest.raises(TrainingConfigError, match="single GPU"):
        ExperimentConfig.from_mapping({**MINIMAL, "distributed": {"enabled": True}})


def test_empty_lora_target_modules_is_rejected():
    with pytest.raises(TrainingConfigError, match="target_modules"):
        ExperimentConfig.from_mapping({**MINIMAL, "lora": {"target_modules": []}})


def test_unsupported_dpo_loss_is_rejected():
    with pytest.raises(TrainingConfigError, match="loss_type"):
        ExperimentConfig.from_mapping({**MINIMAL, "dpo": {"loss_type": "ipo"}})


def test_unsupported_quant_type_is_rejected():
    with pytest.raises(TrainingConfigError, match="quant_type"):
        ExperimentConfig.from_mapping({**MINIMAL, "quantization": {"quant_type": "int4"}})


def test_max_prompt_length_must_be_below_max_length():
    with pytest.raises(TrainingConfigError, match="max_prompt_length"):
        ExperimentConfig.from_mapping(
            {**MINIMAL, "training": {"max_length": 512, "max_prompt_length": 512}}
        )


def test_zero_max_steps_is_rejected():
    with pytest.raises(TrainingConfigError, match="max_steps"):
        ExperimentConfig.from_mapping({**MINIMAL, "training": {"max_steps": 0}})


def test_negative_one_max_steps_means_use_epochs():
    config = ExperimentConfig.from_mapping({**MINIMAL, "training": {"max_steps": -1}})
    assert config.training.max_steps == -1


def test_warmup_ratio_must_be_a_fraction():
    with pytest.raises(TrainingConfigError, match="warmup_ratio"):
        ExperimentConfig.from_mapping({**MINIMAL, "training": {"warmup_ratio": 1.5}})


def test_bad_scheduler_is_rejected():
    with pytest.raises(TrainingConfigError, match="lr_scheduler_type"):
        ExperimentConfig.from_mapping(
            {**MINIMAL, "training": {"lr_scheduler_type": "quadratic"}}
        )


def test_nonpositive_learning_rate_is_rejected():
    with pytest.raises(TrainingConfigError, match="learning_rate"):
        ExperimentConfig.from_mapping({**MINIMAL, "training": {"learning_rate": 0}})


# ------------------------------------------------------------------------------- load


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(TrainingConfigError, match="not found"):
        ExperimentConfig.load(tmp_path / "nope.yaml")


def test_load_invalid_yaml_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model: [unclosed", encoding="utf-8")
    with pytest.raises(TrainingConfigError, match="invalid YAML"):
        ExperimentConfig.load(path)


def test_load_non_mapping_root_raises(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(TrainingConfigError, match="mapping"):
        ExperimentConfig.load(path)


def test_load_round_trips_through_to_dict(tmp_path):
    original = ExperimentConfig.from_mapping(MINIMAL)
    path = write_config(tmp_path, original.to_dict())
    assert ExperimentConfig.load(path) == original


# -------------------------------------------------------------------------- overrides


def test_overrides_apply_and_leave_the_original_untouched():
    config = ExperimentConfig.from_mapping(MINIMAL)
    updated = config.with_overrides(learning_rate=2e-5, beta=0.5, lora_r=8)
    assert updated.training.learning_rate == 2e-5
    assert updated.dpo.beta == 0.5
    assert updated.lora.r == 8
    assert config.training.learning_rate == 1.0e-5  # unchanged
    assert config.lora.r == 16


def test_none_overrides_are_ignored():
    config = ExperimentConfig.from_mapping(MINIMAL)
    assert config.with_overrides(learning_rate=None, beta=None) == config


def test_override_routes_preference_run_id_into_the_dataset_section():
    config = ExperimentConfig.from_mapping(MINIMAL)
    updated = config.with_overrides(preference_run_id="pref_x")
    assert updated.dataset.preference_run_id == "pref_x"


def test_unknown_override_is_rejected():
    config = ExperimentConfig.from_mapping(MINIMAL)
    with pytest.raises(TrainingConfigError, match="unknown configuration override"):
        config.with_overrides(nonsense=1)


def test_an_override_cannot_bypass_validation():
    # Overrides re-run the full schema, so a flag can no more produce an invalid config
    # than the YAML could.
    config = ExperimentConfig.from_mapping(MINIMAL)
    with pytest.raises(TrainingConfigError, match="learning_rate"):
        config.with_overrides(learning_rate=-1.0)
