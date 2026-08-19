"""Tests for the experiment configuration and its four-level hierarchy
(spec 12 sections 8, 9, 10, 24, 25)."""

from __future__ import annotations

import pytest
import yaml

from python_dpo.pipeline.config import (
    ExperimentConfig,
    StageConfig,
    apply_overrides,
    apply_smoke_test,
)
from python_dpo.pipeline.errors import ExperimentConfigError
from python_dpo.pipeline.stages import STAGE_NAMES

MINIMAL_YAML = """
experiment:
  name: qwen-python-dpo-v1
  seed: 42

problem_dataset:
  enabled: true
  problem_count: 10

candidate_generation:
  enabled: true
  candidates_per_problem: 5

candidate_execution:
  enabled: true

candidate_evaluation:
  enabled: true

preference_generation:
  enabled: true
  policy: strict

dpo_training:
  enabled: true
  config: configs/training/dpo_qlora.yaml

model_evaluation:
  enabled: true
  benchmark: python_eval_v1
  num_samples: 10

error_analysis:
  enabled: false

packaging:
  enabled: true
"""


def write_config(tmp_path, text=MINIMAL_YAML):
    path = tmp_path / "experiment.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------------------ loading


def test_load_parses_every_stage_section(tmp_path):
    config = ExperimentConfig.load(write_config(tmp_path))
    assert config.name == "qwen-python-dpo-v1"
    assert config.seed == 42
    assert set(config.stages) == set(STAGE_NAMES)
    assert config.stage("problem_dataset").get("problem_count") == 10
    assert config.stage("error_analysis").enabled is False


def test_load_raises_on_missing_file(tmp_path):
    with pytest.raises(ExperimentConfigError, match="not found"):
        ExperimentConfig.load(tmp_path / "does_not_exist.yaml")


def test_load_raises_on_invalid_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("experiment: [this is not, valid: yaml", encoding="utf-8")
    with pytest.raises(ExperimentConfigError, match="invalid YAML"):
        ExperimentConfig.load(path)


def test_load_raises_when_a_stage_section_is_entirely_absent(tmp_path):
    # A section that never appears in the file is indistinguishable from an empty
    # mapping, so it fails the same way an empty section does: 'enabled' is required.
    text = MINIMAL_YAML.replace("packaging:\n  enabled: true\n", "")
    with pytest.raises(ExperimentConfigError, match="packaging.enabled is required"):
        ExperimentConfig.load(write_config(tmp_path, text))


def test_experiment_config_rejects_incomplete_stages_mapping_on_direct_construction():
    # Defends direct dataclass construction (bypassing from_mapping) against an
    # incomplete `stages` dict.
    from python_dpo.pipeline.config import StageConfig
    from python_dpo.pipeline.stages import STAGE_NAMES

    incomplete = {
        name: StageConfig(name=name, enabled=True)
        for name in STAGE_NAMES
        if name != "packaging"
    }
    with pytest.raises(ExperimentConfigError, match="missing stage section"):
        ExperimentConfig(name="x", stages=incomplete)


def test_load_raises_on_unknown_top_level_key(tmp_path):
    text = MINIMAL_YAML + "\nbogus_section:\n  x: 1\n"
    with pytest.raises(ExperimentConfigError, match="unknown top-level key"):
        ExperimentConfig.load(write_config(tmp_path, text))


def test_load_raises_when_stage_enabled_is_missing(tmp_path):
    text = MINIMAL_YAML.replace(
        "packaging:\n  enabled: true\n", "packaging:\n  some_key: 1\n"
    )
    with pytest.raises(ExperimentConfigError, match="enabled is required"):
        ExperimentConfig.load(write_config(tmp_path, text))


def test_enabled_stages_excludes_disabled_ones(tmp_path):
    config = ExperimentConfig.load(write_config(tmp_path))
    assert "error_analysis" not in config.enabled_stages()
    assert "dpo_training" in config.enabled_stages()


# ------------------------------------------------------------------------------ overrides


def test_set_override_takes_priority_over_the_file(tmp_path):
    config = ExperimentConfig.load(
        write_config(tmp_path), overrides=["dpo_training.beta=0.2"]
    )
    assert config.stage("dpo_training").get("beta") == 0.2


def test_set_override_parses_scalar_types():
    data = {"dpo_training": {"enabled": True}}
    result = apply_overrides(
        data,
        [
            "dpo_training.beta=0.2",
            "dpo_training.max_steps=5",
            "dpo_training.debug=true",
            "dpo_training.label=custom",
        ],
    )
    assert result["dpo_training"]["beta"] == 0.2
    assert result["dpo_training"]["max_steps"] == 5
    assert result["dpo_training"]["debug"] is True
    assert result["dpo_training"]["label"] == "custom"


def test_set_override_creates_missing_nested_keys():
    result = apply_overrides({}, ["a.b.c=1"])
    assert result == {"a": {"b": {"c": 1}}}


def test_set_override_does_not_mutate_input():
    data = {"a": {"b": 1}}
    apply_overrides(data, ["a.b=2"])
    assert data == {"a": {"b": 1}}


def test_set_override_rejects_malformed_entry():
    with pytest.raises(ExperimentConfigError, match="key.path=value"):
        apply_overrides({}, ["not-an-assignment"])


def test_smoke_test_reduces_scale_but_preserves_the_full_stage_set(tmp_path):
    config = ExperimentConfig.load(write_config(tmp_path), smoke_test=True)
    assert config.stage("problem_dataset").get("problem_count") == 3
    assert config.stage("candidate_generation").get("candidates_per_problem") == 2
    assert config.stage("dpo_training").get("max_steps") == 1
    assert config.stage("model_evaluation").get("num_samples") == 2
    assert set(config.stages) == set(STAGE_NAMES)  # nothing dropped, per section 25


def test_set_override_wins_over_smoke_test(tmp_path):
    config = ExperimentConfig.load(
        write_config(tmp_path),
        smoke_test=True,
        overrides=["model_evaluation.num_samples=7"],
    )
    assert config.stage("model_evaluation").get("num_samples") == 7


def test_apply_smoke_test_does_not_mutate_input():
    data = yaml.safe_load(MINIMAL_YAML)
    before = yaml.safe_dump(data)
    apply_smoke_test(data)
    assert yaml.safe_dump(data) == before


# --------------------------------------------------------------------------- round trip


def test_to_dict_round_trips_through_from_mapping(tmp_path):
    config = ExperimentConfig.load(write_config(tmp_path))
    restored = ExperimentConfig.from_mapping(config.to_dict())
    assert restored == config


def test_to_dict_round_trips_with_hypothesis_and_success_criteria():
    config = ExperimentConfig.from_mapping(
        {
            "experiment": {"name": "x", "seed": 1},
            "hypothesis": {"description": "DP coverage will help."},
            "success_criteria": {"pass_at_1_delta": 0.02},
            **{name: {"enabled": False} for name in STAGE_NAMES},
        }
    )
    restored = ExperimentConfig.from_mapping(config.to_dict())
    assert restored == config
    assert restored.hypothesis == "DP coverage will help."
    assert restored.success_criteria == {"pass_at_1_delta": 0.02}


# -------------------------------------------------------------------------- StageConfig


def test_stage_config_rejects_unknown_stage_name():
    with pytest.raises(ExperimentConfigError):
        StageConfig(name="not_a_stage", enabled=True)


def test_stage_config_get_returns_default_for_missing_key():
    config = StageConfig(name="dpo_training", enabled=True, settings={"beta": 0.1})
    assert config.get("missing", "fallback") == "fallback"
