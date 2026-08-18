"""Tests for the evaluation configuration section (spec 06 sections 33-35, 53)."""

from __future__ import annotations

from typing import Any

import pytest

from python_dpo.config import Config, ConfigError
from python_dpo.evaluation import EvaluationConfig, EvaluationConfigError


def make_config(**overrides: Any) -> EvaluationConfig:
    return EvaluationConfig(**overrides)


def test_defaults_match_the_specification():
    config = EvaluationConfig()
    assert config.image == "python-dpo-evaluator:1.0"
    assert config.timeout_seconds == 30
    assert config.startup_grace_seconds == 10
    assert config.auto_pull is False


def test_from_mapping_applies_defaults_for_absent_keys():
    config = EvaluationConfig.from_mapping({"timeout_seconds": 60})
    assert config.timeout_seconds == 60
    assert config.image == "python-dpo-evaluator:1.0"


def test_from_mapping_accepts_none_as_an_absent_section():
    assert EvaluationConfig.from_mapping(None) == EvaluationConfig()


def test_from_mapping_rejects_unknown_keys():
    with pytest.raises(EvaluationConfigError, match="unknown key"):
        EvaluationConfig.from_mapping({"gpus": 1})


def test_from_mapping_rejects_a_non_mapping():
    with pytest.raises(EvaluationConfigError, match="expected a mapping"):
        EvaluationConfig.from_mapping([1, 2])


# ------------------------------------------------------------------------- image pinning


def test_latest_tag_is_rejected():
    with pytest.raises(EvaluationConfigError, match="latest"):
        make_config(image="python-dpo-evaluator:latest")


def test_unpinned_image_without_a_tag_is_rejected():
    with pytest.raises(EvaluationConfigError, match="pinned"):
        make_config(image="python-dpo-evaluator")


@pytest.mark.parametrize(
    "image",
    ["python-dpo-evaluator:1.0", "python-dpo-evaluator@sha256:" + "a" * 64],
)
def test_pinned_images_are_accepted(image):
    assert make_config(image=image).image == image


# ----------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"image": ""}, "image"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"timeout_seconds": True}, "timeout_seconds"),
        ({"startup_grace_seconds": 0}, "startup_grace_seconds"),
        ({"auto_pull": "yes"}, "auto_pull"),
    ],
)
def test_invalid_values_are_rejected(overrides, match):
    with pytest.raises(EvaluationConfigError, match=match):
        make_config(**overrides)


# ---------------------------------------------------------------- integration with config.py


def test_real_config_yaml_exposes_an_evaluation_section():
    config = Config.load()
    assert isinstance(config.evaluation, EvaluationConfig)
    assert config.evaluation.image == "python-dpo-evaluator:1.0"


def test_config_loader_wraps_evaluation_errors_as_config_errors(tmp_path):
    # evaluation/ must not import python_dpo.config, so it raises EvaluationConfigError;
    # the loader is what turns that into the ConfigError the rest of the CLI expects.
    bad = tmp_path / "config.yaml"
    bad.write_text(
        "project:\n  name: x\n"
        "paths:\n  raw_data: data/raw\n  problems: data/problems\n"
        "  candidates: data/candidates\n  evaluations: data/evaluations\n"
        "  rankings: data/rankings\n"
        "  preferences: data/preferences\n  training: data/training\n"
            "  reports: data/reports\n"
        "logging:\n  level: INFO\n"
        "model:\n  provider: mock\n  name: mock/x\n"
        "generation:\n  candidates_per_problem: 1\n"
        "evaluation:\n  image: python-dpo-evaluator:latest\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="latest"):
        Config.load(path=bad)
