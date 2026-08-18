"""Tests for the sandbox configuration section (spec 05 section 53)."""

from __future__ import annotations

from typing import Any

import pytest

from python_dpo.config import Config, ConfigError
from python_dpo.sandbox import SandboxConfig, SandboxConfigError

DIGEST = "sha256:" + "a" * 64


def make_config(**overrides: Any) -> SandboxConfig:
    return SandboxConfig(**overrides)


def test_defaults_match_the_specification():
    config = SandboxConfig()
    assert config.image == "python:3.12-slim"
    assert config.network_mode == "none"
    assert config.cpus == 1.0
    assert config.memory == "512m"
    assert config.pids_limit == 64
    assert config.timeout_seconds == 5
    assert config.max_output_bytes == 1_000_000
    assert config.read_only_root is True
    assert config.run_as_non_root is True
    assert config.drop_capabilities is True


def test_from_mapping_applies_defaults_for_absent_keys():
    config = SandboxConfig.from_mapping({"memory": "256m"})
    assert config.memory == "256m"
    assert config.pids_limit == 64


def test_from_mapping_accepts_none_as_an_absent_section():
    assert SandboxConfig.from_mapping(None) == SandboxConfig()


def test_from_mapping_rejects_unknown_keys():
    with pytest.raises(SandboxConfigError, match="unknown key"):
        SandboxConfig.from_mapping({"gpus": 1})


def test_from_mapping_rejects_a_non_mapping():
    with pytest.raises(SandboxConfigError, match="expected a mapping"):
        SandboxConfig.from_mapping([1, 2])


# ------------------------------------------------------------------------- image pinning


def test_latest_tag_is_rejected():
    # Spec section 9: :latest can change underneath us, silently changing what every
    # recorded result means.
    with pytest.raises(SandboxConfigError, match="latest"):
        make_config(image="python:latest")


def test_unpinned_image_without_a_tag_is_rejected():
    with pytest.raises(SandboxConfigError, match="pinned"):
        make_config(image="python")


@pytest.mark.parametrize("image", ["python:3.12-slim", "python:3.13.1-slim-bookworm", f"python@{DIGEST}"])
def test_pinned_images_are_accepted(image):
    assert make_config(image=image).image == image


def test_image_reference_prefers_the_digest_when_configured():
    config = make_config(image="python:3.12-slim", image_digest=DIGEST)
    assert config.image_reference == f"python@{DIGEST}"


def test_image_reference_falls_back_to_the_tag():
    assert make_config(image="python:3.12-slim").image_reference == "python:3.12-slim"


def test_malformed_digest_is_rejected():
    with pytest.raises(SandboxConfigError, match="image_digest"):
        make_config(image_digest="sha256:short")


# ----------------------------------------------------------------------- network is a MUST


def test_only_network_mode_none_is_accepted():
    # Spec section 12 makes network isolation mandatory, so a value we would have to
    # ignore is rejected rather than silently honoured.
    with pytest.raises(SandboxConfigError, match="network_mode"):
        make_config(network_mode="bridge")
    with pytest.raises(SandboxConfigError, match="network_mode"):
        make_config(network_mode="host")


# ----------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"cpus": 0}, "cpus"),
        ({"cpus": -1}, "cpus"),
        ({"cpus": "one"}, "cpus"),
        ({"memory": "lots"}, "memory"),
        ({"memory": ""}, "memory"),
        ({"pids_limit": 0}, "pids_limit"),
        ({"pids_limit": True}, "pids_limit"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"max_output_bytes": 0}, "max_output_bytes"),
        ({"read_only_root": "yes"}, "read_only_root"),
        ({"run_as_non_root": "yes"}, "run_as_non_root"),
        ({"drop_capabilities": 1}, "drop_capabilities"),
        ({"tmpfs_size": "big"}, "tmpfs_size"),
        ({"auto_pull": "yes"}, "auto_pull"),
    ],
)
def test_invalid_values_are_rejected(overrides, match):
    with pytest.raises(SandboxConfigError, match=match):
        make_config(**overrides)


def test_root_uid_is_rejected():
    # Spec sections 19/20: candidate code must never run as uid 0.
    with pytest.raises(SandboxConfigError, match="must not be UID 0"):
        make_config(user="0")
    with pytest.raises(SandboxConfigError, match="must not be UID 0"):
        make_config(user="0:0")


def test_named_user_is_rejected_when_running_non_root():
    # The stock image has no named non-root user, so a name would silently fail at runtime.
    with pytest.raises(SandboxConfigError, match="numeric UID"):
        make_config(user="sandbox")


@pytest.mark.parametrize("user", ["65534", "65534:65534", "1000:1000"])
def test_numeric_non_root_users_are_accepted(user):
    assert make_config(user=user).user == user


# ------------------------------------------------------------- spec section 52 config record


def test_to_dict_records_the_execution_environment():
    record = make_config(image_digest=DIGEST).to_dict()
    for key in (
        "image",
        "image_digest",
        "memory",
        "cpus",
        "timeout_seconds",
        "pids_limit",
        "network_mode",
    ):
        assert key in record, f"spec section 52 requires {key} in the environment record"
    assert record["image_reference"] == f"python@{DIGEST}"


# ---------------------------------------------------------------- integration with config.py


def test_real_config_yaml_exposes_a_sandbox_section():
    config = Config.load()
    assert isinstance(config.sandbox, SandboxConfig)
    assert config.sandbox.network_mode == "none"


def test_config_loader_wraps_sandbox_errors_as_config_errors(tmp_path):
    # sandbox/ must not import python_dpo.config, so it raises SandboxConfigError; the
    # loader is what turns that into the ConfigError the rest of the CLI expects.
    bad = tmp_path / "config.yaml"
    bad.write_text(
        "project:\n  name: x\n"
        "paths:\n  raw_data: data/raw\n  problems: data/problems\n"
        "  candidates: data/candidates\n  evaluations: data/evaluations\n"
        "  rankings: data/rankings\n"
        "  preferences: data/preferences\n  training: data/training\n"
            "  model_evaluations: data/model_evaluations\n"
            "  reports: data/reports\n"
        "logging:\n  level: INFO\n"
        "model:\n  provider: mock\n  name: mock/x\n"
        "generation:\n  candidates_per_problem: 1\n"
        "sandbox:\n  network_mode: host\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="network_mode"):
        Config.load(path=bad)
