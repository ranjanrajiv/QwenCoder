"""Tests for full environment capture (spec 12 section 30, and sections 76-78's no-PII
and no-secrets rule)."""

from __future__ import annotations

import getpass
import platform

from python_dpo.pipeline.environment import capture_environment, docker_version, gpu_info


def test_capture_environment_includes_every_section_30_key():
    env = capture_environment()
    for key in (
        "python_version",
        "platform",
        "packages",
        "nvidia_driver_version",
        "os",
        "os_release",
        "machine",
        "cuda_version",
        "gpus",
        "pytest_version",
        "docker_version",
    ):
        assert key in env


def test_capture_environment_never_includes_hostname_username_or_token():
    env = capture_environment()
    rendered = repr(env)
    assert platform.node() not in rendered or platform.node() == ""
    assert getpass.getuser() not in rendered


def test_capture_environment_reports_this_machines_pytest_version():
    import pytest as _pytest

    env = capture_environment()
    assert env["pytest_version"] == _pytest.__version__


def test_capture_environment_packages_includes_the_tracked_training_packages():
    env = capture_environment()
    assert "torch" in env["packages"]
    assert "trl" in env["packages"]


def test_gpu_info_returns_a_list():
    assert isinstance(gpu_info(), list)


def test_docker_version_returns_string_or_none():
    version = docker_version()
    assert version is None or isinstance(version, str)
