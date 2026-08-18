"""Tests for package/driver version capture (spec 09 sections 61, 71)."""

from __future__ import annotations

from importlib import metadata

import pytest

from python_dpo.training import versions


def test_tracked_packages_cover_the_spec_list():
    assert set(versions.TRACKED_PACKAGES) == {
        "torch",
        "transformers",
        "trl",
        "peft",
        "bitsandbytes",
        "accelerate",
        "datasets",
        "safetensors",
    }


def test_installed_package_reports_a_version():
    assert versions.package_version("pytest") is not None


def test_missing_package_is_none_not_an_error():
    """A missing optional package must not break `train hardware-check`."""
    assert versions.package_version("definitely-not-installed-xyz") is None


def test_capture_package_versions_covers_every_tracked_name(monkeypatch):
    monkeypatch.setattr(versions, "package_version", lambda name: f"{name}-1.0")
    captured = versions.capture_package_versions()
    assert set(captured) == set(versions.TRACKED_PACKAGES)
    assert captured["torch"] == "torch-1.0"


def test_capture_package_versions_records_missing_as_none(monkeypatch):
    def fake(name: str):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", fake)
    captured = versions.capture_package_versions()
    assert all(value is None for value in captured.values())


def test_driver_version_is_none_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(versions.shutil, "which", lambda name: None)
    assert versions.nvidia_driver_version() is None


def test_driver_version_parses_nvidia_smi_output(monkeypatch):
    class FakeCompleted:
        returncode = 0
        stdout = "580.82.07\n"

    monkeypatch.setattr(versions.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(versions.subprocess, "run", lambda *a, **k: FakeCompleted())
    assert versions.nvidia_driver_version() == "580.82.07"


def test_driver_version_is_none_on_failure(monkeypatch):
    class FakeCompleted:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(versions.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(versions.subprocess, "run", lambda *a, **k: FakeCompleted())
    assert versions.nvidia_driver_version() is None


def test_capture_environment_shape():
    environment = versions.capture_environment()
    assert "python_version" in environment
    assert "platform" in environment
    assert set(environment["packages"]) == set(versions.TRACKED_PACKAGES)
    assert "nvidia_driver_version" in environment
