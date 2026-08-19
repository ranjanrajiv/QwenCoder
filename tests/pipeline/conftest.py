"""Shared fixtures for the pipeline test suite."""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any

import pytest

from python_dpo.config import Config, Paths
from python_dpo.pipeline.repository import ExperimentRunRepository
from python_dpo.pipeline.stages import STAGE_NAMES
from python_dpo.pipeline.stages._context import StageResult


@pytest.fixture
def project_config(tmp_path) -> Config:
    """A real, valid Config (model/generation/sandbox/evaluation/preferences all parsed
    from the actual root config.yaml) relocated entirely under ``tmp_path``.

    ``project_root`` is redirected as well as ``paths``, not just the latter. Several
    locations are derived from ``project_root`` rather than from a ``paths`` entry --
    ``benchmarks/`` (preflight, the model_evaluation stage) and ``models/registry.json``
    (packaging, the experiment report) -- so leaving it pointing at the real checkout let
    tests write into tracked files. That is what made ``benchmarks/*/manifest.json`` show
    up modified after every test run.
    """
    base = Config.load()
    paths = Paths(
        raw=tmp_path / "raw",
        problems=tmp_path / "problems",
        candidates=tmp_path / "candidates",
        evaluations=tmp_path / "evaluations",
        rankings=tmp_path / "rankings",
        preferences=tmp_path / "preferences",
        training=tmp_path / "training",
        model_evaluations=tmp_path / "model_evaluations",
        experiments=tmp_path / "experiments",
        analysis=tmp_path / "analysis",
        reports=tmp_path / "reports",
    )
    paths.ensure_exists()
    return dataclasses.replace(base, paths=paths, project_root=tmp_path)


@pytest.fixture
def experiment_repo(project_config: Config) -> ExperimentRunRepository:
    return ExperimentRunRepository(project_config.paths.experiments / "runs")


def make_stage_section(enabled: bool = True, **settings: Any) -> dict[str, Any]:
    return {"enabled": enabled, **settings}


def full_experiment_mapping(
    name: str = "test-experiment", *, enabled: dict[str, bool] | None = None
) -> dict[str, Any]:
    """Every stage present and enabled, except ``error_analysis`` (matching the shipped
    default: disabled pending Stage 11), overridable per-stage via ``enabled``."""
    enabled = enabled or {}
    data: dict[str, Any] = {"experiment": {"name": name, "seed": 42}}
    for stage_name in STAGE_NAMES:
        default_enabled = stage_name != "error_analysis"
        data[stage_name] = make_stage_section(enabled.get(stage_name, default_enabled))
    return data


def install_stub_adapter(
    monkeypatch: pytest.MonkeyPatch,
    stage_name: str,
    *,
    stage_run_id: str | None = None,
    output_artifacts: dict[str, str] | None = None,
    raises: BaseException | None = None,
    calls: list[str] | None = None,
) -> None:
    """Replace one stage's real adapter with a stub that needs no GPU, Docker, or model
    weights -- used to test orchestration mechanics in isolation from stage business
    logic (which each stage's own module already tests, and which the real
    --smoke-test run exercises end to end)."""
    module = importlib.import_module(f"python_dpo.pipeline.stages.{stage_name}")

    def fake_run(context):
        if calls is not None:
            calls.append(stage_name)
        if raises is not None:
            raise raises
        return StageResult(
            stage_run_id=stage_run_id or f"stub_{stage_name}",
            output_artifacts=output_artifacts or {},
        )

    monkeypatch.setattr(module, "run", fake_run)


ALL_ADAPTER_STAGES = tuple(n for n in STAGE_NAMES if n != "error_analysis")


def install_all_stub_adapters(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    for stage_name in ALL_ADAPTER_STAGES:
        install_stub_adapter(monkeypatch, stage_name, calls=calls)
