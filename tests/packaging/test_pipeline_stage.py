"""Tests for the shared packaging body (spec 12 section 5, item 9) that both the
``packaging`` pipeline stage adapter and the ``model package`` CLI command call.

Real :class:`~python_dpo.evaluation.CandidateEvaluator` and
:class:`~python_dpo.evaluation.repository.EvaluationRepository` are used throughout, with
only the sandbox boundary (``PytestRunner``) and the model backend (``AdapterModelRunner``)
faked -- torch and Docker are both unavailable in this environment, and this is the same
seam ``tests/evaluation/test_executor.py`` and ``tests/packaging/test_verify.py`` already
fake, so these tests exercise the real classification and persistence code, not a mock of
it.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

import python_dpo.packaging.pipeline_stage as pipeline_stage
from python_dpo.config import Config, Paths
from python_dpo.packaging.errors import VerificationError
from python_dpo.packaging.registry import ModelRegistry
from python_dpo.packaging.verify import VERIFICATION_PROBLEM
from python_dpo.pipeline.config import StageConfig
from python_dpo.pipeline.repository import ExperimentRunRepository
from python_dpo.pipeline.stages._context import StageContext
from python_dpo.sandbox.result import ExecutionResult
from python_dpo.training.run_repository import TrainingRunRepository

from .conftest import make_training_run


class _FakeAdapterRunner:
    def __init__(self, *, model_name, model_revision, adapter_dir, quantization, generation):
        pass

    def ensure_loaded(self):
        pass

    def generate(self, prompt, *, seed):
        @dataclasses.dataclass
        class _Gen:
            text: str

        return _Gen(text="```python\ndef add_two(a, b):\n    return a + b\n```")

    def unload(self):
        pass


class _ScriptedPytestRunner:
    """Stands in for ``PytestRunner(SandboxExecutor(...))`` (spec's sandbox boundary)."""

    def __init__(self, executor):
        self._passing = True

    def run(self, job, *, job_id=None, run_id=None):
        lines = [
            f"{job.nonce} " + json.dumps({
                "kind": "test", "test_case_id": f"{VERIFICATION_PROBLEM.id}_{tc.id}",
                "status": "passed" if self._passing else "failed", "duration_ms": 1,
                "error_type": None, "error_message": None, "stdout": "", "stderr": "",
            })
            for tc in VERIFICATION_PROBLEM.tests
        ]
        lines.append(
            f"{job.nonce} " + json.dumps({
                "kind": "session",
                "testscollected": len(VERIFICATION_PROBLEM.tests),
                "testsfailed": 0 if self._passing else len(VERIFICATION_PROBLEM.tests),
                "exitstatus": 0 if self._passing else 1,
            })
        )
        return ExecutionResult(
            status="success", exit_code=0 if self._passing else 1,
            stdout="\n".join(lines), stderr="", duration_ms=5, container_id="abc123",
        )


class _FailingPytestRunner(_ScriptedPytestRunner):
    def __init__(self, executor):
        super().__init__(executor)
        self._passing = False


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch):
    monkeypatch.setattr(pipeline_stage, "AdapterModelRunner", _FakeAdapterRunner)
    monkeypatch.setattr(pipeline_stage, "PytestRunner", _ScriptedPytestRunner)


def test_package_and_verify_packages_and_registers_a_passing_model(tmp_path):
    from python_dpo.sandbox.config import SandboxConfig

    repo, training_run_id = make_training_run(tmp_path)
    registry = ModelRegistry(tmp_path / "models" / "registry.json")

    result = pipeline_stage.package_and_verify(
        model_id="exp_x",
        training_run_id=training_run_id,
        training_run_repo=repo,
        dest_dir=tmp_path / "package",
        verification_dir=tmp_path / "verification",
        sandbox_config=SandboxConfig(),
        registry=registry,
    )

    assert result.verification.ok is True
    assert result.registry_entry.status == "EXPERIMENTAL"
    assert registry.get("exp_x").training_run_id == training_run_id
    assert (tmp_path / "package" / "adapter" / "adapter_config.json").is_file()


def test_package_and_verify_never_registers_a_failing_model(tmp_path, monkeypatch):
    from python_dpo.sandbox.config import SandboxConfig

    monkeypatch.setattr(pipeline_stage, "PytestRunner", _FailingPytestRunner)
    repo, training_run_id = make_training_run(tmp_path)
    registry = ModelRegistry(tmp_path / "models" / "registry.json")

    with pytest.raises(VerificationError):
        pipeline_stage.package_and_verify(
            model_id="exp_x",
            training_run_id=training_run_id,
            training_run_repo=repo,
            dest_dir=tmp_path / "package",
            verification_dir=tmp_path / "verification",
            sandbox_config=SandboxConfig(),
            registry=registry,
        )

    with pytest.raises(Exception):
        registry.get("exp_x")


def _project_config(tmp_path) -> Config:
    base = Config.load()
    paths = Paths(
        raw=tmp_path / "raw", problems=tmp_path / "problems", candidates=tmp_path / "candidates",
        evaluations=tmp_path / "evaluations", rankings=tmp_path / "rankings",
        preferences=tmp_path / "preferences", training=tmp_path / "training",
        model_evaluations=tmp_path / "model_evaluations", experiments=tmp_path / "experiments",
        analysis=tmp_path / "analysis",
        reports=tmp_path / "reports",
    )
    paths.ensure_exists()
    return dataclasses.replace(base, paths=paths, project_root=tmp_path)


def test_run_stage_adapter_produces_a_stage_result(tmp_path, monkeypatch):
    config = _project_config(tmp_path)
    experiment_repo = ExperimentRunRepository(config.paths.experiments / "runs")
    experiment_repo.create_run(experiment_name="test", configuration_hash="h", experiment_run_id="exp_x")

    training_run_repo = TrainingRunRepository(config.paths.training / "runs")
    monkeypatch.setattr(pipeline_stage, "TrainingRunRepository", lambda root: training_run_repo)
    make_training_run(tmp_path, training_run_id="dpo_x")
    # make_training_run built its own TrainingRunRepository at tmp_path/training/runs,
    # which is exactly `config.paths.training / "runs"` -- both point at the same run.

    context = StageContext(
        experiment_run_id="exp_x",
        stage_config=StageConfig(name="packaging", enabled=True),
        project_config=config,
        experiment_repo=experiment_repo,
        upstream={},
    )
    # upstream_run_id("dpo_training") needs a manifest recording the training run id.
    from python_dpo.pipeline.manifest import StageManifest

    context.upstream["dpo_training"] = StageManifest(
        stage_name="dpo_training", stage_run_id="dpo_x", status="COMPLETED", code_version="0.0.0",
    )

    result = pipeline_stage.run(context)

    assert result.stage_run_id == "exp_x_packaging"
    assert "packaging" in result.output_artifacts
    registry = ModelRegistry(config.project_root / "models" / "registry.json")
    assert registry.get("exp_x").status == "EXPERIMENTAL"
