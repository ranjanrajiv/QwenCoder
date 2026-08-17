"""Tests for build_evaluation_sandbox_config and PytestRunner (spec 06 sections 33-43, 78)."""

from __future__ import annotations

from typing import Any

from python_dpo.evaluation.config import EvaluationConfig
from python_dpo.evaluation.pytest_runner import (
    PYTEST_COMMAND,
    PytestRunner,
    build_evaluation_sandbox_config,
)
from python_dpo.evaluation.test_generator import TestJob
from python_dpo.sandbox import ExecutionResult, SandboxConfig


# --------------------------------------------------------- build_evaluation_sandbox_config


def test_overlays_only_the_four_evaluation_specific_fields():
    base = SandboxConfig()
    evaluation = EvaluationConfig(
        image="python-dpo-evaluator:1.0",
        timeout_seconds=30,
        startup_grace_seconds=10,
        auto_pull=False,
    )
    derived = build_evaluation_sandbox_config(base, evaluation)

    assert derived.image == "python-dpo-evaluator:1.0"
    assert derived.timeout_seconds == 30
    assert derived.startup_grace_seconds == 10
    assert derived.auto_pull is False


def test_every_isolation_setting_is_inherited_unchanged_from_base():
    # Spec section 78: adding pytest must not weaken network mode, user, capabilities, or
    # resource limits. A non-default base proves these are copied, not defaulted.
    base = SandboxConfig(
        network_mode="none",
        cpus=2.0,
        memory="1g",
        pids_limit=32,
        read_only_root=True,
        run_as_non_root=True,
        drop_capabilities=True,
        user="65534:65534",
        tmpfs_size="128m",
        max_output_bytes=500_000,
    )
    derived = build_evaluation_sandbox_config(base, EvaluationConfig())

    assert derived.network_mode == base.network_mode
    assert derived.cpus == base.cpus
    assert derived.memory == base.memory
    assert derived.pids_limit == base.pids_limit
    assert derived.read_only_root == base.read_only_root
    assert derived.run_as_non_root == base.run_as_non_root
    assert derived.drop_capabilities == base.drop_capabilities
    assert derived.user == base.user
    assert derived.tmpfs_size == base.tmpfs_size
    assert derived.max_output_bytes == base.max_output_bytes


def test_derived_config_is_still_a_sandbox_config():
    derived = build_evaluation_sandbox_config(SandboxConfig(), EvaluationConfig())
    assert isinstance(derived, SandboxConfig)


# ------------------------------------------------------------------------------ PytestRunner


class FakeExecutor:
    """Duck-typed stand-in for SandboxExecutor: records the call, returns a scripted result."""

    def __init__(self, config: SandboxConfig, result: ExecutionResult) -> None:
        self.config = config
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def execute_job(self, *, files, command, job_id=None, run_id=None):
        self.calls.append({"files": files, "command": command, "job_id": job_id, "run_id": run_id})
        return self._result


def make_job(**overrides: Any) -> TestJob:
    fields: dict[str, Any] = {
        "files": {
            "candidate.py": "def f():\n    return 1\n",
            "test_candidate.py": "def test_p001_t001():\n    pass\n",
            "conftest.py": "# reporting plugin\n",
        },
        "nonce": "deadbeef" * 4,
        "expected_test_case_ids": ("p001_t001",),
    }
    fields.update(overrides)
    return TestJob(**fields)


def make_execution_result(**overrides: Any) -> ExecutionResult:
    fields: dict[str, Any] = {
        "status": "success",
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "duration_ms": 10,
        "timed_out": False,
        "container_id": "abc123",
    }
    fields.update(overrides)
    return ExecutionResult(**fields)


def test_run_delegates_to_execute_job_with_the_fixed_pytest_command():
    result = make_execution_result()
    executor = FakeExecutor(SandboxConfig(), result)
    job = make_job()

    outcome = PytestRunner(executor).run(job, job_id="job-1", run_id="run-1")

    assert outcome is result
    assert len(executor.calls) == 1
    call = executor.calls[0]
    assert call["files"] == job.files
    assert call["command"] == PYTEST_COMMAND
    assert call["job_id"] == "job-1"
    assert call["run_id"] == "run-1"


def test_run_passes_every_job_file_through_unchanged():
    executor = FakeExecutor(SandboxConfig(), make_execution_result())
    job = make_job(files={"candidate.py": "x = 1\n", "test_candidate.py": "y = 2\n", "conftest.py": "z = 3\n"})

    PytestRunner(executor).run(job)

    assert executor.calls[0]["files"] == job.files


def test_config_property_exposes_the_executors_config():
    config = SandboxConfig(image="python-dpo-evaluator:1.0")
    executor = FakeExecutor(config, make_execution_result())
    assert PytestRunner(executor).config is config


def test_pytest_command_disables_the_cache_provider_and_targets_the_test_module():
    # -p no:cacheprovider matters because the workspace is read-only; the target must be
    # test_candidate.py, matching what TestGenerator writes.
    assert "-p" in PYTEST_COMMAND
    assert "no:cacheprovider" in PYTEST_COMMAND
    assert PYTEST_COMMAND[-1] == "test_candidate.py"
    assert PYTEST_COMMAND[0] == "python"
