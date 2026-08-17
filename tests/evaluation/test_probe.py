"""Tests for probe_versions (spec 06 section 74) — no Docker required."""

from __future__ import annotations

from typing import Any

import pytest

from python_dpo.evaluation.errors import EvaluationError
from python_dpo.evaluation.probe import probe_versions
from python_dpo.sandbox import SandboxConfig


class FakeExecutor:
    def __init__(self, result) -> None:
        self._result = result
        self.calls: list[str] = []

    def execute(self, code: str, **kwargs: Any):
        self.calls.append(code)
        return self._result


def make_result(**overrides: Any):
    from python_dpo.sandbox import ExecutionResult

    fields: dict[str, Any] = {
        "status": "success",
        "exit_code": 0,
        "stdout": '{"python_version": "3.12.7", "pytest_version": "8.3.4"}',
        "stderr": "",
        "duration_ms": 5,
        "timed_out": False,
        "container_id": "abc",
    }
    fields.update(overrides)
    return ExecutionResult(**fields)


def test_returns_the_parsed_versions():
    executor = FakeExecutor(make_result())
    python_version, pytest_version = probe_versions(SandboxConfig(), executor)
    assert python_version == "3.12.7"
    assert pytest_version == "8.3.4"


def test_probe_runs_exactly_one_program():
    executor = FakeExecutor(make_result())
    probe_versions(SandboxConfig(), executor)
    assert len(executor.calls) == 1
    assert "pytest" in executor.calls[0]


def test_a_failed_probe_raises_evaluation_error():
    executor = FakeExecutor(make_result(status="infrastructure_error", stdout="", stderr="boom"))
    with pytest.raises(EvaluationError, match="probe"):
        probe_versions(SandboxConfig(), executor)


def test_malformed_probe_output_raises_evaluation_error():
    executor = FakeExecutor(make_result(stdout="not json"))
    with pytest.raises(EvaluationError, match="unexpected output"):
        probe_versions(SandboxConfig(), executor)


def test_probe_output_missing_a_key_raises_evaluation_error():
    executor = FakeExecutor(make_result(stdout='{"python_version": "3.12.7"}'))
    with pytest.raises(EvaluationError, match="unexpected output"):
        probe_versions(SandboxConfig(), executor)
