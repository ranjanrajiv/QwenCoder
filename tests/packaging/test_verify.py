"""Tests for packaging verification (spec 12 section 38): generated code is executed
only through :class:`~python_dpo.evaluation.CandidateEvaluator` over a sandbox runner --
this module never calls ``exec`` on model output (CLAUDE.md's Security rule)."""

from __future__ import annotations

from typing import Any

import pytest

from python_dpo.evaluation.executor import CandidateEvaluator
from python_dpo.evaluation.repository import EvaluationRepository
from python_dpo.packaging.errors import VerificationError
from python_dpo.packaging.verify import VERIFICATION_PROBLEM, verify_package
from python_dpo.sandbox.result import ExecutionResult

EVALUATION_RUN_ID = "verify_test"


class _NonceCapturingRunner:
    """Mirrors ``tests/evaluation/test_executor.py``'s FakeRunner: records the job it was
    given and scripts an ExecutionResult from the job's own nonce."""

    def __init__(self, *, passing: bool) -> None:
        self.passing = passing
        self.jobs: list[Any] = []
        self.config = None

    def run(self, job, *, job_id=None, run_id=None):
        self.jobs.append(job)
        status = "passed" if self.passing else "failed"
        lines = [
            _test_event(job.nonce, f"{VERIFICATION_PROBLEM.id}_{tc.id}", status)
            for tc in VERIFICATION_PROBLEM.tests
        ]
        lines.append(
            f"{job.nonce} "
            + _json_dumps(
                kind="session",
                testscollected=len(VERIFICATION_PROBLEM.tests),
                testsfailed=0 if self.passing else len(VERIFICATION_PROBLEM.tests),
                exitstatus=0 if self.passing else 1,
            )
        )
        return ExecutionResult(
            status="success", exit_code=0 if self.passing else 1,
            stdout="\n".join(lines), stderr="", duration_ms=10, container_id="abc123",
        )


def _json_dumps(**payload: Any) -> str:
    import json

    return json.dumps(payload)


def _test_event(nonce: str, test_case_id: str, status: str) -> str:
    payload = {
        "kind": "test", "test_case_id": test_case_id, "status": status, "duration_ms": 1,
        "error_type": None, "error_message": None, "stdout": "", "stderr": "",
    }
    return f"{nonce} " + _json_dumps(**payload)


def make_evaluator(tmp_path, *, passing: bool) -> CandidateEvaluator:
    return CandidateEvaluator(
        runner=_NonceCapturingRunner(passing=passing),
        repository=EvaluationRepository(tmp_path),
    )


def test_verify_package_passes_when_the_generated_code_passes_its_tests(tmp_path):
    evaluator = make_evaluator(tmp_path, passing=True)

    result = verify_package(
        package_id="exp_x",
        model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
        generate=lambda prompt: "```python\ndef add_two(a, b):\n    return a + b\n```",
        evaluator=evaluator,
        evaluation_run_id=EVALUATION_RUN_ID,
    )

    assert result.ok is True
    assert result.tests_passed == result.tests_total == len(VERIFICATION_PROBLEM.tests)
    assert result.extracted_code == "def add_two(a, b):\n    return a + b"


def test_verify_package_raises_when_the_generated_code_fails_its_tests(tmp_path):
    evaluator = make_evaluator(tmp_path, passing=False)

    with pytest.raises(VerificationError, match="failed"):
        verify_package(
            package_id="exp_x",
            model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
            generate=lambda prompt: "```python\ndef add_two(a, b):\n    return 0\n```",
            evaluator=evaluator,
            evaluation_run_id=EVALUATION_RUN_ID,
        )


def test_verify_package_raises_when_no_code_can_be_extracted(tmp_path):
    evaluator = make_evaluator(tmp_path, passing=True)

    with pytest.raises(VerificationError, match="no extractable"):
        verify_package(
            package_id="exp_x",
            model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
            generate=lambda prompt: "I refuse to write code.",
            evaluator=evaluator,
            evaluation_run_id=EVALUATION_RUN_ID,
        )
    # The sandbox runner was never invoked -- extraction failed before evaluate() could.
    assert evaluator._runner.jobs == []  # noqa: SLF001


def test_verify_package_never_executes_model_output_directly(tmp_path, monkeypatch):
    """The generated code must reach the sandbox only through CandidateEvaluator -- this
    test asserts no stdlib exec/eval path is taken by patching them to explode."""
    import builtins

    def _boom(*args, **kwargs):
        raise AssertionError("verify_package must never call exec()/eval() directly")

    monkeypatch.setattr(builtins, "exec", _boom)
    monkeypatch.setattr(builtins, "eval", _boom)

    evaluator = make_evaluator(tmp_path, passing=True)
    result = verify_package(
        package_id="exp_x",
        model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
        generate=lambda prompt: "```python\ndef add_two(a, b):\n    return a + b\n```",
        evaluator=evaluator,
        evaluation_run_id=EVALUATION_RUN_ID,
    )
    assert result.ok is True
