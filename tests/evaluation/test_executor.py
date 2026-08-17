"""Tests for CandidateEvaluator — every classification path, driven by a fake sandbox
runner (spec 06 sections 23-29, 45-47, 63-64, 69).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from python_dpo.candidates.models import Candidate
from python_dpo.evaluation.errors import InvalidProblemError
from python_dpo.evaluation.executor import CandidateEvaluator
from python_dpo.evaluation.repository import EvaluationRepository
from python_dpo.evaluation.test_generator import TestGenerator
from python_dpo.problems.models import Problem, TestCase
from python_dpo.sandbox.result import ExecutionResult

EVALUATION_RUN_ID = "eval_20260817_154500_a12f"


def make_problem(**overrides: Any) -> Problem:
    fields: dict[str, Any] = {
        "id": "p001",
        "prompt": "Return the sum of the even integers in a list.",
        "signature": "def sum_even(numbers):",
        "entry_point": "sum_even",
        "category": "lists",
        "difficulty": "easy",
        "reference_solution": "def sum_even(numbers):\n    return 0\n",
        "tests": (
            TestCase(id="t001", input={"numbers": [1, 2, 3, 4]}, expected=6),
            TestCase(id="t002", input={"numbers": []}, expected=0),
        ),
    }
    fields.update(overrides)
    return Problem(**fields)


def make_candidate(**overrides: Any) -> Candidate:
    fields: dict[str, Any] = {
        "candidate_id": "p001_c001",
        "problem_id": "p001",
        "run_id": "run_20260817_055411",
        "generation_index": 1,
        "strategy": "normal",
        "model": "mock/deterministic-coder",
        "provider": "mock",
        "prompt_version": "v1",
        "prompt": "Solve it.",
        "raw_output": "```python\ndef sum_even(numbers):\n    return 0\n```",
        "code": "def sum_even(numbers):\n    return 0\n",
        "extraction_format": "python_fence",
        "syntax_valid": True,
        "function_name_valid": True,
        "generation_config": {},
        "created_at": "2026-08-17T12:00:00Z",
    }
    fields.update(overrides)
    return Candidate.create(**fields)


def nonce_line(nonce: str, **payload) -> str:
    return f"{nonce} " + json.dumps(payload)


def make_test_event(nonce: str, test_case_id: str, status: str, **fields) -> str:
    payload = {
        "kind": "test",
        "test_case_id": test_case_id,
        "status": status,
        "duration_ms": 1,
        "error_type": None,
        "error_message": None,
        "stdout": "",
        "stderr": "",
    }
    payload.update(fields)
    return nonce_line(nonce, **payload)


class FakeRunner:
    """Records the job it was given and returns a scripted ExecutionResult."""

    def __init__(self, result: ExecutionResult) -> None:
        self._result = result
        self.jobs = []
        self.config = None  # unused by the evaluator

    def run(self, job, *, job_id=None, run_id=None):
        self.jobs.append(job)
        return self._result


def make_evaluator(tmp_path, exec_result: ExecutionResult):
    repository = EvaluationRepository(tmp_path)
    runner = FakeRunner(exec_result)
    evaluator = CandidateEvaluator(runner=runner, repository=repository, test_generator=TestGenerator())
    return evaluator, repository, runner


def exec_result(**overrides: Any) -> ExecutionResult:
    fields: dict[str, Any] = {
        "status": "success",
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "duration_ms": 50,
        "container_id": "abc123",
    }
    fields.update(overrides)
    return ExecutionResult(**fields)


# --------------------------------------------------------------------------------- passed


def test_passed_when_every_test_passes(tmp_path):
    problem = make_problem()
    candidate = make_candidate()
    # We don't know the nonce ahead of time, so build the scripted stdout from the job's
    # actual nonce once the runner receives it.

    class CapturingRunner(FakeRunner):
        def run(self, job, *, job_id=None, run_id=None):
            self._result = ExecutionResult(
                status="success",
                exit_code=0,
                stdout="\n".join(
                    [
                        make_test_event(job.nonce, "p001_t001", "passed"),
                        make_test_event(job.nonce, "p001_t002", "passed"),
                        nonce_line(job.nonce, kind="session", testscollected=2, testsfailed=0, exitstatus=0),
                    ]
                ),
                stderr="",
                duration_ms=42,
                container_id="abc123",
            )
            self.jobs.append(job)
            return self._result

    repository = EvaluationRepository(tmp_path)
    runner = CapturingRunner(exec_result())
    evaluator = CandidateEvaluator(runner=runner, repository=repository)

    result = evaluator.evaluate(candidate, problem, evaluation_run_id=EVALUATION_RUN_ID)

    assert result.status == "passed"
    assert result.tests_total == 2
    assert result.tests_passed == 2
    assert repository.load_all() == [result]
    assert len(repository.load_test_results()) == 2


# ---------------------------------------------------------------------------- wrong answer


def test_failed_with_a_wrong_answer(tmp_path):
    problem = make_problem()
    candidate = make_candidate()

    class Runner(FakeRunner):
        def run(self, job, *, job_id=None, run_id=None):
            result = exec_result(
                exit_code=1,
                stdout="\n".join(
                    [
                        make_test_event(job.nonce, "p001_t001", "passed"),
                        make_test_event(job.nonce, "p001_t002", "failed", error_type="AssertionError"),
                        nonce_line(job.nonce, kind="session", testscollected=2, testsfailed=1, exitstatus=1),
                    ]
                ),
            )
            self.jobs.append(job)
            return result

    repository = EvaluationRepository(tmp_path)
    evaluator = CandidateEvaluator(runner=Runner(exec_result()), repository=repository)
    result = evaluator.evaluate(candidate, problem, evaluation_run_id=EVALUATION_RUN_ID)

    assert result.status == "failed"
    assert result.tests_passed == 1
    assert result.tests_failed == 1
    assert result.runtime_error is False  # a wrong answer, not a candidate exception


# ------------------------------------------------------------------------- runtime error


def test_failed_with_a_runtime_exception_sets_runtime_error(tmp_path):
    # Spec section 27: a candidate exception during a test is "failed" at candidate
    # level with runtime_error=true — never mistaken for infrastructure trouble.
    problem = make_problem()
    candidate = make_candidate()

    class Runner(FakeRunner):
        def run(self, job, *, job_id=None, run_id=None):
            result = exec_result(
                exit_code=1,
                stdout="\n".join(
                    [
                        make_test_event(job.nonce, "p001_t001", "error", error_type="ValueError", error_message="boom"),
                        make_test_event(job.nonce, "p001_t002", "passed"),
                        nonce_line(job.nonce, kind="session", testscollected=2, testsfailed=1, exitstatus=1),
                    ]
                ),
            )
            self.jobs.append(job)
            return result

    repository = EvaluationRepository(tmp_path)
    evaluator = CandidateEvaluator(runner=Runner(exec_result()), repository=repository)
    result = evaluator.evaluate(candidate, problem, evaluation_run_id=EVALUATION_RUN_ID)

    assert result.status == "failed"
    assert result.tests_error == 1
    assert result.runtime_error is True


# --------------------------------------------------------------------------------- timeout


def test_timeout_is_distinct_from_failed(tmp_path):
    problem = make_problem()
    candidate = make_candidate()

    class Runner(FakeRunner):
        def run(self, job, *, job_id=None, run_id=None):
            result = ExecutionResult(
                status="timeout",
                exit_code=137,
                stdout=make_test_event(job.nonce, "p001_t001", "passed"),  # partial: t002 never ran
                stderr="",
                duration_ms=5000,
                timed_out=True,
            )
            self.jobs.append(job)
            return result

    repository = EvaluationRepository(tmp_path)
    evaluator = CandidateEvaluator(runner=Runner(exec_result()), repository=repository)
    result = evaluator.evaluate(candidate, problem, evaluation_run_id=EVALUATION_RUN_ID)

    assert result.status == "timeout"
    assert result.timeout is True
    # Spec section 46/68: the un-run test is still accounted for, not silently dropped.
    assert result.tests_total == 2
    test_results = repository.load_test_results()
    missing = next(t for t in test_results if t.test_case_id == "p001_t002")
    assert missing.status == "error"
    assert missing.error_type == "Timeout"


# --------------------------------------------------------------------------- syntax error


def test_collection_failure_is_syntax_error(tmp_path):
    problem = make_problem()
    candidate = make_candidate(syntax_valid=False, code="def broken(:\n", function_name_valid=False)

    class Runner(FakeRunner):
        def run(self, job, *, job_id=None, run_id=None):
            result = exec_result(
                exit_code=2,
                stdout=nonce_line(
                    job.nonce,
                    kind="collect_error",
                    nodeid="test_candidate.py",
                    message="candidate.py:1: SyntaxError: invalid syntax",
                ),
            )
            self.jobs.append(job)
            return result

    repository = EvaluationRepository(tmp_path)
    evaluator = CandidateEvaluator(runner=Runner(exec_result()), repository=repository)
    result = evaluator.evaluate(candidate, problem, evaluation_run_id=EVALUATION_RUN_ID)

    assert result.status == "syntax_error"
    assert result.syntax_error is True
    assert result.tests_total == 2
    assert all(t.status == "error" for t in repository.load_test_results())


def test_non_syntax_collection_failure_is_failed_not_syntax_error(tmp_path):
    problem = make_problem()
    candidate = make_candidate()

    class Runner(FakeRunner):
        def run(self, job, *, job_id=None, run_id=None):
            result = exec_result(
                exit_code=2,
                stdout=nonce_line(
                    job.nonce,
                    kind="collect_error",
                    nodeid="test_candidate.py",
                    message="ModuleNotFoundError: No module named 'numpy'",
                ),
            )
            self.jobs.append(job)
            return result

    repository = EvaluationRepository(tmp_path)
    evaluator = CandidateEvaluator(runner=Runner(exec_result()), repository=repository)
    result = evaluator.evaluate(candidate, problem, evaluation_run_id=EVALUATION_RUN_ID)

    assert result.status == "failed"
    assert result.syntax_error is False


# --------------------------------------------------------------------- infrastructure error


def test_infrastructure_failure_is_never_a_candidate_outcome(tmp_path):
    # Spec sections 29, 81: the candidate must not be judged badly because Docker failed.
    problem = make_problem()
    candidate = make_candidate()
    infra = ExecutionResult.infrastructure_failure(
        error_type="DockerUnavailableError", error_message="daemon down", duration_ms=5
    )
    evaluator, repository, runner = make_evaluator(tmp_path, infra)
    result = evaluator.evaluate(candidate, problem, evaluation_run_id=EVALUATION_RUN_ID)

    assert result.status == "infrastructure_error"
    assert result.infrastructure_error is True
    assert result.tests_total == 0
    assert repository.load_test_results() == []


# ------------------------------------------------------------------------- resource limits


def test_output_flood_is_classified_failed_with_resource_reason(tmp_path):
    problem = make_problem()
    candidate = make_candidate()

    class Runner(FakeRunner):
        def run(self, job, *, job_id=None, run_id=None):
            result = exec_result(
                status="resource_exceeded", exit_code=137, stdout_truncated=True,
                stdout=make_test_event(job.nonce, "p001_t001", "passed"),
            )
            self.jobs.append(job)
            return result

    repository = EvaluationRepository(tmp_path)
    evaluator = CandidateEvaluator(runner=Runner(exec_result()), repository=repository)
    result = evaluator.evaluate(candidate, problem, evaluation_run_id=EVALUATION_RUN_ID)

    assert result.status == "failed"
    missing = next(t for t in repository.load_test_results() if t.test_case_id == "p001_t002")
    assert missing.error_type == "ResourceExceeded"


# ------------------------------------------------------------------------------ §45 empty


def test_zero_test_problem_is_a_machinery_failure_not_a_pass(tmp_path):
    problem = make_problem()
    object.__setattr__(problem, "tests", ())
    candidate = make_candidate()
    repository = EvaluationRepository(tmp_path)
    evaluator = CandidateEvaluator(runner=FakeRunner(exec_result()), repository=repository)

    with pytest.raises(InvalidProblemError):
        evaluator.evaluate(candidate, problem, evaluation_run_id=EVALUATION_RUN_ID)
    assert repository.load_all() == []  # nothing attempted, nothing to persist as a result


# -------------------------------------------------------------------------- evaluate_many


def test_evaluate_many_records_a_machinery_failure_for_a_missing_problem(tmp_path):
    candidate = make_candidate(problem_id="p999", candidate_id="p999_c001")
    repository = EvaluationRepository(tmp_path)
    evaluator = CandidateEvaluator(runner=FakeRunner(exec_result()), repository=repository)

    summary = evaluator.evaluate_many([candidate], {}, evaluation_run_id=EVALUATION_RUN_ID)

    assert summary.machinery_failed == 1
    assert summary.evaluated == 0
    failures = repository.load_failures()
    assert failures[0].error_type == "problem_not_found"


def test_evaluate_many_skips_already_evaluated_candidates(tmp_path):
    problem = make_problem()
    candidate = make_candidate()
    repository = EvaluationRepository(tmp_path)

    class Runner(FakeRunner):
        def run(self, job, *, job_id=None, run_id=None):
            result = exec_result(
                stdout="\n".join(
                    [
                        make_test_event(job.nonce, "p001_t001", "passed"),
                        make_test_event(job.nonce, "p001_t002", "passed"),
                    ]
                )
            )
            self.jobs.append(job)
            return result

    runner = Runner(exec_result())
    evaluator = CandidateEvaluator(runner=runner, repository=repository)
    evaluator.evaluate_many([candidate], {"p001": problem}, evaluation_run_id=EVALUATION_RUN_ID)
    assert len(runner.jobs) == 1

    summary = evaluator.evaluate_many([candidate], {"p001": problem}, evaluation_run_id=EVALUATION_RUN_ID)
    assert summary.skipped == 1
    assert summary.evaluated == 0
    assert len(runner.jobs) == 1, "resume must not call the sandbox again"


def test_evaluate_many_does_not_retry_a_structural_failure_on_resume(tmp_path):
    # Unlike Stage 4's transient generation failures, an evaluation machinery failure is
    # deterministic and must not be retried indefinitely (spec section 56).
    candidate = make_candidate(problem_id="p999", candidate_id="p999_c001")
    repository = EvaluationRepository(tmp_path)
    evaluator = CandidateEvaluator(runner=FakeRunner(exec_result()), repository=repository)

    evaluator.evaluate_many([candidate], {}, evaluation_run_id=EVALUATION_RUN_ID)
    summary = evaluator.evaluate_many([candidate], {}, evaluation_run_id=EVALUATION_RUN_ID)

    assert summary.skipped == 1
    assert summary.machinery_failed == 0
    assert len(repository.load_failures()) == 1


# -------------------------------------------------------------------------- §63/64 discrepancy


def test_metadata_discrepancy_is_recorded_without_overwriting_stage3_data(tmp_path):
    problem = make_problem()
    # Stage 3 said this parsed fine, but the sandbox disagrees.
    candidate = make_candidate(syntax_valid=True)

    class Runner(FakeRunner):
        def run(self, job, *, job_id=None, run_id=None):
            result = exec_result(
                exit_code=2,
                stdout=nonce_line(
                    job.nonce, kind="collect_error", nodeid="x",
                    message="candidate.py:1: SyntaxError: invalid syntax",
                ),
            )
            self.jobs.append(job)
            return result

    repository = EvaluationRepository(tmp_path)
    evaluator = CandidateEvaluator(runner=Runner(exec_result()), repository=repository)
    result = evaluator.evaluate(candidate, problem, evaluation_run_id=EVALUATION_RUN_ID)

    assert result.metadata_discrepancy is True
    assert "syntax_valid=true" in result.discrepancy_reason
    # Stage 3's own record is never touched by this package.
    assert candidate.syntax_valid is True


# -------------------------------------------------------------------------------- §69 immutability


def test_candidate_source_is_never_modified(tmp_path):
    problem = make_problem()
    candidate = make_candidate()
    before = candidate.code

    class Runner(FakeRunner):
        def run(self, job, *, job_id=None, run_id=None):
            self.jobs.append(job)
            return exec_result(
                stdout="\n".join(
                    [
                        make_test_event(job.nonce, "p001_t001", "passed"),
                        make_test_event(job.nonce, "p001_t002", "passed"),
                    ]
                )
            )

    repository = EvaluationRepository(tmp_path)
    evaluator = CandidateEvaluator(runner=Runner(exec_result()), repository=repository)
    evaluator.evaluate(candidate, problem, evaluation_run_id=EVALUATION_RUN_ID)

    assert candidate.code == before
