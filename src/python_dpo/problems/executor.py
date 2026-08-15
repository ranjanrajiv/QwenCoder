"""Execution seam for running reference solutions against their test cases.

``ProblemValidator``-style code depends only on the ``ReferenceExecutor`` protocol, so
Stage 3 can substitute a Docker-backed executor for untrusted candidates without
touching the validation logic (spec 02 sections 23-24).
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Protocol

from .models import Problem, ProblemError, TestCase, TestResult


class ReferenceExecutor(Protocol):
    """Runs one test case against a problem's solution and reports the outcome."""

    def run(self, problem: Problem, test_case: TestCase) -> TestResult: ...


class InProcessReferenceExecutor:
    """Executes a problem's reference solution in this process.

    SECURITY: this executor is for TRUSTED, manually authored reference solutions
    only. It compiles and runs the stored source directly on the host, which is
    sanctioned for reference code (spec 02 section 23) precisely because that code
    ships with the repository and is reviewed like any other source file.

    Model-generated candidates must NEVER be passed to this class. Executing them
    requires the isolated sandbox introduced in Stage 3; see CLAUDE.md.
    """

    def run(self, problem: Problem, test_case: TestCase) -> TestResult:
        test_id = f"{problem.id}_{test_case.id}"
        try:
            function = self._load_function(problem)
            actual = self._call(function, test_case.input)
        except Exception as exc:  # noqa: BLE001 - any failure is a test outcome
            return self._result_for_exception(test_id, test_case, exc)

        if test_case.expected_exception is not None:
            return TestResult(
                test_id=test_id,
                passed=False,
                actual=actual,
                expected=test_case.expected_exception,
                error_type=None,
                error_message=(
                    f"expected {test_case.expected_exception} to be raised, "
                    f"but the call returned {actual!r}"
                ),
            )

        if _values_match(actual, test_case.expected):
            return TestResult(
                test_id=test_id, passed=True, actual=actual, expected=test_case.expected
            )

        return TestResult(
            test_id=test_id,
            passed=False,
            actual=actual,
            expected=test_case.expected,
            error_type="AssertionError",
            error_message=f"expected {test_case.expected!r}, got {actual!r}",
        )

    def _load_function(self, problem: Problem) -> Any:
        namespace: dict[str, Any] = {"__name__": f"python_dpo_reference_{problem.id}"}
        code = compile(problem.reference_solution, f"<reference:{problem.id}>", "exec")
        exec(code, namespace)  # noqa: S102 - trusted reference solutions only; see class docstring

        if problem.entry_point not in namespace:
            raise ProblemError(
                f"{problem.id}: reference solution does not define {problem.entry_point!r}"
            )
        return namespace[problem.entry_point]

    def _call(self, function: Any, kwargs: dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(function):
            return _materialize(asyncio.run(function(**kwargs)))
        return _materialize(function(**kwargs))

    def _result_for_exception(
        self, test_id: str, test_case: TestCase, exc: BaseException
    ) -> TestResult:
        error_type = type(exc).__name__

        if test_case.expected_exception is not None:
            matched = error_type == test_case.expected_exception
            return TestResult(
                test_id=test_id,
                passed=matched,
                actual=error_type,
                expected=test_case.expected_exception,
                error_type=None if matched else error_type,
                error_message=None if matched else str(exc),
            )

        return TestResult(
            test_id=test_id,
            passed=False,
            actual=None,
            expected=test_case.expected,
            error_type=error_type,
            error_message=str(exc),
        )


def _materialize(value: Any) -> Any:
    """Turn a generator result into a list so it can be compared and serialized.

    Generator problems declare their expected output as a list of chunks; whether the
    solution is a real generator (rather than a function returning a list) is checked
    separately in the test suite, since it is not expressible as an input/output pair.
    """
    if inspect.isgenerator(value):
        return list(value)
    return value


def _values_match(actual: Any, expected: Any) -> bool:
    """Compare results without letting ``True == 1`` slip through as a pass."""
    if isinstance(actual, bool) != isinstance(expected, bool):
        return False
    return bool(actual == expected)
