"""Turns a problem's declared test cases and a candidate's code into an isolated pytest
job (spec 06 sections 10-20, 42-47).

Nothing here executes anything (spec section 47: validation happens structurally, via
``ast.parse``, never by running the generated file). Test data comes from the trusted
problem dataset and is serialized with ``repr()`` — never string-interpolated as source
text and never reconstructed with ``eval()`` (spec sections 16, 44).

The comparison semantics deliberately replicate
:class:`python_dpo.problems.executor.InProcessReferenceExecutor` exactly: the same
generator-materialization, the same async handling, and the same boolean-strict equality
check. This is load-bearing, not cosmetic — the dataset's expected values were *validated*
under those exact rules (spec section 9), so any divergence here would make a genuinely
correct candidate fail. See ``tests/evaluation/test_integration.py``'s
reference-solution self-check for the test that keeps this honest.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ..candidates.models import Candidate
from ..problems.models import Problem, TestCase
from ..sandbox.workspace import CANDIDATE_FILENAME
from .errors import InvalidProblemError, TestGenerationError
from .result_parser import CONFTEST_FILENAME, new_nonce, render_conftest

TEST_GENERATOR_VERSION = "v1"
TEST_FILENAME = "test_candidate.py"

# Mirrors InProcessReferenceExecutor exactly (spec 02 problems/executor.py):
#   _materialize  - a generator-typed result is drained into a list before comparison,
#                    since Problem test cases declare expected output as a list of chunks.
#   _call         - a coroutine-function entry point is awaited via asyncio.run(); no
#                    pytest-asyncio dependency needed since only one call is ever made.
#   _matches      - bool-strict equality, so True does not silently pass as 1.
# Duplicated here (not imported) because generated code runs inside the sandbox, which has
# no python_dpo installed and must not import project source (spec section 40).
_HELPERS = '''\
import asyncio
import inspect

import candidate
import pytest


def _materialize(value):
    if inspect.isgenerator(value):
        return list(value)
    return value


def _call(fn, kwargs):
    if inspect.iscoroutinefunction(fn):
        return _materialize(asyncio.run(fn(**kwargs)))
    return _materialize(fn(**kwargs))


def _matches(actual, expected):
    if isinstance(actual, bool) != isinstance(expected, bool):
        return False
    return actual == expected
'''


def test_case_id(problem_id: str, test_case: TestCase) -> str:
    """The fully-qualified id used everywhere a test case is referenced outside its
    problem — matches ``InProcessReferenceExecutor``'s ``f"{problem.id}_{test_case.id}"``
    convention (spec 02), not the ``_tc``-suffixed shape in the spec's illustrative
    examples.
    """
    return f"{problem_id}_{test_case.id}"


def _test_function_name(problem_id: str, test_case: TestCase) -> str:
    name = f"test_{test_case_id(problem_id, test_case)}"
    if not name.isidentifier():
        # Defense in depth: Problem/TestCase ids are validated on load, but a generated
        # Python identifier is cheap to double-check before it becomes source code.
        raise TestGenerationError(
            f"generated test function name {name!r} is not a valid Python identifier"
        )
    return name


def _render_test_function(problem: Problem, case: TestCase) -> str:
    name = _test_function_name(problem.id, case)
    # repr() of a JSON-native dict/list/str/int/float/bool/None value round-trips to an
    # exact Python literal — never string interpolation of untrusted text, never eval()
    # (spec sections 16, 44). Stage 2 already restricts test data to JSON-native values.
    call = f"_call(candidate.{problem.entry_point}, {case.input!r})"

    if case.expected_exception is not None:
        return (
            f"def {name}():\n"
            f"    with pytest.raises(Exception) as _exc_info:\n"
            f"        {call}\n"
            f"    assert type(_exc_info.value).__name__ == {case.expected_exception!r}\n"
        )

    return f"def {name}():\n    _result = {call}\n    assert _matches(_result, {case.expected!r})\n"


def render_test_module(problem: Problem) -> str:
    """The complete ``test_candidate.py`` source: one function per test case (spec
    section 20), deterministic given the same problem (spec section 13).
    """
    parts = [_HELPERS]
    parts.extend(_render_test_function(problem, case) for case in problem.tests)
    return "\n\n".join(parts) + "\n"


@dataclass(frozen=True)
class TestJob:
    """Everything needed to run one evaluation job in the sandbox."""

    # Not a pytest test class despite the name; keeps collection warnings away.
    __test__ = False

    files: dict[str, str]
    nonce: str
    expected_test_case_ids: tuple[str, ...]


class TestGenerator:
    """Builds and structurally validates a deterministic pytest job."""

    # Not a pytest test class despite the name; keeps collection warnings away.
    __test__ = False

    def build(self, problem: Problem, candidate: Candidate) -> TestJob:
        """Build a job. Raises :class:`InvalidProblemError` for a problem with no test
        cases (spec section 45) and :class:`TestGenerationError` if the candidate does not
        belong to this problem or the generated job fails validation (spec section 47).
        Never executes anything.
        """
        if candidate.problem_id != problem.id:
            raise TestGenerationError(
                f"candidate {candidate.candidate_id!r} belongs to problem "
                f"{candidate.problem_id!r}, not {problem.id!r}; refusing to evaluate a "
                "candidate against an unrelated problem"
            )
        if not problem.tests:
            raise InvalidProblemError(f"problem {problem.id!r} has no test cases")

        nonce = new_nonce()
        job = TestJob(
            files={
                # Written exactly as persisted — never repaired, reformatted, or modified
                # (spec sections 12, 69).
                CANDIDATE_FILENAME: candidate.code,
                TEST_FILENAME: render_test_module(problem),
                CONFTEST_FILENAME: render_conftest(nonce),
            },
            nonce=nonce,
            expected_test_case_ids=tuple(test_case_id(problem.id, case) for case in problem.tests),
        )
        self.validate(job, problem)
        return job

    def validate(self, job: TestJob, problem: Problem) -> None:
        """Structural validation only (spec sections 46, 47) — parses with ``ast``,
        which builds a syntax tree and runs nothing, exactly like Stage 3's syntax check.
        Never executes the generated file on the host.
        """
        test_source = job.files.get(TEST_FILENAME)
        if not test_source:
            raise TestGenerationError(f"{TEST_FILENAME} was not generated")

        try:
            tree = ast.parse(test_source)
        except SyntaxError as exc:
            raise TestGenerationError(
                f"generated {TEST_FILENAME} does not parse: {exc}"
            ) from exc

        defined_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        expected_names = {_test_function_name(problem.id, case) for case in problem.tests}

        missing = expected_names - defined_names
        if missing:
            raise TestGenerationError(
                f"generated test file is missing expected test(s): {', '.join(sorted(missing))}"
            )
        if len(expected_names) != len(problem.tests):
            # Two test cases mapping to the same function name would silently overwrite
            # one another at import time — caught here instead of a quietly-short suite.
            raise TestGenerationError(
                f"problem {problem.id!r} produces duplicate generated test function names"
            )
        if len(job.expected_test_case_ids) != len(problem.tests):
            raise TestGenerationError(
                f"expected {len(problem.tests)} test case id(s), got "
                f"{len(job.expected_test_case_ids)}"
            )


__all__ = [
    "TEST_FILENAME",
    "TEST_GENERATOR_VERSION",
    "TestGenerator",
    "TestJob",
    "render_test_module",
    "test_case_id",
]
