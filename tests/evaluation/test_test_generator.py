"""Tests for TestGenerator (spec 06 sections 10-20, 42-47)."""

from __future__ import annotations

import ast
from typing import Any

import pytest

from python_dpo.candidates.models import Candidate
from python_dpo.evaluation import test_generator as tg
from python_dpo.evaluation.errors import InvalidProblemError, TestGenerationError
from python_dpo.evaluation.test_generator import TEST_FILENAME, TestGenerator, render_test_module
from python_dpo.problems.models import Problem, TestCase
from python_dpo.sandbox.workspace import CANDIDATE_FILENAME

CODE = "def sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)\n"


def make_problem(**overrides: Any) -> Problem:
    fields: dict[str, Any] = {
        "id": "p001",
        "prompt": "Return the sum of the even integers in a list.",
        "signature": "def sum_even(numbers):",
        "entry_point": "sum_even",
        "category": "lists",
        "difficulty": "easy",
        "reference_solution": CODE,
        "tests": (
            TestCase(id="t001", input={"numbers": [1, 2, 3, 4]}, expected=6),
            TestCase(id="t002", input={"numbers": []}, expected=0),
        ),
    }
    fields.update(overrides)
    return Problem(**fields)


def make_async_problem(**overrides: Any) -> Problem:
    fields: dict[str, Any] = {
        "id": "p010",
        "prompt": "Gather results in order.",
        "signature": "async def gather_in_order(operations):",
        "entry_point": "gather_in_order",
        "category": "async",
        "difficulty": "medium",
        "reference_solution": "async def gather_in_order(operations):\n    return []\n",
        "tests": (TestCase(id="t001", input={"operations": []}, expected=[]),),
    }
    fields.update(overrides)
    return Problem(**fields)


def make_exception_problem(**overrides: Any) -> Problem:
    fields: dict[str, Any] = {
        "id": "p006",
        "prompt": "Compute a factorial, rejecting negatives.",
        "signature": "def factorial(n):",
        "entry_point": "factorial",
        "category": "recursion",
        "difficulty": "easy",
        "reference_solution": (
            "def factorial(n):\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be non-negative')\n"
            "    return 1 if n == 0 else n * factorial(n - 1)\n"
        ),
        "tests": (
            TestCase(id="t001", input={"n": 0}, expected=1),
            TestCase(id="t002", input={"n": -1}, expected_exception="ValueError"),
        ),
    }
    fields.update(overrides)
    return Problem(**fields)


def make_generator_problem(**overrides: Any) -> Problem:
    fields: dict[str, Any] = {
        "id": "p009",
        "prompt": "Chunk a sequence.",
        "signature": "def chunk_sequence(sequence, size):",
        "entry_point": "chunk_sequence",
        "category": "generators",
        "difficulty": "medium",
        "reference_solution": (
            "def chunk_sequence(sequence, size):\n"
            "    for i in range(0, len(sequence), size):\n"
            "        yield sequence[i : i + size]\n"
        ),
        "tests": (
            TestCase(id="t001", input={"sequence": [1, 2, 3, 4, 5], "size": 2}, expected=[[1, 2], [3, 4], [5]]),
        ),
    }
    fields.update(overrides)
    return Problem(**fields)


def make_candidate(problem_id: str = "p001", code: str = CODE, **overrides: Any) -> Candidate:
    fields: dict[str, Any] = {
        "candidate_id": f"{problem_id}_c001",
        "problem_id": problem_id,
        "run_id": "run_20260817_120000_ab12",
        "generation_index": 1,
        "strategy": "normal",
        "model": "mock/deterministic-coder",
        "provider": "mock",
        "prompt_version": "v1",
        "prompt": "Solve the problem.",
        "raw_output": f"```python\n{code}\n```",
        "code": code,
        "extraction_format": "python_fence",
        "syntax_valid": True,
        "function_name_valid": True,
        "generation_config": {},
        "created_at": "2026-08-17T12:00:00Z",
    }
    fields.update(overrides)
    return Candidate.create(**fields)


# --------------------------------------------------------------------- test_case_id


def test_test_case_id_matches_reference_executor_convention():
    problem = make_problem()
    assert tg.test_case_id(problem.id, problem.tests[0]) == "p001_t001"


# --------------------------------------------------------------------- render_test_module


def test_generated_module_is_valid_python():
    ast.parse(render_test_module(make_problem()))


def test_generated_module_is_deterministic():
    problem = make_problem()
    assert render_test_module(problem) == render_test_module(problem)


def test_one_function_per_test_case():
    problem = make_problem()
    source = render_test_module(problem)
    assert "def test_p001_t001():" in source
    assert "def test_p001_t002():" in source


def test_call_uses_keyword_arguments_not_positional():
    # TestCase.input is a kwargs mapping, not a positional argument list (spec sections
    # 10/13's examples show positional calls, but the dataset's actual shape is kwargs).
    source = render_test_module(make_problem())
    assert "candidate.sum_even, {'numbers': [1, 2, 3, 4]}" in source
    assert "candidate.sum_even([1, 2, 3, 4])" not in source


def test_literals_are_reproduced_via_repr_for_every_json_type():
    problem = make_problem(
        tests=(
            TestCase(
                id="t001",
                input={
                    "a": 1,
                    "b": 1.5,
                    "c": "text",
                    "d": True,
                    "e": None,
                    "f": [1, "x", None, {"nested": [True, 2.0]}],
                },
                expected={"ok": True, "value": None},
            ),
        )
    )
    source = render_test_module(problem)
    ast.parse(source)  # must still be valid Python with these types embedded
    assert "'a': 1" in source
    assert "'b': 1.5" in source
    assert "'c': 'text'" in source
    assert "'d': True" in source
    assert "'e': None" in source
    assert "'nested': [True, 2.0]" in source
    assert "{'ok': True, 'value': None}" in source


def test_no_eval_or_exec_in_generated_source():
    # Spec section 44: test data must never be treated as executable source.
    source = render_test_module(make_problem())
    assert "eval(" not in source
    assert "exec(" not in source


def test_async_entry_point_is_awaited():
    source = render_test_module(make_async_problem())
    assert "asyncio.run" in source
    assert "_call(candidate.gather_in_order" in source


def test_generator_result_is_materialized():
    source = render_test_module(make_generator_problem())
    # The _call helper materializes any generator result via list(); this just confirms
    # the helper is present and wired into the call, matching InProcessReferenceExecutor.
    assert "_materialize" in source
    assert "inspect.isgenerator" in source


def test_expected_exception_uses_pytest_raises():
    source = render_test_module(make_exception_problem())
    assert "pytest.raises(Exception)" in source
    assert "type(_exc_info.value).__name__ == 'ValueError'" in source


def test_bool_strict_equality_guard_is_present():
    # True must not satisfy an expected value of 1, mirroring
    # InProcessReferenceExecutor._values_match exactly.
    source = render_test_module(make_problem())
    assert "isinstance(actual, bool) != isinstance(expected, bool)" in source


# -------------------------------------------------------------------------------- build


def test_build_produces_the_three_expected_files():
    job = TestGenerator().build(make_problem(), make_candidate())
    assert set(job.files) == {CANDIDATE_FILENAME, TEST_FILENAME, "conftest.py"}


def test_build_writes_candidate_code_unchanged():
    # Spec sections 12, 69: never modified, never repaired.
    weird_code = "def sum_even(numbers):\n    return sum(n for n in numbers if n%2==0)  # no repair\n"
    job = TestGenerator().build(make_problem(), make_candidate(code=weird_code))
    assert job.files[CANDIDATE_FILENAME] == weird_code


def test_build_records_the_full_expected_test_case_id_list():
    job = TestGenerator().build(make_problem(), make_candidate())
    assert job.expected_test_case_ids == ("p001_t001", "p001_t002")


def test_build_assigns_a_fresh_nonce_per_job():
    generator = TestGenerator()
    job_a = generator.build(make_problem(), make_candidate())
    job_b = generator.build(make_problem(), make_candidate())
    assert job_a.nonce != job_b.nonce


def test_build_rejects_a_candidate_from_a_different_problem():
    # Spec section 6: never evaluate a candidate against an unrelated problem.
    mismatched = make_candidate(problem_id="p002")
    with pytest.raises(TestGenerationError, match="p002"):
        TestGenerator().build(make_problem(), mismatched)


def test_build_rejects_a_problem_with_no_tests():
    empty_problem = make_problem()
    object.__setattr__(empty_problem, "tests", ())
    with pytest.raises(InvalidProblemError):
        TestGenerator().build(empty_problem, make_candidate())


# ----------------------------------------------------------------------------- validate


def test_validate_accepts_a_well_formed_job():
    generator = TestGenerator()
    problem = make_problem()
    job = generator.build(problem, make_candidate())
    generator.validate(job, problem)  # must not raise


def test_validate_rejects_a_missing_test_file():
    generator = TestGenerator()
    problem = make_problem()
    job = generator.build(problem, make_candidate())
    job.files.pop(TEST_FILENAME)
    with pytest.raises(TestGenerationError, match="was not generated"):
        generator.validate(job, problem)


def test_validate_rejects_a_short_test_suite():
    # Spec section 46: the generated file must not silently contain fewer tests than
    # the problem declares.
    generator = TestGenerator()
    problem = make_problem()
    job = generator.build(problem, make_candidate())
    job.files[TEST_FILENAME] = "def test_p001_t001():\n    pass\n"  # missing t002
    with pytest.raises(TestGenerationError, match="missing expected test"):
        generator.validate(job, problem)


def test_validate_rejects_an_unparseable_test_file():
    generator = TestGenerator()
    problem = make_problem()
    job = generator.build(problem, make_candidate())
    job.files[TEST_FILENAME] = "def broken(:\n"
    with pytest.raises(TestGenerationError, match="does not parse"):
        generator.validate(job, problem)


def test_validate_never_executes_the_generated_file(monkeypatch):
    # ast.parse builds a syntax tree and runs nothing; a test file that would raise if
    # executed must still validate cleanly.
    generator = TestGenerator()
    problem = make_problem()
    job = generator.build(problem, make_candidate())
    job.files[TEST_FILENAME] = (
        "raise SystemExit('this must never actually run')\n\n"
        "def test_p001_t001():\n    pass\n\n"
        "def test_p001_t002():\n    pass\n"
    )
    generator.validate(job, problem)  # must not raise SystemExit
