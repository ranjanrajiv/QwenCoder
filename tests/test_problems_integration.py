"""Integration tests: the full build -> validate -> persist -> reload -> validate loop."""

import asyncio
import inspect
import time

from python_dpo.problems import (
    CATEGORIES,
    EXPECTED_PROBLEM_COUNT,
    MIN_TESTS_PER_PROBLEM,
    InProcessReferenceExecutor,
    build_catalog,
    dataset_path,
    format_report,
    load_problems,
    save_problems,
    validate_dataset,
)
from python_dpo.problems import references


def test_full_round_trip(tmp_path):
    executor = InProcessReferenceExecutor()

    problems = build_catalog()
    first_report = validate_dataset(problems, executor)
    assert first_report.valid, format_report(first_report)
    assert first_report.failed_tests == 0

    path = dataset_path(tmp_path)
    save_problems(problems, path)

    reloaded = load_problems(path)
    assert reloaded == problems

    second_report = validate_dataset(reloaded, executor)
    assert second_report.valid, format_report(second_report)
    assert second_report.total_tests == first_report.total_tests
    assert "Dataset validation: PASS" in format_report(second_report)


def test_build_is_deterministic(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    save_problems(build_catalog(), first)
    save_problems(build_catalog(), second)

    assert first.read_bytes() == second.read_bytes()


def test_catalog_shape_matches_the_specification():
    problems = build_catalog()

    assert len(problems) == EXPECTED_PROBLEM_COUNT
    assert [problem.id for problem in problems] == [f"p{n:03d}" for n in range(1, 11)]
    assert {problem.category for problem in problems} == set(CATEGORIES)

    difficulties = [problem.difficulty for problem in problems]
    assert difficulties.count("easy") == 5
    assert difficulties.count("medium") == 4
    assert difficulties.count("hard") == 1

    for problem in problems:
        assert len(problem.tests) >= MIN_TESTS_PER_PROBLEM
        assert problem.source == "manual"


def test_every_reference_solution_is_stored_verbatim():
    """The persisted source must be the code that actually runs."""
    for problem in build_catalog():
        function = getattr(references, problem.entry_point)
        assert problem.reference_solution == inspect.getsource(function)


def test_chunk_sequence_reference_is_a_real_generator():
    """Spec 02 section 18: chunks must be yielded, not returned as one list."""
    assert inspect.isgeneratorfunction(references.chunk_sequence)

    chunks = references.chunk_sequence([1, 2, 3, 4, 5], 2)
    assert inspect.isgenerator(chunks)
    assert next(chunks) == [1, 2]
    assert list(chunks) == [[3, 4], [5]]


def test_gather_in_order_runs_operations_concurrently():
    """Spec 02 section 19: concurrent, not sequential, with input ordering preserved.

    Five 0.05s operations take ~0.25s sequentially and ~0.05s concurrently. The 0.15s
    ceiling leaves a wide margin so a busy machine cannot make this flake.
    """
    operations = [[index, 0.05] for index in range(5)]

    started = time.perf_counter()
    results = asyncio.run(references.gather_in_order(operations))
    elapsed = time.perf_counter() - started

    assert results == [0, 1, 2, 3, 4]
    assert elapsed < 0.15


def test_gather_in_order_returns_input_order_not_completion_order():
    results = asyncio.run(references.gather_in_order([["slow", 0.03], ["fast", 0.0]]))
    assert results == ["slow", "fast"]
