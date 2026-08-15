"""Unit tests for the problem schema, storage, executor, and validation."""

import json

import pytest

from python_dpo.problems import (
    CATEGORIES,
    DATASET_VERSION,
    DatasetError,
    InProcessReferenceExecutor,
    Problem,
    ProblemError,
    TestCase,
    build_catalog,
    dataset_path,
    format_report,
    load_problems,
    save_problems,
    validate_dataset,
    validate_problem,
)

REFERENCE_SOURCE = "def add_one(value):\n    return value + 1\n"


def make_problem(**overrides):
    defaults = dict(
        id="p999",
        prompt="Add one to the given value.",
        signature="def add_one(value):",
        entry_point="add_one",
        category="lists",
        difficulty="easy",
        reference_solution=REFERENCE_SOURCE,
        tests=tuple(
            TestCase(id=f"t00{index}", input={"value": index}, expected=index + 1)
            for index in range(1, 6)
        ),
    )
    defaults.update(overrides)
    return Problem(**defaults)


# --------------------------------------------------------------------------- schema


def test_valid_problem_is_constructible():
    problem = make_problem()
    assert problem.id == "p999"
    assert problem.source == "manual"
    assert problem.dataset_version == DATASET_VERSION
    assert len(problem.tests) == 5


@pytest.mark.parametrize("field", ["prompt", "signature", "reference_solution"])
def test_empty_required_string_is_rejected(field):
    with pytest.raises(ProblemError, match=field):
        make_problem(**{field: "   "})


def test_missing_required_field_is_rejected():
    record = make_problem().to_dict()
    del record["signature"]
    with pytest.raises(ProblemError, match="missing required field"):
        Problem.from_dict(record)


def test_invalid_category_is_rejected():
    with pytest.raises(ProblemError, match="unknown category"):
        make_problem(category="algebra")


def test_invalid_difficulty_is_rejected():
    with pytest.raises(ProblemError, match="unknown difficulty"):
        make_problem(difficulty="trivial")


def test_signature_must_declare_the_entry_point():
    with pytest.raises(ProblemError, match="does not declare the entry point"):
        make_problem(entry_point="something_else", signature="def add_one(value):")


def test_unknown_field_is_rejected():
    record = make_problem().to_dict()
    record["surprise"] = True
    with pytest.raises(ProblemError, match="unknown field"):
        Problem.from_dict(record)


def test_problem_without_tests_is_rejected():
    with pytest.raises(ProblemError, match="must not be empty"):
        make_problem(tests=())


# ------------------------------------------------------------------------ test cases


def test_valid_test_case():
    case = TestCase(id="t001", input={"value": 1}, expected=2)
    assert case.to_dict()["expected"] == 2
    assert case.expected_exception is None


def test_test_case_input_must_be_a_mapping():
    with pytest.raises(ProblemError, match="mapping of keyword arguments"):
        TestCase(id="t001", input=[1, 2, 3])


def test_test_case_cannot_expect_a_value_and_an_exception():
    with pytest.raises(ProblemError, match="cannot set both"):
        TestCase(id="t001", input={"value": 1}, expected=2, expected_exception="ValueError")


def test_malformed_test_case_is_rejected_on_load():
    with pytest.raises(ProblemError, match="missing required field"):
        TestCase.from_dict({"id": "t001"})


@pytest.mark.parametrize(
    "case_kwargs",
    [
        {"input": {"value": (1, 2)}, "expected": 1},
        {"input": {"value": 1}, "expected": (1, 2)},
        {"input": {"value": 1}, "expected": {1, 2}},
    ],
    ids=["tuple-in-input", "tuple-expected", "set-expected"],
)
def test_non_json_values_are_rejected(case_kwargs):
    """Tuples and sets would come back as lists, breaking round-trip equality."""
    with pytest.raises(ProblemError, match="not a JSON type"):
        TestCase(id="t001", **case_kwargs)


def test_problem_tests_must_be_a_sequence():
    with pytest.raises(ProblemError, match="must be a sequence"):
        make_problem(tests=5)


def test_duplicate_test_ids_within_a_problem_are_rejected():
    duplicated = (
        TestCase(id="t001", input={"value": 1}, expected=2),
        TestCase(id="t001", input={"value": 2}, expected=3),
    )
    with pytest.raises(ProblemError, match="duplicate test id"):
        make_problem(tests=duplicated)


# ------------------------------------------------------------------------- executor


def test_executor_reports_a_passing_test():
    problem = make_problem()
    result = InProcessReferenceExecutor().run(problem, problem.tests[0])
    assert result.passed
    assert result.test_id == "p999_t001"
    assert result.actual == 2
    assert result.error_type is None


def test_executor_reports_a_failing_test():
    problem = make_problem(
        tests=(TestCase(id="t001", input={"value": 1}, expected=99),)
        + make_problem().tests[1:]
    )
    result = InProcessReferenceExecutor().run(problem, problem.tests[0])
    assert not result.passed
    assert result.actual == 2
    assert result.expected == 99
    assert result.error_type == "AssertionError"


def test_executor_reports_an_unexpected_exception():
    problem = make_problem(
        reference_solution="def add_one(value):\n    raise RuntimeError('boom')\n"
    )
    result = InProcessReferenceExecutor().run(problem, problem.tests[0])
    assert not result.passed
    assert result.error_type == "RuntimeError"
    assert "boom" in result.error_message


def test_executor_matches_an_expected_exception():
    problem = make_problem(
        reference_solution="def add_one(value):\n    raise ValueError('nope')\n",
        tests=(TestCase(id="t001", input={"value": 1}, expected_exception="ValueError"),),
    )
    result = InProcessReferenceExecutor().run(problem, problem.tests[0])
    assert result.passed
    assert result.actual == "ValueError"


def test_executor_flags_a_missing_exception():
    problem = make_problem(
        tests=(TestCase(id="t001", input={"value": 1}, expected_exception="ValueError"),)
    )
    result = InProcessReferenceExecutor().run(problem, problem.tests[0])
    assert not result.passed
    assert "expected ValueError to be raised" in result.error_message


def test_executor_rejects_a_missing_entry_point():
    problem = make_problem(reference_solution="def other(value):\n    return value\n")
    result = InProcessReferenceExecutor().run(problem, problem.tests[0])
    assert not result.passed
    assert result.error_type == "ProblemError"


def test_executor_materializes_generators():
    problem = make_problem(
        reference_solution="def add_one(value):\n    yield value\n    yield value + 1\n",
        tests=(TestCase(id="t001", input={"value": 1}, expected=[1, 2]),),
    )
    result = InProcessReferenceExecutor().run(problem, problem.tests[0])
    assert result.passed
    assert result.actual == [1, 2]


def test_executor_runs_coroutine_functions():
    problem = make_problem(
        signature="async def add_one(value):",
        reference_solution="async def add_one(value):\n    return value + 1\n",
        tests=(TestCase(id="t001", input={"value": 1}, expected=2),),
    )
    result = InProcessReferenceExecutor().run(problem, problem.tests[0])
    assert result.passed
    assert result.actual == 2


def test_executor_does_not_confuse_booleans_with_integers():
    problem = make_problem(
        reference_solution="def add_one(value):\n    return True\n",
        tests=(TestCase(id="t001", input={"value": 1}, expected=1),),
    )
    result = InProcessReferenceExecutor().run(problem, problem.tests[0])
    assert not result.passed


# ------------------------------------------------------------------------ validation


def test_problem_with_too_few_tests_is_invalid():
    problem = make_problem(tests=(TestCase(id="t001", input={"value": 1}, expected=2),))
    report = validate_problem(problem, InProcessReferenceExecutor())
    assert not report.valid
    assert any("at least 5" in error for error in report.errors)


def test_duplicate_problem_ids_are_detected():
    problems = [make_problem(), make_problem()]
    report = validate_dataset(problems, InProcessReferenceExecutor(), expected_count=2)
    assert not report.valid
    assert any("duplicate problem id" in error for error in report.errors)


def test_wrong_problem_count_is_detected():
    report = validate_dataset([make_problem()], InProcessReferenceExecutor())
    assert not report.valid
    assert any("expected exactly 10 problems" in error for error in report.errors)


def test_failing_reference_solution_makes_the_dataset_invalid():
    problem = make_problem(reference_solution="def add_one(value):\n    return 0\n")
    report = validate_dataset([problem], InProcessReferenceExecutor(), expected_count=1)
    assert not report.valid
    assert report.failed_tests == 5
    assert "Dataset validation: FAIL" in format_report(report)


def test_report_lists_every_category_present():
    report = validate_dataset(build_catalog(), InProcessReferenceExecutor())
    assert report.valid
    assert set(report.categories) == set(CATEGORIES)


# ------------------------------------------------------------------------- storage


def test_round_trip_preserves_problems(tmp_path):
    problems = build_catalog()
    path = dataset_path(tmp_path)
    save_problems(problems, path)

    assert load_problems(path) == problems


def test_writer_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "deeper" / "problems.jsonl"
    save_problems([make_problem()], path)
    assert path.is_file()


def test_jsonl_holds_one_object_per_line(tmp_path):
    path = dataset_path(tmp_path)
    save_problems(build_catalog(), path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10
    for line in lines:
        assert json.loads(line)["id"].startswith("p")


def test_loader_rejects_malformed_json_with_a_line_number(tmp_path):
    path = dataset_path(tmp_path)
    save_problems(build_catalog(), path)

    lines = path.read_text(encoding="utf-8").splitlines()
    lines[3] = "{not valid json"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(DatasetError, match=r":4: invalid JSON"):
        load_problems(path)


def test_loader_rejects_an_invalid_record(tmp_path):
    path = dataset_path(tmp_path)
    record = make_problem().to_dict()
    record["category"] = "nonsense"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="unknown category"):
        load_problems(path)


def test_loader_rejects_duplicate_ids(tmp_path):
    path = dataset_path(tmp_path)
    line = json.dumps(make_problem().to_dict())
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="duplicate problem id"):
        load_problems(path)


def test_loader_reports_a_missing_file(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_problems(dataset_path(tmp_path))
