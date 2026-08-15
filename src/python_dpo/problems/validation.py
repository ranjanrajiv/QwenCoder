"""Dataset integrity validation and human-readable reporting.

A problem counts as valid only when its schema checks pass, its reference solution
loads, and every one of its tests passes (spec 02 section 22). Validation talks to
solutions exclusively through the ``ReferenceExecutor`` protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .executor import ReferenceExecutor
from .models import (
    CATEGORIES,
    DIFFICULTIES,
    EXPECTED_PROBLEM_COUNT,
    MIN_TESTS_PER_PROBLEM,
    Problem,
    TestResult,
)


@dataclass(frozen=True)
class ProblemReport:
    """Validation outcome for a single problem."""

    problem_id: str
    category: str
    errors: tuple[str, ...]
    results: tuple[TestResult, ...]

    @property
    def valid(self) -> bool:
        return not self.errors and all(result.passed for result in self.results)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed(self) -> int:
        return sum(1 for result in self.results if not result.passed)


@dataclass(frozen=True)
class DatasetReport:
    """Validation outcome for a whole dataset."""

    problems: tuple[ProblemReport, ...]
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors and all(report.valid for report in self.problems)

    @property
    def total_problems(self) -> int:
        return len(self.problems)

    @property
    def valid_problems(self) -> int:
        return sum(1 for report in self.problems if report.valid)

    @property
    def invalid_problems(self) -> int:
        return self.total_problems - self.valid_problems

    @property
    def total_tests(self) -> int:
        return sum(len(report.results) for report in self.problems)

    @property
    def passed_tests(self) -> int:
        return sum(report.passed for report in self.problems)

    @property
    def failed_tests(self) -> int:
        return sum(report.failed for report in self.problems)

    @property
    def categories(self) -> dict[str, int]:
        """Category counts in dataset order of first appearance."""
        counts: dict[str, int] = {}
        for report in self.problems:
            counts[report.category] = counts.get(report.category, 0) + 1
        return counts

    def failure_messages(self) -> list[str]:
        """Every dataset- and problem-level failure, ready for logging."""
        messages = list(self.errors)
        for report in self.problems:
            messages.extend(f"{report.problem_id}: {error}" for error in report.errors)
            for result in report.results:
                if not result.passed:
                    detail = result.error_message or "test failed"
                    error_type = f"{result.error_type}: " if result.error_type else ""
                    messages.append(f"{result.test_id}: {error_type}{detail}")
        return messages


def validate_problem(problem: Problem, executor: ReferenceExecutor) -> ProblemReport:
    """Check one problem's schema invariants and run its reference tests."""
    errors: list[str] = []

    if not problem.prompt.strip():
        errors.append("prompt must not be empty")
    if not problem.signature.strip():
        errors.append("signature must not be empty")
    if not problem.reference_solution.strip():
        errors.append("reference solution must not be empty")
    if problem.category not in CATEGORIES:
        errors.append(f"unknown category {problem.category!r}")
    if problem.difficulty not in DIFFICULTIES:
        errors.append(f"unknown difficulty {problem.difficulty!r}")
    if len(problem.tests) < MIN_TESTS_PER_PROBLEM:
        errors.append(
            f"has {len(problem.tests)} test(s); at least {MIN_TESTS_PER_PROBLEM} are required"
        )

    test_ids = [test.id for test in problem.tests]
    duplicates = sorted({test_id for test_id in test_ids if test_ids.count(test_id) > 1})
    if duplicates:
        errors.append(f"duplicate test id(s): {', '.join(duplicates)}")

    results = tuple(executor.run(problem, test) for test in problem.tests)

    return ProblemReport(
        problem_id=problem.id,
        category=problem.category,
        errors=tuple(errors),
        results=results,
    )


def validate_dataset(
    problems: Sequence[Problem],
    executor: ReferenceExecutor,
    *,
    expected_count: int = EXPECTED_PROBLEM_COUNT,
) -> DatasetReport:
    """Validate a whole dataset: counts, unique ids, and every reference test."""
    errors: list[str] = []

    if len(problems) != expected_count:
        errors.append(f"expected exactly {expected_count} problems, found {len(problems)}")

    ids = [problem.id for problem in problems]
    duplicates = sorted({problem_id for problem_id in ids if ids.count(problem_id) > 1})
    if duplicates:
        errors.append(f"duplicate problem id(s): {', '.join(duplicates)}")

    reports = tuple(validate_problem(problem, executor) for problem in problems)

    return DatasetReport(problems=reports, errors=tuple(errors))


def format_report(report: DatasetReport) -> str:
    """Render a dataset report as the summary block described in spec 02 section 31."""
    lines = [
        _row("Problems:", report.total_problems),
        _row("Valid:", report.valid_problems),
        _row("Invalid:", report.invalid_problems),
        "",
        "Categories:",
    ]
    lines.extend(
        _row(f"{category}:", count, indent=2) for category, count in report.categories.items()
    )
    lines.extend(
        [
            "",
            "Reference tests:",
            _row("Total:", report.total_tests, indent=2),
            _row("Passed:", report.passed_tests, indent=2),
            _row("Failed:", report.failed_tests, indent=2),
        ]
    )

    failures = report.failure_messages()
    if failures:
        lines.extend(["", "Failures:"])
        lines.extend(f"  {message}" for message in failures)

    lines.extend(["", f"Dataset validation: {'PASS' if report.valid else 'FAIL'}"])
    return "\n".join(lines) + "\n"


def _row(label: str, value: int, *, indent: int = 0) -> str:
    padding = " " * indent
    return f"{padding}{label:<{22 - indent}}{value:>3}"
