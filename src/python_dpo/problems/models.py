"""Typed schema for problems, test cases, and reference test results.

The models are frozen dataclasses that validate in ``__post_init__``, matching the
configuration layer from Stage 1. Constructing an invalid object is impossible: every
entry point into the dataset (the catalog, the JSONL loader) therefore produces
already-validated records.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# Version of the problem dataset format and content. Deliberately independent of
# python_dpo.__version__ so dataset revisions and software releases can move apart.
DATASET_VERSION = "0.1.0"

CATEGORIES = frozenset(
    {
        "lists",
        "dictionaries",
        "strings",
        "sets",
        "sorting",
        "recursion",
        "edge_cases",
        "exceptions",
        "generators",
        "async",
    }
)

DIFFICULTIES = frozenset({"easy", "medium", "hard"})

MIN_TESTS_PER_PROBLEM = 5
EXPECTED_PROBLEM_COUNT = 10

_TEST_CASE_FIELDS = frozenset({"id", "input", "expected", "expected_exception"})
_PROBLEM_FIELDS = frozenset(
    {
        "id",
        "prompt",
        "signature",
        "entry_point",
        "category",
        "difficulty",
        "reference_solution",
        "tests",
        "description",
        "tags",
        "source",
        "metadata",
        "dataset_version",
    }
)


class ProblemError(Exception):
    """Raised when a problem or test case fails schema validation."""


def _require_json_native(value: Any, label: str) -> None:
    """Reject values that would not survive a JSONL round trip unchanged.

    Tuples and sets are the trap: ``json`` happily writes them out as arrays, but they
    come back as lists, so a reloaded dataset would no longer equal the original.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for item in value:
            _require_json_native(item, label)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProblemError(f"{label}: JSON object keys must be strings, got {key!r}")
            _require_json_native(item, label)
        return
    raise ProblemError(
        f"{label}: {type(value).__name__} is not a JSON type; use lists, dicts, strings, "
        "numbers, booleans, or None so the dataset round-trips unchanged"
    )


@dataclass(frozen=True)
class TestCase:
    """A single executable check for a problem.

    ``input`` is a mapping of keyword arguments for the problem's entry point, kept as
    structured JSON-native values rather than serialized Python expressions. A case
    either asserts a return value (``expected``) or an exception type name
    (``expected_exception``), never both.
    """

    # Not a pytest test class despite the name; keeps collection warnings away.
    __test__ = False

    id: str
    input: dict[str, Any]
    expected: Any = None
    expected_exception: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ProblemError("test case: 'id' must be a non-empty string")
        if not isinstance(self.input, dict):
            raise ProblemError(f"{self.id}: 'input' must be a mapping of keyword arguments")
        for key in self.input:
            if not isinstance(key, str):
                raise ProblemError(f"{self.id}: 'input' keys must be strings (argument names)")
        if self.expected_exception is not None:
            if not isinstance(self.expected_exception, str) or not self.expected_exception.strip():
                raise ProblemError(
                    f"{self.id}: 'expected_exception' must be an exception type name, "
                    "for example 'ValueError'"
                )
            if self.expected is not None:
                raise ProblemError(
                    f"{self.id}: a test case cannot set both 'expected' and 'expected_exception'"
                )

        _require_json_native(self.input, f"{self.id}: 'input'")
        _require_json_native(self.expected, f"{self.id}: 'expected'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input": self.input,
            "expected": self.expected,
            "expected_exception": self.expected_exception,
        }

    @classmethod
    def from_dict(cls, data: Any) -> TestCase:
        if not isinstance(data, dict):
            raise ProblemError("test case: expected a JSON object")
        unknown = sorted(set(data) - _TEST_CASE_FIELDS)
        if unknown:
            raise ProblemError(f"test case: unknown field(s): {', '.join(unknown)}")
        missing = sorted({"id", "input"} - set(data))
        if missing:
            raise ProblemError(f"test case: missing required field(s): {', '.join(missing)}")
        return cls(
            id=data["id"],
            input=data["input"],
            expected=data.get("expected"),
            expected_exception=data.get("expected_exception"),
        )


@dataclass(frozen=True)
class Problem:
    """A programming task with a trusted reference solution and executable tests."""

    id: str
    prompt: str
    signature: str
    entry_point: str
    category: str
    difficulty: str
    reference_solution: str
    tests: tuple[TestCase, ...]
    description: str | None = None
    tags: tuple[str, ...] = ()
    source: str = "manual"
    metadata: dict[str, Any] = field(default_factory=dict)
    dataset_version: str = DATASET_VERSION

    def __post_init__(self) -> None:
        # Accept sequences from JSON and normalize to tuples so the record stays frozen.
        for name in ("tests", "tags"):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
                raise ProblemError(f"{self._label}: {name!r} must be a sequence")
            object.__setattr__(self, name, tuple(value))

        for name in ("id", "prompt", "signature", "entry_point", "reference_solution"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ProblemError(f"{self._label}: '{name}' must be a non-empty string")

        if self.category not in CATEGORIES:
            raise ProblemError(
                f"{self._label}: unknown category {self.category!r}; "
                f"expected one of {', '.join(sorted(CATEGORIES))}"
            )
        if self.difficulty not in DIFFICULTIES:
            raise ProblemError(
                f"{self._label}: unknown difficulty {self.difficulty!r}; "
                f"expected one of {', '.join(sorted(DIFFICULTIES))}"
            )
        if not self.entry_point.isidentifier():
            raise ProblemError(
                f"{self._label}: 'entry_point' must be a Python identifier, got {self.entry_point!r}"
            )
        if self.entry_point not in self.signature:
            raise ProblemError(
                f"{self._label}: 'signature' does not declare the entry point "
                f"{self.entry_point!r}"
            )
        if not self.tests:
            raise ProblemError(f"{self._label}: 'tests' must not be empty")

        for test in self.tests:
            if not isinstance(test, TestCase):
                raise ProblemError(f"{self._label}: 'tests' must contain TestCase objects")
        seen: set[str] = set()
        for test in self.tests:
            if test.id in seen:
                raise ProblemError(f"{self._label}: duplicate test id {test.id!r}")
            seen.add(test.id)

        if not isinstance(self.metadata, dict):
            raise ProblemError(f"{self._label}: 'metadata' must be a mapping")

    @property
    def _label(self) -> str:
        return self.id if isinstance(self.id, str) and self.id else "problem"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "signature": self.signature,
            "entry_point": self.entry_point,
            "category": self.category,
            "difficulty": self.difficulty,
            "reference_solution": self.reference_solution,
            "tests": [test.to_dict() for test in self.tests],
            "description": self.description,
            "tags": list(self.tags),
            "source": self.source,
            "metadata": self.metadata,
            "dataset_version": self.dataset_version,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Problem:
        if not isinstance(data, dict):
            raise ProblemError("problem: expected a JSON object")
        unknown = sorted(set(data) - _PROBLEM_FIELDS)
        if unknown:
            raise ProblemError(f"problem: unknown field(s): {', '.join(unknown)}")
        required = {
            "id",
            "prompt",
            "signature",
            "entry_point",
            "category",
            "difficulty",
            "reference_solution",
            "tests",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ProblemError(f"problem: missing required field(s): {', '.join(missing)}")

        raw_tests = data["tests"]
        if not isinstance(raw_tests, list):
            raise ProblemError(f"{data['id']}: 'tests' must be a list")

        return cls(
            id=data["id"],
            prompt=data["prompt"],
            signature=data["signature"],
            entry_point=data["entry_point"],
            category=data["category"],
            difficulty=data["difficulty"],
            reference_solution=data["reference_solution"],
            tests=tuple(TestCase.from_dict(test) for test in raw_tests),
            description=data.get("description"),
            tags=tuple(data.get("tags") or ()),
            source=data.get("source", "manual"),
            metadata=data.get("metadata") or {},
            dataset_version=data.get("dataset_version", DATASET_VERSION),
        )


@dataclass(frozen=True)
class TestResult:
    """Outcome of executing one test case against a solution."""

    # Not a pytest test class despite the name; keeps collection warnings away.
    __test__ = False

    test_id: str
    passed: bool
    actual: Any = None
    expected: Any = None
    error_type: str | None = None
    error_message: str | None = None
