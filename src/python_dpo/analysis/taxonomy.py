"""The deterministic failure taxonomy (spec 11 sections 10, 11, 12).

Section 12 is explicit that classification is **deterministic and never an LLM**. The
evidence is pytest's own status, the raw exception class, the exit code and the timeout
flag -- all facts Stage 6 and Stage 10 already recorded. An LLM judge is forbidden as the
primary classifier precisely because this stage's output drives what gets trained next: a
probabilistic label there would make every downstream recommendation unfalsifiable.

The **coarse category** is taken from Stage 10's ``EvaluationRecord.error_type`` rather
than re-derived. Section 12 prefers the pytest result, and re-deriving it here would risk
Stage 11 disagreeing with Stage 10 about what failed -- two components reporting different
causes for the same run is worse than either being slightly coarse. Two categories Stage 10
has no name for are resolved locally, from the generation record:

* ``generation_failure``      -- the model errored before producing text at all.
* ``code_extraction_failure`` -- text came back, but no Python could be extracted from it.

``memory_error`` is promoted out of ``runtime_error`` when the subcategory says
``MemoryError``, because section 10 lists it separately and the distinction changes the
recommendation (a memory error is a resource problem, not a logic one).

The **subcategory** (section 11) is the raw exception class from the per-test records --
``TypeError``, ``IndexError``, ``KeyError`` and so on. Anything unrecognised maps to
``other`` rather than being invented: the real data's ``Failed`` (pytest's own
``pytest.fail``, not a builtin) is the case that proves this branch is needed.

When a candidate fails several tests with *different* exceptions, the subcategory is the
**most frequent, ties broken alphabetically** -- deterministic, and the full per-subcategory
counts are recorded alongside it so nothing is discarded (CLAUDE.md's Data Integrity rule).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# Section 10's closed set. `generation_failure` and `code_extraction_failure` are Stage 11
# additions; the remaining seven mirror Stage 10's EVALUATION_ERROR_TYPES plus memory_error.
ERROR_CATEGORIES: tuple[str, ...] = (
    "generation_failure",
    "code_extraction_failure",
    "syntax_error",
    "import_error",
    "runtime_error",
    "assertion_failure",
    "timeout",
    "memory_error",
    "infrastructure_error",
)

# Section 11's recognised exception classes. Anything outside this set becomes `other`
# rather than being passed through -- an open-ended subcategory would let a typo in an
# upstream field silently become a taxonomy entry.
KNOWN_SUBCATEGORIES: tuple[str, ...] = (
    "AssertionError",
    "AttributeError",
    "IndexError",
    "KeyError",
    "MemoryError",
    "NameError",
    "RecursionError",
    "StopIteration",
    "SyntaxError",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
)

OTHER_SUBCATEGORY = "other"
NO_FAILURE = "none"


def normalize_subcategory(error_type: str | None) -> str:
    """Map a raw exception class onto section 11's recognised set, or ``other``."""
    if not error_type:
        return OTHER_SUBCATEGORY
    return error_type if error_type in KNOWN_SUBCATEGORIES else OTHER_SUBCATEGORY


def dominant_subcategory(error_types: Sequence[str | None]) -> tuple[str, dict[str, int]]:
    """The most frequent subcategory across a candidate's failing tests, plus full counts.

    Ties are broken alphabetically so the result never depends on iteration order. Returns
    ``(NO_FAILURE, {})`` when nothing failed.
    """
    normalized = [normalize_subcategory(e) for e in error_types if e is not None]
    if not normalized:
        return NO_FAILURE, {}
    counts = Counter(normalized)
    # -count first so higher counts sort earlier; the name second gives the alphabetical
    # tie-break.
    winner = min(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return winner, dict(sorted(counts.items()))


@dataclass(frozen=True)
class ErrorClassification:
    """One (problem, variant, sample)'s failure, classified (sections 10, 11)."""

    problem_id: str
    model_variant: str
    sample_index: int
    category: str
    subcategory: str
    subcategory_counts: dict[str, int] = field(default_factory=dict)
    tests_total: int = 0
    tests_passed: int = 0
    passed: bool = False

    def __post_init__(self) -> None:
        if self.category not in ERROR_CATEGORIES and self.category != NO_FAILURE:
            raise ValueError(
                f"category must be one of {', '.join((*ERROR_CATEGORIES, NO_FAILURE))}, "
                f"got {self.category!r}"
            )
        if not isinstance(self.subcategory_counts, dict):
            raise ValueError("subcategory_counts must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "model_variant": self.model_variant,
            "sample_index": self.sample_index,
            "category": self.category,
            "subcategory": self.subcategory,
            "subcategory_counts": dict(self.subcategory_counts),
            "tests_total": self.tests_total,
            "tests_passed": self.tests_passed,
            "passed": self.passed,
        }


def classify(
    generation_record: Any,
    evaluation_record: Any | None,
    test_results: Sequence[Any] = (),
) -> ErrorClassification:
    """Classify one sample's outcome (sections 10, 11, 12).

    ``evaluation_record`` is ``None`` when the sample never reached the sandbox -- which is
    exactly the generation-failure case, and why the generation record is the first thing
    consulted rather than a fallback.
    """
    problem_id = generation_record.problem_id
    variant = generation_record.model_variant
    sample_index = generation_record.sample_index

    # Section 10: a generation that never produced usable code is classified from the
    # generation record alone; there is no execution evidence to consult.
    if generation_record.status == "generation_error":
        return ErrorClassification(
            problem_id=problem_id, model_variant=variant, sample_index=sample_index,
            category="generation_failure", subcategory=OTHER_SUBCATEGORY,
        )
    if generation_record.extracted_code is None:
        # Text came back but nothing extractable was in it -- distinct from the model
        # erroring outright, and it points at a different fix (prompt/format, not decoding).
        category = (
            "code_extraction_failure"
            if (generation_record.raw_response or "").strip()
            else "generation_failure"
        )
        return ErrorClassification(
            problem_id=problem_id, model_variant=variant, sample_index=sample_index,
            category=category, subcategory=OTHER_SUBCATEGORY,
        )

    if evaluation_record is None:
        raise ValueError(
            f"{problem_id}/{variant}/{sample_index}: code was extracted but no evaluation "
            "record exists; the sample's outcome cannot be classified"
        )

    subcategory, counts = dominant_subcategory(
        [r.error_type for r in test_results if r.status in ("failed", "error")]
    )

    tests_total = evaluation_record.tests_total
    tests_passed = evaluation_record.tests_passed
    passed = (
        evaluation_record.status == "passed"
        if hasattr(evaluation_record, "status")
        else tests_total > 0 and tests_passed == tests_total
    )

    if passed:
        return ErrorClassification(
            problem_id=problem_id, model_variant=variant, sample_index=sample_index,
            category=NO_FAILURE, subcategory=NO_FAILURE,
            tests_total=tests_total, tests_passed=tests_passed, passed=True,
        )

    category = evaluation_record.error_type or "runtime_error"
    # Section 10 lists memory_error separately; Stage 10 folds it into runtime_error
    # because its own taxonomy has no such bucket.
    if category == "runtime_error" and subcategory == "MemoryError":
        category = "memory_error"
    if category not in ERROR_CATEGORIES:
        category = "runtime_error"

    return ErrorClassification(
        problem_id=problem_id, model_variant=variant, sample_index=sample_index,
        category=category, subcategory=subcategory, subcategory_counts=counts,
        tests_total=tests_total, tests_passed=tests_passed, passed=False,
    )


__all__ = [
    "ERROR_CATEGORIES",
    "KNOWN_SUBCATEGORIES",
    "NO_FAILURE",
    "OTHER_SUBCATEGORY",
    "ErrorClassification",
    "classify",
    "dominant_subcategory",
    "normalize_subcategory",
]
