"""Tests for the deterministic failure taxonomy (spec 11 sections 10-12, 113)."""

from __future__ import annotations

import pytest

from python_dpo.analysis.taxonomy import (
    NO_FAILURE,
    OTHER_SUBCATEGORY,
    classify,
    dominant_subcategory,
    normalize_subcategory,
)

from .conftest import FakeEvaluation, FakeGeneration, FakeTestResult


# --------------------------------------------------------------- section 113's named cases


def test_syntax_error_classifies_as_syntax_error():
    generation = FakeGeneration("p001", "base", 0)
    evaluation = FakeEvaluation("p001", "base", 0, tests_passed=0, status="failed",
                                error_type="syntax_error")
    assert classify(generation, evaluation, []).category == "syntax_error"


def test_type_error_classifies_as_runtime_error_with_typeerror_subcategory():
    generation = FakeGeneration("p001", "base", 0)
    evaluation = FakeEvaluation("p001", "base", 0, tests_passed=0, status="failed",
                                error_type="runtime_error")
    results = [FakeTestResult("p001", "p001_c001", "t1", status="failed", error_type="TypeError")]
    classification = classify(generation, evaluation, results)
    assert classification.category == "runtime_error"
    assert classification.subcategory == "TypeError"


def test_timeout_classifies_as_timeout():
    generation = FakeGeneration("p001", "base", 0)
    evaluation = FakeEvaluation("p001", "base", 0, tests_passed=0, status="timeout",
                                error_type="timeout")
    assert classify(generation, evaluation, []).category == "timeout"


# ------------------------------------------------------------------ the additional branches


def test_memory_error_is_promoted_out_of_runtime_error():
    """Section 10 lists memory_error separately; Stage 10 has no such bucket and folds it
    into runtime_error, so Stage 11 promotes it back out."""
    generation = FakeGeneration("p001", "base", 0)
    evaluation = FakeEvaluation("p001", "base", 0, tests_passed=0, status="failed",
                                error_type="runtime_error")
    results = [FakeTestResult("p001", "p001_c001", "t1", status="failed", error_type="MemoryError")]
    assert classify(generation, evaluation, results).category == "memory_error"


def test_unrecognised_exception_maps_to_other():
    """The real data's `Failed` (pytest.fail, not a builtin) is why this branch exists."""
    assert normalize_subcategory("Failed") == OTHER_SUBCATEGORY
    assert normalize_subcategory("SomethingInvented") == OTHER_SUBCATEGORY


def test_generation_failure_is_distinct_from_extraction_failure():
    errored = FakeGeneration("p001", "base", 0, status="generation_error", extracted_code=None)
    assert classify(errored, None, []).category == "generation_failure"

    prose = FakeGeneration(
        "p001", "base", 0, extracted_code=None, raw_response="I cannot help with that."
    )
    assert classify(prose, None, []).category == "code_extraction_failure"


def test_empty_response_is_a_generation_failure_not_an_extraction_failure():
    empty = FakeGeneration("p001", "base", 0, extracted_code=None, raw_response="   ")
    assert classify(empty, None, []).category == "generation_failure"


def test_passing_sample_classifies_as_no_failure():
    generation = FakeGeneration("p001", "base", 0)
    evaluation = FakeEvaluation("p001", "base", 0)
    classification = classify(generation, evaluation, [])
    assert classification.category == NO_FAILURE
    assert classification.passed is True


def test_extracted_code_without_an_evaluation_record_is_an_error_not_a_guess():
    generation = FakeGeneration("p001", "base", 0)
    with pytest.raises(ValueError, match="cannot be classified"):
        classify(generation, None, [])


# ------------------------------------------------------------------ subcategory selection


def test_most_frequent_subcategory_wins():
    winner, counts = dominant_subcategory(["TypeError", "KeyError", "KeyError"])
    assert winner == "KeyError"
    assert counts == {"KeyError": 2, "TypeError": 1}


def test_subcategory_ties_break_alphabetically():
    """Deterministic rather than dependent on iteration order."""
    winner, _ = dominant_subcategory(["ValueError", "KeyError"])
    assert winner == "KeyError"


def test_full_subcategory_counts_are_retained():
    """Data Integrity: the losing subcategories are recorded, not discarded."""
    _, counts = dominant_subcategory(["TypeError", "KeyError", "KeyError", "IndexError"])
    assert counts == {"IndexError": 1, "KeyError": 2, "TypeError": 1}


def test_no_failing_tests_yields_no_failure_subcategory():
    assert dominant_subcategory([]) == (NO_FAILURE, {})
