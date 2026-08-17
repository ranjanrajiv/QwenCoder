"""End-to-end generation tests driven by the mock model (spec 03 sections 46, 47).

No test here loads the real Qwen model, and none needs a GPU, weights, or a network.
"""

from __future__ import annotations

import pytest

from python_dpo.candidates import CandidateRepository
from python_dpo.generation import (
    PROMPT_VERSION,
    STRATEGIES,
    CandidateGenerator,
    resolve_strategies,
)
from python_dpo.models import GenerationConfig, InferenceError, MockModelClient, ModelLoadError
from python_dpo.problems import Problem, TestCase

GOOD_CODE = "def sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)"
GOOD_OUTPUT = f"```python\n{GOOD_CODE}\n```"
BROKEN_OUTPUT = "```python\ndef sum_even(numbers:\n    return 0\n```"
PROSE_OUTPUT = "I am not able to help with that request."
WRONG_NAME_OUTPUT = "```python\ndef solve(numbers):\n    return 0\n```"


@pytest.fixture
def problem() -> Problem:
    return Problem(
        id="p001",
        prompt="Return the sum of the even integers in a list.",
        signature="def sum_even(numbers):",
        entry_point="sum_even",
        category="lists",
        difficulty="easy",
        reference_solution="def sum_even(numbers):\n    return 0\n",
        tests=(TestCase(id="t001", input={"numbers": [2]}, expected=2),),
    )


def build(tmp_path, client=None):
    repository = CandidateRepository(tmp_path)
    generator = CandidateGenerator(
        client=client or MockModelClient(),
        repository=repository,
        generation_config=GenerationConfig(),
    )
    return generator, repository


def run(generator, problem, *, count=5, run_id="20260817_100000", force=False, override=None):
    strategies = resolve_strategies(STRATEGIES, count, override=override)
    return generator.generate(
        [problem], count=count, strategies=strategies, run_id=run_id, force=force
    )


# ------------------------------------------------------------------- integration (§47)


def test_one_problem_five_candidates(tmp_path, problem):
    generator, repository = build(tmp_path)
    summary = run(generator, problem)

    assert summary.generated == 5
    assert summary.failed == 0

    records = repository.load_all()
    assert len(records) == 5
    assert {r.problem_id for r in records} == {"p001"}
    assert len({r.candidate_id for r in records}) == 5
    assert {r.strategy for r in records} == set(STRATEGIES)
    assert [r.generation_index for r in records] == [1, 2, 3, 4, 5]
    assert all(r.raw_output for r in records)
    assert all(r.code for r in records)
    assert all(r.syntax_valid for r in records)
    assert all(r.function_name_valid for r in records)
    assert all(r.prompt_version == PROMPT_VERSION for r in records)
    assert all(r.run_id == "20260817_100000" for r in records)
    assert all(r.generation_config == GenerationConfig().to_dict() for r in records)
    assert not repository.failures_path.exists()


def test_candidate_ids_follow_the_documented_shape(tmp_path, problem):
    generator, repository = build(tmp_path)
    run(generator, problem)
    assert [r.candidate_id for r in repository.load_all()] == [
        "p001_c001",
        "p001_c002",
        "p001_c003",
        "p001_c004",
        "p001_c005",
    ]


def test_raw_output_is_kept_alongside_extracted_code(tmp_path, problem):
    generator, repository = build(tmp_path, MockModelClient(script=[GOOD_OUTPUT]))
    run(generator, problem, count=1)

    record = repository.load_all()[0]
    assert record.raw_output == GOOD_OUTPUT
    assert record.code == GOOD_CODE
    assert record.raw_output != record.code
    assert record.extraction_format == "python_fence"


def test_strategy_override_applies_to_every_candidate(tmp_path, problem):
    generator, repository = build(tmp_path)
    run(generator, problem, count=3, override=["edge_case_focused"])
    assert {r.strategy for r in repository.load_all()} == {"edge_case_focused"}


def test_mismatched_strategy_count_is_a_programming_error(tmp_path, problem):
    generator, _ = build(tmp_path)
    with pytest.raises(ValueError, match="resolved strategies"):
        generator.generate([problem], count=5, strategies=("normal",), run_id="r1")


# ------------------------------------------------------------------------------ resume


def test_second_run_resumes_and_regenerates_nothing(tmp_path, problem):
    client = MockModelClient()
    generator, repository = build(tmp_path, client)
    run(generator, problem)
    assert client.call_count == 5

    summary = run(generator, problem, run_id="20260817_110000")
    assert summary.generated == 0
    assert summary.skipped == 5
    assert client.call_count == 5, "resume must not call the model again"
    assert len(repository.load_all()) == 5


def test_force_appends_a_new_run_without_touching_the_old_one(tmp_path, problem):
    generator, repository = build(tmp_path)
    run(generator, problem)
    first_run = repository.load_all()

    summary = run(generator, problem, run_id="20260817_110000", force=True)
    assert summary.generated == 5
    assert summary.skipped == 0

    records = repository.load_all()
    assert len(records) == 10
    assert records[:5] == first_run, "the earlier run must be preserved verbatim"
    assert {r.run_id for r in records} == {"20260817_100000", "20260817_110000"}

    latest = repository.latest_by_candidate_id()
    assert len(latest) == 5
    assert {c.run_id for c in latest.values()} == {"20260817_110000"}


def test_resume_retries_a_generation_that_previously_failed(tmp_path, problem):
    # First attempt: candidate 2 produces nothing extractable.
    generator, repository = build(
        tmp_path, MockModelClient(script=[GOOD_OUTPUT, PROSE_OUTPUT])
    )
    run(generator, problem, count=3)
    assert {r.generation_index for r in repository.load_all()} == {1, 3}

    generator, repository = build(tmp_path, MockModelClient())
    summary = run(generator, problem, count=3, run_id="20260817_110000")
    assert summary.generated == 1
    assert summary.skipped == 2
    assert {r.generation_index for r in repository.load_all()} == {1, 2, 3}


# ---------------------------------------------------------------------- failure paths


def test_empty_response_is_a_failure_not_a_candidate(tmp_path, problem):
    generator, repository = build(tmp_path, MockModelClient(script=["   "]))
    summary = run(generator, problem, count=2)

    assert summary.generated == 1
    assert summary.failed == 1
    assert len(repository.load_all()) == 1

    failures = repository.load_failures()
    assert len(failures) == 1
    assert failures[0].error_type == "empty_output"
    assert failures[0].generation_index == 1
    assert failures[0].strategy == "normal"
    assert failures[0].run_id == "20260817_100000"


def test_unextractable_output_is_a_failure_not_a_candidate(tmp_path, problem):
    generator, repository = build(tmp_path, MockModelClient(script=[PROSE_OUTPUT]))
    summary = run(generator, problem, count=2)

    assert summary.generated == 1
    assert summary.failed == 1
    assert repository.load_failures()[0].error_type == "code_extraction"
    assert "p001_c001" not in {r.candidate_id for r in repository.load_all()}


def test_inference_error_is_recorded_and_the_run_continues(tmp_path, problem):
    generator, repository = build(
        tmp_path, MockModelClient(script=[InferenceError("CUDA blew up")])
    )
    summary = run(generator, problem, count=5)

    assert summary.failed == 1
    assert summary.generated == 4
    failure = repository.load_failures()[0]
    assert failure.error_type == "inference"
    assert "CUDA blew up" in failure.error_message


def test_model_load_failure_aborts_the_run(tmp_path, problem):
    # Section 26.2: no candidate in the run can succeed, so retrying per candidate would
    # only emit one identical failure per generation.
    generator, repository = build(
        tmp_path, MockModelClient(script=[ModelLoadError("no weights")])
    )
    with pytest.raises(ModelLoadError):
        run(generator, problem, count=5)

    failures = repository.load_failures()
    assert len(failures) == 1
    assert failures[0].error_type == "model_load"
    assert repository.load_all() == []


def test_syntax_error_produces_a_candidate_and_no_failure_record(tmp_path, problem):
    # Section 19.1: the record is the model's actual output, malformed or not.
    generator, repository = build(tmp_path, MockModelClient(script=[BROKEN_OUTPUT]))
    summary = run(generator, problem, count=1)

    assert summary.generated == 1
    assert summary.failed == 0

    record = repository.load_all()[0]
    assert record.syntax_valid is False
    assert record.syntax_error
    assert record.function_name_valid is False
    assert record.code == "def sum_even(numbers:\n    return 0"
    assert not repository.failures_path.exists()


def test_wrong_function_name_is_recorded_not_rejected(tmp_path, problem):
    generator, repository = build(tmp_path, MockModelClient(script=[WRONG_NAME_OUTPUT]))
    run(generator, problem, count=1)

    record = repository.load_all()[0]
    assert record.syntax_valid is True
    assert record.function_name_valid is False


# -------------------------------------------------------------------------- duplicates


def test_exact_duplicates_are_flagged_and_kept(tmp_path, problem):
    generator, repository = build(
        tmp_path, MockModelClient(script=[GOOD_OUTPUT, GOOD_OUTPUT, GOOD_OUTPUT])
    )
    summary = run(generator, problem, count=3)

    assert summary.generated == 3
    assert summary.duplicates == 2

    records = repository.load_all()
    assert len(records) == 3, "duplicates are kept for analysis, never deleted"
    assert records[0].duplicate_of is None
    assert records[1].duplicate_of == "p001_c001"
    assert records[2].duplicate_of == "p001_c001"


def test_duplicate_detection_spans_runs(tmp_path, problem):
    # The realistic case: a run dies partway, and the resumed run produces code identical
    # to a candidate the earlier run already stored.
    generator, repository = build(
        tmp_path, MockModelClient(script=[GOOD_OUTPUT, PROSE_OUTPUT])
    )
    run(generator, problem, count=2)
    assert {r.candidate_id for r in repository.load_all()} == {"p001_c001"}

    generator, repository = build(tmp_path, MockModelClient(script=[GOOD_OUTPUT]))
    run(generator, problem, count=2, run_id="20260817_110000")

    resumed = repository.load_all()[1]
    assert resumed.candidate_id == "p001_c002"
    assert resumed.run_id == "20260817_110000"
    assert resumed.duplicate_of == "p001_c001"


def test_regenerating_the_same_index_is_not_a_duplicate_of_itself(tmp_path, problem):
    generator, repository = build(tmp_path, MockModelClient(script=[GOOD_OUTPUT]))
    run(generator, problem, count=1)

    generator, repository = build(tmp_path, MockModelClient(script=[GOOD_OUTPUT]))
    summary = run(generator, problem, count=1, run_id="20260817_110000", force=True)

    records = repository.load_all()
    assert len(records) == 2
    assert records[1].candidate_id == records[0].candidate_id
    assert records[1].duplicate_of is None
    assert summary.duplicates == 0


def test_distinct_code_is_not_flagged(tmp_path, problem):
    generator, repository = build(tmp_path)
    run(generator, problem)
    assert all(record.duplicate_of is None for record in repository.load_all())
