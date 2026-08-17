"""Tests for the candidate schema and the append-only JSONL repository."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from python_dpo.candidates import (
    Candidate,
    CandidateError,
    CandidateRepository,
    CandidateStoreError,
    GenerationFailure,
    build_candidate_id,
    utc_now_iso,
)

CODE = "def sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)"


def make_candidate(**overrides: Any) -> Candidate:
    fields: dict[str, Any] = {
        "candidate_id": "p001_c001",
        "problem_id": "p001",
        "run_id": "20260817_120000",
        "generation_index": 1,
        "strategy": "normal",
        "model": "mock/deterministic-coder",
        "provider": "mock",
        "prompt_version": "v1",
        "prompt": "Solve the problem.",
        "raw_output": f"```python\n{CODE}\n```",
        "code": CODE,
        "extraction_format": "python_fence",
        "syntax_valid": True,
        "function_name_valid": True,
        "generation_config": {"temperature": 0.8, "seed": 42},
        "created_at": utc_now_iso(),
    }
    fields.update(overrides)
    return Candidate(**fields)


def make_failure(**overrides: Any) -> GenerationFailure:
    fields: dict[str, Any] = {
        "run_id": "20260817_120000",
        "problem_id": "p001",
        "generation_index": 3,
        "strategy": "alternative",
        "error_type": "code_extraction",
        "error_message": "No Python code detected",
        "timestamp": utc_now_iso(),
    }
    fields.update(overrides)
    return GenerationFailure(**fields)


# ------------------------------------------------------------------------------ schema


def test_candidate_ids_are_deterministic_and_zero_padded():
    assert build_candidate_id("p001", 1) == "p001_c001"
    assert build_candidate_id("p010", 12) == "p010_c012"


def test_valid_candidate_round_trips_through_dict():
    candidate = make_candidate()
    assert Candidate.from_dict(candidate.to_dict()) == candidate


def test_candidate_with_invalid_syntax_is_still_a_valid_record():
    # Spec 03 section 19.1: unparseable output is persisted as a candidate, not discarded.
    candidate = make_candidate(
        code="def foo(:", syntax_valid=False, syntax_error="invalid syntax (line 1)"
    )
    assert candidate.syntax_valid is False
    assert Candidate.from_dict(candidate.to_dict()) == candidate


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"candidate_id": ""}, "candidate_id"),
        ({"candidate_id": "p002_c001"}, "does not belong"),
        ({"generation_index": 0}, "generation_index"),
        ({"generation_index": True}, "generation_index"),
        ({"code": ""}, "code"),
        ({"extraction_format": "unknown"}, "extraction_format"),
        ({"extraction_format": "guessed"}, "extraction_format"),
        ({"syntax_valid": "yes"}, "syntax_valid"),
        ({"syntax_error": "boom"}, "syntax_error must be null"),
        ({"duplicate_of": "p001_c001"}, "different candidate"),
        ({"generation_config": []}, "generation_config"),
        ({"model": ""}, "model"),
        ({"prompt_version": ""}, "prompt_version"),
    ],
)
def test_invalid_candidate_is_rejected(overrides, match):
    with pytest.raises(CandidateError, match=match):
        make_candidate(**overrides)


def test_candidate_from_dict_rejects_unknown_and_missing_fields():
    payload = make_candidate().to_dict()
    with pytest.raises(CandidateError, match="unknown field"):
        Candidate.from_dict({**payload, "temperature": 0.8})
    del payload["code"]
    with pytest.raises(CandidateError, match="missing required field"):
        Candidate.from_dict(payload)


def test_generation_failure_round_trips():
    failure = make_failure()
    assert GenerationFailure.from_dict(failure.to_dict()) == failure


def test_generation_failure_rejects_unknown_error_type():
    # syntax_error is deliberately absent from the closed set (section 26.1).
    with pytest.raises(CandidateError, match="error_type"):
        make_failure(error_type="syntax_error")


# -------------------------------------------------------------------------- repository


def test_append_and_load_round_trip(tmp_path):
    repo = CandidateRepository(tmp_path)
    assert repo.load_all() == []

    first = make_candidate()
    second = make_candidate(candidate_id="p001_c002", generation_index=2, code=CODE + "\n")
    repo.append(first)
    repo.append(second)

    assert repo.load_all() == [first, second]
    assert repo.candidates_path.name == "candidates.jsonl"


def test_records_are_readable_before_the_run_finishes(tmp_path):
    # Records are flushed per append, so a killed run leaves a usable file behind.
    repo = CandidateRepository(tmp_path)
    repo.append(make_candidate())
    assert len(CandidateRepository(tmp_path).load_all()) == 1


def test_failures_are_persisted_separately(tmp_path):
    repo = CandidateRepository(tmp_path)
    failure = make_failure()
    repo.append_failure(failure)
    assert repo.load_failures() == [failure]
    assert repo.failures_path.name == "generation_failures.jsonl"
    assert not repo.candidates_path.exists()


def test_existing_keys_drive_resume(tmp_path):
    repo = CandidateRepository(tmp_path)
    repo.append(make_candidate())
    repo.append(make_candidate(candidate_id="p001_c003", generation_index=3))
    assert repo.existing_keys() == {("p001", 1), ("p001", 3)}


def test_failed_generations_do_not_block_a_retry(tmp_path):
    repo = CandidateRepository(tmp_path)
    repo.append_failure(make_failure(generation_index=2))
    assert ("p001", 2) not in repo.existing_keys()


def test_code_index_reports_the_earliest_match_per_problem(tmp_path):
    repo = CandidateRepository(tmp_path)
    repo.append(make_candidate())
    repo.append(make_candidate(candidate_id="p001_c002", generation_index=2))
    repo.append(make_candidate(candidate_id="p002_c001", problem_id="p002"))

    index = repo.code_index()
    assert index["p001"][CODE] == "p001_c001"
    # Identical code under a different problem is a coincidence, not a duplicate.
    assert index["p002"][CODE] == "p002_c001"


def test_latest_by_candidate_id_prefers_the_newer_run(tmp_path):
    repo = CandidateRepository(tmp_path)
    repo.append(make_candidate(run_id="20260817_120000"))
    repo.append(make_candidate(run_id="20260817_130000", code=CODE + "\n# v2"))

    latest = repo.latest_by_candidate_id()
    assert set(latest) == {"p001_c001"}
    assert latest["p001_c001"].run_id == "20260817_130000"


def test_run_ids_are_unique_within_the_same_second(tmp_path):
    repo = CandidateRepository(tmp_path)
    moment = datetime(2026, 8, 17, 10, 30, 0, tzinfo=timezone.utc)
    assert repo.new_run_id(moment) == "20260817_103000"

    repo.append(make_candidate(run_id="20260817_103000"))
    assert repo.new_run_id(moment) == "20260817_103000_2"

    repo.append(make_candidate(candidate_id="p001_c002", generation_index=2, run_id="20260817_103000_2"))
    assert repo.new_run_id(moment) == "20260817_103000_3"


def test_run_ids_also_account_for_failure_only_runs(tmp_path):
    repo = CandidateRepository(tmp_path)
    moment = datetime(2026, 8, 17, 10, 30, 0, tzinfo=timezone.utc)
    repo.append_failure(make_failure(run_id="20260817_103000"))
    assert repo.new_run_id(moment) == "20260817_103000_2"


@pytest.mark.parametrize(
    "content, match",
    [
        ("not json\n", "invalid JSON"),
        ('{"candidate_id": "p001_c001"}\n', "missing required field"),
        ("[1, 2]\n", "expected a JSON object"),
        ('{"candidate_id": "x"}\n\n', "blank line"),
    ],
)
def test_malformed_lines_are_rejected_with_a_line_number(tmp_path, content, match):
    repo = CandidateRepository(tmp_path)
    repo.candidates_path.parent.mkdir(parents=True, exist_ok=True)
    repo.candidates_path.write_text(content, encoding="utf-8")

    with pytest.raises(CandidateStoreError, match=match):
        repo.load_all()
