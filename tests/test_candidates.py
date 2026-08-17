"""Tests for the candidate schema and the durable, run-scoped repository."""

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
    PromptRecord,
    build_candidate_id,
    sha256_text,
    utc_now_iso,
)

CODE = "def sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)"
PROMPT = "Solve the problem."
RAW_OUTPUT = f"```python\n{CODE}\n```"


def make_candidate(**overrides: Any) -> Candidate:
    fields: dict[str, Any] = {
        "candidate_id": "p001_c001",
        "problem_id": "p001",
        "run_id": "run_20260817_120000_ab12",
        "generation_index": 1,
        "strategy": "normal",
        "model": "mock/deterministic-coder",
        "provider": "mock",
        "prompt_version": "v1",
        "prompt": PROMPT,
        "raw_output": RAW_OUTPUT,
        "code": CODE,
        "extraction_format": "python_fence",
        "syntax_valid": True,
        "function_name_valid": True,
        "generation_config": {"temperature": 0.8, "seed": 42},
        "created_at": utc_now_iso(),
    }
    fields.update(overrides)
    return Candidate.create(**fields)


def make_legacy_candidate(**overrides: Any) -> Candidate:
    """A schema_version 1.0 record: no hash fields, constructed directly (not via create)."""
    fields: dict[str, Any] = {
        "candidate_id": "p001_c001",
        "problem_id": "p001",
        "run_id": "20260817_120000",
        "generation_index": 1,
        "strategy": "normal",
        "model": "mock/deterministic-coder",
        "provider": "mock",
        "prompt_version": "v1",
        "prompt": PROMPT,
        "raw_output": RAW_OUTPUT,
        "code": CODE,
        "extraction_format": "python_fence",
        "syntax_valid": True,
        "function_name_valid": True,
        "generation_config": {"temperature": 0.8, "seed": 42},
        "created_at": utc_now_iso(),
        "schema_version": "1.0",
    }
    fields.update(overrides)
    return Candidate(**fields)


def make_failure(**overrides: Any) -> GenerationFailure:
    fields: dict[str, Any] = {
        "run_id": "run_20260817_120000_ab12",
        "problem_id": "p001",
        "generation_index": 3,
        "strategy": "alternative",
        "error_type": "code_extraction",
        "error_message": "No Python code detected",
        "timestamp": utc_now_iso(),
        "prompt_sha256": sha256_text(PROMPT),
    }
    fields.update(overrides)
    return GenerationFailure(**fields)


def make_prompt_record(**overrides: Any) -> PromptRecord:
    fields: dict[str, Any] = {
        "run_id": "run_20260817_120000_ab12",
        "problem_id": "p001",
        "generation_index": 1,
        "strategy": "normal",
        "attempt": 1,
        "prompt": PROMPT,
        "prompt_sha256": sha256_text(PROMPT),
    }
    fields.update(overrides)
    return PromptRecord(**fields)


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


# ----------------------------------------------------------------- schema versioning (04)


def test_create_computes_all_three_hashes():
    candidate = make_candidate()
    assert candidate.schema_version == "2.0"
    assert candidate.code_sha256 == sha256_text(CODE)
    assert candidate.prompt_sha256 == sha256_text(PROMPT)
    assert candidate.raw_output_sha256 == sha256_text(RAW_OUTPUT)


def test_tampered_code_hash_is_rejected():
    with pytest.raises(CandidateError, match="code_sha256 does not match"):
        Candidate(
            candidate_id="p001_c001",
            problem_id="p001",
            run_id="run_20260817_120000_ab12",
            generation_index=1,
            strategy="normal",
            model="mock/deterministic-coder",
            provider="mock",
            prompt_version="v1",
            prompt=PROMPT,
            raw_output=RAW_OUTPUT,
            code=CODE,
            extraction_format="python_fence",
            syntax_valid=True,
            function_name_valid=True,
            generation_config={},
            created_at=utc_now_iso(),
            code_sha256="0" * 64,
            prompt_sha256=sha256_text(PROMPT),
            raw_output_sha256=sha256_text(RAW_OUTPUT),
        )


def test_schema_2_record_requires_all_hashes():
    with pytest.raises(CandidateError, match="code_sha256 is required"):
        Candidate(
            candidate_id="p001_c001",
            problem_id="p001",
            run_id="run_20260817_120000_ab12",
            generation_index=1,
            strategy="normal",
            model="mock/deterministic-coder",
            provider="mock",
            prompt_version="v1",
            prompt=PROMPT,
            raw_output=RAW_OUTPUT,
            code=CODE,
            extraction_format="python_fence",
            syntax_valid=True,
            function_name_valid=True,
            generation_config={},
            created_at=utc_now_iso(),
        )


def test_legacy_1_0_record_has_null_hashes_and_loads():
    candidate = make_legacy_candidate()
    assert candidate.code_sha256 is None
    assert candidate.prompt_sha256 is None
    assert candidate.raw_output_sha256 is None
    assert Candidate.from_dict(candidate.to_dict()) == candidate


def test_legacy_record_missing_schema_version_field_reads_as_1_0():
    payload = make_legacy_candidate().to_dict()
    del payload["schema_version"]
    del payload["code_sha256"]
    del payload["prompt_sha256"]
    del payload["raw_output_sha256"]
    del payload["attempt"]
    loaded = Candidate.from_dict(payload)
    assert loaded.schema_version == "1.0"
    assert loaded.code_sha256 is None


def test_legacy_record_rejects_a_populated_hash_field():
    with pytest.raises(CandidateError, match="must be null on a schema_version 1.0"):
        make_legacy_candidate(code_sha256=sha256_text(CODE))


def test_failure_prompt_hash_links_to_the_prompt_artifact():
    failure = make_failure()
    assert failure.prompt_sha256 == sha256_text(PROMPT)


def test_legacy_failure_rejects_a_populated_prompt_hash():
    with pytest.raises(CandidateError, match="must be null on a schema_version 1.0"):
        make_failure(schema_version="1.0", prompt_sha256=sha256_text(PROMPT))


def test_legacy_failure_missing_schema_version_and_hash_reads_cleanly():
    payload = make_failure(schema_version="1.0", prompt_sha256=None).to_dict()
    del payload["schema_version"]
    del payload["attempt"]
    del payload["prompt_sha256"]
    del payload["traceback"]
    loaded = GenerationFailure.from_dict(payload)
    assert loaded.schema_version == "1.0"
    assert loaded.prompt_sha256 is None


# -------------------------------------------------------------------------- repository


def test_save_and_load_round_trip(tmp_path):
    repo = CandidateRepository(tmp_path)
    assert repo.load_all() == []

    first = make_candidate()
    second = make_candidate(candidate_id="p001_c002", generation_index=2, code=CODE + "\n")
    repo.save(first)
    repo.save(second)

    assert repo.load_all() == [first, second]
    assert repo.candidates_path.name == "candidates.jsonl"


def test_records_are_readable_before_the_run_finishes(tmp_path):
    # Records are flushed and fsynced per append, so a killed run leaves a usable file.
    repo = CandidateRepository(tmp_path)
    repo.save(make_candidate())
    assert len(CandidateRepository(tmp_path).load_all()) == 1


def test_failures_are_persisted_separately(tmp_path):
    repo = CandidateRepository(tmp_path)
    failure = make_failure()
    repo.save_failure(failure)
    assert repo.load_failures() == [failure]
    assert repo.failures_path.name == "failures.jsonl"
    assert not repo.candidates_path.exists()


def test_prompts_are_persisted_and_loadable(tmp_path):
    repo = CandidateRepository(tmp_path)
    record = make_prompt_record()
    repo.append_prompt(record)
    loaded = repo.load_prompts()
    assert len(loaded) == 1
    assert loaded[0].prompt == PROMPT
    assert loaded[0].prompt_sha256 == sha256_text(PROMPT)


def test_existing_keys_drive_resume(tmp_path):
    repo = CandidateRepository(tmp_path)
    repo.save(make_candidate())
    repo.save(make_candidate(candidate_id="p001_c003", generation_index=3))
    assert repo.existing_keys() == {("p001", 1), ("p001", 3)}


def test_failed_generations_do_not_block_a_retry(tmp_path):
    repo = CandidateRepository(tmp_path)
    repo.save_failure(make_failure(generation_index=2))
    assert ("p001", 2) not in repo.existing_keys()


def test_code_index_reports_the_earliest_match_per_problem(tmp_path):
    repo = CandidateRepository(tmp_path)
    repo.save(make_candidate())
    repo.save(make_candidate(candidate_id="p001_c002", generation_index=2))
    repo.save(make_candidate(candidate_id="p002_c001", problem_id="p002"))

    index = repo.code_index()
    assert index["p001"][sha256_text(CODE)] == "p001_c001"
    # Identical code under a different problem is a coincidence, not a duplicate.
    assert index["p002"][sha256_text(CODE)] == "p002_c001"


# ------------------------------------------------------------------ spec 04 section 23 API


def test_get_returns_the_matching_candidate(tmp_path):
    repo = CandidateRepository(tmp_path)
    repo.save(make_candidate())
    assert repo.get("p001_c001").candidate_id == "p001_c001"
    assert repo.get("does-not-exist") is None


def test_exists_reflects_get(tmp_path):
    repo = CandidateRepository(tmp_path)
    repo.save(make_candidate())
    assert repo.exists("p001_c001") is True
    assert repo.exists("p001_c999") is False


def test_list_and_count(tmp_path):
    repo = CandidateRepository(tmp_path)
    repo.save(make_candidate())
    repo.save(make_candidate(candidate_id="p001_c002", generation_index=2))
    assert repo.count() == 2
    assert [c.candidate_id for c in repo.list()] == ["p001_c001", "p001_c002"]


def test_find_by_problem(tmp_path):
    repo = CandidateRepository(tmp_path)
    repo.save(make_candidate())
    repo.save(make_candidate(candidate_id="p002_c001", problem_id="p002"))
    assert [c.candidate_id for c in repo.find_by_problem("p001")] == ["p001_c001"]


def test_find_by_hash(tmp_path):
    repo = CandidateRepository(tmp_path)
    a = make_candidate()
    b = make_candidate(candidate_id="p001_c002", generation_index=2, code="def sum_even(x):\n    return 0")
    repo.save(a)
    repo.save(b)
    assert [c.candidate_id for c in repo.find_by_hash(sha256_text(CODE))] == ["p001_c001"]


def test_run_id_minting_moved_to_the_run_repository(tmp_path):
    # new_run_id has moved to RunRepository; the candidate repository no longer mints ids.
    assert not hasattr(CandidateRepository(tmp_path), "new_run_id")


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


def test_truncated_final_line_is_rejected(tmp_path):
    repo = CandidateRepository(tmp_path)
    repo.save(make_candidate())
    with repo.candidates_path.open("a", encoding="utf-8") as handle:
        handle.write('{"candidate_id": "p001_c002"')  # torn write, no trailing newline

    with pytest.raises(CandidateStoreError, match="truncated final line"):
        repo.load_all()
