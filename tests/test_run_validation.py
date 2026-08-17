"""Tests for run integrity validation (spec 04 sections 22, 38, 51).

Each test builds one good, valid run directory via the real generator and mock model,
then mutates exactly one thing about it and asserts ``validate_run`` catches it, names
the offending record, and that a clean run still passes.
"""

from __future__ import annotations

import json

import pytest

from python_dpo.atomic_io import atomic_write_json
from python_dpo.candidates.repository import (
    CANDIDATES_FILENAME,
    FAILURES_FILENAME,
    PROMPTS_DIRNAME,
    PROMPTS_FILENAME,
)
from python_dpo.generation import CandidateGenerator, PROMPT_VERSION
from python_dpo.models import GenerationConfig, MockModelClient
from python_dpo.problems.models import Problem, TestCase
from python_dpo.runs import RunRepository, RunStatistics, validate_run
from python_dpo.runs.repository import MANIFEST_FILENAME, STATISTICS_FILENAME

PROBLEM = Problem(
    id="p001",
    prompt="Return the sum of the even integers in a list.",
    signature="def sum_even(numbers):",
    entry_point="sum_even",
    category="lists",
    difficulty="easy",
    reference_solution="def sum_even(numbers):\n    return 0\n",
    tests=(TestCase(id="t001", input={"numbers": [2]}, expected=2),),
)
KNOWN_PROBLEM_IDS = {"p001"}


def build_good_run(tmp_path, *, count=2, mark_completed=True):
    """A real, valid, completed run directory built through the actual pipeline."""
    run_repo = RunRepository(tmp_path / "runs")
    manifest = run_repo.create_run(
        requested_problem_ids=["p001"],
        requested_candidates_per_problem=count,
        strategies=["normal"] if count == 1 else ["normal", "optimized"][:count],
        model_config={"provider": "mock", "name": "mock/deterministic-coder"},
        generation_config=GenerationConfig().to_dict(),
        prompt_version=PROMPT_VERSION,
        retry={"max_attempts": 1},
    )
    run_repo.start_run(manifest.run_id)
    repository = run_repo.candidates(manifest.run_id)
    generator = CandidateGenerator(client=MockModelClient(), repository=repository)
    generator.generate([PROBLEM], manifest)

    stats = RunStatistics.from_records(manifest, repository.load_all(), repository.load_failures())
    run_repo.write_statistics(stats)
    if mark_completed:
        run_repo.complete_run(manifest.run_id)

    return run_repo, manifest.run_id


def read_lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def write_lines(path, lines):
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def test_a_clean_run_passes(tmp_path):
    run_repo, run_id = build_good_run(tmp_path)
    report = validate_run(run_repo.run_dir(run_id), KNOWN_PROBLEM_IDS)
    assert report.valid
    assert report.issues == ()


def test_missing_manifest_is_reported(tmp_path):
    run_repo, run_id = build_good_run(tmp_path)
    (run_repo.run_dir(run_id) / MANIFEST_FILENAME).unlink()

    report = validate_run(run_repo.run_dir(run_id), KNOWN_PROBLEM_IDS)
    assert not report.valid
    assert any(i.check == "manifest" for i in report.issues)


def test_malformed_json_is_reported(tmp_path):
    run_repo, run_id = build_good_run(tmp_path)
    path = run_repo.run_dir(run_id) / CANDIDATES_FILENAME
    lines = read_lines(path)
    lines[0] = "not json"
    write_lines(path, lines)

    report = validate_run(run_repo.run_dir(run_id), KNOWN_PROBLEM_IDS)
    assert not report.valid
    assert any(i.check == "jsonl" for i in report.issues)


def test_duplicate_candidate_id_is_reported(tmp_path):
    run_repo, run_id = build_good_run(tmp_path)
    path = run_repo.run_dir(run_id) / CANDIDATES_FILENAME
    lines = read_lines(path)
    write_lines(path, lines + [lines[0]])

    report = validate_run(run_repo.run_dir(run_id), KNOWN_PROBLEM_IDS)
    assert not report.valid
    assert any(i.check == "duplicate_id" for i in report.issues)


def test_wrong_code_hash_is_reported_and_names_the_candidate(tmp_path):
    run_repo, run_id = build_good_run(tmp_path)
    path = run_repo.run_dir(run_id) / CANDIDATES_FILENAME
    lines = read_lines(path)
    record = json.loads(lines[0])
    record["code_sha256"] = "0" * 64
    lines[0] = json.dumps(record)
    write_lines(path, lines)

    report = validate_run(run_repo.run_dir(run_id), KNOWN_PROBLEM_IDS)
    assert not report.valid
    schema_issues = [i for i in report.issues if i.check == "schema"]
    assert schema_issues
    assert record["candidate_id"] in schema_issues[0].message
    assert "code_sha256" in schema_issues[0].message


def test_missing_required_field_is_reported(tmp_path):
    run_repo, run_id = build_good_run(tmp_path)
    path = run_repo.run_dir(run_id) / CANDIDATES_FILENAME
    lines = read_lines(path)
    record = json.loads(lines[0])
    del record["code"]
    lines[0] = json.dumps(record)
    write_lines(path, lines)

    report = validate_run(run_repo.run_dir(run_id), KNOWN_PROBLEM_IDS)
    assert not report.valid
    assert any("missing required field" in i.message for i in report.issues)


def test_mismatched_run_id_is_reported(tmp_path):
    run_repo, run_id = build_good_run(tmp_path)
    path = run_repo.run_dir(run_id) / CANDIDATES_FILENAME
    lines = read_lines(path)
    record = json.loads(lines[0])
    record["run_id"] = "run_wrong"
    lines[0] = json.dumps(record)
    write_lines(path, lines)

    report = validate_run(run_repo.run_dir(run_id), KNOWN_PROBLEM_IDS)
    assert not report.valid
    assert any(i.check == "run_id" for i in report.issues)


def test_unknown_problem_id_is_reported(tmp_path):
    run_repo, run_id = build_good_run(tmp_path)
    report = validate_run(run_repo.run_dir(run_id), known_problem_ids={"p999"})
    assert not report.valid
    assert any(i.check == "problem_id" for i in report.issues)


def test_drifted_statistics_is_reported(tmp_path):
    run_repo, run_id = build_good_run(tmp_path)
    stats_path = run_repo.run_dir(run_id) / STATISTICS_FILENAME
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    payload["candidates_generated"] = payload["candidates_generated"] + 1
    atomic_write_json(stats_path, payload)

    report = validate_run(run_repo.run_dir(run_id), KNOWN_PROBLEM_IDS)
    assert not report.valid
    assert any(i.check == "statistics" for i in report.issues)


def test_truncated_tail_is_reported(tmp_path):
    run_repo, run_id = build_good_run(tmp_path)
    path = run_repo.run_dir(run_id) / CANDIDATES_FILENAME
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"candidate_id": "p001_c003"')  # torn write, no newline

    report = validate_run(run_repo.run_dir(run_id), KNOWN_PROBLEM_IDS)
    assert not report.valid
    assert any("truncated" in i.message for i in report.issues)


def test_dangling_duplicate_of_is_reported(tmp_path):
    run_repo, run_id = build_good_run(tmp_path)
    path = run_repo.run_dir(run_id) / CANDIDATES_FILENAME
    lines = read_lines(path)
    record = json.loads(lines[0])
    record["duplicate_of"] = "p001_c999"
    lines[0] = json.dumps(record)
    write_lines(path, lines)

    report = validate_run(run_repo.run_dir(run_id), KNOWN_PROBLEM_IDS)
    assert not report.valid
    assert any(i.check == "duplicate_of" for i in report.issues)


def test_completed_status_with_missing_work_is_reported(tmp_path):
    run_repo, run_id = build_good_run(tmp_path, count=2, mark_completed=False)
    # Drop the second candidate's line so the run is missing work, then mark it
    # completed anyway — a lie the validator must catch.
    path = run_repo.run_dir(run_id) / CANDIDATES_FILENAME
    lines = read_lines(path)
    write_lines(path, lines[:1])

    stats = RunStatistics.from_records(
        run_repo.get_run(run_id),
        run_repo.candidates(run_id).load_all(),
        run_repo.candidates(run_id).load_failures(),
    )
    run_repo.write_statistics(stats)
    run_repo.complete_run(run_id)

    report = validate_run(run_repo.run_dir(run_id), KNOWN_PROBLEM_IDS)
    assert not report.valid
    assert any(i.check == "incomplete" for i in report.issues)


def test_prompt_missing_from_prompts_artifact_is_reported(tmp_path):
    run_repo, run_id = build_good_run(tmp_path)
    prompts_path = run_repo.run_dir(run_id) / PROMPTS_DIRNAME / PROMPTS_FILENAME
    prompts_path.write_text("", encoding="utf-8")

    report = validate_run(run_repo.run_dir(run_id), KNOWN_PROBLEM_IDS)
    assert not report.valid
    assert any(i.check == "prompt_missing" for i in report.issues)


def test_format_run_report_success_message(tmp_path):
    from python_dpo.runs import format_run_report

    run_repo, run_id = build_good_run(tmp_path)
    report = validate_run(run_repo.run_dir(run_id), KNOWN_PROBLEM_IDS)
    assert format_run_report(report) == "Run validation passed.\n"


def test_format_run_report_failure_message_lists_issues(tmp_path):
    from python_dpo.runs import format_run_report

    run_repo, run_id = build_good_run(tmp_path)
    (run_repo.run_dir(run_id) / MANIFEST_FILENAME).unlink()
    report = validate_run(run_repo.run_dir(run_id), KNOWN_PROBLEM_IDS)
    rendered = format_run_report(report)
    assert rendered.startswith("Run validation failed:")
    assert "manifest.json" in rendered
