"""Tests for migrating the Stage 3 flat candidates.jsonl into run directories."""

from __future__ import annotations

import json

import pytest

from python_dpo.candidates import Candidate, utc_now_iso
from python_dpo.runs import MigrationError, RunRepository, migrate_flat_file, validate_run

CODE_A = "def solve(x):\n    return x"
CODE_B = "def solve(x):\n    return x + 1"


def legacy_record(**overrides):
    fields = {
        "candidate_id": "p001_c001",
        "problem_id": "p001",
        "run_id": "20260817_055411",
        "generation_index": 1,
        "strategy": "normal",
        "model": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "model_revision": None,
        "provider": "transformers",
        "prompt_version": "v1",
        "prompt": "Solve it.",
        "raw_output": f"```python\n{CODE_A}\n```",
        "code": CODE_A,
        "extraction_format": "python_fence",
        "syntax_valid": True,
        "syntax_error": None,
        "function_name_valid": True,
        "duplicate_of": None,
        "generation_config": {"temperature": 0.8, "seed": 42},
        "created_at": utc_now_iso(),
    }
    fields.update(overrides)
    return fields


def write_legacy_file(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_migrates_a_single_run_with_hashes_backfilled(tmp_path):
    source = tmp_path / "candidates.jsonl"
    records = [
        legacy_record(candidate_id="p001_c001", generation_index=1),
        legacy_record(
            candidate_id="p001_c002",
            generation_index=2,
            strategy="optimized",
            code=CODE_B,
            raw_output=f"```python\n{CODE_B}\n```",
        ),
    ]
    write_legacy_file(source, records)

    run_repo = RunRepository(tmp_path / "runs")
    manifests = migrate_flat_file(source, run_repo)

    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest.run_id == "20260817_055411"
    assert manifest.source == "migrated"
    assert manifest.status == "completed"
    assert manifest.candidate_schema_version == "2.0"

    repository = run_repo.candidates(manifest.run_id)
    migrated = repository.load_all()
    assert len(migrated) == 2
    assert all(c.schema_version == "2.0" for c in migrated)
    assert all(c.code_sha256 for c in migrated)


def test_source_file_is_left_byte_identical(tmp_path):
    source = tmp_path / "candidates.jsonl"
    write_legacy_file(source, [legacy_record()])
    before = source.read_bytes()

    run_repo = RunRepository(tmp_path / "runs")
    migrate_flat_file(source, run_repo)

    assert source.read_bytes() == before


def test_migrated_run_passes_validation(tmp_path):
    source = tmp_path / "candidates.jsonl"
    write_legacy_file(
        source,
        [
            legacy_record(candidate_id="p001_c001", generation_index=1),
            legacy_record(candidate_id="p002_c001", problem_id="p002", generation_index=1),
        ],
    )

    run_repo = RunRepository(tmp_path / "runs")
    manifest = migrate_flat_file(source, run_repo)[0]

    report = validate_run(run_repo.run_dir(manifest.run_id), {"p001", "p002"})
    assert report.valid, report.issues


def test_migrating_twice_without_force_refuses_to_clobber(tmp_path):
    source = tmp_path / "candidates.jsonl"
    write_legacy_file(source, [legacy_record()])

    run_repo = RunRepository(tmp_path / "runs")
    migrate_flat_file(source, run_repo)

    with pytest.raises(MigrationError, match="already exists"):
        migrate_flat_file(source, run_repo)


def test_migrating_twice_with_force_overwrites_cleanly(tmp_path):
    source = tmp_path / "candidates.jsonl"
    write_legacy_file(source, [legacy_record()])

    run_repo = RunRepository(tmp_path / "runs")
    migrate_flat_file(source, run_repo)
    manifests = migrate_flat_file(source, run_repo, force=True)

    repository = run_repo.candidates(manifests[0].run_id)
    assert repository.count() == 1, "force must overwrite, not append duplicates"


def test_multiple_run_ids_in_the_source_file_produce_multiple_runs(tmp_path):
    source = tmp_path / "candidates.jsonl"
    write_legacy_file(
        source,
        [
            legacy_record(run_id="20260817_055411", candidate_id="p001_c001"),
            legacy_record(run_id="20260817_070000", candidate_id="p001_c001"),
        ],
    )

    run_repo = RunRepository(tmp_path / "runs")
    manifests = migrate_flat_file(source, run_repo)

    assert {m.run_id for m in manifests} == {"20260817_055411", "20260817_070000"}
    assert len(run_repo.list_runs()) == 2
