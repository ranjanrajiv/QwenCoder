"""Tests for experiment archiving and inspection (spec 12 sections 73, 74)."""

from __future__ import annotations

import tarfile

import pytest

from python_dpo.pipeline.archive import archive_experiment, inspect_archive
from python_dpo.pipeline.errors import ArchiveError
from python_dpo.pipeline.repository import ExperimentRunRepository


def make_run(tmp_path) -> tuple[ExperimentRunRepository, str]:
    repo = ExperimentRunRepository(tmp_path / "experiments" / "runs")
    manifest = repo.create_run(experiment_name="test", configuration_hash="h" * 64)
    run_id = manifest.experiment_run_id
    run_dir = repo.run_dir(run_id)
    (run_dir / "resolved_config.yaml").write_text("experiment:\n  name: test\n", encoding="utf-8")
    (run_dir / "stages").mkdir()
    (run_dir / "stages" / "note.txt").write_text("hello", encoding="utf-8")
    return repo, run_id


def test_archive_experiment_produces_a_readable_tar_gz(tmp_path):
    repo, run_id = make_run(tmp_path)
    dest = tmp_path / "archives"

    archive_path = archive_experiment(repo, run_id, dest)

    assert archive_path == dest / f"{run_id}.tar.gz"
    assert archive_path.is_file()
    with tarfile.open(archive_path, "r:gz") as tar:
        names = tar.getnames()
    assert f"{run_id}/manifest.json" in names
    assert f"{run_id}/resolved_config.yaml" in names
    assert f"{run_id}/archive_manifest.json" in names


def test_archive_experiment_raises_for_a_missing_run(tmp_path):
    repo = ExperimentRunRepository(tmp_path / "experiments" / "runs")
    with pytest.raises(ArchiveError, match="no experiment run directory"):
        archive_experiment(repo, "does-not-exist", tmp_path / "archives")


def test_inspect_archive_reads_the_manifest_without_full_extraction(tmp_path):
    repo, run_id = make_run(tmp_path)
    archive_path = archive_experiment(repo, run_id, tmp_path / "archives")

    manifest = inspect_archive(archive_path)

    assert manifest.experiment_run_id == run_id
    assert manifest.file_count >= 3
    assert "resolved_config.yaml" in manifest.files
    assert manifest.total_bytes > 0


def test_inspect_archive_hashes_match_the_real_files(tmp_path):
    from python_dpo.pipeline.hashing import sha256_file

    repo, run_id = make_run(tmp_path)
    run_dir = repo.run_dir(run_id)
    archive_path = archive_experiment(repo, run_id, tmp_path / "archives")

    manifest = inspect_archive(archive_path)

    assert manifest.files["resolved_config.yaml"] == sha256_file(run_dir / "resolved_config.yaml")


def test_inspect_archive_raises_for_a_missing_file(tmp_path):
    with pytest.raises(ArchiveError, match="no archive"):
        inspect_archive(tmp_path / "missing.tar.gz")


def test_inspect_archive_raises_for_a_corrupt_file(tmp_path):
    bad = tmp_path / "corrupt.tar.gz"
    bad.write_bytes(b"not a tarball")
    with pytest.raises(ArchiveError, match="not a valid archive"):
        inspect_archive(bad)
