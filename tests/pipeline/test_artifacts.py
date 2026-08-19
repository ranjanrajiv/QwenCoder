"""Tests for the artifact pointer manifest (spec 12 sections 69, 70)."""

from __future__ import annotations

import pytest

from python_dpo.pipeline.artifacts import (
    ArtifactError,
    ArtifactRef,
    make_artifact_ref,
    read_artifact_manifest,
    write_artifact_manifest,
)
from python_dpo.pipeline.hashing import sha256_file, sha256_tree


def test_artifact_ref_round_trip():
    ref = ArtifactRef(name="adapter", path="model/adapter", sha256="a" * 64, bytes=1024)
    assert ArtifactRef.from_dict("adapter", ref.to_dict()) == ref


def test_artifact_ref_rejects_negative_bytes():
    with pytest.raises(ArtifactError):
        ArtifactRef(name="x", path="p", sha256="a" * 64, bytes=-1)


def test_artifact_ref_from_dict_rejects_unknown_field():
    with pytest.raises(ArtifactError, match="unknown field"):
        ArtifactRef.from_dict("x", {"path": "p", "sha256": "a" * 64, "bytes": 1, "bogus": 1})


def test_artifact_ref_from_dict_rejects_missing_field():
    with pytest.raises(ArtifactError, match="missing required field"):
        ArtifactRef.from_dict("x", {"path": "p", "sha256": "a" * 64})


def test_make_artifact_ref_for_a_file_matches_sha256_file(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    ref = make_artifact_ref("manifest", path)
    assert ref.sha256 == sha256_file(path)
    assert ref.bytes == path.stat().st_size
    assert ref.path == str(path)


def test_make_artifact_ref_for_a_directory_matches_sha256_tree(tmp_path):
    root = tmp_path / "adapter"
    root.mkdir()
    (root / "weights.bin").write_bytes(b"x" * 100)
    (root / "config.json").write_text("{}", encoding="utf-8")
    ref = make_artifact_ref("adapter", root)
    assert ref.sha256 == sha256_tree(root)
    assert ref.bytes == 100 + len("{}")


def test_make_artifact_ref_uses_relative_path_when_given(tmp_path):
    root = tmp_path / "exp"
    (root / "model").mkdir(parents=True)
    (root / "model" / "manifest.json").write_text("{}", encoding="utf-8")
    ref = make_artifact_ref("model_manifest", root / "model" / "manifest.json", relative_to=root)
    assert ref.path == "model/manifest.json"


def test_make_artifact_ref_raises_when_path_does_not_exist(tmp_path):
    with pytest.raises(ArtifactError, match="no such file"):
        make_artifact_ref("missing", tmp_path / "does_not_exist")


def test_write_and_read_artifact_manifest_round_trip(tmp_path):
    path = tmp_path / "artifacts.json"
    refs = {
        "problems": ArtifactRef(name="problems", path="data/problems", sha256="a" * 64, bytes=10),
        "adapter": ArtifactRef(name="adapter", path="model/adapter", sha256="b" * 64, bytes=20),
    }
    write_artifact_manifest(path, refs)
    assert read_artifact_manifest(path) == refs


def test_read_artifact_manifest_returns_empty_dict_when_missing(tmp_path):
    assert read_artifact_manifest(tmp_path / "does_not_exist.json") == {}


def test_write_artifact_manifest_sha256_matches_file_on_disk(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"content")
    ref = make_artifact_ref("source", source)

    manifest_path = tmp_path / "artifacts.json"
    write_artifact_manifest(manifest_path, {"source": ref})

    restored = read_artifact_manifest(manifest_path)["source"]
    assert restored.sha256 == sha256_file(source)
