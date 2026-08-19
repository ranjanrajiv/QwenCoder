"""Tests for content hashing (spec 12 sections 18, 69, 70)."""

from __future__ import annotations

import hashlib

import pytest

from python_dpo.pipeline.hashing import canonical_json, config_hash, sha256_file, sha256_tree


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"hello world")
    assert sha256_file(path) == hashlib.sha256(b"hello world").hexdigest()


def test_sha256_file_is_deterministic(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"x" * 5_000_000)  # exercise the chunked read path
    assert sha256_file(path) == sha256_file(path)


def test_sha256_tree_changes_when_a_file_changes(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.txt").write_text("one")
    (root / "b.txt").write_text("two")
    before = sha256_tree(root)

    (root / "b.txt").write_text("two-edited")
    after = sha256_tree(root)

    assert before != after


def test_sha256_tree_changes_when_a_file_is_added(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.txt").write_text("one")
    before = sha256_tree(root)

    (root / "c.txt").write_text("three")
    after = sha256_tree(root)

    assert before != after


def test_sha256_tree_is_independent_of_filesystem_iteration_order(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    # Written in opposite orders.
    (root_a / "1.txt").write_text("one")
    (root_a / "2.txt").write_text("two")
    (root_b / "2.txt").write_text("two")
    (root_b / "1.txt").write_text("one")

    assert sha256_tree(root_a) == sha256_tree(root_b)


def test_sha256_tree_covers_nested_subdirectories(tmp_path):
    root = tmp_path / "tree"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "deep.txt").write_text("deep")
    before = sha256_tree(root)

    (root / "nested" / "deep.txt").write_text("deep-edited")
    after = sha256_tree(root)

    assert before != after


def test_sha256_tree_raises_on_non_directory(tmp_path):
    path = tmp_path / "not_a_dir.txt"
    path.write_text("x")
    with pytest.raises(NotADirectoryError):
        sha256_tree(path)


def test_config_hash_is_stable_under_key_reordering():
    assert config_hash({"b": 1, "a": 2}) == config_hash({"a": 2, "b": 1})


def test_config_hash_differs_on_value_change():
    assert config_hash({"beta": 0.1}) != config_hash({"beta": 0.2})


def test_config_hash_is_stable_under_nested_key_reordering():
    left = {"outer": {"b": 1, "a": 2}, "z": 1}
    right = {"z": 1, "outer": {"a": 2, "b": 1}}
    assert config_hash(left) == config_hash(right)


def test_canonical_json_has_no_incidental_whitespace():
    assert canonical_json({"a": 1, "b": 2}) == '{"a":1,"b":2}'
