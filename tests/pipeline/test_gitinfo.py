"""Tests for git version capture and the dirty-tree policy (spec 12 section 29)."""

from __future__ import annotations

import subprocess

import pytest

from python_dpo.pipeline.gitinfo import GitInfoError, capture_git_info, enforce_dirty_policy


def _init_repo(path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)


def test_capture_git_info_on_a_clean_repo(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=tmp_path, check=True)

    info = capture_git_info(tmp_path)
    assert info["dirty"] is False
    assert isinstance(info["sha"], str) and len(info["sha"]) == 40
    assert info["branch"] is not None


def test_capture_git_info_detects_dirty_tree(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=tmp_path, check=True)

    (tmp_path / "a.txt").write_text("changed", encoding="utf-8")
    info = capture_git_info(tmp_path)
    assert info["dirty"] is True


def test_capture_git_info_on_non_repo_returns_none_fields(tmp_path):
    info = capture_git_info(tmp_path)
    assert info["sha"] is None
    assert info["branch"] is None
    assert info["dirty"] is None


def test_enforce_dirty_policy_warn_never_raises_on_dirty():
    enforce_dirty_policy({"sha": "x", "branch": "main", "dirty": True}, on_dirty="warn")


def test_enforce_dirty_policy_fail_raises_on_dirty():
    with pytest.raises(GitInfoError):
        enforce_dirty_policy({"sha": "x", "branch": "main", "dirty": True}, on_dirty="fail")


def test_enforce_dirty_policy_fail_passes_on_clean_tree():
    enforce_dirty_policy({"sha": "x", "branch": "main", "dirty": False}, on_dirty="fail")


def test_enforce_dirty_policy_fail_raises_on_indeterminate_dirty():
    # git unavailable/not a repo: an unverifiable tree is treated as dirty under "fail".
    with pytest.raises(GitInfoError):
        enforce_dirty_policy({"sha": None, "branch": None, "dirty": None}, on_dirty="fail")


def test_enforce_dirty_policy_rejects_unknown_policy():
    with pytest.raises(GitInfoError, match="warn.*fail"):
        enforce_dirty_policy({"dirty": False}, on_dirty="explode")
