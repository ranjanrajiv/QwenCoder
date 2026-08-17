"""Tests for the per-execution job workspace (spec 05 sections 15, 16, 41)."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from python_dpo.sandbox import CANDIDATE_FILENAME, SandboxWorkspace, WorkspaceError, new_job_id

CODE = 'print("hello")\n'

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_job_ids_are_sortable_and_unique():
    # Spec section 37 rules out purely random names: the timestamp is what makes a stray
    # container or directory traceable back to when it ran.
    first, second = new_job_id(), new_job_id()
    assert first != second
    assert len(first.split("_")) == 3


def test_workspace_creates_a_directory_and_writes_the_candidate(tmp_path):
    with SandboxWorkspace(root=tmp_path) as workspace:
        workspace.write_candidate(CODE)
        assert workspace.path.is_dir()
        assert workspace.candidate_path.name == CANDIDATE_FILENAME
        assert workspace.candidate_path.read_text(encoding="utf-8") == CODE


def test_candidate_source_is_written_verbatim(tmp_path):
    # The stored bytes must be what the model produced; a workspace that reformatted or
    # normalized code would make the whole experiment mean something else.
    weird = "x = 1\r\n\tif True:\n  pass\n\n\n"
    with SandboxWorkspace(root=tmp_path) as workspace:
        workspace.write_candidate(weird)
        # Compared as bytes: text-mode newline translation would silently rewrite \r\n.
        assert workspace.candidate_path.read_bytes() == weird.encode("utf-8")


def test_permissions_allow_a_non_root_container_user_to_read(tmp_path):
    # tempfile.mkdtemp creates 0o700, which the container's non-root UID cannot traverse.
    # Without the explicit chmod every execution would fail with a confusing permission
    # error rather than running the candidate.
    with SandboxWorkspace(root=tmp_path) as workspace:
        workspace.write_candidate(CODE)
        dir_mode = stat.S_IMODE(workspace.path.stat().st_mode)
        file_mode = stat.S_IMODE(workspace.candidate_path.stat().st_mode)

    assert dir_mode & stat.S_IROTH, "other must be able to read the job directory"
    assert dir_mode & stat.S_IXOTH, "other must be able to traverse the job directory"
    assert file_mode & stat.S_IROTH, "other must be able to read candidate.py"


def test_cleanup_removes_the_directory_on_success(tmp_path):
    with SandboxWorkspace(root=tmp_path) as workspace:
        workspace.write_candidate(CODE)
        path = workspace.path
    assert not path.exists()


def test_cleanup_removes_the_directory_when_the_body_raises(tmp_path):
    # Spec section 16: cleanup happens even when the candidate crashes, the run times out,
    # Docker errors, or an exception propagates to the caller.
    path = None
    with pytest.raises(RuntimeError, match="boom"):
        with SandboxWorkspace(root=tmp_path) as workspace:
            workspace.write_candidate(CODE)
            path = workspace.path
            raise RuntimeError("boom")
    assert path is not None
    assert not path.exists()


def test_cleanup_is_idempotent(tmp_path):
    workspace = SandboxWorkspace(root=tmp_path)
    workspace.create()
    workspace.cleanup()
    workspace.cleanup()  # must not raise


def test_path_before_creation_is_an_error(tmp_path):
    workspace = SandboxWorkspace(root=tmp_path)
    with pytest.raises(WorkspaceError, match="not been created"):
        _ = workspace.path


def test_creating_twice_is_an_error(tmp_path):
    workspace = SandboxWorkspace(root=tmp_path)
    workspace.create()
    try:
        with pytest.raises(WorkspaceError, match="already exists"):
            workspace.create()
    finally:
        workspace.cleanup()


def test_non_string_code_is_rejected(tmp_path):
    with SandboxWorkspace(root=tmp_path) as workspace:
        with pytest.raises(WorkspaceError, match="must be a string"):
            workspace.write_candidate(b"print('hi')")  # type: ignore[arg-type]


def test_workspace_holds_only_the_candidate_file(tmp_path):
    # Spec section 15: the workspace contains only what that execution needs. Anything else
    # would become visible inside the container.
    with SandboxWorkspace(root=tmp_path) as workspace:
        workspace.write_candidate(CODE)
        assert [p.name for p in workspace.path.iterdir()] == [CANDIDATE_FILENAME]


def test_default_workspace_lives_outside_the_project_tree():
    # Spec section 14: the project directory is never what gets mounted.
    with SandboxWorkspace() as workspace:
        assert PROJECT_ROOT not in workspace.path.parents


def test_write_file_supports_multiple_files(tmp_path):
    # Stage 6's evaluation job needs candidate.py + test_candidate.py + conftest.py in
    # one workspace.
    with SandboxWorkspace(root=tmp_path) as workspace:
        workspace.write_file("candidate.py", "print('a')\n")
        workspace.write_file("test_candidate.py", "def test_x(): pass\n")
        workspace.write_file("conftest.py", "# plugin\n")
        names = sorted(p.name for p in workspace.path.iterdir())
    assert names == ["candidate.py", "conftest.py", "test_candidate.py"]


def test_write_file_rejects_a_path_with_a_directory_separator(tmp_path):
    with SandboxWorkspace(root=tmp_path) as workspace:
        with pytest.raises(WorkspaceError, match="safe bare filename"):
            workspace.write_file("subdir/evil.py", "x = 1\n")


def test_write_file_rejects_parent_directory_traversal(tmp_path):
    with SandboxWorkspace(root=tmp_path) as workspace:
        with pytest.raises(WorkspaceError, match="safe bare filename"):
            workspace.write_file("../escape.py", "x = 1\n")


def test_write_candidate_is_a_thin_wrapper_over_write_file(tmp_path):
    with SandboxWorkspace(root=tmp_path) as workspace:
        path_a = workspace.write_candidate(CODE)
        path_b = workspace.write_file(CANDIDATE_FILENAME, CODE)
    assert path_a == path_b


def test_workspace_never_executes_anything(tmp_path):
    # Spec section 41: the workspace writes files and nothing else. Code that would raise
    # if executed must be written out untouched.
    with SandboxWorkspace(root=tmp_path) as workspace:
        workspace.write_candidate("raise SystemExit(99)\n")
        assert "SystemExit" in workspace.candidate_path.read_text(encoding="utf-8")
