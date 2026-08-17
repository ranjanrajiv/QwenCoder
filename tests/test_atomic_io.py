"""Tests for the durable-write primitives (spec 04 section 21)."""

from __future__ import annotations

from pathlib import Path

import pytest

from python_dpo.atomic_io import (
    JsonlError,
    append_jsonl,
    atomic_write_json,
    iter_jsonl,
    read_json,
    repair_truncated_tail,
)


# --------------------------------------------------------------------- atomic_write_json


def test_atomic_write_json_creates_the_file(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"a": 1})
    assert read_json(path) == {"a": 1}


def test_atomic_write_json_leaves_no_tmp_file_behind(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"a": 1})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]


def test_atomic_write_json_replaces_existing_content(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"a": 1})
    atomic_write_json(path, {"a": 2, "b": 3})
    assert read_json(path) == {"a": 2, "b": 3}


def test_atomic_write_json_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "manifest.json"
    atomic_write_json(path, {"a": 1})
    assert read_json(path) == {"a": 1}


def test_atomic_write_json_failure_leaves_original_intact(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"a": 1})

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("os.replace", _boom)
    with pytest.raises(OSError):
        atomic_write_json(path, {"a": 2})

    assert read_json(path) == {"a": 1}


# -------------------------------------------------------------------------- append_jsonl


def test_append_jsonl_writes_one_complete_line(tmp_path):
    path = tmp_path / "candidates.jsonl"
    append_jsonl(path, {"id": 1})
    append_jsonl(path, {"id": 2})

    assert path.read_text(encoding="utf-8").endswith("\n")
    records = list(iter_jsonl(path))
    assert [record for _, record in records] == [{"id": 1}, {"id": 2}]


def test_append_jsonl_creates_parent_directories(tmp_path):
    path = tmp_path / "runs" / "run_1" / "candidates.jsonl"
    append_jsonl(path, {"id": 1})
    assert list(iter_jsonl(path)) == [(1, {"id": 1})]


# ---------------------------------------------------------------------------- iter_jsonl


def test_iter_jsonl_on_missing_file_yields_nothing(tmp_path):
    assert list(iter_jsonl(tmp_path / "missing.jsonl")) == []


def test_iter_jsonl_rejects_invalid_json(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(JsonlError, match="invalid JSON"):
        list(iter_jsonl(path))


def test_iter_jsonl_rejects_a_non_object_line(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(JsonlError, match="expected a JSON object"):
        list(iter_jsonl(path))


def test_iter_jsonl_rejects_a_blank_line(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")
    with pytest.raises(JsonlError, match="blank line"):
        list(iter_jsonl(path))


def test_iter_jsonl_detects_a_truncated_final_line_with_its_line_number(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_bytes(b'{"id": 1}\n{"id": 2, "code"')  # torn mid-write, no trailing \n
    with pytest.raises(JsonlError, match=r"candidates\.jsonl:2: truncated final line"):
        list(iter_jsonl(path))


def test_iter_jsonl_does_not_flag_a_properly_terminated_file(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
    assert [record for _, record in iter_jsonl(path)] == [{"id": 1}, {"id": 2}]


def test_iter_jsonl_corrupt_line_mid_file_is_an_error(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text('{"id": 1}\nnot json\n{"id": 3}\n', encoding="utf-8")
    with pytest.raises(JsonlError, match=r"candidates\.jsonl:2: invalid JSON"):
        list(iter_jsonl(path))


# --------------------------------------------------------------------- repair_truncated_tail


def test_repair_truncated_tail_removes_exactly_the_torn_bytes(tmp_path):
    path = tmp_path / "candidates.jsonl"
    good = b'{"id": 1}\n{"id": 2}\n'
    torn = b'{"id": 3, "code"'
    path.write_bytes(good + torn)

    removed = repair_truncated_tail(path)

    assert removed == len(torn)
    assert path.read_bytes() == good
    assert [record for _, record in iter_jsonl(path)] == [{"id": 1}, {"id": 2}]


def test_repair_truncated_tail_is_a_noop_on_a_well_formed_file(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text('{"id": 1}\n', encoding="utf-8")
    assert repair_truncated_tail(path) == 0
    assert path.read_text(encoding="utf-8") == '{"id": 1}\n'


def test_repair_truncated_tail_on_pure_garbage_truncates_to_empty(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_bytes(b'{"incomplete')
    removed = repair_truncated_tail(path)
    assert removed == len(b'{"incomplete')
    assert path.read_bytes() == b""


def test_repair_truncated_tail_does_not_touch_a_corrupt_mid_file_line(tmp_path):
    # repair only ever addresses a torn tail; a bad line earlier in the file must still
    # surface as an error rather than being silently dropped.
    path = tmp_path / "candidates.jsonl"
    path.write_text('not json\n{"id": 2}\n', encoding="utf-8")
    assert repair_truncated_tail(path) == 0
    with pytest.raises(JsonlError, match="invalid JSON"):
        list(iter_jsonl(path))
