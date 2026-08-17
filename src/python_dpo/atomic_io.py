"""Durable-write primitives shared by the run and candidate repositories.

Two distinct durability shapes are needed (spec 04 section 21):

* Whole-file artifacts (``manifest.json``, ``statistics.json``) are rewritten completely
  on every update. :func:`atomic_write_json` writes to a temp file in the same directory,
  fsyncs it, and ``os.replace``s it into place, so a reader never observes a partial file.
* Append-only artifacts (``candidates.jsonl``, ``failures.jsonl``, ``prompts.jsonl``) grow
  one record at a time. Rewriting the whole file per append would be quadratic against the
  2,500-candidate target (section 53), so :func:`append_jsonl` instead writes one complete
  line and fsyncs just that write. :func:`iter_jsonl` is the matching reader: it treats a
  final line with no trailing newline as a torn write and raises rather than guessing.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class JsonlError(Exception):
    """Raised when a JSONL file is malformed, truncated, or holds a non-object line."""


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace ``path`` with ``payload`` such that a reader never sees a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)

    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(tmp_path, path)
    _fsync_dir(path.parent)


def read_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise JsonlError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise JsonlError(f"{path}: expected a JSON object")
    return data


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one durable, complete JSON line to ``path``, creating it if necessary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, ensure_ascii=False)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")
        handle.flush()
        os.fsync(handle.fileno())


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, obj)`` pairs. Yields nothing if ``path`` does not exist.

    Raises :class:`JsonlError` on invalid JSON, a non-object line, a blank line, or a
    final line with no trailing newline (a torn write) — never silently skips a bad line.
    """
    path = Path(path)
    if not path.is_file():
        return

    with path.open("rb") as handle:
        raw = handle.read()

    if not raw:
        return

    ends_with_newline = raw.endswith(b"\n")
    text = raw.decode("utf-8")
    lines = text.split("\n")
    # split() on a trailing newline leaves one empty trailing element; drop it, since it
    # is not a line, only the terminator of the last real one.
    if ends_with_newline and lines and lines[-1] == "":
        lines.pop()

    for number, line in enumerate(lines, start=1):
        is_last = number == len(lines)
        if is_last and not ends_with_newline:
            raise JsonlError(
                f"{path}:{number}: truncated final line (no trailing newline); "
                "this looks like a torn write"
            )
        stripped = line.strip()
        if not stripped:
            raise JsonlError(
                f"{path}:{number}: blank line; JSONL must hold exactly one object per line"
            )
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise JsonlError(f"{path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise JsonlError(f"{path}:{number}: expected a JSON object")
        yield number, record


def repair_truncated_tail(path: Path) -> int:
    """Truncate ``path`` at the last complete newline. Returns the bytes removed.

    Only ever called explicitly (``runs validate --repair``), never automatically — a
    torn tail is reported by :func:`iter_jsonl`, and repairing it is a deliberate,
    auditable action, not something persistence does on its own (spec 04 section 38).
    """
    path = Path(path)
    with path.open("rb") as handle:
        raw = handle.read()

    if not raw or raw.endswith(b"\n"):
        return 0

    last_newline = raw.rfind(b"\n")
    keep = raw[: last_newline + 1] if last_newline != -1 else b""
    removed = len(raw) - len(keep)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        handle.write(keep)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    _fsync_dir(path.parent)

    return removed


def _fsync_dir(directory: Path) -> None:
    """Fsync a directory so a rename inside it survives a crash.

    Not supported on Windows (no directory file descriptors); a no-op there is the right
    fallback since the rename itself is still atomic on that filesystem.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


__all__ = [
    "JsonlError",
    "append_jsonl",
    "atomic_write_json",
    "iter_jsonl",
    "read_json",
    "repair_truncated_tail",
]
