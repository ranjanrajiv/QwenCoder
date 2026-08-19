"""Content hashing for artifacts and configuration (spec 12 sections 18, 69, 70).

Three hash shapes, all SHA-256, matching :func:`python_dpo.candidates.hashing.sha256_text`
in spirit: same input, same digest, every time.

* :func:`sha256_file` -- a single artifact file (a manifest, a report, an adapter safetensor).
* :func:`sha256_tree` -- a whole directory (the adapter directory), as one deterministic
  digest over every file's relative path and content, sorted so the digest never depends on
  filesystem iteration order.
* :func:`config_hash` -- canonical-JSON hash of a resolved configuration mapping, used to
  build the cache key (section 18) and to detect a hyperparameter change (section 19).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    """Content-addressed digest of every regular file under ``root``.

    Each file contributes ``"<relative-posix-path>\\0<sha256-of-its-bytes>\\n"`` to an
    outer digest, sorted by relative path -- so adding, removing, renaming, or editing any
    file changes the tree digest, and the result never depends on directory iteration order.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    entries = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )

    digest = hashlib.sha256()
    for relative_path in entries:
        file_digest = sha256_file(root / relative_path)
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_json(data: Any) -> str:
    """Stable JSON rendering: sorted keys, no incidental whitespace differences."""
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_hash(data: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


__all__ = ["canonical_json", "config_hash", "sha256_file", "sha256_tree"]
