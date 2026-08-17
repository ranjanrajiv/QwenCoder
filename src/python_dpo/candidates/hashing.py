"""One hash function, so every hash in the system is computed identically.

Used for ``code_sha256``, ``prompt_sha256``, and ``raw_output_sha256`` (spec 04 sections
16, 17, 18) — exact duplicate detection, prompt-identity verification, and integrity
checks without needing to inspect raw text.
"""

from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = ["sha256_text"]
