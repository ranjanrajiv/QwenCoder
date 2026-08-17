"""Pull Python source out of whatever the model actually returned.

Models are asked for bare code and return markdown fences, prose preambles, or both, so
extraction is a required layer rather than a nicety (spec 03 sections 16, 17).

**This module never repairs code** (sections 18, 44). It locates a likely block, strips
surrounding whitespace, and hands back the bytes untouched — internal formatting is
preserved exactly. Guessing at an unterminated fence or patching a missing colon would
mean the stored candidate is no longer what the model produced, which would quietly
corrupt the preference experiment.

An extraction that finds nothing returns ``extracted=False``. It never invents a
successful result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NO_CODE_DETECTED = "No Python code detected"

# ```python / ```py / ```python3, then a newline, then the block.
_PYTHON_FENCE_RE = re.compile(
    r"```[ \t]*(?:python|py|python3)[ \t]*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# A bare ``` fence. The mandatory newline right after the backticks keeps this from
# matching a language-tagged fence.
_GENERIC_FENCE_RE = re.compile(r"```[ \t]*\r?\n(.*?)```", re.DOTALL)

# A line that starts a Python definition or import. Cheap structural evidence that a
# block is source code rather than prose or a shell transcript.
_CODE_LINE_RE = re.compile(r"^[ \t]*(?:async[ \t]+def|def|class|import|from)[ \t]", re.MULTILINE)


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of extraction, including how the code was found."""

    code: str | None
    extracted: bool
    source_format: str
    error: str | None = None


def _normalize(block: str) -> str:
    """Trim blank lines and trailing whitespace around a block, leaving it otherwise intact."""
    return block.strip("\r\n").rstrip()


def _looks_like_code(block: str) -> bool:
    return bool(_CODE_LINE_RE.search(block))


def extract_code(raw_output: str) -> ExtractionResult:
    """Extract Python source from a raw model response."""
    if not isinstance(raw_output, str) or not raw_output.strip():
        return ExtractionResult(
            code=None, extracted=False, source_format="unknown", error=NO_CODE_DETECTED
        )

    for match in _PYTHON_FENCE_RE.finditer(raw_output):
        block = _normalize(match.group(1))
        if block:
            return ExtractionResult(code=block, extracted=True, source_format="python_fence")

    for match in _GENERIC_FENCE_RE.finditer(raw_output):
        block = _normalize(match.group(1))
        if block and _looks_like_code(block):
            return ExtractionResult(code=block, extracted=True, source_format="generic_fence")

    # Bare code, no fences at all. A stray ``` means the model opened a fence it never
    # closed; treating the remainder as plain code would be a repair, so it fails instead.
    if "```" not in raw_output and _looks_like_code(raw_output):
        block = _normalize(raw_output)
        if block:
            return ExtractionResult(code=block, extracted=True, source_format="plain")

    return ExtractionResult(
        code=None, extracted=False, source_format="unknown", error=NO_CODE_DETECTED
    )
