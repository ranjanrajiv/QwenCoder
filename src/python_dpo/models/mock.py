"""A deterministic model client for tests and offline pipeline runs.

Unit tests must never need a GPU, model weights, Hugging Face authentication, or a
network connection (spec 03 section 32). This client satisfies the same
:class:`~python_dpo.models.base.ModelClient` interface as
:class:`~python_dpo.models.qwen.QwenModelClient` and returns predictable output, so the
prompt builder, extractor, validators, repository, resume logic, and failure handling can
all be exercised end to end.

It lives in ``src/`` rather than ``tests/`` so the CLI can drive it too, via
``generate --mock-model``.

Two modes:

**Synthesized (default)** — output is derived from a hash of the prompt. Same prompt in,
same bytes out; a different strategy produces a different prompt and therefore different
output. The wrapper format cycles through the four shapes the extractor must handle, so a
plain mock run covers markdown fences, generic fences, prose preambles, and bare code.

**Scripted** — ``MockModelClient(script=[...])`` returns each item in order, raising any
item that is an exception. This is how tests inject an empty response, prose with no code,
a syntax error, or an inference failure at one specific generation index.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any

from .base import PROVIDER_MOCK, GenerationConfig, RawGeneration

DEFAULT_MOCK_NAME = "mock/deterministic-coder"

# Matches the "Required function signature:" line of a built prompt.
_SIGNATURE_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)

_FALLBACK_ENTRY_POINT = "solution"


def _wrap_python_fence(code: str) -> str:
    return f"```python\n{code}\n```"


def _wrap_prose_fence(code: str) -> str:
    return f"Here is the implementation:\n\n```python\n{code}\n```"


def _wrap_generic_fence(code: str) -> str:
    return f"```\n{code}\n```"


def _wrap_plain(code: str) -> str:
    return code


_WRAPPERS = (_wrap_python_fence, _wrap_prose_fence, _wrap_generic_fence, _wrap_plain)


class MockModelClient:
    """Deterministic stand-in for a real model."""

    def __init__(
        self,
        *,
        script: Sequence[str | BaseException] | None = None,
        name: str = DEFAULT_MOCK_NAME,
        revision: str | None = None,
    ) -> None:
        self._script = list(script) if script is not None else None
        self._name = name
        self._revision = revision
        self._calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def revision(self) -> str | None:
        return self._revision

    @property
    def provider(self) -> str:
        return PROVIDER_MOCK

    @property
    def call_count(self) -> int:
        """Number of ``generate`` calls made, for assertions about resume behavior."""
        return self._calls

    def generate(self, prompt: str, generation_config: GenerationConfig) -> RawGeneration:
        index = self._calls
        self._calls += 1

        if self._script is not None and index < len(self._script):
            item: Any = self._script[index]
            if isinstance(item, BaseException):
                raise item
            text = str(item)
        else:
            text = _synthesize(prompt)

        return RawGeneration(
            text=text,
            prompt_token_count=len(prompt.split()),
            completion_token_count=len(text.split()),
            finish_reason="stop",
        )


def entry_point_from_prompt(prompt: str) -> str:
    """Recover the target function name from a built prompt.

    Keyed off the prompt text rather than a ``Problem`` object so the mock stays a pure
    ``ModelClient`` — it sees exactly what a real model sees.
    """
    match = _SIGNATURE_RE.search(prompt)
    return match.group(1) if match else _FALLBACK_ENTRY_POINT


def _synthesize(prompt: str) -> str:
    """Build deterministic output for a prompt.

    ``hashlib`` rather than ``hash()``: string hashing is randomized per interpreter
    process, which would make output differ between runs and break the determinism this
    class exists to provide.
    """
    entry_point = entry_point_from_prompt(prompt)
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    variant = digest[0] % 1000
    wrapper = _WRAPPERS[digest[1] % len(_WRAPPERS)]

    code = (
        f"def {entry_point}(*args, **kwargs):\n"
        f'    """Mock implementation variant {variant}."""\n'
        f"    return {variant}\n"
    )
    return wrapper(code.rstrip("\n"))
