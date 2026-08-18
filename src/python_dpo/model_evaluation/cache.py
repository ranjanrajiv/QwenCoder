"""Content-addressed cache keys for generation and evaluation reuse (spec sections 92-94,
138, 139).

Two closed key shapes, each reduced to one ``digest`` -- a SHA-256 hex string over the
key's canonical JSON. **Model identity is always part of the generation key** (spec
section 94): the same prompt under ``base`` and ``dpo`` always digests differently, so the
two variants can never collide into one cache entry merely because the rendered prompt
text matched.

This module defines the keys and a minimal, explicit key-value store. Reuse across
runs (spec section 138 -- caching a Base Qwen baseline across several DPO experiments) is
available through :class:`JsonCacheStore` but is opt-in: nothing in
:mod:`python_dpo.model_evaluation.generation` or ``.evaluation`` consults a cache
automatically, so a cache is never a hidden source of truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..candidates.hashing import sha256_text


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True)
class GenerationCacheKey:
    """Identifies one generation request (spec section 92).

    ``adapter_identity`` is ``None`` for the base variant and the adapter's path (or a
    hash of its weights) for DPO -- this alone is what spec section 94 requires to keep
    base and DPO from ever sharing an entry.
    """

    model_variant: str
    model_name: str
    model_revision: str | None
    adapter_identity: str | None
    prompt_sha256: str
    generation_config: dict[str, Any]
    seed: int

    @property
    def digest(self) -> str:
        return sha256_text(
            _canonical(
                {
                    "model_variant": self.model_variant,
                    "model_name": self.model_name,
                    "model_revision": self.model_revision,
                    "adapter_identity": self.adapter_identity,
                    "prompt_sha256": self.prompt_sha256,
                    "generation_config": self.generation_config,
                    "seed": self.seed,
                }
            )
        )


@dataclass(frozen=True)
class EvaluationCacheKey:
    """Identifies one candidate evaluation (spec section 93)."""

    candidate_code_sha256: str
    problem_id: str
    test_suite_hash: str
    evaluator_version: str
    sandbox_config: dict[str, Any]

    @property
    def digest(self) -> str:
        return sha256_text(
            _canonical(
                {
                    "candidate_code_sha256": self.candidate_code_sha256,
                    "problem_id": self.problem_id,
                    "test_suite_hash": self.test_suite_hash,
                    "evaluator_version": self.evaluator_version,
                    "sandbox_config": self.sandbox_config,
                }
            )
        )


class JsonCacheStore:
    """A flat digest -> JSON-payload store backed by one file.

    Deliberately simple: :meth:`get`/:meth:`put` only, no eviction, no TTL. Suitable for
    the spec section 138 baseline-reuse case (one cache file per shared Base Qwen result),
    not for a general-purpose cache service.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._data: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._data is None:
            if self.path.is_file():
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            else:
                self._data = {}
        return self._data

    def get(self, digest: str) -> Any | None:
        return self._load().get(digest)

    def put(self, digest: str, payload: Any) -> None:
        data = self._load()
        data[digest] = payload
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp_path.replace(self.path)


__all__ = ["EvaluationCacheKey", "GenerationCacheKey", "JsonCacheStore"]
