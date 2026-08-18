"""Canonical prompt reconstruction and lineage verification (spec 08 decision 1).

Every candidate of a problem was generated under a *different* per-candidate prompt (it
embeds the generation strategy — see ``generation/prompt_builder.py``), so no two
candidates of the same problem share a ``prompt_sha256``. Applied literally, spec section
41's "``chosen.prompt_sha256 == rejected.prompt_sha256`` or reject the pair" would produce
zero preference pairs under every policy.

The resolution is to build the pair's ``prompt`` from a *canonical*, strategy-free
rendering of the problem (:func:`~python_dpo.generation.prompt_builder.build_canonical_prompt`)
instead of either candidate's own generation-time prompt. That still satisfies section 10's
"do not invent a new problem statement": :func:`verify_prompt_lineage` proves, for every
candidate a problem's pairs will be built from, that its stored ``prompt_sha256`` is
exactly what the *current* prompt template produces for its own strategy — so the
canonical prompt is demonstrably a rendering of the same template, never a fabrication.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..candidates.hashing import sha256_text
from ..candidates.models import Candidate
from ..generation.prompt_builder import PROMPT_VERSION, build_canonical_prompt, build_prompt
from ..problems.models import Problem
from .errors import PreferenceError


class PromptLineageError(PreferenceError):
    """Raised when a candidate's stored prompt cannot be traced back to the current
    generation prompt template for its problem and strategy.

    Spec section 42's candidate integrity check, specialized to the prompt: a candidate
    that fails this check is excluded from pairing (recorded as an ``integrity_failure``
    rejection), never silently paired using an unverifiable prompt.
    """


def verify_prompt_lineage(
    problem: Problem, candidates: Sequence[Candidate]
) -> tuple[str, str]:
    """Rebuild ``problem``'s canonical prompt and verify every candidate in ``candidates``
    was actually generated from the current template.

    Returns ``(canonical_prompt, canonical_prompt_sha256)``. Raises
    :class:`PromptLineageError` if any candidate belongs to a different problem, was
    generated under a stale ``prompt_version``, or its stored ``prompt_sha256`` does not
    match a freshly rebuilt prompt for its own strategy — a tampered or stale record, never
    silently trusted or silently dropped.
    """
    canonical_prompt = build_canonical_prompt(problem)
    canonical_prompt_sha256 = sha256_text(canonical_prompt)

    for candidate in candidates:
        if candidate.problem_id != problem.id:
            raise PromptLineageError(
                f"candidate {candidate.candidate_id} belongs to problem "
                f"{candidate.problem_id!r}, not {problem.id!r}"
            )
        if candidate.prompt_version != PROMPT_VERSION:
            raise PromptLineageError(
                f"candidate {candidate.candidate_id}: prompt_version "
                f"{candidate.prompt_version!r} does not match the current generation "
                f"template {PROMPT_VERSION!r}"
            )
        expected = sha256_text(build_prompt(problem, candidate.strategy))
        if candidate.prompt_sha256 != expected:
            raise PromptLineageError(
                f"candidate {candidate.candidate_id}: stored prompt_sha256 does not match "
                "a freshly rebuilt prompt for its own strategy"
            )

    return canonical_prompt, canonical_prompt_sha256


__all__ = ["PromptLineageError", "verify_prompt_lineage"]
