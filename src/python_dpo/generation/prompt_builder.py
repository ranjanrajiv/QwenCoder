"""Prompt construction (spec 03 sections 14, 15).

``build_prompt`` is pure: the same problem and strategy always produce byte-identical
text. No timestamps, no randomness, no model state. That is what makes a candidate
reproducible from its stored record.

**Versioning rule (section 15):** changing ``_TEMPLATE`` — including whitespace — changes
what the model sees and therefore what the dataset means. Bump :data:`PROMPT_VERSION`
whenever the template changes, so datasets generated under different templates stay
distinguishable. The same rule applies to :data:`CANONICAL_PROMPT_VERSION` and
``build_canonical_prompt`` below.

``build_canonical_prompt`` (spec 08 section 10, decision 1) renders the same problem
without a per-candidate strategy instruction — a Stage 8 preference pair's ``prompt`` must
be identical for both the chosen and rejected candidate, but every candidate of a problem
was generated under a *different* strategy block, so the raw per-candidate prompt can never
satisfy that. The canonical template is derived from ``_TEMPLATE`` itself (by stripping the
strategy section as a substring, not by hand-authoring a second literal), so the two
renderings cannot drift apart independently of one another.
"""

from __future__ import annotations

from ..problems.models import Problem
from .strategies import instruction_for

PROMPT_VERSION = "v1"
CANONICAL_PROMPT_VERSION = "v1"

_TEMPLATE = """You are an expert Python programmer.

Solve the following programming problem.

Problem:
{problem}

Required function signature:
{signature}

Strategy:
{strategy_instruction}

Requirements:
- Implement the requested function.
- Follow the function signature.
- Handle the specified edge cases.
- Use Python.
- Return only the implementation.
- Do not provide an explanation.
- Do not use eval().
- Do not use exec().
- Do not perform network operations.
- Do not read or write files."""

# The exact substring covering the strategy section, including its surrounding blank
# lines. Removing it from _TEMPLATE derives the canonical (strategy-free) template from
# the single source of truth above, rather than a second hand-authored literal that could
# silently drift from it.
_STRATEGY_SECTION = "\n\nStrategy:\n{strategy_instruction}"
assert _STRATEGY_SECTION in _TEMPLATE  # noqa: S101 - fails import if _TEMPLATE ever moves
_CANONICAL_TEMPLATE = _TEMPLATE.replace(_STRATEGY_SECTION, "")


def build_prompt(problem: Problem, strategy: str) -> str:
    """Render the prompt for one problem under one generation strategy."""
    return _TEMPLATE.format(
        problem=problem.prompt.strip(),
        signature=problem.signature.strip(),
        strategy_instruction=instruction_for(strategy),
    )


def build_canonical_prompt(problem: Problem) -> str:
    """Render the strategy-free prompt for one problem (spec 08 decision 1).

    Identical for every candidate of ``problem`` regardless of which strategy generated
    it, which is what lets a Stage 8 preference pair satisfy "the prompt used for chosen
    and rejected must be identical" (spec 08 section 11) without inventing a new problem
    statement (spec 08 section 10).
    """
    return _CANONICAL_TEMPLATE.format(
        problem=problem.prompt.strip(),
        signature=problem.signature.strip(),
    )
