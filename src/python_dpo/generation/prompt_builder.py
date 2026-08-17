"""Prompt construction (spec 03 sections 14, 15).

``build_prompt`` is pure: the same problem and strategy always produce byte-identical
text. No timestamps, no randomness, no model state. That is what makes a candidate
reproducible from its stored record.

**Versioning rule (section 15):** changing ``_TEMPLATE`` — including whitespace — changes
what the model sees and therefore what the dataset means. Bump :data:`PROMPT_VERSION`
whenever the template changes, so datasets generated under different templates stay
distinguishable.
"""

from __future__ import annotations

from ..problems.models import Problem
from .strategies import instruction_for

PROMPT_VERSION = "v1"

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


def build_prompt(problem: Problem, strategy: str) -> str:
    """Render the prompt for one problem under one generation strategy."""
    return _TEMPLATE.format(
        problem=problem.prompt.strip(),
        signature=problem.signature.strip(),
        strategy_instruction=instruction_for(strategy),
    )
