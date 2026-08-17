"""The five generation strategies (spec 03 section 13).

Each strategy is a different instruction to the model, chosen so that five candidates for
one problem explore genuinely different implementations rather than five samples of the
same one. Diversity is what makes a preference pair informative.

These are **generation prompts, not correctness labels**. Nothing here claims the
``optimized`` candidate will actually be faster or the ``edge_case_focused`` one actually
more careful — later evaluation decides that.
"""

from __future__ import annotations

from collections.abc import Sequence

STRATEGIES: tuple[str, ...] = (
    "normal",
    "straightforward",
    "edge_case_focused",
    "alternative",
    "optimized",
)

STRATEGY_INSTRUCTIONS: dict[str, str] = {
    "normal": "Solve the problem using a clear, correct Python implementation.",
    "straightforward": "Provide a simple and easy-to-understand Python implementation.",
    "edge_case_focused": (
        "Pay particular attention to empty inputs, boundary conditions, duplicates, and "
        "other edge cases specified by the problem."
    ),
    "alternative": (
        "Solve the problem using an alternative reasonable algorithm or implementation "
        "approach."
    ),
    "optimized": (
        "Produce an efficient implementation appropriate for the problem constraints."
    ),
}


class StrategyError(Exception):
    """Raised when an unknown strategy is requested or a count makes no sense."""


def instruction_for(strategy: str) -> str:
    try:
        return STRATEGY_INSTRUCTIONS[strategy]
    except KeyError:
        raise StrategyError(
            f"unknown strategy {strategy!r}; expected one of {', '.join(STRATEGIES)}"
        ) from None


def resolve_strategies(
    configured: Sequence[str],
    count: int,
    override: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return exactly ``count`` strategy names, one per candidate.

    With the default five strategies and ``count=5`` this is one candidate per strategy
    (spec 03 section 31). ``count`` beyond the list length cycles, so asking for ten
    candidates gives two passes over the five strategies rather than an error.
    ``override`` is the ``--strategy`` flag and replaces the configured list entirely.
    """
    names = tuple(override) if override else tuple(configured)
    if not names:
        raise StrategyError("no generation strategies configured")

    for name in names:
        instruction_for(name)

    if count < 1:
        raise StrategyError(f"candidate count must be at least 1, got {count}")

    return tuple(names[index % len(names)] for index in range(count))
