"""Static checks on generated code (spec 03 sections 19, 42, 43).

**Nothing here executes the candidate.** ``ast.parse`` builds a syntax tree and stops;
it does not run module-level statements, import anything, or evaluate expressions. That
is the whole point — generated code is untrusted (CLAUDE.md, Security), and the only
component permitted to execute Python is ``InProcessReferenceExecutor``, which handles
manually authored reference solutions exclusively.

These checks establish *structure*, never correctness. ``def factorial(n): return 123``
passes both of them and is still wrong; deciding that is a later stage's job.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class SyntaxCheck:
    """Whether code parses, and why not when it doesn't."""

    valid: bool
    error_message: str | None = None


def _format_syntax_error(exc: SyntaxError) -> str:
    message = exc.msg or "invalid syntax"
    if exc.lineno is None:
        return message
    return f"{message} (line {exc.lineno})"


def check_syntax(code: str) -> SyntaxCheck:
    """Parse ``code`` with the AST parser. Never executes it."""
    if not isinstance(code, str):
        return SyntaxCheck(valid=False, error_message="code must be a string")
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return SyntaxCheck(valid=False, error_message=_format_syntax_error(exc))
    except ValueError as exc:
        # Raised for source containing null bytes, which is not a SyntaxError.
        return SyntaxCheck(valid=False, error_message=f"{type(exc).__name__}: {exc}")
    return SyntaxCheck(valid=True)


def check_function_name(code: str, entry_point: str) -> bool:
    """Report whether ``code`` defines a function named ``entry_point``.

    Uses ``Problem.entry_point`` from the Stage 2 schema rather than re-parsing the
    signature string. Both ``def`` and ``async def`` count — p010 is an async problem.
    Nested definitions count too: a helper-wrapped implementation is a structural
    oddity, not a missing function.

    Unparseable code returns ``False`` rather than raising; the caller has already
    recorded *why* it does not parse via :func:`check_syntax`.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return False

    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point
        for node in ast.walk(tree)
    )
