"""The evaluation exception hierarchy (spec 06).

Failures here are always about the evaluation *machinery* — problem/candidate lookup,
test generation, or result parsing. A failure that says something about a *candidate*
(a runtime exception in a generated test, a timeout, a syntax error) is never raised as
an exception; it is recorded as an :class:`~python_dpo.evaluation.models.EvaluationResult`
with the appropriate status. That distinction mirrors the sandbox's own separation of
candidate outcomes from infrastructure failures (spec 05 section 81, spec 06 section 29).
"""

from __future__ import annotations


class EvaluationError(Exception):
    """Base class for every evaluation failure."""


class EvaluationConfigError(EvaluationError):
    """Raised when the ``evaluation:`` configuration section is invalid.

    Deliberately not ``python_dpo.config.ConfigError`` — this package must not import the
    configuration layer, matching how ``sandbox.errors.SandboxConfigError`` keeps that
    dependency one-way. ``config.py`` catches and re-raises as ``ConfigError``.
    """


class ProblemNotFoundError(EvaluationError):
    """Raised when a candidate references a problem id that does not exist."""


class CandidateNotFoundError(EvaluationError):
    """Raised when a requested candidate id does not exist in the given generation run."""


class InvalidProblemError(EvaluationError):
    """Raised when a problem has no test cases to evaluate against (spec section 45)."""


class TestGenerationError(EvaluationError):
    """Raised when a generated pytest job fails structural validation (spec section 47).

    Never raised because the *candidate's* Python failed to compile — that is a normal,
    recorded ``syntax_error`` outcome. This is raised only when the evaluator's own
    generated test file is wrong, which is an evaluator bug, not a candidate outcome.
    """

    # Not a pytest test class despite the name; keeps collection warnings away.
    __test__ = False


class ResultParseError(EvaluationError):
    """Raised when pytest's structured output cannot be parsed at all.

    A parse failure is an evaluation-machinery problem (the reporting plugin misbehaved),
    not a statement about the candidate.
    """


__all__ = [
    "CandidateNotFoundError",
    "EvaluationConfigError",
    "EvaluationError",
    "InvalidProblemError",
    "ProblemNotFoundError",
    "ResultParseError",
    "TestGenerationError",
]
