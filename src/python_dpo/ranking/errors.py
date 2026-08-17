"""The ranking exception hierarchy (spec 07).

Ranking is pure computation over already-persisted evidence (spec 07 section 2) — it
never calls a model, never touches Docker, never executes candidate code. Every failure
here is therefore about the ranking *machinery* — a missing evaluation run, a malformed
manifest, a broken transition — never a statement about a candidate. A candidate that
cannot be assessed becomes an ``indeterminate`` :class:`~python_dpo.ranking.models.
CandidateAssessment` with a reason, not an exception (spec sections 12, 70, 71).
"""

from __future__ import annotations


class RankingError(Exception):
    """Base class for every ranking failure."""


class RankingConfigError(RankingError):
    """Raised when a ranking configuration value is invalid."""


class EvaluationRunNotFoundError(RankingError):
    """Raised when the evaluation run a ranking is requested against does not exist.

    Distinct from :class:`python_dpo.ranking.run_repository.RankingRunNotFoundError`,
    which is about *this* package's own ranking run directories — this is about the
    Stage 6 evaluation run a new ranking run is being built from.
    """


__all__ = [
    "EvaluationRunNotFoundError",
    "RankingConfigError",
    "RankingError",
]
