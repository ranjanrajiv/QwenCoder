"""The preferences exception hierarchy (spec 08).

Preference-pair generation is pure computation over already-persisted evidence (spec 08
section 87): it never calls a model, never touches Docker, never re-executes candidate
code. Every failure here is therefore about the preference-generation *machinery* — a
missing ranking run, a malformed manifest, an unknown selection policy — never a statement
about a candidate pair. A candidate pair that cannot be turned into a preference becomes a
recorded :class:`~python_dpo.preferences.models.PreferenceRejection` with a reason, not an
exception (spec section 77, CLAUDE.md's data-integrity rule).

Unlike Stage 7, every preference-specific exception lives here rather than being split
across ``errors.py``/``repository.py``/``run_repository.py`` — a deliberate consolidation
recorded in the Stage 8 plan.
"""

from __future__ import annotations


class PreferenceError(Exception):
    """Base class for every preference-generation failure."""


class PreferenceConfigError(PreferenceError):
    """Raised when the ``preferences:`` configuration section is invalid."""


class PreferencePolicyError(PreferenceError):
    """Raised when an unknown selection policy is requested."""


class RankingRunNotFoundError(PreferenceError):
    """Raised when the ranking run a preference generation is requested against does not
    exist.

    Distinct from :class:`~python_dpo.preferences.run_repository.PreferenceRunNotFoundError`,
    which is about *this* package's own preference run directories — this is about the
    Stage 7 ranking run a new preference run is being built from. Mirrors
    :class:`python_dpo.ranking.errors.EvaluationRunNotFoundError`.
    """


class PreferenceRunNotFoundError(PreferenceError):
    """Raised when a preference run id has no corresponding directory."""


class PreferenceStoreError(PreferenceError):
    """Raised when a persisted preference file is malformed."""


__all__ = [
    "PreferenceConfigError",
    "PreferenceError",
    "PreferencePolicyError",
    "PreferenceRunNotFoundError",
    "PreferenceStoreError",
    "RankingRunNotFoundError",
]
