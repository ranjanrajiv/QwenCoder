"""Preference-pair selection policies (spec 08 sections 24-28).

Each policy answers one question: given a decisive comparison (never a tie, never
indeterminate — the builder filters those out before a policy ever sees a pair), is this
candidate pair confident enough to become a DPO preference pair? A policy is pure and
stateless, and returns the rejection *reason* rather than a bare ``False`` (spec section
77), so the builder never has to re-derive why a pair was declined.

The universal exclusions that apply regardless of policy — ties, indeterminate
candidates, identical code, cross-problem pairs, invalid prompt provenance, integrity
failures — live in ``builder.py``, not here, so no policy can accidentally omit one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..ranking.models import CandidateAssessment
from .errors import PreferencePolicyError

# Spec section 22: the default v1 margin. Configurable, never hard-coded in the builder.
DEFAULT_MINIMUM_SCORE_MARGIN = 0.2

_MARGIN_EPSILON = 1e-9


class PreferencePolicy(Protocol):
    """A pair-selection policy (spec section 24)."""

    name: str
    version: str

    def admits(
        self,
        chosen: CandidateAssessment,
        rejected: CandidateAssessment,
        *,
        minimum_score_margin: float,
    ) -> tuple[bool, str | None]:
        """Return ``(admitted, rejection_reason)``. ``chosen.score > rejected.score`` is
        already guaranteed true by the caller before a policy is ever consulted.
        """
        ...


@dataclass(frozen=True)
class StrictPolicy:
    """Correct vs incorrect only, no margin gate (spec section 25, decision 2).

    The highest-confidence policy: a pair is admitted purely on the correctness
    classification, independent of how large the score margin happens to be. Decision 2
    keeps this deliberately independent of :class:`MarginPolicy` so the two datasets a
    single ranking run produces are a genuine comparison rather than one being a subset of
    the other.
    """

    name: str = "strict"
    version: str = "strict_v1"

    def admits(
        self,
        chosen: CandidateAssessment,
        rejected: CandidateAssessment,
        *,
        minimum_score_margin: float,
    ) -> tuple[bool, str | None]:
        if chosen.correctness == "correct" and rejected.correctness == "incorrect":
            return True, None
        return False, "not_correct_vs_incorrect"


@dataclass(frozen=True)
class MarginPolicy:
    """Any objectively better pair clearing ``minimum_score_margin`` (spec section 26)."""

    name: str = "margin"
    version: str = "margin_v1"

    def admits(
        self,
        chosen: CandidateAssessment,
        rejected: CandidateAssessment,
        *,
        minimum_score_margin: float,
    ) -> tuple[bool, str | None]:
        margin = chosen.score - rejected.score
        if margin + _MARGIN_EPSILON >= minimum_score_margin:
            return True, None
        return False, "insufficient_margin"


@dataclass(frozen=True)
class AllBetterPolicy:
    """Every objectively better pair, no margin gate (spec section 27).

    Useful for experimentation; never the default (spec section 27) because it accepts
    every decisive comparison, however small the margin.
    """

    name: str = "all_better"
    version: str = "all_better_v1"

    def admits(
        self,
        chosen: CandidateAssessment,
        rejected: CandidateAssessment,
        *,
        minimum_score_margin: float,
    ) -> tuple[bool, str | None]:
        del minimum_score_margin  # unused: this policy has no margin gate
        return True, None


_POLICIES_BY_NAME: dict[str, type[PreferencePolicy]] = {
    "strict": StrictPolicy,
    "margin": MarginPolicy,
    "all_better": AllBetterPolicy,
}


def make_policy(name: str) -> PreferencePolicy:
    """Instantiate the named policy (spec section 24)."""
    try:
        policy_cls = _POLICIES_BY_NAME[name]
    except KeyError:
        raise PreferencePolicyError(
            f"unknown selection policy {name!r}; expected one of "
            f"{', '.join(sorted(_POLICIES_BY_NAME))}"
        ) from None
    return policy_cls()


__all__ = [
    "DEFAULT_MINIMUM_SCORE_MARGIN",
    "AllBetterPolicy",
    "MarginPolicy",
    "PreferencePolicy",
    "StrictPolicy",
    "make_policy",
]
