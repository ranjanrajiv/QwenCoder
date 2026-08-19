"""Stage-level state machine (spec 12 section 13).

Mirrors the shape of :mod:`python_dpo.runs.models`'s ``RUN_STATUS_TRANSITIONS``: a closed
set of states and an explicit adjacency table, so an illegal transition is a data error
caught at the call site rather than a manifest silently drifting into nonsense.

This machine sits *above* the six existing per-stage run repositories (candidates, runs,
evaluations, rankings, preferences, training, model_evaluations). A ``StageState`` tracks
whether the orchestrator considers a pipeline stage done for this experiment; the
underlying repository's own run status (``created``/``running``/``completed``/...) is a
separate, independent concern recorded in the stage manifest alongside it.
"""

from __future__ import annotations

STAGE_STATES = frozenset(
    {"PENDING", "RUNNING", "COMPLETED", "FAILED", "SKIPPED", "CANCELLED", "BLOCKED"}
)

# Terminal states (COMPLETED, SKIPPED, CANCELLED) have no outgoing edges except the single
# explicit invalidation path out of COMPLETED (spec section 22, `--force`) -- everything
# else about a terminal state's history is immutable.
STAGE_STATE_TRANSITIONS: dict[str, frozenset[str]] = {
    "PENDING": frozenset({"RUNNING", "SKIPPED", "BLOCKED", "CANCELLED"}),
    "RUNNING": frozenset({"COMPLETED", "FAILED", "CANCELLED"}),
    "BLOCKED": frozenset({"PENDING", "RUNNING", "CANCELLED"}),
    "FAILED": frozenset({"RUNNING", "CANCELLED"}),
    # `--force` explicitly invalidates a completed stage back to PENDING (spec section 22);
    # this is the only edge out of a terminal state, and it is never taken implicitly.
    "COMPLETED": frozenset({"PENDING"}),
    "SKIPPED": frozenset(),
    "CANCELLED": frozenset(),
}


class StateError(Exception):
    """Raised on an illegal stage-state transition."""


def validate_transition(current: str, target: str) -> None:
    """Raise :class:`StateError` unless ``current -> target`` is a legal edge."""
    if current not in STAGE_STATES:
        raise StateError(f"unknown stage state {current!r}")
    if target not in STAGE_STATES:
        raise StateError(f"unknown stage state {target!r}")
    if target not in STAGE_STATE_TRANSITIONS[current]:
        raise StateError(f"illegal stage-state transition: {current!r} -> {target!r}")


__all__ = ["STAGE_STATE_TRANSITIONS", "STAGE_STATES", "StateError", "validate_transition"]
