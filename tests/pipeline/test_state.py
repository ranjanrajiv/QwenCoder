"""Tests for the stage-level state machine (spec 12 section 13)."""

from __future__ import annotations

import pytest

from python_dpo.pipeline.state import (
    STAGE_STATE_TRANSITIONS,
    STAGE_STATES,
    StateError,
    validate_transition,
)


def test_seven_states_defined():
    assert STAGE_STATES == {
        "PENDING",
        "RUNNING",
        "COMPLETED",
        "FAILED",
        "SKIPPED",
        "CANCELLED",
        "BLOCKED",
    }


@pytest.mark.parametrize(
    ("current", "target"),
    [
        ("PENDING", "RUNNING"),
        ("PENDING", "SKIPPED"),
        ("PENDING", "BLOCKED"),
        ("RUNNING", "COMPLETED"),
        ("RUNNING", "FAILED"),
        ("BLOCKED", "PENDING"),
        ("BLOCKED", "RUNNING"),
        ("FAILED", "RUNNING"),  # retry
        ("COMPLETED", "PENDING"),  # --force invalidation
    ],
)
def test_legal_transitions(current, target):
    validate_transition(current, target)  # must not raise


@pytest.mark.parametrize(
    ("current", "target"),
    [
        ("COMPLETED", "RUNNING"),
        ("COMPLETED", "FAILED"),
        ("SKIPPED", "RUNNING"),
        ("CANCELLED", "RUNNING"),
        ("PENDING", "COMPLETED"),
        ("FAILED", "COMPLETED"),
    ],
)
def test_illegal_transitions_raise(current, target):
    with pytest.raises(StateError, match="illegal"):
        validate_transition(current, target)


def test_unknown_current_state_raises():
    with pytest.raises(StateError, match="unknown"):
        validate_transition("NOT_A_STATE", "RUNNING")


def test_unknown_target_state_raises():
    with pytest.raises(StateError, match="unknown"):
        validate_transition("PENDING", "NOT_A_STATE")


def test_every_state_has_a_transition_entry():
    assert set(STAGE_STATE_TRANSITIONS) == STAGE_STATES


def test_terminal_states_have_no_outgoing_edges_except_completed_force_invalidation():
    assert STAGE_STATE_TRANSITIONS["SKIPPED"] == frozenset()
    assert STAGE_STATE_TRANSITIONS["CANCELLED"] == frozenset()
    assert STAGE_STATE_TRANSITIONS["COMPLETED"] == frozenset({"PENDING"})
