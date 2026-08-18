"""The three distinct deduplication notions a preference dataset needs (spec sections 32,
33, 72, 73) — kept as separate functions because conflating any two of them is how a
dataset silently loses rows it should keep, or keeps rows it should collapse.

+------------------+---------------------------------------------------+--------------+
| ``pair_key``     | ``(problem_id, chosen_id, rejected_id)`` — ordered | sections 32, |
|                  | pair identity; ``A>B`` and ``B>A`` are NOT the     | 73           |
|                  | same key (a ``B>A`` alongside an ``A>B`` is an     |              |
|                  | invalid reverse preference, section 71, not a      |              |
|                  | duplicate to silently merge)                       |              |
+------------------+---------------------------------------------------+--------------+
| ``code_identical``| candidate-level: do two candidates share code?    | 33, 34, 74   |
+------------------+---------------------------------------------------+--------------+
| ``training_key`` | ``(prompt, chosen, rejected)`` — the text-level    | 72           |
|                  | identity a DPO trainer actually sees               |              |
+------------------+---------------------------------------------------+--------------+

Spec section 74's asymmetry: two candidates with identical code may not pair with *each
other* (no informative preference between identical text), but each may still pair
against a third, different candidate. ``code_identical`` only ever gates a single pair; it
never removes a candidate from the pool.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from ..candidates.models import Candidate
from .models import PreferencePair


def pair_key(pair: PreferencePair) -> tuple[str, str, str]:
    """The ordered, directional identity of one preference pair (spec sections 32, 73)."""
    return (pair.problem_id, pair.chosen_candidate_id, pair.rejected_candidate_id)


def code_identical(a: Candidate, b: Candidate) -> bool:
    """Whether two candidates have byte-identical code (spec sections 33, 34, 74)."""
    return a.code_sha256 == b.code_sha256


def training_key(pair: PreferencePair) -> tuple[str, str, str]:
    """The text-level identity a DPO trainer sees (spec section 72) — the one that
    collapses many candidate-id pairs into few distinct training records when candidates
    share code across strategies (spec sections 34, 74).
    """
    return (pair.prompt, pair.chosen, pair.rejected)


def dedupe_training_records(pairs: Sequence[PreferencePair]) -> list[PreferencePair]:
    """Mark every pair beyond the first occurrence of its :func:`training_key` as a
    duplicate training record (spec section 72, decision 3).

    "First occurrence" is decided by :attr:`PreferencePair.preference_id` order, not input
    order (spec plan decision 3: "first occurrence in preference_id order wins") — so the
    result is independent of how the caller happened to enumerate problems. Every pair is
    kept in the output (nothing is dropped, CLAUDE.md's data-integrity rule); duplicates
    are only flagged via ``duplicate_training_record``/``canonical_preference_id`` so
    ``metadata.jsonl`` still records all of them while ``preferences.jsonl`` can filter to
    the survivors.
    """
    groups: dict[tuple[str, str, str], list[PreferencePair]] = {}
    for pair in pairs:
        groups.setdefault(training_key(pair), []).append(pair)

    survivor_by_key: dict[tuple[str, str, str], str] = {}
    for key, group in groups.items():
        survivor = min(group, key=lambda p: p.preference_id)
        survivor_by_key[key] = survivor.preference_id

    result: list[PreferencePair] = []
    for pair in pairs:
        canonical_id = survivor_by_key[training_key(pair)]
        if pair.preference_id == canonical_id:
            result.append(
                dataclasses.replace(
                    pair, duplicate_training_record=False, canonical_preference_id=None
                )
            )
        else:
            result.append(
                dataclasses.replace(
                    pair,
                    duplicate_training_record=True,
                    canonical_preference_id=canonical_id,
                )
            )
    return result


__all__ = ["code_identical", "dedupe_training_records", "pair_key", "training_key"]
