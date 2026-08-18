"""Tests for the three deduplication notions (spec 08 sections 32, 33, 72, 73, 74)."""

from __future__ import annotations

from typing import Any

from python_dpo.candidates.models import Candidate
from python_dpo.generation.prompt_builder import PROMPT_VERSION
from python_dpo.preferences.dedup import (
    code_identical,
    dedupe_training_records,
    pair_key,
    training_key,
)

from .test_models import make_pair


def make_candidate(candidate_id: str, code: str, **overrides: Any) -> Candidate:
    fields: dict[str, Any] = {
        "candidate_id": candidate_id,
        "problem_id": "p001",
        "run_id": "run_20260817_055411",
        "generation_index": 1,
        "strategy": "normal",
        "model": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "provider": "transformers",
        "prompt_version": PROMPT_VERSION,
        "prompt": "prompt text",
        "raw_output": f"```python\n{code}\n```",
        "code": code,
        "extraction_format": "python_fence",
        "syntax_valid": True,
        "function_name_valid": True,
        "generation_config": {},
        "created_at": "2026-08-17T05:54:18Z",
    }
    fields.update(overrides)
    return Candidate.create(**fields)


# ---------------------------------------------------------------------------- pair_key


def test_pair_key_is_directional():
    a_beats_b = make_pair(
        preference_id="pref_p001_c001__p001_c002",
        chosen_candidate_id="p001_c001",
        rejected_candidate_id="p001_c002",
    )
    b_beats_a = make_pair(
        preference_id="pref_p001_c002__p001_c001",
        chosen_candidate_id="p001_c002",
        rejected_candidate_id="p001_c001",
        chosen_score=0.9,
        rejected_score=0.1,
        score_margin=0.8,
    )
    # Spec section 32: A>B and B>A are different keys, not duplicates of each other.
    assert pair_key(a_beats_b) != pair_key(b_beats_a)


# ------------------------------------------------------------------------ code_identical


def test_code_identical_true_for_same_sha256():
    a = make_candidate("p001_c001", "def f():\n    return 1")
    b = make_candidate("p001_c002", "def f():\n    return 1")
    assert code_identical(a, b)


def test_code_identical_false_for_different_code():
    a = make_candidate("p001_c001", "def f():\n    return 1")
    b = make_candidate("p001_c002", "def f():\n    return 2")
    assert not code_identical(a, b)


def test_section_74_worked_example():
    # A = code X, B = code X, C = code Y. A>C and B>C are each valid (different code);
    # A vs B would not be (same code) — dedup only ever gates a single pair, never
    # removes a candidate from the pool entirely.
    a = make_candidate("p001_c001", "def f():\n    return 1")
    b = make_candidate("p001_c002", "def f():\n    return 1")
    c = make_candidate("p001_c003", "def f():\n    return 2")
    assert code_identical(a, b)
    assert not code_identical(a, c)
    assert not code_identical(b, c)


# ------------------------------------------------------------------------ training_key


def test_training_key_collapses_identical_text_triples():
    p1 = make_pair(preference_id="pref_p001_c001__p001_c002")
    p2 = make_pair(preference_id="pref_p001_c003__p001_c004")  # same prompt/chosen/rejected text
    assert training_key(p1) == training_key(p2)


def test_same_prompt_different_code_is_not_a_duplicate():
    # Spec section 73: sharing a prompt alone is never grounds for treating two records
    # as duplicates — only the full (prompt, chosen, rejected) triple is.
    p1 = make_pair(preference_id="pref_p001_c001__p001_c002")
    p2 = make_pair(
        preference_id="pref_p001_c003__p001_c004",
        chosen="def f():\n    return 3",
        rejected="def f():\n    return 4",
        chosen_code_sha256="f" * 64,
        rejected_code_sha256="g" * 64,
    )
    assert p1.prompt == p2.prompt
    assert training_key(p1) != training_key(p2)


# ------------------------------------------------------------------- dedupe_training_records


def test_dedupe_marks_all_but_the_lexicographically_first_preference_id():
    survivor = make_pair(preference_id="pref_p001_c001__p001_c002")
    later = make_pair(preference_id="pref_p001_c003__p001_c004")  # identical text
    result = dedupe_training_records([later, survivor])  # deliberately out of order
    by_id = {p.preference_id: p for p in result}
    assert not by_id["pref_p001_c001__p001_c002"].duplicate_training_record
    assert by_id["pref_p001_c003__p001_c004"].duplicate_training_record
    assert (
        by_id["pref_p001_c003__p001_c004"].canonical_preference_id
        == "pref_p001_c001__p001_c002"
    )


def test_dedupe_is_a_noop_for_distinct_triples():
    p1 = make_pair(preference_id="pref_p001_c001__p001_c002")
    p2 = make_pair(
        preference_id="pref_p001_c003__p001_c004",
        chosen="def f():\n    return 3",
        rejected="def f():\n    return 4",
        chosen_code_sha256="f" * 64,
        rejected_code_sha256="g" * 64,
    )
    result = dedupe_training_records([p1, p2])
    assert not any(p.duplicate_training_record for p in result)


def test_dedupe_preserves_pair_count():
    pairs = [
        make_pair(preference_id="pref_p001_c001__p001_c002"),
        make_pair(preference_id="pref_p001_c003__p001_c004"),
        make_pair(preference_id="pref_p001_c005__p001_c006"),
    ]
    result = dedupe_training_records(pairs)
    assert len(result) == len(pairs)
    assert {p.preference_id for p in result} == {p.preference_id for p in pairs}
