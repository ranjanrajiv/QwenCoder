"""Tests for ProblemSplitter (spec 08 sections 62-68, 89)."""

from __future__ import annotations

import pytest

from python_dpo.preferences.splitter import (
    DEFAULT_SPLIT_RATIOS,
    ProblemSplitter,
    SplitConfigError,
    SplitManifest,
)

TEN_PROBLEMS = [f"p{i:03d}" for i in range(1, 11)]


def test_split_is_deterministic_for_a_fixed_seed():
    a = ProblemSplitter(seed=42).split(TEN_PROBLEMS)
    b = ProblemSplitter(seed=42).split(TEN_PROBLEMS)
    assert a == b


def test_split_is_independent_of_input_order():
    forward = ProblemSplitter(seed=42).split(TEN_PROBLEMS)
    reversed_input = ProblemSplitter(seed=42).split(list(reversed(TEN_PROBLEMS)))
    assert forward == reversed_input


def test_a_different_seed_gives_a_different_split():
    a = ProblemSplitter(seed=1).split(TEN_PROBLEMS)
    b = ProblemSplitter(seed=2).split(TEN_PROBLEMS)
    assert a.train_problem_ids != b.train_problem_ids


def test_no_problem_appears_in_more_than_one_split():
    manifest = ProblemSplitter(seed=42).split(TEN_PROBLEMS)
    all_ids = (
        manifest.train_problem_ids + manifest.validation_problem_ids + manifest.test_problem_ids
    )
    assert len(set(all_ids)) == len(all_ids) == len(TEN_PROBLEMS)


def test_ratios_are_approximately_respected_at_ten_problems():
    manifest = ProblemSplitter(seed=42).split(TEN_PROBLEMS)
    assert len(manifest.train_problem_ids) == 8
    assert len(manifest.validation_problem_ids) == 1
    assert len(manifest.test_problem_ids) == 1


def test_floor_rule_keeps_train_non_empty_at_small_pool_size():
    # 2 problems * 0.8 floors to 1 (already non-empty); the exact seed-42 shape
    # (train=[p008], validation=[], test=[p004]) is asserted directly, matching the
    # real strict-policy pool measured in the Stage 8 plan.
    manifest = ProblemSplitter(seed=42).split(["p004", "p008"])
    assert manifest.train_problem_ids == ("p008",)
    assert manifest.validation_problem_ids == ()
    assert manifest.test_problem_ids == ("p004",)


def test_floor_rule_fires_when_the_arithmetic_floor_would_be_zero():
    # A single-problem pool: floor(1 * 0.8) == 0, but train must never be empty when the
    # pool is non-empty.
    manifest = ProblemSplitter(seed=42).split(["p001"])
    assert manifest.train_problem_ids == ("p001",)
    assert manifest.validation_problem_ids == ()
    assert manifest.test_problem_ids == ()


def test_empty_pool_produces_an_empty_split():
    manifest = ProblemSplitter(seed=42).split([])
    assert manifest.train_problem_ids == ()
    assert manifest.validation_problem_ids == ()
    assert manifest.test_problem_ids == ()


def test_duplicate_input_ids_are_deduplicated_before_splitting():
    manifest = ProblemSplitter(seed=42).split(["p001", "p001", "p002"])
    all_ids = (
        manifest.train_problem_ids + manifest.validation_problem_ids + manifest.test_problem_ids
    )
    assert sorted(all_ids) == ["p001", "p002"]


def test_seed_and_ratios_are_persisted_in_the_manifest():
    manifest = ProblemSplitter(seed=7).split(TEN_PROBLEMS)
    assert manifest.seed == 7
    assert manifest.split_ratios == DEFAULT_SPLIT_RATIOS


def test_split_of_reports_membership():
    manifest = ProblemSplitter(seed=42).split(TEN_PROBLEMS)
    for problem_id in manifest.train_problem_ids:
        assert manifest.split_of(problem_id) == "train"
    for problem_id in manifest.validation_problem_ids:
        assert manifest.split_of(problem_id) == "validation"
    for problem_id in manifest.test_problem_ids:
        assert manifest.split_of(problem_id) == "test"
    assert manifest.split_of("unknown_problem") is None


# ------------------------------------------------------------------------- configuration


def test_ratios_must_sum_to_one():
    with pytest.raises(SplitConfigError):
        ProblemSplitter(ratios={"train": 0.5, "validation": 0.1, "test": 0.1})


def test_ratios_must_include_all_three_splits():
    with pytest.raises(SplitConfigError):
        ProblemSplitter(ratios={"train": 0.9, "validation": 0.1})


# --------------------------------------------------------------------------- SplitManifest


def test_split_manifest_rejects_a_problem_in_two_splits():
    with pytest.raises(Exception):
        SplitManifest(
            train_problem_ids=("p001",),
            validation_problem_ids=("p001",),
            test_problem_ids=(),
            seed=42,
            split_ratios=DEFAULT_SPLIT_RATIOS,
        )


def test_split_manifest_round_trip():
    manifest = ProblemSplitter(seed=42).split(TEN_PROBLEMS)
    restored = SplitManifest.from_dict(manifest.to_dict())
    assert restored == manifest
