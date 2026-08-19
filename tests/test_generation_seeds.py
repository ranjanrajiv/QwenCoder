"""Tests for the per-candidate sampling seed schedule (spec 03 sections 11, 30).

The bug these guard against: one configured ``generation.seed`` fed to every ``generate()``
call reseeds the sampler identically per candidate, so a problem's candidates come back
byte-identical and preference-pair construction has nothing to compare.
"""

from __future__ import annotations

from python_dpo.generation import compute_candidate_seed


def test_distinct_generation_indices_get_distinct_seeds():
    """The actual bug: candidate 1 and candidate 2 of the same problem must not sample
    from the same RNG state."""
    seeds = [compute_candidate_seed(42, "p001", i) for i in range(1, 6)]
    assert len(set(seeds)) == 5


def test_distinct_problems_get_distinct_seeds():
    seeds = [compute_candidate_seed(42, f"p{i:03d}", 1) for i in range(1, 11)]
    assert len(set(seeds)) == 10


def test_distinct_base_seeds_get_distinct_seeds():
    assert compute_candidate_seed(42, "p001", 1) != compute_candidate_seed(43, "p001", 1)


def test_seed_is_deterministic():
    """A run stays reproducible from config.yaml's base seed alone."""
    assert compute_candidate_seed(42, "p001", 3) == compute_candidate_seed(42, "p001", 3)


def test_seed_does_not_depend_on_which_other_problems_are_in_the_run():
    """Derived from problem_id, not ordinal position -- so resuming a run over a subset
    of problems (--problem-ids) does not silently reseed the candidates still to come."""
    full_run = {p: compute_candidate_seed(42, p, 1) for p in ("p001", "p002", "p003")}
    subset_run = {p: compute_candidate_seed(42, p, 1) for p in ("p003",)}
    assert subset_run["p003"] == full_run["p003"]


def test_seed_fits_in_the_non_negative_signed_32_bit_range():
    """`transformers.set_seed` feeds numpy, whose legacy seeding rejects wider values."""
    for problem in ("p001", "p002", "p010"):
        for index in range(1, 20):
            seed = compute_candidate_seed(2**31, problem, index)
            assert 0 <= seed < 2**31 - 1


def test_seeds_are_well_spread_across_a_realistic_run():
    """10 problems x 5 candidates must yield 50 distinct seeds -- no collisions at the
    scale this pipeline actually generates at."""
    seeds = {
        compute_candidate_seed(42, f"p{p:03d}", i)
        for p in range(1, 11)
        for i in range(1, 6)
    }
    assert len(seeds) == 50
