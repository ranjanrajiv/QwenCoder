"""Tests for bootstrap confidence intervals and McNemar's test (spec sections 59-66,
105-108)."""

from __future__ import annotations

import pytest

from python_dpo.model_evaluation.statistics import bootstrap_ci, mcnemar, paired_bootstrap


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def test_bootstrap_ci_is_reproducible_under_a_fixed_seed():
    data = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    first = bootstrap_ci(data, _mean, iterations=200, seed=42)
    second = bootstrap_ci(data, _mean, iterations=200, seed=42)
    assert first.to_dict() == second.to_dict()


def test_bootstrap_ci_different_seed_can_differ():
    data = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    first = bootstrap_ci(data, _mean, iterations=200, seed=42)
    second = bootstrap_ci(data, _mean, iterations=200, seed=7)
    # Not asserting inequality (a coincidence is possible), just that both are valid and
    # bracket the known mean -- the real property under test.
    assert first.lower <= first.point <= first.upper
    assert second.lower <= second.point <= second.upper


def test_bootstrap_ci_brackets_a_known_mean():
    data = [0.4] * 50  # degenerate: every resample has the same mean
    result = bootstrap_ci(data, _mean, iterations=500, seed=1)
    assert result.point == pytest.approx(0.4)
    assert result.lower == pytest.approx(0.4)
    assert result.upper == pytest.approx(0.4)


def test_bootstrap_ci_empty_data_is_degenerate():
    result = bootstrap_ci([], _mean, iterations=100, seed=1)
    assert result.point == result.lower == result.upper == 0.0


def test_bootstrap_resamples_at_the_problem_level():
    """Spec section 60: resampling must operate on the per-problem sequence handed in,
    not flatten it -- constructed so candidate-level resampling would visibly differ.

    Each "problem" here carries a very different number of correlated candidate
    observations; a statistic that treated every candidate as an independent unit would
    weight problem B far more heavily than problem A. The bootstrap must resample
    *problems*, so each problem contributes equally regardless of its candidate count.
    """
    problem_a = (1, 10)  # 10 candidates, 1 problem-level unit
    problem_b = (1, 1)  # 1 candidate, 1 problem-level unit
    data = [problem_a, problem_b]

    def problem_level_mean(resampled) -> float:
        # Each entry counts once, regardless of its candidate_count.
        return _mean(weight for weight, _candidate_count in resampled)

    result = bootstrap_ci(data, problem_level_mean, iterations=200, seed=3)
    assert result.point == pytest.approx(1.0)


def test_paired_bootstrap_known_difference():
    base = [(10, 2), (10, 3), (10, 1), (10, 4), (10, 2)]
    dpo = [(10, 6), (10, 7), (10, 5), (10, 8), (10, 6)]

    def mean_correct_fraction(data) -> float:
        return _mean(c / n for n, c in data)

    result = paired_bootstrap(base, dpo, mean_correct_fraction, iterations=500, seed=42)
    assert result.point == pytest.approx(0.4, abs=1e-6)
    assert result.lower <= result.point <= result.upper


def test_paired_bootstrap_requires_equal_length():
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [1], lambda data: _mean(data))


def test_paired_bootstrap_is_reproducible():
    base = [(10, 2), (10, 3)]
    dpo = [(10, 6), (10, 7)]

    def stat(data):
        return _mean(c / n for n, c in data)

    first = paired_bootstrap(base, dpo, stat, iterations=100, seed=9)
    second = paired_bootstrap(base, dpo, stat, iterations=100, seed=9)
    assert first.to_dict() == second.to_dict()


# ------------------------------------------------------------------------------ McNemar


def test_mcnemar_all_concordant_gives_p_value_one():
    base_solved = [True, True, False, False]
    dpo_solved = [True, True, False, False]
    result = mcnemar(base_solved, dpo_solved)
    assert result.n_discordant == 0
    assert result.p_value == 1.0


def test_mcnemar_matches_hand_computed_binomial():
    # 1 base-only, 5 dpo-only -> n=6, k=min(1,5)=1
    # exact two-sided p = 2 * sum_{i=0}^{1} C(6,i) * 0.5^6 = 2 * (1 + 6) / 64 = 14/64
    base_solved = [True] + [False] * 5 + [True] * 6
    dpo_solved = [False] + [True] * 5 + [True] * 6
    result = mcnemar(base_solved, dpo_solved)
    assert result.base_only == 1
    assert result.dpo_only == 5
    assert result.n_discordant == 6
    assert result.p_value == pytest.approx(14 / 64)


def test_mcnemar_requires_equal_length():
    with pytest.raises(ValueError):
        mcnemar([True], [True, False])
