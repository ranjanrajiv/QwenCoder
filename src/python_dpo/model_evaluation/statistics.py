"""Bootstrap confidence intervals and McNemar's test (spec sections 59-66, 105-108).

Pure stdlib (``random.Random``, ``math.comb``) -- no numpy or scipy, matching
``python_dpo.training.lengths``'s precedent and keeping PyYAML the project's only
runtime dependency.

Every resampling step operates at the **problem level** (spec section 60), never the
candidate level: each problem contributes ``k`` correlated samples, so resampling
candidates directly would treat correlated observations as independent and understate
variance.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")

DEFAULT_ITERATIONS = 1000
DEFAULT_SEED = 42
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate plus its bootstrap confidence interval (spec sections 59, 108)."""

    point: float
    lower: float
    upper: float
    iterations: int
    seed: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "iterations": self.iterations,
            "seed": self.seed,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class McNemarResult:
    """Exact McNemar's test over discordant paired outcomes (spec section 63)."""

    base_only: int
    """Problems the base model solved and DPO did not (a DPO loss)."""
    dpo_only: int
    """Problems DPO solved and the base model did not (a DPO win)."""
    n_discordant: int
    p_value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_only": self.base_only,
            "dpo_only": self.dpo_only,
            "n_discordant": self.n_discordant,
            "p_value": self.p_value,
        }


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile of an already-sorted sequence."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100) * (len(sorted_values) - 1)
    lower_idx = math.floor(rank)
    upper_idx = math.ceil(rank)
    if lower_idx == upper_idx:
        return sorted_values[int(rank)]
    fraction = rank - lower_idx
    return sorted_values[lower_idx] + (sorted_values[upper_idx] - sorted_values[lower_idx]) * fraction


def bootstrap_ci(
    data: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> BootstrapResult:
    """Bootstrap a confidence interval for ``statistic(data)`` (spec sections 59-61).

    ``data`` is one entry per **problem** (spec section 60); ``statistic`` reduces a
    (possibly resampled) sequence of problem-level entries to one number, e.g. a pass@k
    computed from a list of ``(n, c)`` pairs.
    """
    n = len(data)
    point = statistic(data)
    if n == 0:
        return BootstrapResult(point=point, lower=point, upper=point, iterations=iterations, seed=seed, confidence=confidence)

    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        resample = [data[rng.randrange(n)] for _ in range(n)]
        samples.append(statistic(resample))
    samples.sort()

    alpha = 1 - confidence
    lower = _percentile(samples, 100 * alpha / 2)
    upper = _percentile(samples, 100 * (1 - alpha / 2))
    return BootstrapResult(
        point=point, lower=lower, upper=upper, iterations=iterations, seed=seed, confidence=confidence
    )


def paired_bootstrap(
    base_data: Sequence[T],
    dpo_data: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> BootstrapResult:
    """Bootstrap a CI for ``statistic(dpo_data) - statistic(base_data)`` (spec sections
    62, 107).

    ``base_data[i]`` and ``dpo_data[i]`` must describe the **same problem** -- each
    bootstrap iteration draws one set of problem indices and applies it to both sequences,
    so the resample stays paired rather than treating base and DPO as independent samples.
    """
    if len(base_data) != len(dpo_data):
        raise ValueError("base_data and dpo_data must be the same length (a paired problem set)")
    n = len(base_data)
    point = statistic(dpo_data) - statistic(base_data)
    if n == 0:
        return BootstrapResult(point=point, lower=point, upper=point, iterations=iterations, seed=seed, confidence=confidence)

    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        base_resample = [base_data[i] for i in indices]
        dpo_resample = [dpo_data[i] for i in indices]
        diffs.append(statistic(dpo_resample) - statistic(base_resample))
    diffs.sort()

    alpha = 1 - confidence
    lower = _percentile(diffs, 100 * alpha / 2)
    upper = _percentile(diffs, 100 * (1 - alpha / 2))
    return BootstrapResult(
        point=point, lower=lower, upper=upper, iterations=iterations, seed=seed, confidence=confidence
    )


def mcnemar(base_solved: Sequence[bool], dpo_solved: Sequence[bool]) -> McNemarResult:
    """Exact two-sided McNemar's test over paired binary outcomes (spec section 63).

    Only discordant pairs matter: problems both models solved or both failed carry no
    information about which model is better. The exact binomial tail (via ``math.comb``)
    is used rather than the chi-square approximation, which is unreliable for the small
    discordant counts this project's benchmark sizes will produce.
    """
    if len(base_solved) != len(dpo_solved):
        raise ValueError("base_solved and dpo_solved must be the same length")

    base_only = sum(1 for b, d in zip(base_solved, dpo_solved) if b and not d)
    dpo_only = sum(1 for b, d in zip(base_solved, dpo_solved) if d and not b)
    n = base_only + dpo_only

    if n == 0:
        return McNemarResult(base_only=0, dpo_only=0, n_discordant=0, p_value=1.0)

    k = min(base_only, dpo_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    p_value = min(1.0, 2 * tail)
    return McNemarResult(base_only=base_only, dpo_only=dpo_only, n_discordant=n, p_value=p_value)


__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_ITERATIONS",
    "DEFAULT_SEED",
    "BootstrapResult",
    "McNemarResult",
    "bootstrap_ci",
    "mcnemar",
    "paired_bootstrap",
]
