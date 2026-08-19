"""Per-variant error profiles and the rate comparison (spec 11 sections 14, 15, 16).

Both variants are profiled through the **same** code path, so ``base_error_profile.json``
and ``dpo_error_profile.json`` are structurally comparable by construction rather than by
convention -- the same reasoning that made Stage 10 build both runners from one
quantization config.

Infrastructure errors are excluded from correctness rates and reported separately, matching
Stage 10's section 120 treatment: a Docker fault says nothing about the model, and folding
it into an error rate would make the harness's own flakiness look like a model regression.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from .models import ErrorProfile, ErrorRateComparison
from .taxonomy import ERROR_CATEGORIES, NO_FAILURE, ErrorClassification, classify


def classify_variant(
    generations: Sequence[Any],
    evaluations: Sequence[Any],
    test_results: Sequence[Any],
) -> list[ErrorClassification]:
    """Classify every sample of one variant (sections 10-12)."""
    evaluations_by_key = {(e.problem_id, e.sample_index): e for e in evaluations}

    # Per-test rows carry `candidate_id`, which Stage 10 mints as "<problem>_c<NNN>" with
    # the candidate number derived from the sample index. Grouping by problem and matching
    # on the ordinal keeps this robust to the exact id format.
    results_by_problem: dict[str, list[Any]] = {}
    for row in test_results:
        results_by_problem.setdefault(row.problem_id, []).append(row)

    classifications: list[ErrorClassification] = []
    for generation in generations:
        key = (generation.problem_id, generation.sample_index)
        evaluation = evaluations_by_key.get(key)
        if evaluation is None and generation.extracted_code is not None:
            # An extracted candidate with no evaluation record is a Stage 10 data gap, not
            # something to classify silently -- skip it and let the counts disagree loudly.
            continue
        rows = [
            r
            for r in results_by_problem.get(generation.problem_id, [])
            if _sample_index_of(r.candidate_id) == generation.sample_index
        ]
        classifications.append(classify(generation, evaluation, rows))
    return classifications


def _sample_index_of(candidate_id: str) -> int | None:
    """Map ``<problem>_c<NNN>`` back to its zero-based ``sample_index``.

    The two identifiers use different bases: Stage 10 numbers samples from 0, while the
    candidate id it mints for the sandbox numbers them from 1 (``c001`` is sample 0).
    Joining them without this correction silently attributes every sample's failures to
    its neighbour and drops one sample per problem entirely.
    """
    _, separator, tail = candidate_id.rpartition("_c")
    if not separator:
        return None
    try:
        ordinal = int(tail)
    except ValueError:
        return None
    return ordinal - 1 if ordinal >= 1 else None


def build_error_profile(
    variant: str, classifications: Sequence[ErrorClassification]
) -> ErrorProfile:
    """Sections 14, 15: the seven counters plus the hierarchical breakdown."""
    categories: Counter[str] = Counter()
    subcategories: Counter[str] = Counter()
    passed = 0
    for item in classifications:
        if item.passed:
            passed += 1
            continue
        categories[item.category] += 1
        for name, count in item.subcategory_counts.items():
            subcategories[name] += count
        if not item.subcategory_counts and item.subcategory != NO_FAILURE:
            subcategories[item.subcategory] += 1

    return ErrorProfile(
        model_variant=variant,
        total_samples=len(classifications),
        passed=passed,
        counts_by_category={c: categories.get(c, 0) for c in ERROR_CATEGORIES},
        counts_by_subcategory=dict(sorted(subcategories.items())),
        infrastructure_errors=categories.get("infrastructure_error", 0),
    )


def compare_error_rates(
    base: ErrorProfile, dpo: ErrorProfile
) -> list[ErrorRateComparison]:
    """Section 16: per-category rates for both variants, in the taxonomy's own order."""
    return [
        ErrorRateComparison(
            category=category,
            base_rate=base.rate_for(category),
            dpo_rate=dpo.rate_for(category),
        )
        for category in ERROR_CATEGORIES
    ]


__all__ = [
    "build_error_profile",
    "classify_variant",
    "compare_error_rates",
]
