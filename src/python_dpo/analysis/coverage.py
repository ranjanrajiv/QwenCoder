"""Training-vs-benchmark coverage gaps (spec 11 sections 32-42).

On this dataset this is the one module with a strong, unambiguous finding, and it is
structural rather than statistical: the categories DPO trained on and the categories the
benchmark measures do not overlap at all.

Two subtleties decide whether the numbers mean anything.

**The training population is the train split's pairs, not every pair in the run.** A pair
built on a problem that landed in the test or validation split was never trained on;
counting it would overstate coverage of exactly the categories the analysis is trying to
find holes in.

**``coverage_ratio`` is ``float | None``.** Training share over benchmark share is
undefined twice over in the real data -- a category present in training but absent from the
benchmark divides by zero, and one absent from both is 0/0. Neither ``Infinity`` nor ``NaN``
is representable in JSON, so the ratio is ``None`` and an explicit five-value verdict
carries the meaning instead.

Section 38's wording rule is enforced at the point of authorship: this module emits
``potential data gap`` and never a causal claim. Correlation between a weak category and a
failure rate is not evidence that the gap caused the failure, and on a benchmark with one
problem per category it could not be.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from .models import CategoryGap


def _verdict(
    training_share: float,
    benchmark_share: float,
    under: float,
    over: float,
) -> tuple[float | None, str]:
    """Section 37's ratio and verdict, with both degenerate cases handled explicitly."""
    if benchmark_share == 0.0 and training_share == 0.0:
        return None, "absent_from_both"
    if benchmark_share == 0.0:
        # Trained on, never measured. The ratio would be infinite; the verdict is the
        # informative part anyway.
        return None, "not_in_benchmark"
    ratio = training_share / benchmark_share
    if ratio < under:
        return ratio, "underrepresented"
    if ratio > over:
        return ratio, "overrepresented"
    return ratio, "balanced"


def _shares(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {name: count / total for name, count in counter.items()}


def training_problem_ids(split_manifest: Any | None, pairs: Sequence[Any]) -> set[str]:
    """The problems actually trained on.

    Prefers the split manifest's ``train`` list -- the authoritative record of what reached
    the trainer. Falls back to every pair-bearing problem only when no split was recorded,
    and that fallback is deliberately conservative about being an over-count.
    """
    if split_manifest is not None:
        train_ids = getattr(split_manifest, "train_problem_ids", None)
        if train_ids:
            return set(train_ids)
    return {p.problem_id for p in pairs}


def build_gaps(
    *,
    attribute: str,
    pairs: Sequence[Any],
    problems: dict[str, Any],
    benchmark_problem_ids: Sequence[str],
    trained_problem_ids: set[str],
    under: float = 0.5,
    over: float = 2.0,
) -> list[CategoryGap]:
    """Sections 35-37 for either ``category`` or ``difficulty``."""
    training_counter: Counter[str] = Counter()
    for pair in pairs:
        if pair.problem_id not in trained_problem_ids:
            continue
        problem = problems.get(pair.problem_id)
        value = getattr(problem, attribute, None)
        if value:
            training_counter[value] += 1

    benchmark_counter: Counter[str] = Counter()
    for problem_id in benchmark_problem_ids:
        problem = problems.get(problem_id)
        value = getattr(problem, attribute, None)
        if value:
            benchmark_counter[value] += 1

    training_shares = _shares(training_counter)
    benchmark_shares = _shares(benchmark_counter)

    # The universe is every value the *catalog* carries, not just those appearing in
    # training or the benchmark. A category the dataset defines but neither side uses is
    # exactly what `absent_from_both` reports, and omitting it would hide the fact that
    # the catalog has coverage nobody is exercising.
    universe = {
        value for problem in problems.values()
        if (value := getattr(problem, attribute, None))
    }

    gaps: list[CategoryGap] = []
    for name in sorted(universe | set(training_shares) | set(benchmark_shares)):
        training_share = training_shares.get(name, 0.0)
        benchmark_share = benchmark_shares.get(name, 0.0)
        ratio, verdict = _verdict(training_share, benchmark_share, under, over)
        gaps.append(
            CategoryGap(
                name=name,
                training_share=training_share,
                benchmark_share=benchmark_share,
                coverage_ratio=ratio,
                verdict=verdict,
            )
        )
    return gaps


def preference_coverage(
    pairs: Sequence[Any], problems: dict[str, Any], trained_problem_ids: set[str]
) -> dict[str, Any]:
    """Sections 32, 33, 39: how the preference dataset is distributed."""
    trained = [p for p in pairs if p.problem_id in trained_problem_ids]
    margins = [p.score_margin for p in trained if p.score_margin is not None]
    return {
        "total_pairs": len(pairs),
        "trained_pairs": len(trained),
        "problems_with_pairs": sorted({p.problem_id for p in pairs}),
        "trained_problem_ids": sorted(trained_problem_ids),
        "problems_without_pairs": sorted(set(problems) - {p.problem_id for p in pairs}),
        "mean_score_margin": (sum(margins) / len(margins)) if margins else None,
        "pairs_by_category": dict(
            sorted(
                Counter(
                    getattr(problems.get(p.problem_id), "category", "unknown") for p in trained
                ).items()
            )
        ),
    }


def strategy_gaps(pairs: Sequence[Any], trained_problem_ids: set[str]) -> dict[str, Any]:
    """Sections 41, 42: which generation strategies win and lose.

    The real strategy set is five values (``normal``, ``straightforward``,
    ``edge_case_focused``, ``alternative``, ``optimized``), not the spec's four-value
    shorthand -- counted from the data rather than a hard-coded list so it stays correct
    if the strategy set changes.
    """
    trained = [p for p in pairs if p.problem_id in trained_problem_ids]
    return {
        "chosen": dict(sorted(Counter(p.chosen_strategy for p in trained).items())),
        "rejected": dict(sorted(Counter(p.rejected_strategy for p in trained).items())),
    }


def correlate_errors_with_coverage(
    gaps: Sequence[CategoryGap], outcomes: Sequence[Any]
) -> list[dict[str, Any]]:
    """Section 38: pair each weak category with its observed failure rate.

    Emitted with the mandated *potential data gap* wording and no causal verb. On a
    benchmark carrying one problem per category this is arithmetic over cells of size one,
    and the report is required to say so next to the table.
    """
    failures_by_category: Counter[str] = Counter()
    totals_by_category: Counter[str] = Counter()
    for outcome in outcomes:
        if outcome.category is None:
            continue
        totals_by_category[outcome.category] += 1
        if not outcome.dpo_solved:
            failures_by_category[outcome.category] += 1

    rows: list[dict[str, Any]] = []
    for gap in gaps:
        if gap.verdict not in ("underrepresented", "absent_from_both"):
            continue
        total = totals_by_category.get(gap.name, 0)
        if total == 0:
            continue
        rows.append(
            {
                "category": gap.name,
                "coverage_verdict": gap.verdict,
                "coverage_ratio": gap.coverage_ratio,
                "benchmark_problems": total,
                "unsolved_by_dpo": failures_by_category.get(gap.name, 0),
                "observation": (
                    f"{gap.name}: potential data gap -- training share "
                    f"{gap.training_share:.0%} against benchmark share "
                    f"{gap.benchmark_share:.0%}, with "
                    f"{failures_by_category.get(gap.name, 0)}/{total} benchmark problem(s) "
                    "unsolved by the DPO model"
                ),
            }
        )
    return rows


__all__ = [
    "build_gaps",
    "correlate_errors_with_coverage",
    "preference_coverage",
    "strategy_gaps",
    "training_problem_ids",
]
