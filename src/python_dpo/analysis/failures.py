"""Test-level failure frequencies (spec 11 sections 45-49).

Where the outcome analysis asks *"which problems changed?"*, this asks *"which individual
assertions fail, and for which variant?"* -- the difference between knowing a problem is
hard and knowing which edge case the model keeps missing.

Three flags come out of it:

* **hard test** (section 47) -- both variants fail it at least ``hard_test_failure_rate``
  of the time. That is a property of the problem, not of either model, and it is the
  signal that a problem may be miscalibrated rather than the model deficient.
* **DPO-specific** (section 48) -- DPO fails it materially more than base. A regression
  candidate.
* **base-specific** (section 49) -- base fails it materially more than DPO. Evidence that
  training helped something specific, even when aggregate pass@k did not move.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from .models import TestFailureStat

_FAILED_STATUSES = ("failed", "error")


def build_test_failure_stats(
    test_results: dict[str, Sequence[Any]],
    *,
    hard_test_failure_rate: float = 0.5,
    variant_specific_delta: float = 0.2,
) -> list[TestFailureStat]:
    """Section 46's frequencies plus the three flags of sections 47-49."""
    failures: dict[str, Counter[tuple[str, str]]] = {}
    runs: dict[str, Counter[tuple[str, str]]] = {}

    for variant, rows in test_results.items():
        failures[variant] = Counter()
        runs[variant] = Counter()
        for row in rows:
            key = (row.problem_id, row.test_case_id)
            runs[variant][key] += 1
            if row.status in _FAILED_STATUSES:
                failures[variant][key] += 1

    keys = sorted(set().union(*(set(c) for c in runs.values())) if runs else set())

    stats: list[TestFailureStat] = []
    for problem_id, test_case_id in keys:
        key = (problem_id, test_case_id)
        base_fail = failures.get("base", Counter()).get(key, 0)
        base_runs = runs.get("base", Counter()).get(key, 0)
        dpo_fail = failures.get("dpo", Counter()).get(key, 0)
        dpo_runs = runs.get("dpo", Counter()).get(key, 0)

        base_rate = base_fail / base_runs if base_runs else 0.0
        dpo_rate = dpo_fail / dpo_runs if dpo_runs else 0.0

        stats.append(
            TestFailureStat(
                problem_id=problem_id,
                test_case_id=test_case_id,
                base_failures=base_fail,
                base_runs=base_runs,
                dpo_failures=dpo_fail,
                dpo_runs=dpo_runs,
                hard_test=(
                    base_rate >= hard_test_failure_rate and dpo_rate >= hard_test_failure_rate
                ),
                dpo_specific=(dpo_rate - base_rate) >= variant_specific_delta,
                base_specific=(base_rate - dpo_rate) >= variant_specific_delta,
            )
        )
    return stats


def interesting(stats: Sequence[TestFailureStat]) -> list[TestFailureStat]:
    """Only the rows a reader would act on -- any flag set, or any failure at all."""
    return [
        s for s in stats
        if s.hard_test or s.dpo_specific or s.base_specific or s.base_failures or s.dpo_failures
    ]


__all__ = ["build_test_failure_stats", "interesting"]
