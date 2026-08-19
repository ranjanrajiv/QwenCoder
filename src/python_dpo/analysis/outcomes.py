"""Per-problem improvement and regression classification (spec 11 sections 17-25, 50-52).

The outcome table, verbatim from sections 19-23:

| Base            | DPO                      | Outcome                |
|-----------------|--------------------------|------------------------|
| 0 tests passed  | all passed               | `complete_improvement` |
| all passed      | 0 passed                 | `complete_regression`  |
| lower rate      | higher, not all          | `partial_improvement`  |
| all passed      | above 0, fewer           | `partial_regression`   |
| equal           | equal                    | `unchanged`            |

Two definitional choices are load-bearing and asserted in tests. Section 25 defines a
problem's score as the **maximum** test-pass rate across samples, not the mean, and
``solved`` as *any* sample passing everything. Using the mean would let a model that solves
a problem once in ten attempts look worse than one that never solves it but fails
gracefully -- which is backwards for pass@k-style evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .models import ProblemOutcome


def _score(records: Sequence[Any]) -> tuple[float, bool]:
    """Section 25: ``(best test-pass rate, solved)`` over one problem's samples."""
    valid = [r for r in records if r.error_type != "infrastructure_error"]
    if not valid:
        return 0.0, False
    rates = [(r.tests_passed / r.tests_total if r.tests_total else 0.0) for r in valid]
    solved = any(r.tests_total > 0 and r.tests_passed == r.tests_total for r in valid)
    return max(rates), solved


def _classify_outcome(base_score: float, dpo_score: float) -> str:
    if base_score == dpo_score:
        return "unchanged"
    if dpo_score > base_score:
        return "complete_improvement" if base_score == 0.0 and dpo_score == 1.0 else "partial_improvement"
    return "complete_regression" if base_score == 1.0 and dpo_score == 0.0 else "partial_regression"


def _severity(outcome: str, delta: float, regression_threshold: float) -> str:
    """Sections 50, 51: `high` is reserved for the complete cases; everything else is
    graded off the magnitude of the change against the configured threshold."""
    if outcome in ("complete_improvement", "complete_regression"):
        return "high"
    if outcome == "unchanged":
        return "none"
    return "medium" if abs(delta) >= regression_threshold else "low"


def build_problem_outcomes(
    evaluations: dict[str, Sequence[Any]],
    problems: dict[str, Any],
    *,
    regression_threshold: float = 0.2,
) -> list[ProblemOutcome]:
    """One :class:`ProblemOutcome` per problem evaluated for *both* variants.

    Problems evaluated for only one variant are skipped: an outcome is a comparison, and
    inventing a zero for the missing side would manufacture an improvement or a regression
    that was never measured.
    """
    base_records = evaluations.get("base", [])
    dpo_records = evaluations.get("dpo", [])

    base_by_problem: dict[str, list[Any]] = {}
    dpo_by_problem: dict[str, list[Any]] = {}
    for record in base_records:
        base_by_problem.setdefault(record.problem_id, []).append(record)
    for record in dpo_records:
        dpo_by_problem.setdefault(record.problem_id, []).append(record)

    outcomes: list[ProblemOutcome] = []
    for problem_id in sorted(set(base_by_problem) & set(dpo_by_problem)):
        base_score, base_solved = _score(base_by_problem[problem_id])
        dpo_score, dpo_solved = _score(dpo_by_problem[problem_id])
        outcome = _classify_outcome(base_score, dpo_score)
        problem = problems.get(problem_id)
        outcomes.append(
            ProblemOutcome(
                problem_id=problem_id,
                outcome=outcome,
                base_best_score=base_score,
                dpo_best_score=dpo_score,
                base_solved=base_solved,
                dpo_solved=dpo_solved,
                severity=_severity(outcome, dpo_score - base_score, regression_threshold),
                category=getattr(problem, "category", None),
                difficulty=getattr(problem, "difficulty", None),
            )
        )
    return outcomes


def partition(outcomes: Sequence[ProblemOutcome]) -> dict[str, list[ProblemOutcome]]:
    """Split into the three buckets sections 17/18 report separately."""
    return {
        "improvements": [o for o in outcomes if o.outcome.endswith("improvement")],
        "regressions": [o for o in outcomes if o.outcome.endswith("regression")],
        "unchanged": [o for o in outcomes if o.outcome == "unchanged"],
    }


__all__ = ["build_problem_outcomes", "partition"]
