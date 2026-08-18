"""Problem-level and aggregate base-vs-DPO comparison (spec sections 54-58, 109, 121-123, 127).

``compare`` is the single place that turns two variants' :class:`~python_dpo.model_evaluation.models.EvaluationRecord`
lists into a win/tie/loss verdict. Everything downstream -- the report, the paired
bootstrap -- is built from its output, so it is where spec section 122's paired-subset
rule and section 121's mismatch-is-an-error rule both live.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import IncompleteBenchmarkError
from .models import EvaluationRecord


def _valid(records: Sequence[EvaluationRecord]) -> list[EvaluationRecord]:
    """Drop infrastructure failures (spec section 120): they say nothing about the
    candidate, so they must never count as either a pass or a fail."""
    return [r for r in records if not r.is_infrastructure_error]


def _mean_pass_rate(records: Sequence[EvaluationRecord]) -> float:
    if not records:
        return 0.0
    return sum(r.pass_rate for r in records) / len(records)


@dataclass(frozen=True)
class ProblemComparison:
    """Base vs DPO for one problem (spec sections 54, 57)."""

    problem_id: str
    base_n: int
    base_c: int
    dpo_n: int
    dpo_c: int
    base_pass: bool
    dpo_pass: bool
    improvement: int
    base_test_pass_rate: float
    dpo_test_pass_rate: float
    test_pass_delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "base_n": self.base_n,
            "base_c": self.base_c,
            "dpo_n": self.dpo_n,
            "dpo_c": self.dpo_c,
            "base_pass": self.base_pass,
            "dpo_pass": self.dpo_pass,
            "improvement": self.improvement,
            "base_test_pass_rate": self.base_test_pass_rate,
            "dpo_test_pass_rate": self.dpo_test_pass_rate,
            "test_pass_delta": self.test_pass_delta,
        }


@dataclass(frozen=True)
class ComparisonResult:
    """The full spec section 123 completeness table plus win/tie/loss (spec sections 55, 56)."""

    benchmark_problems: int
    base_successfully_evaluated: int
    dpo_successfully_evaluated: int
    paired_problems: int
    per_problem: tuple[ProblemComparison, ...]
    dpo_wins: int
    ties: int
    dpo_losses: int
    dpo_win_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_problems": self.benchmark_problems,
            "base_successfully_evaluated": self.base_successfully_evaluated,
            "dpo_successfully_evaluated": self.dpo_successfully_evaluated,
            "paired_problems": self.paired_problems,
            "per_problem": [p.to_dict() for p in self.per_problem],
            "dpo_wins": self.dpo_wins,
            "ties": self.ties,
            "dpo_losses": self.dpo_losses,
            "dpo_win_rate": self.dpo_win_rate,
        }


def compare(
    problem_ids: Sequence[str],
    base_records: Mapping[str, Sequence[EvaluationRecord]],
    dpo_records: Mapping[str, Sequence[EvaluationRecord]],
    *,
    allow_incomplete: bool = False,
) -> ComparisonResult:
    """Compare base and DPO across ``problem_ids`` (spec sections 112, 113, 121, 122).

    ``base_records``/``dpo_records`` map ``problem_id -> its EvaluationRecords`` for that
    variant; a problem absent from a mapping (or mapped to an empty list) was not
    successfully evaluated for that variant.

    Raises :class:`IncompleteBenchmarkError` when the two variants do not cover the same
    problems, unless ``allow_incomplete`` is set -- silently narrowing to the intersection
    is exactly what spec section 121 forbids by default.
    """
    problem_id_set = set(problem_ids)
    base_evaluated = {pid for pid in problem_id_set if base_records.get(pid)}
    dpo_evaluated = {pid for pid in problem_id_set if dpo_records.get(pid)}
    paired = base_evaluated & dpo_evaluated

    missing_from_base = sorted(problem_id_set - base_evaluated)
    missing_from_dpo = sorted(problem_id_set - dpo_evaluated)
    if (missing_from_base or missing_from_dpo) and not allow_incomplete:
        parts = []
        if missing_from_base:
            parts.append(f"missing from base: {', '.join(missing_from_base)}")
        if missing_from_dpo:
            parts.append(f"missing from dpo: {', '.join(missing_from_dpo)}")
        raise IncompleteBenchmarkError(
            "base and DPO do not cover the same problems ("
            + "; ".join(parts)
            + "); either rerun the missing evaluations or pass allow_incomplete=True to "
            "explicitly compare only the paired subset"
        )

    per_problem: list[ProblemComparison] = []
    dpo_wins = ties = dpo_losses = 0
    for problem_id in sorted(paired):
        base_valid = _valid(base_records[problem_id])
        dpo_valid = _valid(dpo_records[problem_id])
        base_c = sum(1 for r in base_valid if r.correct)
        dpo_c = sum(1 for r in dpo_valid if r.correct)
        base_pass = base_c > 0
        dpo_pass = dpo_c > 0

        if dpo_pass and not base_pass:
            improvement = 1
            dpo_wins += 1
        elif base_pass and not dpo_pass:
            improvement = -1
            dpo_losses += 1
        else:
            improvement = 0
            ties += 1

        base_rate = _mean_pass_rate(base_valid)
        dpo_rate = _mean_pass_rate(dpo_valid)

        per_problem.append(
            ProblemComparison(
                problem_id=problem_id,
                base_n=len(base_valid),
                base_c=base_c,
                dpo_n=len(dpo_valid),
                dpo_c=dpo_c,
                base_pass=base_pass,
                dpo_pass=dpo_pass,
                improvement=improvement,
                base_test_pass_rate=base_rate,
                dpo_test_pass_rate=dpo_rate,
                test_pass_delta=dpo_rate - base_rate,
            )
        )

    decided = dpo_wins + dpo_losses
    win_rate = dpo_wins / decided if decided > 0 else 0.0

    return ComparisonResult(
        benchmark_problems=len(problem_id_set),
        base_successfully_evaluated=len(base_evaluated),
        dpo_successfully_evaluated=len(dpo_evaluated),
        paired_problems=len(paired),
        per_problem=tuple(per_problem),
        dpo_wins=dpo_wins,
        ties=ties,
        dpo_losses=dpo_losses,
        dpo_win_rate=win_rate,
    )


__all__ = ["ComparisonResult", "ProblemComparison", "compare"]
