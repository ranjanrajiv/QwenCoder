"""Pairwise candidate comparison (spec 07 sections 33-38, 66-68, 74).

``compare()`` never invents a preference (spec section 35): a tie stays a tie, and either
side being indeterminate makes the whole comparison indeterminate, regardless of the
other side's score. Neutral terminology throughout — ``A_BETTER``/``B_BETTER``/``TIE``/
``INDETERMINATE``, never ``chosen``/``rejected`` (spec section 67).
"""

from __future__ import annotations

from itertools import combinations

from .models import CandidateAssessment, ComparisonResult

COMPARATOR_VERSION = "v1"

# Two assessments' scores are floats derived from integer test counts (spec section 15);
# this tolerance treats floating-point noise as equality without conflating genuinely
# different pass rates. It plays the same role as EvaluationResult's own 1e-9 pass_rate
# check.
_SCORE_EPSILON = 1e-9


class CandidateComparator:
    """Compares candidates within one problem. Never compares across problems (spec
    section 8) — :meth:`compare` raises if the two assessments disagree on ``problem_id``.
    """

    def compare(
        self, ranking_run_id: str, a: CandidateAssessment, b: CandidateAssessment
    ) -> ComparisonResult:
        if a.problem_id != b.problem_id:
            raise ValueError(
                f"cannot compare candidates from different problems: "
                f"{a.problem_id!r} vs {b.problem_id!r}"
            )

        score_margin = abs(a.score - b.score)

        if a.correctness == "indeterminate" or b.correctness == "indeterminate":
            relation = "INDETERMINATE"
        elif abs(a.score - b.score) <= _SCORE_EPSILON:
            # Spec section 33: both fully correct, or any identical pass rate, is a tie.
            relation = "TIE"
        elif a.score > b.score:
            relation = "A_BETTER"
        else:
            relation = "B_BETTER"

        return ComparisonResult(
            ranking_run_id=ranking_run_id,
            problem_id=a.problem_id,
            candidate_a=a.candidate_id,
            candidate_b=b.candidate_id,
            relation=relation,
            score_a=a.score,
            score_b=b.score,
            score_margin=score_margin,
            correctness_a=a.correctness,
            correctness_b=b.correctness,
            comparison_eligible=relation in ("A_BETTER", "B_BETTER"),
        )

    def build_matrix(
        self, ranking_run_id: str, assessments: list[CandidateAssessment]
    ) -> list[ComparisonResult]:
        """Every ``N(N-1)/2`` comparison among ``assessments`` (spec section 58), which
        must all belong to the same problem. Order is deterministic: assessments are
        compared in ``candidate_id`` order, never a set/dict iteration order.
        """
        ordered = sorted(assessments, key=lambda a: a.candidate_id)
        return [
            self.compare(ranking_run_id, a, b) for a, b in combinations(ordered, 2)
        ]


__all__ = ["COMPARATOR_VERSION", "CandidateComparator"]
