"""Deterministic per-problem ranking (spec 07 sections 7, 8, 28-31, 39, 46).

Candidates are grouped strictly by ``problem_id`` (spec section 8) — this module never
compares candidates from different problems. Within a problem, ranked candidates
(``correct``/``incorrect``) receive a competition rank (1, 1, 3, 4, 5 — spec section 79's
worked example); ``indeterminate`` candidates are recorded with ``rank = None`` and
excluded from the ordering (spec section 39) rather than dropped (spec section 71).

Tie detection compares the **integer** ``tests_passed``, never a float, because every
ranked candidate within one problem shares the same ``tests_total`` — asserted, not
assumed. This is what makes spec section 46's determinism structural: there is no float
tolerance, no rounding, and no ordering that could depend on IEEE representation.
``candidate_id`` breaks ties only for **presentation** (spec section 31); it is applied
after the tie group has already been formed, so it can never turn a tie into a decision.
"""

from __future__ import annotations

from collections import defaultdict

from .models import CandidateAssessment, RankingResult

RANKING_VERSION = "v1"


class CandidateRanker:
    """Ranks candidate assessments independently within each problem."""

    def rank(
        self, ranking_run_id: str, assessments: list[CandidateAssessment]
    ) -> list[RankingResult]:
        """Rank every assessment, grouped by ``problem_id``. Returns one
        :class:`~python_dpo.ranking.models.RankingResult` per assessment, problems in
        sorted order, so the overall output is fully deterministic.
        """
        by_problem: dict[str, list[CandidateAssessment]] = defaultdict(list)
        for assessment in assessments:
            by_problem[assessment.problem_id].append(assessment)

        results: list[RankingResult] = []
        for problem_id in sorted(by_problem):
            results.extend(
                self.rank_problem(ranking_run_id, problem_id, by_problem[problem_id])
            )
        return results

    def rank_problem(
        self,
        ranking_run_id: str,
        problem_id: str,
        assessments: list[CandidateAssessment],
    ) -> list[RankingResult]:
        """Rank the assessments belonging to exactly one problem."""
        mismatched = [a for a in assessments if a.problem_id != problem_id]
        if mismatched:
            raise ValueError(
                f"rank_problem({problem_id!r}) received assessment(s) for a different "
                f"problem: {sorted({a.problem_id for a in mismatched})}"
            )

        ranked = [a for a in assessments if a.correctness != "indeterminate"]
        indeterminate = [a for a in assessments if a.correctness == "indeterminate"]

        results: list[RankingResult] = []
        results.extend(self._rank_ranked(ranking_run_id, problem_id, ranked))
        results.extend(self._rank_indeterminate(ranking_run_id, problem_id, indeterminate))
        return results

    def _rank_ranked(
        self, ranking_run_id: str, problem_id: str, ranked: list[CandidateAssessment]
    ) -> list[RankingResult]:
        if not ranked:
            return []

        totals = {a.tests_total for a in ranked}
        if len(totals) > 1:
            raise ValueError(
                f"problem {problem_id!r}: ranked candidates disagree on tests_total "
                f"({sorted(totals)}); a problem's declared test suite must be constant"
            )

        # correct (0) sorts before incorrect (1); within each, higher tests_passed first
        # (spec section 28); candidate_id only breaks ties for presentation (section 31).
        order = {"correct": 0, "incorrect": 1}
        ordered = sorted(
            ranked,
            key=lambda a: (order[a.correctness], -a.tests_passed, a.candidate_id),
        )

        results: list[RankingResult] = []
        rank = 1
        index = 0
        group_number = 0
        while index < len(ordered):
            bucket_key = (ordered[index].correctness, ordered[index].tests_passed)
            bucket = [
                a
                for a in ordered[index:]
                if (a.correctness, a.tests_passed) == bucket_key
            ]
            group_number += 1
            tie_group = f"{problem_id}_tg{group_number:03d}"
            tie_group_size = len(bucket)
            tied = tie_group_size > 1

            for assessment in bucket:
                results.append(
                    RankingResult(
                        ranking_run_id=ranking_run_id,
                        evaluation_run_id=assessment.evaluation_run_id,
                        problem_id=problem_id,
                        candidate_id=assessment.candidate_id,
                        score=assessment.score,
                        correctness=assessment.correctness,
                        pass_rate=assessment.pass_rate,
                        all_tests_passed=assessment.all_tests_passed,
                        eligible_for_preference=True,
                        rank=rank,
                        tie_group=tie_group,
                        tie_group_size=tie_group_size,
                        tied=tied,
                    )
                )

            index += tie_group_size
            rank += tie_group_size  # spec section 30: the next rank skips by group size

        return results

    def _rank_indeterminate(
        self, ranking_run_id: str, problem_id: str, indeterminate: list[CandidateAssessment]
    ) -> list[RankingResult]:
        # Presentation-only ordering; indeterminate candidates never receive a rank.
        ordered = sorted(indeterminate, key=lambda a: a.candidate_id)
        return [
            RankingResult(
                ranking_run_id=ranking_run_id,
                evaluation_run_id=assessment.evaluation_run_id,
                problem_id=problem_id,
                candidate_id=assessment.candidate_id,
                score=assessment.score,
                correctness=assessment.correctness,
                pass_rate=assessment.pass_rate,
                all_tests_passed=assessment.all_tests_passed,
                eligible_for_preference=False,
                rank=None,
                tie_group=None,
                tie_group_size=0,
                tied=False,
            )
            for assessment in ordered
        ]


__all__ = ["RANKING_VERSION", "CandidateRanker"]
