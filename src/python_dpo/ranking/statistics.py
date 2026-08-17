"""Text rendering for ranking statistics and per-problem rank tables (spec sections 40,
41, 49, 79).

:class:`~python_dpo.ranking.models.RankingStatistics` itself (including
``from_records``) lives in ``models.py``, alongside every other schema type; this module
holds only the formatters, extracted here (a mild departure from how Stage 4/6 format
statistics inline in the CLI) because ``rankings list`` and ``rankings show`` both render
the same table shape.
"""

from __future__ import annotations

from .models import CandidateAssessment, RankingResult, RankingStatistics


def format_ranking_statistics(stats: RankingStatistics) -> str:
    """The spec section 40 counters, plus the section 41 per-problem distribution."""
    lines = [
        f"Problems: {stats.problems}",
        f"Candidates: {stats.candidates}",
        f"Correct: {stats.correct}",
        f"Incorrect: {stats.incorrect}",
        f"Indeterminate: {stats.indeterminate}",
        f"Fully correct: {stats.fully_correct}",
        f"Partially correct: {stats.partially_correct}",
        f"Zero test pass: {stats.zero_test_pass}",
        f"Tied candidates: {stats.tied_candidates}",
        f"Preference eligible candidates: {stats.preference_eligible_candidates}",
    ]
    if stats.per_problem:
        lines += ["", "Per problem:"]
        header = f"  {'PROBLEM':<10}{'TOTAL':<8}{'CORRECT':<10}{'PARTIAL':<10}{'ZERO':<8}{'INDET.'}"
        lines.append(header)
        for problem_id in sorted(stats.per_problem):
            bucket = stats.per_problem[problem_id]
            lines.append(
                f"  {problem_id:<10}{bucket['total']:<8}{bucket['fully_correct']:<10}"
                f"{bucket['partially_correct']:<10}{bucket['zero_test_pass']:<8}"
                f"{bucket['indeterminate']}"
            )
    return "\n".join(lines) + "\n"


def format_ranking_table(
    rows: list[tuple[RankingResult, CandidateAssessment]],
) -> str:
    """The spec sections 49/79 rank table for one problem.

    ``rows`` pairs each :class:`~python_dpo.ranking.models.RankingResult` with its
    :class:`~python_dpo.ranking.models.CandidateAssessment` (joined by the caller, since
    the two artifacts live in separate files) so the ``TESTS`` column can be rendered.
    Ranked candidates are listed first in rank order; indeterminate candidates (no rank)
    are listed last, ordered by candidate id.
    """
    ordered = sorted(
        rows,
        key=lambda pair: (
            pair[0].rank is None,
            pair[0].rank if pair[0].rank is not None else 0,
            pair[0].candidate_id,
        ),
    )
    header = f"{'RANK':<6}{'CANDIDATE':<14}{'TESTS':<10}{'SCORE':<8}{'STATUS'}"
    lines = [header]
    for ranking, assessment in ordered:
        rank_text = str(ranking.rank) if ranking.rank is not None else "-"
        tests_text = f"{assessment.tests_passed}/{assessment.tests_total}"
        lines.append(
            f"{rank_text:<6}{ranking.candidate_id:<14}{tests_text:<10}"
            f"{ranking.score:<8.2f}{ranking.correctness}"
        )
    return "\n".join(lines) + "\n"


__all__ = ["format_ranking_statistics", "format_ranking_table"]
