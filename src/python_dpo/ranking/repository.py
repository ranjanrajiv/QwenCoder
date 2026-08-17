"""Durable, run-scoped persistence for assessments, rankings, and comparisons.

Mirrors :class:`python_dpo.evaluation.repository.EvaluationRepository`: one instance owns
exactly one ranking run directory, records are written the moment they exist via fsynced
:mod:`python_dpo.atomic_io` appends, and nothing is ever rewritten (spec section 53) — a
killed run leaves a usable, resumable file behind.
"""

from __future__ import annotations

from pathlib import Path

from ..atomic_io import JsonlError, append_jsonl, iter_jsonl
from .models import CandidateAssessment, ComparisonResult, RankingModelError, RankingResult

ASSESSMENTS_FILENAME = "assessments.jsonl"
RANKINGS_FILENAME = "rankings.jsonl"
COMPARISONS_FILENAME = "comparisons.jsonl"


class RankingStoreError(Exception):
    """Raised when a persisted ranking file is malformed."""


def _load(path: Path, build) -> list:
    """Parse a JSONL file, validating every record. Never silently skips a bad line
    (CLAUDE.md Data Integrity), matching ``evaluation/repository.py``'s ``_load``.
    """
    try:
        raw_records = list(iter_jsonl(path))
    except JsonlError as exc:
        raise RankingStoreError(str(exc)) from exc

    records = []
    for number, raw in raw_records:
        try:
            records.append(build(raw))
        except RankingModelError as exc:
            raise RankingStoreError(f"{path}:{number}: {exc}") from exc
    return records


class RankingRepository:
    """Reads and appends the ``assessments``, ``rankings``, and ``comparisons`` artifacts
    of one ranking run.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.assessments_path = self.directory / ASSESSMENTS_FILENAME
        self.rankings_path = self.directory / RANKINGS_FILENAME
        self.comparisons_path = self.directory / COMPARISONS_FILENAME

    # --------------------------------------------------------------------------- write

    def save_assessment(self, assessment: CandidateAssessment) -> None:
        append_jsonl(self.assessments_path, assessment.to_dict())

    def save_ranking(self, ranking: RankingResult) -> None:
        append_jsonl(self.rankings_path, ranking.to_dict())

    def save_comparison(self, comparison: ComparisonResult) -> None:
        append_jsonl(self.comparisons_path, comparison.to_dict())

    def save_assessments(self, assessments: list[CandidateAssessment]) -> None:
        for assessment in assessments:
            self.save_assessment(assessment)

    def save_rankings(self, rankings: list[RankingResult]) -> None:
        for ranking in rankings:
            self.save_ranking(ranking)

    def save_comparisons(self, comparisons: list[ComparisonResult]) -> None:
        for comparison in comparisons:
            self.save_comparison(comparison)

    # ---------------------------------------------------------------------------- read

    def load_assessments(self) -> list[CandidateAssessment]:
        return _load(self.assessments_path, CandidateAssessment.from_dict)

    def load_rankings(self) -> list[RankingResult]:
        return _load(self.rankings_path, RankingResult.from_dict)

    def load_comparisons(self) -> list[ComparisonResult]:
        return _load(self.comparisons_path, ComparisonResult.from_dict)

    # ------------------------------------------------------------------ spec section 53

    def get_assessment(self, candidate_id: str) -> CandidateAssessment | None:
        for assessment in self.load_assessments():
            if assessment.candidate_id == candidate_id:
                return assessment
        return None

    def get_ranking(self, candidate_id: str) -> RankingResult | None:
        for ranking in self.load_rankings():
            if ranking.candidate_id == candidate_id:
                return ranking
        return None

    def list_problem_rankings(self, problem_id: str) -> list[RankingResult]:
        return [r for r in self.load_rankings() if r.problem_id == problem_id]

    def list_all_rankings(self) -> list[RankingResult]:
        return self.load_rankings()

    def count(self) -> int:
        return len(self.load_assessments())

    # ------------------------------------------------------------------------- indexes

    def ranked_problem_ids(self) -> set[str]:
        """Problem ids already settled by this ranking run — the resume index (spec
        section 54). A problem is settled once any of its candidates has a persisted
        ranking; ``rank run --resume`` skips these and only (re)ranks the rest.
        """
        return {r.problem_id for r in self.load_rankings()}


__all__ = [
    "ASSESSMENTS_FILENAME",
    "COMPARISONS_FILENAME",
    "RANKINGS_FILENAME",
    "RankingRepository",
    "RankingStoreError",
]
