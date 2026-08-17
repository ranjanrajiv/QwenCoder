"""Durable, run-scoped persistence for evaluation results, test-level detail, and
failures.

Mirrors :class:`python_dpo.candidates.repository.CandidateRepository`: one instance owns
exactly one evaluation run directory, records are written the moment they exist via
fsynced :mod:`python_dpo.atomic_io` appends (spec 04 section 24's precedent, applied
here), and nothing is ever rewritten or batched — a killed run leaves a usable, resumable
file behind.
"""

from __future__ import annotations

from pathlib import Path

from ..atomic_io import JsonlError, append_jsonl, iter_jsonl
from .models import EvaluationFailure, EvaluationModelError, EvaluationResult, TestCaseResult

EVALUATIONS_FILENAME = "evaluations.jsonl"
TEST_RESULTS_FILENAME = "test_results.jsonl"
FAILURES_FILENAME = "failures.jsonl"


class EvaluationStoreError(Exception):
    """Raised when a persisted evaluation file is malformed."""


def _load(path: Path, build) -> list:
    """Parse a JSONL file, validating every record. Never silently skips a bad line
    (CLAUDE.md Data Integrity), matching ``candidates/repository.py``'s ``_load``.
    """
    try:
        raw_records = list(iter_jsonl(path))
    except JsonlError as exc:
        raise EvaluationStoreError(str(exc)) from exc

    records = []
    for number, raw in raw_records:
        try:
            records.append(build(raw))
        except EvaluationModelError as exc:
            raise EvaluationStoreError(f"{path}:{number}: {exc}") from exc
    return records


class EvaluationRepository:
    """Reads and appends the ``evaluations``, ``test_results``, and ``failures``
    artifacts of one evaluation run.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.evaluations_path = self.directory / EVALUATIONS_FILENAME
        self.test_results_path = self.directory / TEST_RESULTS_FILENAME
        self.failures_path = self.directory / FAILURES_FILENAME

    # --------------------------------------------------------------------------- write

    def save(self, result: EvaluationResult) -> None:
        append_jsonl(self.evaluations_path, result.to_dict())

    def save_failure(self, failure: EvaluationFailure) -> None:
        append_jsonl(self.failures_path, failure.to_dict())

    def append_test_results(self, results: list[TestCaseResult]) -> None:
        for result in results:
            append_jsonl(self.test_results_path, result.to_dict())

    # ---------------------------------------------------------------------------- read

    def load_all(self) -> list[EvaluationResult]:
        """Load every evaluation result in file order. ``[]`` when nothing exists yet."""
        return _load(self.evaluations_path, EvaluationResult.from_dict)

    def load_failures(self) -> list[EvaluationFailure]:
        return _load(self.failures_path, EvaluationFailure.from_dict)

    def load_test_results(self) -> list[TestCaseResult]:
        return _load(self.test_results_path, TestCaseResult.from_dict)

    # ------------------------------------------------------------------ spec section 52

    def get(self, candidate_id: str) -> EvaluationResult | None:
        for result in self.load_all():
            if result.candidate_id == candidate_id:
                return result
        return None

    def list(self) -> list[EvaluationResult]:
        return self.load_all()

    def count(self) -> int:
        return len(self.load_all())

    def find_by_candidate(self, candidate_id: str) -> EvaluationResult | None:
        """Same as :meth:`get` — kept as a distinct name because spec section 52 asks for
        both. The repository is run-scoped, so unlike the spec's literal
        ``find_by_candidate(candidate_run_id, candidate_id)`` signature, there is no
        separate generation-run id to pass: this evaluation run already covers exactly
        one generation run's candidates (spec 04's run-scoping deviation, applied here).
        """
        return self.get(candidate_id)

    def find_by_problem(self, problem_id: str) -> list[EvaluationResult]:
        return [result for result in self.load_all() if result.problem_id == problem_id]

    def test_results_for(self, candidate_id: str) -> list[TestCaseResult]:
        return [t for t in self.load_test_results() if t.candidate_id == candidate_id]

    # ------------------------------------------------------------------------- indexes

    def evaluated_keys(self) -> set[str]:
        """Candidate ids already settled by this evaluation run — the resume index (spec
        section 56).

        Covers **both** persisted results and failures. This is a deliberate departure
        from Stage 4's generation-failure semantics, where a failure is retried on resume
        because it may have been transient (a flaky model call). An
        :class:`~python_dpo.evaluation.models.EvaluationFailure` here is structural and
        deterministic given the same candidate and problem — the problem was missing, had
        no tests, or the generated job failed validation — so retrying it on resume would
        just fail identically every time.
        """
        keys = {result.candidate_id for result in self.load_all()}
        keys |= {failure.candidate_id for failure in self.load_failures()}
        return keys


__all__ = [
    "EVALUATIONS_FILENAME",
    "FAILURES_FILENAME",
    "TEST_RESULTS_FILENAME",
    "EvaluationRepository",
    "EvaluationStoreError",
]
