"""Orchestrates one candidate's evaluation and a whole run of them.

    Candidate + Problem -> TestGenerator -> PytestRunner -> PytestResultParser
                         -> classify -> EvaluationResult + TestCaseResult(s)
                         -> EvaluationRepository

The classification precedence (spec sections 23-29) never conflates candidate outcomes
with infrastructure failures: a Docker fault is ``infrastructure_error`` regardless of
what the candidate's code looked like, and a candidate exception during a test is a
``failed`` candidate with ``tests_error > 0``, never mistaken for infrastructure trouble.

Nothing here computes ``chosen``/``rejected``, a reward, or a ranking (spec section 89) —
only objective execution facts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..candidates.models import Candidate
from ..problems.models import Problem
from ..sandbox.result import ExecutionResult
from .errors import (
    CandidateNotFoundError,
    EvaluationError,
    InvalidProblemError,
    ProblemNotFoundError,
    TestGenerationError,
)
from .models import EvaluationFailure, EvaluationResult, TestCaseResult, utc_now_iso
from .pytest_runner import PytestRunner
from .repository import EvaluationRepository
from .result_parser import PytestResultParser, reconcile
from .test_generator import TestGenerator

logger = logging.getLogger("python_dpo.evaluation")

EVALUATOR_VERSION = "v1"

# Maps the machinery-level exceptions TestGenerator/lookup can raise onto the closed
# EvaluationFailure.error_type set (spec section 46-47, models.EVALUATION_FAILURE_TYPES).
_FAILURE_TYPE_FOR_EXCEPTION: dict[type[EvaluationError], str] = {
    ProblemNotFoundError: "problem_not_found",
    CandidateNotFoundError: "candidate_not_found",
    InvalidProblemError: "empty_test_suite",
    TestGenerationError: "test_generation_error",
}


@dataclass(frozen=True)
class EvaluationSummary:
    """What one ``evaluate_many`` call did, for the CLI to report."""

    evaluation_run_id: str
    evaluated: int = 0
    skipped: int = 0
    machinery_failed: int = 0


class CandidateEvaluator:
    """Evaluates candidates against their problems' declared tests, persisting as it goes.

    Records are written the moment they exist (spec 04 section 24's precedent, applied
    here): ``evaluate_many`` calls the repository after every candidate, not once at the
    end, so a killed run leaves a resumable, trustworthy partial result behind.
    """

    def __init__(
        self,
        *,
        runner: PytestRunner,
        repository: EvaluationRepository,
        test_generator: TestGenerator | None = None,
        result_parser: PytestResultParser | None = None,
    ) -> None:
        self._runner = runner
        self._repository = repository
        self._test_generator = test_generator or TestGenerator()
        self._result_parser = result_parser or PytestResultParser()

    def evaluate(
        self,
        candidate: Candidate,
        problem: Problem,
        *,
        evaluation_run_id: str,
        job_id: str | None = None,
    ) -> EvaluationResult:
        """Evaluate and persist one candidate.

        Raises :class:`TestGenerationError` / :class:`InvalidProblemError` only for
        machinery-level problems that prevented a job from being attempted at all —
        never for anything the candidate itself did. :meth:`evaluate_many` is the
        caller that turns those into a persisted :class:`EvaluationFailure`.
        """
        job = self._test_generator.build(problem, candidate)

        logger.info(
            "Evaluating %s against %s (%d test case(s))",
            candidate.candidate_id,
            problem.id,
            len(problem.tests),
        )

        exec_result = self._runner.run(job, job_id=job_id, run_id=evaluation_run_id)

        result, test_results = self._classify(
            exec_result=exec_result,
            job_nonce=job.nonce,
            expected_test_case_ids=job.expected_test_case_ids,
            candidate=candidate,
            problem=problem,
            evaluation_run_id=evaluation_run_id,
        )

        self._repository.save(result)
        if test_results:
            self._repository.append_test_results(test_results)

        logger.info(
            "%s | %s | %d/%d tests passed",
            candidate.candidate_id,
            result.status,
            result.tests_passed,
            result.tests_total,
        )
        if result.metadata_discrepancy:
            logger.warning(
                "%s: %s", candidate.candidate_id, result.discrepancy_reason
            )
        return result

    def evaluate_many(
        self,
        candidates: list[Candidate],
        problems_by_id: dict[str, Problem],
        *,
        evaluation_run_id: str,
    ) -> EvaluationSummary:
        """Evaluate every candidate not already covered by this evaluation run.

        Resume index: :meth:`EvaluationRepository.evaluated_keys` covers both persisted
        ``EvaluationResult``s and ``EvaluationFailure``s — unlike Stage 4's generation
        failures (which are transient and worth retrying), an evaluation failure here is
        structural and deterministic given the same candidate and problem, so retrying it
        achieves nothing (spec section 56).
        """
        existing = self._repository.evaluated_keys()
        evaluated = skipped = machinery_failed = 0

        for candidate in candidates:
            if candidate.candidate_id in existing:
                logger.info("Skipping %s (already evaluated)", candidate.candidate_id)
                skipped += 1
                continue

            problem = problems_by_id.get(candidate.problem_id)
            try:
                if problem is None:
                    raise ProblemNotFoundError(
                        f"no problem {candidate.problem_id!r} for candidate "
                        f"{candidate.candidate_id!r}"
                    )
                self.evaluate(candidate, problem, evaluation_run_id=evaluation_run_id)
                evaluated += 1
            except (ProblemNotFoundError, InvalidProblemError, TestGenerationError) as exc:
                self._record_failure(candidate, evaluation_run_id, exc)
                machinery_failed += 1

        return EvaluationSummary(
            evaluation_run_id=evaluation_run_id,
            evaluated=evaluated,
            skipped=skipped,
            machinery_failed=machinery_failed,
        )

    def _record_failure(
        self, candidate: Candidate, evaluation_run_id: str, exc: EvaluationError
    ) -> None:
        error_type = _FAILURE_TYPE_FOR_EXCEPTION.get(type(exc), "test_generation_error")
        logger.error("%s: %s: %s", candidate.candidate_id, error_type, exc)
        self._repository.save_failure(
            EvaluationFailure(
                evaluation_run_id=evaluation_run_id,
                candidate_run_id=candidate.run_id,
                candidate_id=candidate.candidate_id,
                problem_id=candidate.problem_id,
                error_type=error_type,
                error_message=str(exc),
                timestamp=utc_now_iso(),
            )
        )

    def _classify(
        self,
        *,
        exec_result: ExecutionResult,
        job_nonce: str,
        expected_test_case_ids: tuple[str, ...],
        candidate: Candidate,
        problem: Problem,
        evaluation_run_id: str,
    ) -> tuple[EvaluationResult, list[TestCaseResult]]:
        """The classification precedence (spec sections 23-29): infrastructure outranks
        everything (the candidate never really ran); a collection failure is the
        candidate's own doing and becomes ``syntax_error`` or ``failed``; a timeout is
        distinct from a plain failure; otherwise every test must have passed for
        ``passed`` — exit code 0 alone never produces it (spec section 24).
        """
        if exec_result.is_infrastructure_failure:
            # Spec sections 29, 81: the candidate is not implicated. No test ever ran, so
            # there is nothing to reconcile — tests_total stays 0.
            result = EvaluationResult.create(
                evaluation_run_id=evaluation_run_id,
                candidate_run_id=candidate.run_id,
                candidate_id=candidate.candidate_id,
                problem_id=problem.id,
                status="infrastructure_error",
                tests_passed=0,
                tests_failed=0,
                tests_error=0,
                tests_skipped=0,
                duration_ms=exec_result.duration_ms,
                stdout=exec_result.stdout,
                stderr=exec_result.stderr,
                exit_code=exec_result.exit_code,
                sandbox_container_id=exec_result.container_id,
            )
            return result, []

        parsed = self._result_parser.parse(exec_result.stdout, job_nonce)

        if parsed.collection_error is not None:
            # The candidate's own file could not even be collected — never a run-time
            # exception, never infrastructure (spec section 26).
            missing_type = "SyntaxError" if parsed.is_syntax_error else "CollectionError"
            test_results = reconcile(
                (),
                expected_test_case_ids,
                evaluation_run_id=evaluation_run_id,
                candidate_run_id=candidate.run_id,
                candidate_id=candidate.candidate_id,
                problem_id=problem.id,
                missing_error_type=missing_type,
                missing_error_message=parsed.collection_error,
            )
            status = "syntax_error" if parsed.is_syntax_error else "failed"
        else:
            if exec_result.timed_out:
                missing_type, missing_message = "Timeout", "sandbox timed out before this test ran"
            elif exec_result.status == "resource_exceeded":
                missing_type, missing_message = (
                    "ResourceExceeded",
                    "sandbox terminated the run for exceeding a resource limit",
                )
            else:
                missing_type, missing_message = "NotRun", "no result was reported for this test"

            test_results = reconcile(
                parsed.test_results,
                expected_test_case_ids,
                evaluation_run_id=evaluation_run_id,
                candidate_run_id=candidate.run_id,
                candidate_id=candidate.candidate_id,
                problem_id=problem.id,
                missing_error_type=missing_type,
                missing_error_message=missing_message,
            )

            if exec_result.timed_out:
                status = "timeout"
            else:
                all_passed = bool(test_results) and all(t.status == "passed" for t in test_results)
                status = "passed" if all_passed else "failed"

        tests_passed = sum(1 for t in test_results if t.status == "passed")
        tests_failed = sum(1 for t in test_results if t.status == "failed")
        tests_error = sum(1 for t in test_results if t.status == "error")
        tests_skipped = sum(1 for t in test_results if t.status == "skipped")

        # Spec sections 63, 64: compare against Stage 3's recorded syntax_valid without
        # ever overwriting it. A discrepancy is only meaningful in the syntax_error
        # direction — Stage 3's static check saying "invalid" while the sandbox actually
        # ran real tests is not a contradiction (the model may have produced different
        # code across separate persisted attempts is not the case here, but a stricter or
        # looser static check would be a generator issue, not evaluated here).
        discrepancy = candidate.syntax_valid and status == "syntax_error"
        discrepancy_reason = (
            "candidate metadata recorded syntax_valid=true (Stage 3's static ast.parse "
            "check), but sandbox execution reported a collection-time syntax error"
            if discrepancy
            else None
        )

        result = EvaluationResult.create(
            evaluation_run_id=evaluation_run_id,
            candidate_run_id=candidate.run_id,
            candidate_id=candidate.candidate_id,
            problem_id=problem.id,
            status=status,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_error=tests_error,
            tests_skipped=tests_skipped,
            duration_ms=exec_result.duration_ms,
            stdout=exec_result.stdout,
            stderr=exec_result.stderr,
            exit_code=exec_result.exit_code,
            sandbox_container_id=exec_result.container_id,
            metadata_discrepancy=discrepancy,
            discrepancy_reason=discrepancy_reason,
        )
        return result, test_results


__all__ = ["EVALUATOR_VERSION", "CandidateEvaluator", "EvaluationSummary"]
