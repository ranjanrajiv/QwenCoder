"""Candidate scoring (spec 07 sections 15-23).

Builds a :class:`~python_dpo.ranking.models.CandidateAssessment` from one candidate's
Stage 6 evidence. The score is deliberately simple (spec section 19): ``score ==
pass_rate``, full stop. Duration, code length, syntax validity, and generation strategy
are recorded as secondary metadata (spec sections 20-23, 57) purely for traceability —
this module never lets them influence ``score``.
"""

from __future__ import annotations

from ..candidates.models import Candidate
from ..evaluation.models import EvaluationResult
from .classifier import CorrectnessClassifier
from .models import CandidateAssessment, compute_pass_rate

SCORING_VERSION = "v1"


class CandidateScorer:
    """Turns one candidate's evidence (plus optional candidate metadata) into an
    assessment."""

    def __init__(self, classifier: CorrectnessClassifier | None = None) -> None:
        self._classifier = classifier or CorrectnessClassifier()

    def score(
        self,
        *,
        ranking_run_id: str,
        evaluation_run_id: str,
        candidate_run_id: str,
        candidate_id: str,
        problem_id: str,
        result: EvaluationResult | None,
        missing_error_type: str | None = None,
        candidate: Candidate | None = None,
    ) -> CandidateAssessment:
        """Build one assessment. Exactly one of ``result`` or ``missing_error_type`` must
        be meaningful: pass ``result=None`` and ``missing_error_type`` set when the
        candidate has no :class:`~python_dpo.evaluation.models.EvaluationResult` at all
        (spec section 70) — an :class:`~python_dpo.evaluation.models.EvaluationFailure`'s
        own ``error_type`` is the natural value to pass.

        ``candidate`` is optional (spec section 20's secondary metadata is never
        required for scoring); when absent, every secondary-metadata field is ``None``.
        """
        if result is not None:
            correctness, reason = self._classifier.classify(result)
        else:
            error_type = missing_error_type or "missing_evaluation_result"
            correctness, reason = self._classifier.classify_missing(error_type)

        tests_total = result.tests_total if result is not None else 0
        tests_passed = result.tests_passed if result is not None else 0
        tests_failed = result.tests_failed if result is not None else 0
        tests_error = result.tests_error if result is not None else 0
        tests_skipped = result.tests_skipped if result is not None else 0
        timeout = result.timeout if result is not None else False
        infrastructure_error = result.infrastructure_error if result is not None else False
        duration = result.duration_ms if result is not None else None

        pass_rate = compute_pass_rate(tests_passed, tests_total)
        all_tests_passed = tests_total > 0 and tests_passed == tests_total

        code_lines = code_chars = None
        code_sha256 = duplicate_of = strategy = None
        syntax_valid = None
        if candidate is not None:
            code_sha256 = candidate.code_sha256
            duplicate_of = candidate.duplicate_of
            strategy = candidate.strategy
            syntax_valid = candidate.syntax_valid
            if candidate.code:
                code_lines = len(candidate.code.splitlines())
                code_chars = len(candidate.code)

        return CandidateAssessment(
            ranking_run_id=ranking_run_id,
            evaluation_run_id=evaluation_run_id,
            candidate_run_id=candidate_run_id,
            candidate_id=candidate_id,
            problem_id=problem_id,
            correctness=correctness,
            all_tests_passed=all_tests_passed,
            pass_rate=pass_rate,
            score=pass_rate,  # spec section 18: score starts as exactly pass_rate
            tests_total=tests_total,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_error=tests_error,
            tests_skipped=tests_skipped,
            timeout=timeout,
            infrastructure_error=infrastructure_error,
            execution_duration_ms=duration,
            indeterminate_reason=reason,
            code_sha256=code_sha256,
            duplicate_of=duplicate_of,
            code_lines=code_lines,
            code_chars=code_chars,
            strategy=strategy,
            syntax_valid=syntax_valid,
        )


__all__ = ["SCORING_VERSION", "CandidateScorer"]
