"""Correctness classification (spec 07 sections 9-14).

Converts Stage 6's objective test evidence into one of three classifications. The
precedence order below is what makes spec section 13 true: a candidate-caused timeout
reaches the final row and is ``incorrect``, never ``indeterminate`` — only a genuine
absence of usable test evidence (no job attempted, an infrastructure failure, or a
problem with zero declared tests) is ``indeterminate``.
"""

from __future__ import annotations

from ..evaluation.models import EvaluationResult

CLASSIFIER_VERSION = "v1"


class CorrectnessClassifier:
    """Classifies one candidate's :class:`~python_dpo.evaluation.models.EvaluationResult`
    (or its absence) as ``correct``, ``incorrect``, or ``indeterminate``.
    """

    def classify(self, result: EvaluationResult | None) -> tuple[str, str | None]:
        """Return ``(correctness, indeterminate_reason)``. ``indeterminate_reason`` is
        ``None`` unless ``correctness == "indeterminate"``.
        """
        if result is None:
            # Spec section 70: a candidate with no evaluation result at all is
            # indeterminate, never assumed bad. The caller supplies the specific reason
            # (typically an EvaluationFailure.error_type) via classify_missing.
            return "indeterminate", "missing_evaluation_result"

        # Spec sections 12, 14: Docker never having run the candidate says nothing about
        # the candidate itself.
        if result.infrastructure_error or result.status == "infrastructure_error":
            return "indeterminate", "infrastructure_error"

        # Spec sections 10, 12, 72: a problem with zero declared tests offers no evidence
        # either way.
        if result.tests_total == 0:
            return "indeterminate", "no_tests_executed"

        # Spec section 10: correct requires every test to have passed, with no failures,
        # errors, skips, or timeout.
        if (
            result.tests_passed == result.tests_total
            and result.tests_failed == 0
            and result.tests_error == 0
            and result.tests_skipped == 0
            and not result.timeout
        ):
            return "correct", None

        # Spec sections 11, 13: everything else — a failing assertion, a candidate
        # exception (tests_error > 0), a skipped test, or a candidate-caused timeout — is
        # incorrect. The candidate executed sufficiently to establish that it did not
        # fully solve the problem.
        return "incorrect", None

    def classify_missing(self, error_type: str) -> tuple[str, str]:
        """Classify a candidate that has an :class:`~python_dpo.evaluation.models.
        EvaluationFailure` instead of a result — no job was ever attempted (spec section
        70), so the failure's own ``error_type`` is the most specific reason available.
        """
        return "indeterminate", error_type


__all__ = ["CLASSIFIER_VERSION", "CorrectnessClassifier"]
