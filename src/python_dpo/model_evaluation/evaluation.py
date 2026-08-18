"""Wraps Stage 6 execution for Stage 10 candidates (spec sections 37-40, 117, 120).

Reuses ``python_dpo.evaluation.executor.CandidateEvaluator`` -- and, through it, the
private classification logic that decides ``passed``/``failed``/``timeout``/
``syntax_error``/``infrastructure_error`` -- wholesale rather than reimplementing
execution or classification (spec section 37): Stage 10 would otherwise risk disagreeing
with Stage 6 about what "passed" means. The in-memory :class:`~python_dpo.candidates.models.Candidate`
each generation needs is built here and handed straight to the evaluator; it is never
persisted through Stage 4's candidate repository (the plan's deliberate deviation --
``CandidateEvaluator.evaluate`` never reads one).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from ..candidates.hashing import sha256_text
from ..candidates.models import Candidate
from ..evaluation.errors import EvaluationError
from ..evaluation.executor import CandidateEvaluator
from ..evaluation.models import EvaluationResult, TestCaseResult
from ..evaluation.repository import EvaluationRepository
from ..generation.prompt_builder import CANONICAL_PROMPT_VERSION, build_canonical_prompt
from ..generation.validation import check_function_name
from ..problems.models import Problem
from .errors import ModelEvaluationError, PairingError
from .models import EvaluationRecord, GenerationRecord, utc_now_iso

logger = logging.getLogger("python_dpo.model_evaluation.evaluation")

_IMPORT_ERROR_TYPES = frozenset({"ImportError", "ModuleNotFoundError"})


def _classify_error_type(result: EvaluationResult, test_results: list[TestCaseResult]) -> str | None:
    """Deterministic failure classification (spec sections 124-126). No LLM anywhere."""
    if result.status == "passed":
        return None
    if result.status in ("infrastructure_error", "timeout", "syntax_error"):
        return result.status

    # status == "failed"
    if result.tests_error > 0:
        error_types = {t.error_type for t in test_results if t.status == "error" and t.error_type}
        if error_types & _IMPORT_ERROR_TYPES:
            return "import_error"
        return "runtime_error"
    return "assertion_failure"


class EvaluationDriver:
    """Turns one variant's :class:`~python_dpo.model_evaluation.models.GenerationRecord`
    list into :class:`~python_dpo.model_evaluation.models.EvaluationRecord`\\ s, through the
    Stage 5/6 sandbox unmodified.
    """

    def __init__(
        self,
        *,
        evaluator: CandidateEvaluator,
        repository: EvaluationRepository,
        evaluation_run_id: str,
        model_variant: str,
        model_name: str,
        model_revision: str | None,
        generation_config: dict,
    ) -> None:
        self._evaluator = evaluator
        self._repository = repository
        self.evaluation_run_id = evaluation_run_id
        self.model_variant = model_variant
        self.model_name = model_name
        self.model_revision = model_revision
        self._generation_config = generation_config

    def run(
        self,
        generation_records: list[GenerationRecord],
        problems_by_id: dict[str, Problem],
        *,
        on_record: Callable[[EvaluationRecord], None] | None = None,
    ) -> list[EvaluationRecord]:
        """Evaluate every generated (extracted) candidate. Generation failures (spec
        section 35) never reach here -- they have no code to run."""
        records: list[EvaluationRecord] = []

        for generation_record in generation_records:
            if generation_record.status != "generated":
                continue
            if generation_record.model_variant != self.model_variant:
                raise ModelEvaluationError(
                    f"generation record for {generation_record.problem_id!r} is variant "
                    f"{generation_record.model_variant!r}, but this driver evaluates "
                    f"{self.model_variant!r}"
                )

            problem = problems_by_id.get(generation_record.problem_id)
            if problem is None:
                raise ModelEvaluationError(
                    f"no problem {generation_record.problem_id!r} for generated candidate "
                    f"{generation_record.candidate_id!r}"
                )

            # Spec sections 27, 114: reconstructing the canonical prompt and checking its
            # hash against the one recorded at generation time makes prompt equality
            # verifiable, not merely assumed.
            prompt = build_canonical_prompt(problem)
            if sha256_text(prompt) != generation_record.prompt_sha256:
                raise PairingError(
                    f"prompt for {problem.id!r} does not match the hash recorded at "
                    "generation time; the problem dataset may have changed mid-run"
                )

            candidate = Candidate.create(
                candidate_id=generation_record.candidate_id,
                problem_id=problem.id,
                run_id=self.evaluation_run_id,
                generation_index=generation_record.sample_index + 1,
                strategy="canonical",
                model=self.model_name,
                provider="transformers",
                prompt_version=CANONICAL_PROMPT_VERSION,
                prompt=prompt,
                raw_output=generation_record.raw_response,
                code=generation_record.extracted_code or "",
                extraction_format=generation_record.extraction_format,
                syntax_valid=bool(generation_record.syntax_valid),
                function_name_valid=check_function_name(
                    generation_record.extracted_code or "", problem.entry_point
                ),
                generation_config={**self._generation_config, "seed": generation_record.seed},
                created_at=utc_now_iso(),
                model_revision=self.model_revision,
            )

            try:
                result = self._evaluator.evaluate(
                    candidate, problem, evaluation_run_id=self.evaluation_run_id
                )
            except EvaluationError as exc:
                raise ModelEvaluationError(
                    f"evaluation machinery failed for {candidate.candidate_id!r}: {exc}"
                ) from exc

            test_results = self._repository.test_results_for(candidate.candidate_id)
            error_type = _classify_error_type(result, test_results)

            evaluation_record = EvaluationRecord(
                evaluation_run_id=self.evaluation_run_id,
                problem_id=problem.id,
                model_variant=self.model_variant,
                sample_index=generation_record.sample_index,
                tests_total=result.tests_total,
                tests_passed=result.tests_passed,
                tests_failed=result.tests_failed,
                tests_error=result.tests_error,
                tests_skipped=result.tests_skipped,
                timeout=result.timeout,
                status=result.status,
                duration_ms=result.duration_ms,
                error_type=error_type,
            )
            records.append(evaluation_record)
            if on_record is not None:
                on_record(evaluation_record)

        return records


__all__ = ["EvaluationDriver"]
