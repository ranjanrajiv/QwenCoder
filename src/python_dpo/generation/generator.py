"""The candidate-generation pipeline.

    Problem -> PromptBuilder -> ModelClient -> RawGeneration -> CodeExtractor
            -> Candidate -> CandidateRepository -> candidates.jsonl

Depends on :class:`~python_dpo.models.base.ModelClient`, never on Qwen or Transformers,
so the same orchestration drives the real model and the mock.

Failure policy (spec 03 sections 19.1, 26, 26.1, 26.2):

======================  ==========================================================
Outcome                 Recorded as
======================  ==========================================================
empty response          GenerationFailure(``empty_output``), no candidate
no extractable code     GenerationFailure(``code_extraction``), no candidate
inference exception     GenerationFailure(``inference``), no candidate, run continues
code that won't parse   **Candidate** with ``syntax_valid=false``, no failure record
model won't load        GenerationFailure(``model_load``), then the run aborts
======================  ==========================================================

Nothing is ever silently dropped, and no fake candidate is invented for a failure.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from ..candidates.models import (
    Candidate,
    GenerationFailure,
    build_candidate_id,
    utc_now_iso,
)
from ..candidates.repository import CandidateRepository
from ..models.base import GenerationConfig, ModelClient, ModelLoadError
from ..problems.models import Problem
from .code_extractor import extract_code
from .prompt_builder import PROMPT_VERSION, build_prompt
from .validation import check_function_name, check_syntax

logger = logging.getLogger("python_dpo.generation")


@dataclass(frozen=True)
class GenerationSummary:
    """What a run did, for the CLI to report."""

    run_id: str
    generated: int = 0
    skipped: int = 0
    failed: int = 0
    duplicates: int = 0

    @property
    def attempted(self) -> int:
        return self.generated + self.failed


class CandidateGenerator:
    """Generates and persists candidates for a set of problems."""

    def __init__(
        self,
        *,
        client: ModelClient,
        repository: CandidateRepository,
        generation_config: GenerationConfig,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        self._client = client
        self._repository = repository
        self._generation_config = generation_config
        self._prompt_version = prompt_version

    def generate(
        self,
        problems: Sequence[Problem],
        *,
        count: int,
        strategies: Sequence[str],
        run_id: str,
        force: bool = False,
    ) -> GenerationSummary:
        """Generate ``count`` candidates for each problem.

        ``strategies`` must already be resolved to exactly ``count`` entries by
        :func:`~python_dpo.generation.strategies.resolve_strategies`.
        """
        if len(strategies) != count:
            raise ValueError(
                f"expected {count} resolved strategies, got {len(strategies)}"
            )

        # Loaded once per run: re-reading the file per candidate would be quadratic, and
        # both indexes are kept current in memory as records are appended.
        existing = self._repository.existing_keys()
        code_index = self._repository.code_index()

        generated = skipped = failed = duplicates = 0
        config_dict = self._generation_config.to_dict()

        for problem in problems:
            for index in range(1, count + 1):
                strategy = strategies[index - 1]
                candidate_id = build_candidate_id(problem.id, index)

                if not force and (problem.id, index) in existing:
                    logger.info("Skipping %s (already generated); use --force to redo", candidate_id)
                    skipped += 1
                    continue

                logger.info(
                    "Generating %s candidate %d/%d | strategy=%s",
                    problem.id,
                    index,
                    count,
                    strategy,
                )
                prompt = build_prompt(problem, strategy)
                logger.debug("Prompt for %s:\n%s", candidate_id, prompt)

                try:
                    raw = self._client.generate(prompt, self._generation_config)
                except ModelLoadError as exc:
                    # Run-level: no candidate in this run can succeed, and retrying per
                    # candidate would emit one identical failure per generation.
                    self._record_failure(
                        run_id, problem.id, index, strategy, "model_load", str(exc)
                    )
                    logger.error("Model loading failed: %s", exc)
                    raise
                except Exception as exc:
                    self._record_failure(
                        run_id,
                        problem.id,
                        index,
                        strategy,
                        "inference",
                        f"{type(exc).__name__}: {exc}",
                    )
                    logger.error("Generation failed for %s: %s", candidate_id, exc)
                    failed += 1
                    continue

                logger.info(
                    "Generated raw output for %s | characters=%d", candidate_id, len(raw.text)
                )

                if not raw.text.strip():
                    self._record_failure(
                        run_id,
                        problem.id,
                        index,
                        strategy,
                        "empty_output",
                        "Model returned an empty response",
                    )
                    logger.error("Empty model response for %s", candidate_id)
                    failed += 1
                    continue

                extraction = extract_code(raw.text)
                if not extraction.extracted or extraction.code is None:
                    self._record_failure(
                        run_id,
                        problem.id,
                        index,
                        strategy,
                        "code_extraction",
                        extraction.error or "No Python code detected",
                    )
                    logger.error("Code extraction failed for %s", candidate_id)
                    failed += 1
                    continue

                syntax = check_syntax(extraction.code)
                function_name_valid = check_function_name(extraction.code, problem.entry_point)
                duplicate_of = code_index.get(problem.id, {}).get(extraction.code)
                if duplicate_of == candidate_id:
                    # --force regenerated this same index and got identical code. That is
                    # the generation reproducing itself, not two candidates colliding, and
                    # a record pointing at its own id would say nothing.
                    duplicate_of = None

                candidate = Candidate(
                    candidate_id=candidate_id,
                    problem_id=problem.id,
                    run_id=run_id,
                    generation_index=index,
                    strategy=strategy,
                    model=self._client.name,
                    model_revision=self._client.revision,
                    provider=self._client.provider,
                    prompt_version=self._prompt_version,
                    prompt=prompt,
                    raw_output=raw.text,
                    code=extraction.code,
                    extraction_format=extraction.source_format,
                    syntax_valid=syntax.valid,
                    syntax_error=syntax.error_message,
                    function_name_valid=function_name_valid,
                    duplicate_of=duplicate_of,
                    generation_config=config_dict,
                    created_at=utc_now_iso(),
                )

                self._repository.append(candidate)
                generated += 1
                existing.add((problem.id, index))
                code_index.setdefault(problem.id, {}).setdefault(extraction.code, candidate_id)

                if duplicate_of is not None:
                    duplicates += 1
                    logger.info("%s duplicates %s (kept for analysis)", candidate_id, duplicate_of)
                if not syntax.valid:
                    logger.warning(
                        "%s has invalid syntax (%s); persisted as a candidate",
                        candidate_id,
                        syntax.error_message,
                    )
                if not function_name_valid:
                    logger.warning(
                        "%s does not define %s()", candidate_id, problem.entry_point
                    )

                logger.info(
                    "Persisted %s | format=%s syntax_valid=%s function_name_valid=%s",
                    candidate_id,
                    extraction.source_format,
                    syntax.valid,
                    function_name_valid,
                )

        return GenerationSummary(
            run_id=run_id,
            generated=generated,
            skipped=skipped,
            failed=failed,
            duplicates=duplicates,
        )

    def _record_failure(
        self,
        run_id: str,
        problem_id: str,
        index: int,
        strategy: str,
        error_type: str,
        error_message: str,
    ) -> None:
        self._repository.append_failure(
            GenerationFailure(
                run_id=run_id,
                problem_id=problem_id,
                generation_index=index,
                strategy=strategy,
                error_type=error_type,
                error_message=error_message,
                timestamp=utc_now_iso(),
            )
        )
