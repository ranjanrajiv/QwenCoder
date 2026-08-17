"""The candidate-generation pipeline.

    Problem -> PromptBuilder -> ModelClient -> RawGeneration -> CodeExtractor
            -> Candidate -> CandidateRepository -> candidates.jsonl

Depends on :class:`~python_dpo.models.base.ModelClient`, never on Qwen or Transformers,
so the same orchestration drives the real model and the mock. Depends on
:class:`~python_dpo.runs.models.RunManifest` for everything about *this* run (which
problems, how many candidates, which strategies, the retry policy) rather than on loose
arguments, so the manifest stays the single source of truth (spec 04 section 34).

Failure policy (spec 03 sections 19.1, 26, 26.1, 26.2; spec 04 section 28):

======================  ==========================================================
Outcome                 Recorded as
======================  ==========================================================
empty response          GenerationFailure(``empty_output``), no candidate, not retried
no extractable code     GenerationFailure(``code_extraction``), no candidate, not retried
inference exception     GenerationFailure(``inference``) per attempt, retried up to
                         ``retry.max_attempts``, run continues
code that won't parse   **Candidate** with ``syntax_valid=false``, no failure record
model won't load        GenerationFailure(``model_load``), then the run aborts
======================  ==========================================================

Nothing is ever silently dropped, and no fake candidate is invented for a failure. Every
prompt is persisted to ``prompts/prompts.jsonl`` before the model is called, so a
generation that fails or is interrupted still has its exact prompt recoverable (spec 04
section 31).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from ..candidates.hashing import sha256_text
from ..candidates.models import Candidate, GenerationFailure, build_candidate_id, utc_now_iso
from ..candidates.repository import CandidateRepository, PromptRecord
from ..models.base import GenerationConfig, ModelClient, ModelLoadError
from ..problems.models import Problem
from ..runs.models import RunManifest
from .code_extractor import extract_code
from .prompt_builder import build_prompt
from .validation import check_function_name, check_syntax

logger = logging.getLogger("python_dpo.generation")


@dataclass(frozen=True)
class GenerationSummary:
    """What one ``generate()`` call did, for the CLI to report.

    This is an in-memory summary of the current invocation, not the authoritative record
    — the persisted ``statistics.json`` is always recomputed from
    ``candidates.jsonl``/``failures.jsonl`` on disk (spec 04 section 25), independent of
    these counters.
    """

    run_id: str
    generated: int = 0
    skipped: int = 0
    failed: int = 0
    duplicates: int = 0
    retries: int = 0

    @property
    def attempted(self) -> int:
        return self.generated + self.failed


class CandidateGenerator:
    """Generates and persists candidates for a set of problems, within one run."""

    def __init__(self, *, client: ModelClient, repository: CandidateRepository) -> None:
        self._client = client
        self._repository = repository

    def generate(
        self,
        problems: Sequence[Problem],
        manifest: RunManifest,
    ) -> GenerationSummary:
        """Generate ``manifest.requested_candidates_per_problem`` candidates for each
        problem, persisting into the repository's (run-scoped) directory.

        The decoding parameters come from ``manifest.generation_config``, never from a
        separately-passed :class:`GenerationConfig` — the manifest is the historical
        source of truth for a run (spec 04 section 34), so a resumed run keeps decoding
        exactly as it started even if ``config.yaml`` has since changed.

        There is no ``force`` flag here: each run is its own directory (spec 04 section
        6), so "regenerate instead of resuming" is a run-management decision — start a
        new, empty run directory (:meth:`RunRepository.create_run_from`) — rather than
        something this method needs to know about. Calling ``generate`` twice against the
        *same* run directory is exactly what resume is: already-persisted
        ``(problem_id, generation_index)`` pairs are skipped, unconditionally.
        """
        count = manifest.requested_candidates_per_problem
        strategies = manifest.strategies
        run_id = manifest.run_id
        max_attempts = manifest.retry["max_attempts"]
        generation_config = GenerationConfig.from_dict(manifest.generation_config)

        # Loaded once per call: re-reading the file per candidate would be quadratic, and
        # both indexes are kept current in memory as records are appended.
        existing, code_index, _ = self._repository.load_index()

        generated = skipped = failed = duplicates = retries = 0
        config_dict = manifest.generation_config

        for problem in problems:
            for index in range(1, count + 1):
                strategy = strategies[index - 1]
                candidate_id = build_candidate_id(problem.id, index)

                if (problem.id, index) in existing:
                    logger.info("Skipping %s (already generated in this run)", candidate_id)
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
                prompt_sha256 = sha256_text(prompt)
                logger.debug("Prompt for %s:\n%s", candidate_id, prompt)

                # Persisted before inference so a failed or interrupted generation still
                # has its exact prompt recoverable (spec 04 section 31).
                self._repository.append_prompt(
                    PromptRecord(
                        run_id=run_id,
                        problem_id=problem.id,
                        generation_index=index,
                        strategy=strategy,
                        attempt=1,
                        prompt=prompt,
                        prompt_sha256=prompt_sha256,
                    )
                )

                raw = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        raw = self._client.generate(prompt, generation_config)
                        break
                    except ModelLoadError as exc:
                        # Run-level: no candidate in this run can succeed, so a single
                        # failure is recorded and the run aborts rather than retrying.
                        self._record_failure(
                            run_id,
                            problem.id,
                            index,
                            strategy,
                            "model_load",
                            str(exc),
                            attempt=attempt,
                            prompt_sha256=prompt_sha256,
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
                            attempt=attempt,
                            prompt_sha256=prompt_sha256,
                        )
                        logger.error(
                            "Generation failed for %s (attempt %d/%d): %s",
                            candidate_id,
                            attempt,
                            max_attempts,
                            exc,
                        )
                        if attempt < max_attempts:
                            retries += 1

                if raw is None:
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
                        attempt=attempt,
                        prompt_sha256=prompt_sha256,
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
                        attempt=attempt,
                        prompt_sha256=prompt_sha256,
                    )
                    logger.error("Code extraction failed for %s", candidate_id)
                    failed += 1
                    continue

                syntax = check_syntax(extraction.code)
                function_name_valid = check_function_name(extraction.code, problem.entry_point)
                code_sha256 = sha256_text(extraction.code)
                duplicate_of = code_index.get(problem.id, {}).get(code_sha256)

                candidate = Candidate.create(
                    candidate_id=candidate_id,
                    problem_id=problem.id,
                    run_id=run_id,
                    generation_index=index,
                    strategy=strategy,
                    model=self._client.name,
                    model_revision=self._client.revision,
                    provider=self._client.provider,
                    prompt_version=manifest.prompt_version,
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
                    attempt=attempt,
                )

                self._repository.save(candidate)
                generated += 1
                existing.add((problem.id, index))
                code_index.setdefault(problem.id, {}).setdefault(code_sha256, candidate_id)

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
                    "Persisted %s | format=%s syntax_valid=%s function_name_valid=%s attempt=%d",
                    candidate_id,
                    extraction.source_format,
                    syntax.valid,
                    function_name_valid,
                    attempt,
                )

        return GenerationSummary(
            run_id=run_id,
            generated=generated,
            skipped=skipped,
            failed=failed,
            duplicates=duplicates,
            retries=retries,
        )

    def _record_failure(
        self,
        run_id: str,
        problem_id: str,
        index: int,
        strategy: str,
        error_type: str,
        error_message: str,
        *,
        attempt: int,
        prompt_sha256: str,
    ) -> None:
        self._repository.save_failure(
            GenerationFailure(
                run_id=run_id,
                problem_id=problem_id,
                generation_index=index,
                strategy=strategy,
                error_type=error_type,
                error_message=error_message,
                timestamp=utc_now_iso(),
                attempt=attempt,
                prompt_sha256=prompt_sha256,
            )
        )
