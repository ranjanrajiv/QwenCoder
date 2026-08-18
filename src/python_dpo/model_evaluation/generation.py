"""Paired candidate generation for one model variant (spec sections 20-36).

:func:`compute_seed` is what makes the base-vs-DPO comparison **paired** rather than
merely simultaneous (spec sections 25, 26): both variants consume the identical
``(problem, sample_index) -> seed`` schedule, so any difference in outcome is attributable
to the model weights and not to divergent sampling.

The prompt is Stage 9's canonical, strategy-free prompt (spec section 29: no per-variant
prompt engineering), and code extraction reuses Stage 3's :func:`extract_code` unmodified
for both variants (spec sections 33, 34) -- a failed extraction becomes
``status="generation_error"`` with the raw response retained (spec section 35) and never
reaches a :class:`~python_dpo.candidates.models.Candidate`, since ``Candidate.create``
rejects empty code.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from ..candidates.hashing import sha256_text
from ..generation.code_extractor import extract_code
from ..generation.prompt_builder import build_canonical_prompt
from ..generation.validation import check_syntax
from .benchmark import Benchmark
from .config import GenerationSettings
from .models import GenerationRecord

logger = logging.getLogger("python_dpo.model_evaluation.generation")


def compute_seed(base_seed: int, problem_index: int, sample_index: int) -> int:
    """The deterministic per-(problem, sample) seed (spec sections 25, 26).

    Both model variants are driven through this same function with the same
    ``problem_index``/``sample_index`` schedule, so ``base.seed[i] == dpo.seed[i]`` for
    every corresponding sample (spec section 116).
    """
    return base_seed + problem_index * 1000 + sample_index


class GenerationDriver:
    """Generates every ``(problem, sample_index)`` combination for one model runner."""

    def __init__(self, evaluation_run_id: str) -> None:
        self.evaluation_run_id = evaluation_run_id

    def run(
        self,
        runner,
        benchmark: Benchmark,
        generation: GenerationSettings,
        *,
        existing: set[tuple[str, int]] | None = None,
        on_record: Callable[[GenerationRecord], None] | None = None,
    ) -> list[GenerationRecord]:
        """Generate for every problem in ``benchmark`` and every sample index.

        ``existing`` is the set of ``(problem_id, sample_index)`` pairs already generated
        by a prior, interrupted run of this same variant -- skipped so the run is
        resumable (CLAUDE.md Reproducibility) without re-spending GPU time.

        ``on_record``, when given, is called immediately after each record is produced --
        so a caller can persist it durably (CLAUDE.md Data Integrity: "written the moment
        it exists," matching every prior stage) before the next, possibly slow, generation
        starts.
        """
        existing = existing or set()
        runner.ensure_loaded()

        ordered_problems = sorted(benchmark.problems, key=lambda p: p.id)
        records: list[GenerationRecord] = []

        for problem_index, problem in enumerate(ordered_problems):
            prompt = build_canonical_prompt(problem)
            prompt_sha256 = sha256_text(prompt)

            for sample_index in range(generation.num_samples):
                if (problem.id, sample_index) in existing:
                    continue

                seed = compute_seed(generation.base_seed, problem_index, sample_index)
                raw = runner.generate(prompt, seed=seed)
                extraction = extract_code(raw.text)

                if extraction.extracted and extraction.code:
                    syntax_valid = check_syntax(extraction.code).valid
                    record = GenerationRecord(
                        evaluation_run_id=self.evaluation_run_id,
                        problem_id=problem.id,
                        model_variant=runner.variant,
                        sample_index=sample_index,
                        seed=seed,
                        prompt_sha256=prompt_sha256,
                        raw_response=raw.text,
                        extracted_code=extraction.code,
                        extraction_format=extraction.source_format,
                        syntax_valid=syntax_valid,
                        generation_time_ms=raw.generation_time_ms,
                        generated_tokens=raw.generated_tokens,
                        status="generated",
                    )
                else:
                    record = GenerationRecord(
                        evaluation_run_id=self.evaluation_run_id,
                        problem_id=problem.id,
                        model_variant=runner.variant,
                        sample_index=sample_index,
                        seed=seed,
                        prompt_sha256=prompt_sha256,
                        raw_response=raw.text,
                        extraction_format="unknown",
                        generation_time_ms=raw.generation_time_ms,
                        generated_tokens=raw.generated_tokens,
                        status="generation_error",
                        error=extraction.error or "no Python code detected",
                    )
                records.append(record)
                if on_record is not None:
                    on_record(record)
                logger.info(
                    "%s | %s sample %d | %s",
                    problem.id,
                    runner.variant,
                    sample_index,
                    record.status,
                )

        return records


__all__ = ["GenerationDriver", "compute_seed"]
