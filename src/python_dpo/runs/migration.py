"""Migrate the Stage 3 flat ``candidates.jsonl`` into per-run directories.

Explicit and non-destructive on the source (spec 04 section 46): the legacy file is only
ever read. Records are grouped by their existing ``run_id``, upgraded to schema 2.0 with
hashes back-filled, and written into a proper run directory via :class:`RunRepository` —
the same code path a real ``generate`` run uses, so a migrated run validates exactly like
any other.

Lives in ``runs/`` rather than ``candidates/``: it depends on both packages, and
``candidates`` must never import from ``runs`` (that would be circular — ``runs`` already
depends on ``candidates``, one direction only).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..atomic_io import iter_jsonl
from ..candidates.hashing import sha256_text
from ..candidates.models import Candidate, CandidateError, GenerationFailure
from ..candidates.repository import LEGACY_FAILURES_FILENAME, PromptRecord
from .models import RunManifest, RunStatistics
from .repository import RunRepository

_UNKNOWN_ENVIRONMENT = {
    "python_version": None,
    "platform": None,
    "transformers_version": None,
    "torch_version": None,
    "cuda_version": None,
}


class MigrationError(Exception):
    """Raised when the legacy file cannot be migrated as a self-consistent set of runs."""


def migrate_flat_file(
    source_path: Path,
    run_repo: RunRepository,
    *,
    force: bool = False,
) -> list[RunManifest]:
    """Migrate every run found in ``source_path`` (and its sibling legacy failures file,
    if any) into ``run_repo``. Returns the created manifests, one per run id found.
    """
    source_path = Path(source_path)

    legacy_candidates: dict[str, list[Candidate]] = {}
    for number, raw in iter_jsonl(source_path):
        try:
            candidate = Candidate.from_dict(raw)
        except CandidateError as exc:
            raise MigrationError(f"{source_path}:{number}: {exc}") from exc
        legacy_candidates.setdefault(candidate.run_id, []).append(candidate)

    legacy_failures: dict[str, list[GenerationFailure]] = {}
    failures_path = source_path.parent / LEGACY_FAILURES_FILENAME
    for number, raw in iter_jsonl(failures_path):
        try:
            failure = GenerationFailure.from_dict(raw)
        except CandidateError as exc:
            raise MigrationError(f"{failures_path}:{number}: {exc}") from exc
        legacy_failures.setdefault(failure.run_id, []).append(failure)

    manifests: list[RunManifest] = []
    for run_id in sorted(set(legacy_candidates) | set(legacy_failures)):
        candidates = legacy_candidates.get(run_id, [])
        failures = legacy_failures.get(run_id, [])
        if not candidates:
            raise MigrationError(
                f"run {run_id!r} has failure records but no candidates; cannot infer a "
                "manifest from failures alone"
            )

        run_dir = run_repo.run_dir(run_id)
        if run_dir.exists():
            if not force:
                raise MigrationError(f"{run_dir} already exists; pass --force to overwrite")
            shutil.rmtree(run_dir)

        manifests.append(_migrate_one_run(run_repo, run_id, candidates, failures))

    return manifests


def _migrate_one_run(
    run_repo: RunRepository,
    run_id: str,
    candidates: list[Candidate],
    failures: list[GenerationFailure],
) -> RunManifest:
    first = candidates[0]
    for candidate in candidates[1:]:
        signature = (candidate.model, candidate.provider, candidate.prompt_version)
        if signature != (first.model, first.provider, first.prompt_version):
            raise MigrationError(
                f"run {run_id!r}: candidates disagree on model/provider/prompt_version; "
                "cannot infer a single manifest"
            )

    candidates_per_problem = max(c.generation_index for c in candidates)
    strategy_by_index: dict[int, str] = {}
    for candidate in candidates:
        seen = strategy_by_index.setdefault(candidate.generation_index, candidate.strategy)
        if seen != candidate.strategy:
            raise MigrationError(
                f"run {run_id!r}: generation_index {candidate.generation_index} has "
                f"inconsistent strategies ({seen!r} vs {candidate.strategy!r}) across problems"
            )
    strategies = [strategy_by_index[index] for index in range(1, candidates_per_problem + 1)]
    requested_problem_ids = sorted({c.problem_id for c in candidates})

    manifest = run_repo.create_run(
        run_id=run_id,
        requested_problem_ids=requested_problem_ids,
        requested_candidates_per_problem=candidates_per_problem,
        strategies=strategies,
        model_config={
            "provider": first.provider,
            "name": first.model,
            "revision": first.model_revision,
        },
        generation_config=first.generation_config,
        prompt_version=first.prompt_version,
        retry={"max_attempts": 1},
        source="migrated",
        environment=dict(_UNKNOWN_ENVIRONMENT),
        candidate_schema_version="2.0",
    )
    run_repo.update_status(run_id, "running", started_at=manifest.created_at)

    repository = run_repo.candidates(run_id)
    for candidate in sorted(candidates, key=lambda c: (c.problem_id, c.generation_index)):
        upgraded = Candidate.create(
            candidate_id=candidate.candidate_id,
            problem_id=candidate.problem_id,
            run_id=candidate.run_id,
            generation_index=candidate.generation_index,
            strategy=candidate.strategy,
            model=candidate.model,
            model_revision=candidate.model_revision,
            provider=candidate.provider,
            prompt_version=candidate.prompt_version,
            prompt=candidate.prompt,
            raw_output=candidate.raw_output,
            code=candidate.code,
            extraction_format=candidate.extraction_format,
            syntax_valid=candidate.syntax_valid,
            syntax_error=candidate.syntax_error,
            function_name_valid=candidate.function_name_valid,
            duplicate_of=candidate.duplicate_of,
            generation_config=candidate.generation_config,
            created_at=candidate.created_at,
            attempt=1,
        )
        repository.save(upgraded)
        repository.append_prompt(
            PromptRecord(
                run_id=run_id,
                problem_id=candidate.problem_id,
                generation_index=candidate.generation_index,
                strategy=candidate.strategy,
                attempt=1,
                prompt=candidate.prompt,
                prompt_sha256=sha256_text(candidate.prompt),
            )
        )

    for failure in failures:
        # Legacy failures carry no prompt text, so their prompt hash cannot be
        # back-filled; they are written through unchanged (schema_version 1.0).
        repository.save_failure(failure)

    stats = RunStatistics.from_records(manifest, repository.load_all(), repository.load_failures())
    run_repo.write_statistics(stats)
    return run_repo.complete_run(run_id)


__all__ = ["MigrationError", "migrate_flat_file"]
