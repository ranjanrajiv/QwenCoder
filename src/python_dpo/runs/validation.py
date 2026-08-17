"""Run integrity validation (spec 04 sections 22, 38, 51).

Reads a run directory's raw JSONL lines directly rather than going through
:class:`~python_dpo.candidates.repository.CandidateRepository`, so a single ``validate``
call can collect *every* problem in the run instead of raising on the first one.

Hash correctness (spec section 16-18) is enforced by :class:`Candidate` construction
itself — ``Candidate.__post_init__`` recomputes and compares every hash, so a record that
fails to build with a "does not match" message *is* the hash-mismatch check; there is no
separate pass needed for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..atomic_io import JsonlError, iter_jsonl, read_json
from ..candidates.models import Candidate, CandidateError, GenerationFailure, build_candidate_id
from ..candidates.repository import (
    CANDIDATES_FILENAME,
    FAILURES_FILENAME,
    PROMPTS_DIRNAME,
    PROMPTS_FILENAME,
)
from .models import MANIFEST_VERSION, RunError, RunManifest, RunStatistics
from .repository import MANIFEST_FILENAME, STATISTICS_FILENAME


@dataclass(frozen=True)
class RunValidationIssue:
    check: str
    message: str


@dataclass(frozen=True)
class RunValidationReport:
    run_id: str
    issues: tuple[RunValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


def validate_run(run_dir: Path, known_problem_ids: set[str] | None = None) -> RunValidationReport:
    """Validate one run directory. ``known_problem_ids`` is the current problem dataset's
    id set; pass ``None`` to skip that check (spec 04 section 38's "referenced problem IDs
    exist" needs a dataset to check against).
    """
    run_dir = Path(run_dir)
    run_id = run_dir.name
    issues: list[RunValidationIssue] = []

    manifest = _check_manifest(run_dir, issues)

    candidates_path = run_dir / CANDIDATES_FILENAME
    failures_path = run_dir / FAILURES_FILENAME
    prompts_path = run_dir / PROMPTS_DIRNAME / PROMPTS_FILENAME

    candidates = _load_candidates(candidates_path, run_id, known_problem_ids, issues)
    _check_duplicate_links(candidates, issues)
    failures = _load_failures(failures_path, run_id, issues)
    _check_prompts(prompts_path, candidates, issues)

    if manifest is not None:
        _check_statistics(run_dir, manifest, candidates, failures, issues)
        _check_completeness(manifest, candidates, failures, issues)

    return RunValidationReport(run_id=run_id, issues=tuple(issues))


def _check_manifest(run_dir: Path, issues: list[RunValidationIssue]) -> RunManifest | None:
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        issues.append(RunValidationIssue("manifest", f"{manifest_path} does not exist"))
        return None
    try:
        manifest = RunManifest.from_dict(read_json(manifest_path))
    except (JsonlError, RunError) as exc:
        issues.append(RunValidationIssue("manifest", f"manifest.json: {exc}"))
        return None
    if manifest.manifest_version != MANIFEST_VERSION:
        issues.append(
            RunValidationIssue(
                "manifest",
                f"manifest.json: unknown manifest_version {manifest.manifest_version!r}",
            )
        )
    return manifest


def _load_candidates(
    path: Path,
    run_id: str,
    known_problem_ids: set[str] | None,
    issues: list[RunValidationIssue],
) -> list[Candidate]:
    try:
        raw_records = list(iter_jsonl(path))
    except JsonlError as exc:
        issues.append(RunValidationIssue("jsonl", str(exc)))
        return []

    candidates: list[Candidate] = []
    seen_ids: set[str] = set()
    for number, raw in raw_records:
        try:
            candidate = Candidate.from_dict(raw)
        except CandidateError as exc:
            label = raw.get("candidate_id", "?") if isinstance(raw, dict) else "?"
            issues.append(RunValidationIssue("schema", f"candidate {label} ({path}:{number}): {exc}"))
            continue
        candidates.append(candidate)

        if candidate.candidate_id in seen_ids:
            issues.append(
                RunValidationIssue(
                    "duplicate_id", f"candidate {candidate.candidate_id}: duplicate candidate_id in this run"
                )
            )
        seen_ids.add(candidate.candidate_id)

        expected_id = build_candidate_id(candidate.problem_id, candidate.generation_index)
        if candidate.candidate_id != expected_id:
            issues.append(
                RunValidationIssue(
                    "id_shape",
                    f"candidate {candidate.candidate_id}: expected id {expected_id!r} from "
                    "problem_id and generation_index",
                )
            )

        if candidate.run_id != run_id:
            issues.append(
                RunValidationIssue(
                    "run_id",
                    f"candidate {candidate.candidate_id}: run_id {candidate.run_id!r} does not "
                    f"match directory {run_id!r}",
                )
            )

        if known_problem_ids is not None and candidate.problem_id not in known_problem_ids:
            issues.append(
                RunValidationIssue(
                    "problem_id",
                    f"candidate {candidate.candidate_id}: unknown problem_id {candidate.problem_id!r}",
                )
            )

    return candidates


def _check_duplicate_links(candidates: list[Candidate], issues: list[RunValidationIssue]) -> None:
    by_id = {c.candidate_id: c for c in candidates}
    for candidate in candidates:
        if candidate.duplicate_of is None:
            continue
        target = by_id.get(candidate.duplicate_of)
        if target is None:
            issues.append(
                RunValidationIssue(
                    "duplicate_of",
                    f"candidate {candidate.candidate_id}: duplicate_of {candidate.duplicate_of!r} "
                    "does not exist in this run",
                )
            )
        elif (
            candidate.code_sha256 is not None
            and target.code_sha256 is not None
            and candidate.code_sha256 != target.code_sha256
        ):
            issues.append(
                RunValidationIssue(
                    "duplicate_of",
                    f"candidate {candidate.candidate_id}: duplicate_of {candidate.duplicate_of!r} "
                    "has different code",
                )
            )


def _load_failures(
    path: Path, run_id: str, issues: list[RunValidationIssue]
) -> list[GenerationFailure]:
    try:
        raw_records = list(iter_jsonl(path))
    except JsonlError as exc:
        issues.append(RunValidationIssue("jsonl", str(exc)))
        return []

    failures: list[GenerationFailure] = []
    for number, raw in raw_records:
        try:
            failure = GenerationFailure.from_dict(raw)
        except CandidateError as exc:
            issues.append(RunValidationIssue("schema", f"{path}:{number}: {exc}"))
            continue
        failures.append(failure)
        if failure.run_id != run_id:
            issues.append(
                RunValidationIssue(
                    "run_id",
                    f"failure {failure.problem_id}#{failure.generation_index}: run_id "
                    f"{failure.run_id!r} does not match directory {run_id!r}",
                )
            )
    return failures


def _check_prompts(
    path: Path, candidates: list[Candidate], issues: list[RunValidationIssue]
) -> None:
    try:
        prompt_hashes = {raw.get("prompt_sha256") for _, raw in iter_jsonl(path)}
    except JsonlError as exc:
        issues.append(RunValidationIssue("jsonl", str(exc)))
        return

    for candidate in candidates:
        if candidate.prompt_sha256 is not None and candidate.prompt_sha256 not in prompt_hashes:
            issues.append(
                RunValidationIssue(
                    "prompt_missing",
                    f"candidate {candidate.candidate_id}: prompt_sha256 not found in prompts.jsonl",
                )
            )


def _check_statistics(
    run_dir: Path,
    manifest: RunManifest,
    candidates: list[Candidate],
    failures: list[GenerationFailure],
    issues: list[RunValidationIssue],
) -> None:
    stats_path = run_dir / STATISTICS_FILENAME
    if not stats_path.is_file():
        issues.append(RunValidationIssue("statistics", f"{stats_path} does not exist"))
        return
    try:
        on_disk = RunStatistics.from_dict(read_json(stats_path))
    except (JsonlError, RunError) as exc:
        issues.append(RunValidationIssue("statistics", f"statistics.json: {exc}"))
        return

    recomputed = RunStatistics.from_records(
        manifest, candidates, failures, computed_at=on_disk.computed_at
    )
    if recomputed != on_disk:
        issues.append(
            RunValidationIssue(
                "statistics",
                "statistics.json does not match a fresh recomputation from "
                "candidates.jsonl/failures.jsonl",
            )
        )


def _check_completeness(
    manifest: RunManifest,
    candidates: list[Candidate],
    failures: list[GenerationFailure],
    issues: list[RunValidationIssue],
) -> None:
    if manifest.status != "completed":
        return
    attempted = {(c.problem_id, c.generation_index) for c in candidates}
    attempted |= {(f.problem_id, f.generation_index) for f in failures}
    for problem_id in manifest.requested_problem_ids:
        for index in range(1, manifest.requested_candidates_per_problem + 1):
            if (problem_id, index) not in attempted:
                issues.append(
                    RunValidationIssue(
                        "incomplete",
                        f"run is marked completed but {problem_id} candidate {index} has "
                        "neither a candidate nor a failure record",
                    )
                )


def format_run_report(report: RunValidationReport) -> str:
    if report.valid:
        return "Run validation passed.\n"
    lines = ["Run validation failed:"]
    lines.extend(f"  {issue.message}" for issue in report.issues)
    return "\n".join(lines) + "\n"


__all__ = ["RunValidationIssue", "RunValidationReport", "format_run_report", "validate_run"]
