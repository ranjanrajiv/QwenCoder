"""Preference dataset integrity validation (spec 08 sections 69-73).

Mirrors :mod:`python_dpo.ranking.validation`: reads a preference run's raw JSONL directly
rather than going through :class:`~python_dpo.preferences.repository.PreferenceRepository`,
so one ``validate`` call collects *every* issue instead of raising on the first bad record.
Every issue is fatal — there is no severity scale — and every check runs to completion.

Internal self-consistency (score/margin arithmetic, strength-vs-correctness agreement)
comes free from :class:`~python_dpo.preferences.models.PreferencePair`'s own
``__post_init__`` — a tampered record simply fails to construct. The checks that earn
their keep are the **cross-artifact** ones (spec section 69's "provenance", "objective
preference"): reloading the candidate run and the ranking run and re-deriving the claim
each pair makes, exactly as Stage 7's own validator re-derives from Stage 6's evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..atomic_io import JsonlError, iter_jsonl, read_json
from ..candidates.repository import CandidateRepository, CandidateStoreError
from ..ranking.comparator import CandidateComparator
from ..ranking.repository import RankingRepository, RankingStoreError
from .errors import PreferenceError
from .models import (
    PreferenceManifest,
    PreferenceModelError,
    PreferencePair,
    PreferenceRejection,
    PreferenceStatistics,
    derive_candidates_considered,
)
from .policies import make_policy
from .repository import METADATA_FILENAME, PREFERENCES_FILENAME, REJECTIONS_FILENAME
from .run_repository import MANIFEST_FILENAME, STATISTICS_FILENAME
from .splitter import SplitManifest


@dataclass(frozen=True)
class PreferenceValidationIssue:
    check: str
    message: str


@dataclass(frozen=True)
class PreferenceValidationReport:
    preference_run_id: str
    issues: tuple[PreferenceValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


def validate_preference_run(
    run_dir: Path,
    ranking_run_dir: Path | None = None,
    candidate_run_dir: Path | None = None,
) -> PreferenceValidationReport:
    """Validate one preference run directory.

    ``ranking_run_dir`` is the Stage 7 ranking run this preference run claims to cover;
    ``candidate_run_dir`` is the Stage 4 candidate run its code and prompts came from.
    Both are optional — passing ``None`` skips the corresponding cross-artifact
    recomputation (spec sections 69, 70) but every other check still runs.
    """
    run_dir = Path(run_dir)
    preference_run_id = run_dir.name
    issues: list[PreferenceValidationIssue] = []

    manifest = _check_manifest(run_dir, issues)

    pairs = _load_pairs(run_dir / METADATA_FILENAME, issues)
    rejections = _load_rejections(run_dir / REJECTIONS_FILENAME, issues)
    _check_duplicate_metadata(pairs, issues)
    _check_no_reverse_pairs(pairs, issues)

    _check_training_file(run_dir / PREFERENCES_FILENAME, issues)

    if candidate_run_dir is not None:
        _check_candidate_provenance(pairs, Path(candidate_run_dir), issues)

    if ranking_run_dir is not None and manifest is not None:
        _check_against_ranking_run(pairs, Path(ranking_run_dir), manifest, issues)

    split_manifest = _check_split_manifest(run_dir, issues)
    if split_manifest is not None:
        _check_split_membership(pairs, split_manifest, issues)
        _check_split_reproducibility(pairs, split_manifest, issues)

    if manifest is not None:
        _check_statistics(run_dir, manifest, pairs, rejections, issues)

    return PreferenceValidationReport(preference_run_id=preference_run_id, issues=tuple(issues))


# --------------------------------------------------------------------------------- loaders


def _check_manifest(run_dir: Path, issues: list[PreferenceValidationIssue]) -> PreferenceManifest | None:
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        issues.append(PreferenceValidationIssue("manifest", f"{manifest_path} does not exist"))
        return None
    try:
        return PreferenceManifest.from_dict(read_json(manifest_path))
    except (JsonlError, PreferenceModelError) as exc:
        issues.append(PreferenceValidationIssue("manifest", f"manifest.json: {exc}"))
        return None


def _load_pairs(path: Path, issues: list[PreferenceValidationIssue]) -> list[PreferencePair]:
    try:
        raw_records = list(iter_jsonl(path))
    except JsonlError as exc:
        issues.append(PreferenceValidationIssue("jsonl", str(exc)))
        return []

    pairs: list[PreferencePair] = []
    for number, raw in raw_records:
        try:
            pairs.append(PreferencePair.from_dict(raw))
        except PreferenceModelError as exc:
            label = raw.get("preference_id", "?") if isinstance(raw, dict) else "?"
            issues.append(PreferenceValidationIssue("schema", f"pair {label} ({path}:{number}): {exc}"))
    return pairs


def _load_rejections(path: Path, issues: list[PreferenceValidationIssue]) -> list[PreferenceRejection]:
    try:
        raw_records = list(iter_jsonl(path))
    except JsonlError as exc:
        issues.append(PreferenceValidationIssue("jsonl", str(exc)))
        return []

    rejections: list[PreferenceRejection] = []
    for number, raw in raw_records:
        try:
            rejections.append(PreferenceRejection.from_dict(raw))
        except PreferenceModelError as exc:
            issues.append(PreferenceValidationIssue("schema", f"{path}:{number}: {exc}"))
    return rejections


# ---------------------------------------------------------------------------------- checks


def _check_duplicate_metadata(
    pairs: list[PreferencePair], issues: list[PreferenceValidationIssue]
) -> None:
    seen: set[str] = set()
    for pair in pairs:
        if pair.preference_id in seen:
            issues.append(
                PreferenceValidationIssue(
                    "duplicate_metadata",
                    f"preference_id {pair.preference_id}: duplicate row in metadata.jsonl",
                )
            )
        seen.add(pair.preference_id)


def _check_no_reverse_pairs(
    pairs: list[PreferencePair], issues: list[PreferenceValidationIssue]
) -> None:
    # Spec section 71: if A > B exists, B > A must not.
    directions = {(p.problem_id, p.chosen_candidate_id, p.rejected_candidate_id) for p in pairs}
    seen_reversed: set[tuple[str, str, str]] = set()
    for problem_id, chosen, rejected in directions:
        reverse = (problem_id, rejected, chosen)
        if reverse in directions and reverse not in seen_reversed:
            seen_reversed.add((problem_id, chosen, rejected))
            issues.append(
                PreferenceValidationIssue(
                    "reverse_pair",
                    f"{problem_id}: both {chosen} > {rejected} and {rejected} > {chosen} "
                    "are present",
                )
            )


def _check_training_file(path: Path, issues: list[PreferenceValidationIssue]) -> None:
    # Spec sections 52, 69, 72, 94: exactly three keys, none empty, chosen != rejected,
    # and no duplicate (prompt, chosen, rejected) triple.
    try:
        raw = path.read_text(encoding="utf-8") if path.is_file() else ""
    except OSError as exc:
        issues.append(PreferenceValidationIssue("training_file", f"{path}: {exc}"))
        return

    seen_triples: set[tuple[str, str, str]] = set()
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            issues.append(PreferenceValidationIssue("training_file", f"{path}:{number}: blank line"))
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(PreferenceValidationIssue("training_file", f"{path}:{number}: invalid JSON: {exc}"))
            continue
        if not isinstance(record, dict) or set(record) != {"prompt", "chosen", "rejected"}:
            issues.append(
                PreferenceValidationIssue(
                    "training_file",
                    f"{path}:{number}: expected exactly the keys prompt/chosen/rejected, "
                    f"got {sorted(record) if isinstance(record, dict) else type(record).__name__}",
                )
            )
            continue
        for key in ("prompt", "chosen", "rejected"):
            if not isinstance(record[key], str) or not record[key]:
                issues.append(
                    PreferenceValidationIssue("training_file", f"{path}:{number}: {key!r} must be non-empty")
                )
        if record.get("chosen") == record.get("rejected"):
            issues.append(
                PreferenceValidationIssue("training_file", f"{path}:{number}: chosen and rejected are identical")
            )
        triple = (record.get("prompt", ""), record.get("chosen", ""), record.get("rejected", ""))
        if triple in seen_triples:
            issues.append(
                PreferenceValidationIssue(
                    "duplicate_training_record", f"{path}:{number}: duplicate (prompt, chosen, rejected)"
                )
            )
        seen_triples.add(triple)


def _check_candidate_provenance(
    pairs: list[PreferencePair], candidate_run_dir: Path, issues: list[PreferenceValidationIssue]
) -> None:
    try:
        repo = CandidateRepository(candidate_run_dir)
        candidates_by_id = {c.candidate_id: c for c in repo.load_all()}
    except CandidateStoreError as exc:
        issues.append(PreferenceValidationIssue("candidate_run", str(exc)))
        return

    for pair in pairs:
        for role, candidate_id, code, code_sha256 in (
            ("chosen", pair.chosen_candidate_id, pair.chosen, pair.chosen_code_sha256),
            ("rejected", pair.rejected_candidate_id, pair.rejected, pair.rejected_code_sha256),
        ):
            candidate = candidates_by_id.get(candidate_id)
            if candidate is None:
                issues.append(
                    PreferenceValidationIssue(
                        "unknown_candidate",
                        f"{pair.preference_id}: {role} candidate {candidate_id} not found "
                        "in the candidate run",
                    )
                )
                continue
            if candidate.problem_id != pair.problem_id:
                issues.append(
                    PreferenceValidationIssue(
                        "candidate_problem_mismatch",
                        f"{pair.preference_id}: {role} candidate {candidate_id} belongs to "
                        f"problem {candidate.problem_id!r}, not {pair.problem_id!r}",
                    )
                )
            if candidate.code_sha256 != code_sha256 or candidate.code != code:
                issues.append(
                    PreferenceValidationIssue(
                        "code_provenance",
                        f"{pair.preference_id}: {role} code does not match candidate "
                        f"{candidate_id}'s stored code",
                    )
                )


def _check_against_ranking_run(
    pairs: list[PreferencePair],
    ranking_run_dir: Path,
    manifest: PreferenceManifest,
    issues: list[PreferenceValidationIssue],
) -> None:
    try:
        ranking_repo = RankingRepository(ranking_run_dir)
        assessments_by_id = {a.candidate_id: a for a in ranking_repo.load_assessments()}
    except RankingStoreError as exc:
        issues.append(PreferenceValidationIssue("ranking_run", str(exc)))
        return

    try:
        policy = make_policy(manifest.selection_policy)
    except PreferenceError as exc:
        issues.append(PreferenceValidationIssue("policy", str(exc)))
        return

    comparator = CandidateComparator()

    for pair in pairs:
        chosen_assessment = assessments_by_id.get(pair.chosen_candidate_id)
        rejected_assessment = assessments_by_id.get(pair.rejected_candidate_id)
        if chosen_assessment is None or rejected_assessment is None:
            issues.append(
                PreferenceValidationIssue(
                    "unknown_candidate",
                    f"{pair.preference_id}: chosen/rejected candidate missing from the "
                    "ranking run's assessments",
                )
            )
            continue

        if abs(chosen_assessment.score - pair.chosen_score) > 1e-9 or abs(
            rejected_assessment.score - pair.rejected_score
        ) > 1e-9:
            issues.append(
                PreferenceValidationIssue(
                    "score_recompute",
                    f"{pair.preference_id}: stored scores do not match the ranking run's "
                    f"assessments (chosen {chosen_assessment.score} vs {pair.chosen_score}, "
                    f"rejected {rejected_assessment.score} vs {pair.rejected_score})",
                )
            )

        comparison = comparator.compare(manifest.ranking_run_id, chosen_assessment, rejected_assessment)
        if comparison.relation != "A_BETTER":
            issues.append(
                PreferenceValidationIssue(
                    "direction_recompute",
                    f"{pair.preference_id}: recomputed comparison is {comparison.relation!r}, "
                    "not A_BETTER (chosen no longer objectively better)",
                )
            )

        admitted, reason = policy.admits(
            chosen_assessment, rejected_assessment, minimum_score_margin=manifest.minimum_score_margin
        )
        if not admitted:
            issues.append(
                PreferenceValidationIssue(
                    "policy_recompute",
                    f"{pair.preference_id}: no longer admitted by policy "
                    f"{manifest.selection_policy!r} ({reason})",
                )
            )


def _check_split_manifest(run_dir: Path, issues: list[PreferenceValidationIssue]) -> SplitManifest | None:
    path = run_dir / "split_manifest.json"
    if not path.is_file():
        issues.append(PreferenceValidationIssue("split_manifest", f"{path} does not exist"))
        return None
    try:
        return SplitManifest.from_dict(read_json(path))
    except (JsonlError, PreferenceError) as exc:
        issues.append(PreferenceValidationIssue("split_manifest", f"split_manifest.json: {exc}"))
        return None


def _check_split_membership(
    pairs: list[PreferencePair], split_manifest: SplitManifest, issues: list[PreferenceValidationIssue]
) -> None:
    training_problem_ids = {p.problem_id for p in pairs if not p.duplicate_training_record}
    all_split_ids = (
        split_manifest.train_problem_ids
        + split_manifest.validation_problem_ids
        + split_manifest.test_problem_ids
    )
    for problem_id in all_split_ids:
        if problem_id not in training_problem_ids:
            issues.append(
                PreferenceValidationIssue(
                    "split_membership",
                    f"{problem_id}: assigned to a split but has no training pairs",
                )
            )


def _check_split_reproducibility(
    pairs: list[PreferencePair], split_manifest: SplitManifest, issues: list[PreferenceValidationIssue]
) -> None:
    from .splitter import ProblemSplitter  # local import: avoids a module cycle at import time

    training_problem_ids = {p.problem_id for p in pairs if not p.duplicate_training_record}
    recomputed = ProblemSplitter(ratios=split_manifest.split_ratios, seed=split_manifest.seed).split(
        training_problem_ids
    )
    if (
        recomputed.train_problem_ids != split_manifest.train_problem_ids
        or recomputed.validation_problem_ids != split_manifest.validation_problem_ids
        or recomputed.test_problem_ids != split_manifest.test_problem_ids
    ):
        issues.append(
            PreferenceValidationIssue(
                "split_reproducibility",
                "split_manifest.json does not match a fresh split of the pair-bearing "
                "problems with the recorded seed and ratios",
            )
        )


def _check_statistics(
    run_dir: Path,
    manifest: PreferenceManifest,
    pairs: list[PreferencePair],
    rejections: list[PreferenceRejection],
    issues: list[PreferenceValidationIssue],
) -> None:
    stats_path = run_dir / STATISTICS_FILENAME
    if not stats_path.is_file():
        issues.append(PreferenceValidationIssue("statistics", f"{stats_path} does not exist"))
        return
    try:
        on_disk = PreferenceStatistics.from_dict(read_json(stats_path))
    except (JsonlError, PreferenceModelError) as exc:
        issues.append(PreferenceValidationIssue("statistics", f"statistics.json: {exc}"))
        return

    recomputed = PreferenceStatistics.from_records(
        manifest,
        pairs,
        rejections,
        candidates_considered=derive_candidates_considered(pairs, rejections),
        computed_at=on_disk.computed_at,
    )
    if recomputed != on_disk:
        issues.append(
            PreferenceValidationIssue(
                "statistics",
                "statistics.json does not match a fresh recomputation from "
                "metadata.jsonl/rejections.jsonl",
            )
        )


def format_preference_report(report: PreferenceValidationReport) -> str:
    if report.valid:
        return "Preference dataset validation passed.\n"
    lines = ["Preference dataset validation failed:"]
    lines.extend(f"  {issue.message}" for issue in report.issues)
    return "\n".join(lines) + "\n"


__all__ = [
    "PreferenceValidationIssue",
    "PreferenceValidationReport",
    "format_preference_report",
    "validate_preference_run",
]
