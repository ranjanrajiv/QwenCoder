"""Ranking run integrity validation (spec 07 sections 50-52).

Mirrors :mod:`python_dpo.runs.validation`: reads a ranking run's raw JSONL directly
rather than going through :class:`~python_dpo.ranking.repository.RankingRepository`, so
one ``validate`` call collects *every* problem instead of raising on the first bad
record. Every issue is fatal — there is no severity scale — and every check runs to
completion, accumulating into one list.

Internal self-consistency (``score == pass_rate``, ``all_tests_passed`` matching the
counts) comes free from :class:`~python_dpo.ranking.models.CandidateAssessment`'s own
``__post_init__`` — a tampered record simply fails to construct. But spec section 51 says
"do not trust the stored score blindly" and section 52 says to verify "from the
evaluation results", which is a **cross-artifact** re-derivation this module performs by
loading the original Stage 6 evaluation run and re-running the classifier against it.
That is the check that catches an assessment which is internally coherent but no longer
reflects the evidence it claims to summarise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..atomic_io import JsonlError, iter_jsonl, read_json
from ..evaluation.models import EvaluationManifest, EvaluationModelError
from ..evaluation.repository import EvaluationRepository, EvaluationStoreError
from .classifier import CorrectnessClassifier
from .models import (
    CandidateAssessment,
    ComparisonResult,
    RankingManifest,
    RankingModelError,
    RankingResult,
    RankingStatistics,
    compute_pass_rate,
)
from .repository import ASSESSMENTS_FILENAME, COMPARISONS_FILENAME, RANKINGS_FILENAME
from .run_repository import MANIFEST_FILENAME, STATISTICS_FILENAME


@dataclass(frozen=True)
class RankingValidationIssue:
    check: str
    message: str


@dataclass(frozen=True)
class RankingValidationReport:
    ranking_run_id: str
    issues: tuple[RankingValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


def validate_ranking_run(
    run_dir: Path,
    evaluation_run_dir: Path | None = None,
    known_problem_ids: set[str] | None = None,
) -> RankingValidationReport:
    """Validate one ranking run directory.

    ``evaluation_run_dir`` is the Stage 6 evaluation run this ranking claims to cover
    (``manifest.json``'s ``evaluation_run_id``'s directory); pass ``None`` to skip the
    section 51/52 cross-artifact recomputation (still validates everything else).
    ``known_problem_ids`` is the current problem dataset's id set; pass ``None`` to skip
    that check, matching :func:`python_dpo.runs.validation.validate_run`.
    """
    run_dir = Path(run_dir)
    ranking_run_id = run_dir.name
    issues: list[RankingValidationIssue] = []

    manifest = _check_manifest(run_dir, issues)

    assessments = _load_assessments(run_dir / ASSESSMENTS_FILENAME, ranking_run_id, issues)
    _check_duplicate_assessments(assessments, issues)
    if known_problem_ids is not None:
        _check_known_problems(assessments, known_problem_ids, issues)

    rankings = _load_rankings(run_dir / RANKINGS_FILENAME, ranking_run_id, issues)
    _check_assessment_ranking_pairing(assessments, rankings, issues)
    _check_ranks(rankings, issues)

    comparisons = _load_comparisons(run_dir / COMPARISONS_FILENAME, ranking_run_id, issues)
    _check_no_tied_pair_has_a_winner(rankings, comparisons, issues)

    if evaluation_run_dir is not None:
        _check_against_evaluation_run(assessments, Path(evaluation_run_dir), issues)

    if manifest is not None:
        _check_statistics(run_dir, manifest, assessments, rankings, issues)

    return RankingValidationReport(ranking_run_id=ranking_run_id, issues=tuple(issues))


def _check_manifest(run_dir: Path, issues: list[RankingValidationIssue]) -> RankingManifest | None:
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        issues.append(RankingValidationIssue("manifest", f"{manifest_path} does not exist"))
        return None
    try:
        return RankingManifest.from_dict(read_json(manifest_path))
    except (JsonlError, RankingModelError) as exc:
        issues.append(RankingValidationIssue("manifest", f"manifest.json: {exc}"))
        return None


def _load_assessments(
    path: Path, ranking_run_id: str, issues: list[RankingValidationIssue]
) -> list[CandidateAssessment]:
    try:
        raw_records = list(iter_jsonl(path))
    except JsonlError as exc:
        issues.append(RankingValidationIssue("jsonl", str(exc)))
        return []

    assessments: list[CandidateAssessment] = []
    for number, raw in raw_records:
        try:
            assessment = CandidateAssessment.from_dict(raw)
        except RankingModelError as exc:
            label = raw.get("candidate_id", "?") if isinstance(raw, dict) else "?"
            issues.append(
                RankingValidationIssue("schema", f"assessment {label} ({path}:{number}): {exc}")
            )
            continue
        if assessment.ranking_run_id != ranking_run_id:
            issues.append(
                RankingValidationIssue(
                    "ranking_run_id",
                    f"assessment {assessment.candidate_id}: ranking_run_id "
                    f"{assessment.ranking_run_id!r} does not match directory {ranking_run_id!r}",
                )
            )
        assessments.append(assessment)
    return assessments


def _load_rankings(
    path: Path, ranking_run_id: str, issues: list[RankingValidationIssue]
) -> list[RankingResult]:
    try:
        raw_records = list(iter_jsonl(path))
    except JsonlError as exc:
        issues.append(RankingValidationIssue("jsonl", str(exc)))
        return []

    rankings: list[RankingResult] = []
    for number, raw in raw_records:
        try:
            ranking = RankingResult.from_dict(raw)
        except RankingModelError as exc:
            label = raw.get("candidate_id", "?") if isinstance(raw, dict) else "?"
            issues.append(
                RankingValidationIssue("schema", f"ranking {label} ({path}:{number}): {exc}")
            )
            continue
        if ranking.ranking_run_id != ranking_run_id:
            issues.append(
                RankingValidationIssue(
                    "ranking_run_id",
                    f"ranking {ranking.candidate_id}: ranking_run_id {ranking.ranking_run_id!r} "
                    f"does not match directory {ranking_run_id!r}",
                )
            )
        rankings.append(ranking)
    return rankings


def _load_comparisons(
    path: Path, ranking_run_id: str, issues: list[RankingValidationIssue]
) -> list[ComparisonResult]:
    try:
        raw_records = list(iter_jsonl(path))
    except JsonlError as exc:
        issues.append(RankingValidationIssue("jsonl", str(exc)))
        return []

    comparisons: list[ComparisonResult] = []
    for number, raw in raw_records:
        try:
            comparison = ComparisonResult.from_dict(raw)
        except RankingModelError as exc:
            issues.append(RankingValidationIssue("schema", f"{path}:{number}: {exc}"))
            continue
        if comparison.ranking_run_id != ranking_run_id:
            issues.append(
                RankingValidationIssue(
                    "ranking_run_id",
                    f"comparison {comparison.candidate_a}/{comparison.candidate_b}: "
                    f"ranking_run_id {comparison.ranking_run_id!r} does not match "
                    f"directory {ranking_run_id!r}",
                )
            )
        comparisons.append(comparison)
    return comparisons


def _check_duplicate_assessments(
    assessments: list[CandidateAssessment], issues: list[RankingValidationIssue]
) -> None:
    # Spec section 43: exactly one assessment per candidate per ranking run.
    seen: set[str] = set()
    for assessment in assessments:
        if assessment.candidate_id in seen:
            issues.append(
                RankingValidationIssue(
                    "duplicate_assessment",
                    f"candidate {assessment.candidate_id}: duplicate assessment in this ranking run",
                )
            )
        seen.add(assessment.candidate_id)


def _check_known_problems(
    assessments: list[CandidateAssessment],
    known_problem_ids: set[str],
    issues: list[RankingValidationIssue],
) -> None:
    for assessment in assessments:
        if assessment.problem_id not in known_problem_ids:
            issues.append(
                RankingValidationIssue(
                    "problem_id",
                    f"candidate {assessment.candidate_id}: unknown problem_id "
                    f"{assessment.problem_id!r}",
                )
            )


def _check_assessment_ranking_pairing(
    assessments: list[CandidateAssessment],
    rankings: list[RankingResult],
    issues: list[RankingValidationIssue],
) -> None:
    assessed_ids = {a.candidate_id for a in assessments}
    ranked_ids = {r.candidate_id for r in rankings}

    for candidate_id in sorted(assessed_ids - ranked_ids):
        issues.append(
            RankingValidationIssue(
                "missing_ranking", f"candidate {candidate_id}: has an assessment but no ranking result"
            )
        )
    for candidate_id in sorted(ranked_ids - assessed_ids):
        issues.append(
            RankingValidationIssue(
                "orphaned_ranking",
                f"candidate {candidate_id}: has a ranking result but no assessment",
            )
        )

    by_id = {a.candidate_id: a for a in assessments}
    for ranking in rankings:
        assessment = by_id.get(ranking.candidate_id)
        if assessment is None:
            continue
        if ranking.correctness != assessment.correctness:
            issues.append(
                RankingValidationIssue(
                    "correctness_mismatch",
                    f"candidate {ranking.candidate_id}: ranking correctness "
                    f"{ranking.correctness!r} does not match assessment {assessment.correctness!r}",
                )
            )
        if abs(ranking.score - assessment.score) > 1e-9:
            issues.append(
                RankingValidationIssue(
                    "score_mismatch",
                    f"candidate {ranking.candidate_id}: ranking score {ranking.score} does not "
                    f"match assessment score {assessment.score}",
                )
            )


def _check_ranks(rankings: list[RankingResult], issues: list[RankingValidationIssue]) -> None:
    # Spec sections 30, 31: ranks must be contiguous competition ranks per problem, and a
    # tie group must be internally consistent (every member shares the same rank and
    # score — a tie group is never partially preferred).
    by_problem: dict[str, list[RankingResult]] = {}
    for ranking in rankings:
        by_problem.setdefault(ranking.problem_id, []).append(ranking)

    for problem_id, group in sorted(by_problem.items()):
        ranked = sorted(
            (r for r in group if r.rank is not None), key=lambda r: (r.rank, r.candidate_id)
        )
        if not ranked:
            continue

        by_tie_group: dict[str, list[RankingResult]] = {}
        for r in ranked:
            by_tie_group.setdefault(r.tie_group, []).append(r)

        for tie_group, members in by_tie_group.items():
            ranks = {m.rank for m in members}
            scores = {round(m.score, 9) for m in members}
            if len(ranks) != 1:
                issues.append(
                    RankingValidationIssue(
                        "rank_consistency",
                        f"{problem_id}: tie group {tie_group!r} has inconsistent ranks {sorted(ranks)}",
                    )
                )
            if len(scores) != 1:
                issues.append(
                    RankingValidationIssue(
                        "rank_consistency",
                        f"{problem_id}: tie group {tie_group!r} has inconsistent scores {sorted(scores)}",
                    )
                )
            if len(members) != members[0].tie_group_size:
                issues.append(
                    RankingValidationIssue(
                        "rank_consistency",
                        f"{problem_id}: tie group {tie_group!r} has {len(members)} member(s) but "
                        f"tie_group_size={members[0].tie_group_size}",
                    )
                )

        expected_rank = 1
        for tie_group in sorted(by_tie_group, key=lambda tg: by_tie_group[tg][0].rank):
            members = by_tie_group[tie_group]
            actual_rank = members[0].rank
            if actual_rank != expected_rank:
                issues.append(
                    RankingValidationIssue(
                        "rank_consistency",
                        f"{problem_id}: expected rank {expected_rank} at this position, "
                        f"found {actual_rank} (tie group {tie_group!r})",
                    )
                )
            expected_rank = (actual_rank or expected_rank) + len(members)


def _check_no_tied_pair_has_a_winner(
    rankings: list[RankingResult],
    comparisons: list[ComparisonResult],
    issues: list[RankingValidationIssue],
) -> None:
    # Spec sections 29, 31, 35: the check that directly protects DPO label quality — a
    # candidate must never be declared better than another candidate it is tied with.
    tie_group_of = {r.candidate_id: r.tie_group for r in rankings if r.tie_group is not None}
    for comparison in comparisons:
        group_a = tie_group_of.get(comparison.candidate_a)
        group_b = tie_group_of.get(comparison.candidate_b)
        if group_a is not None and group_a == group_b and comparison.relation != "TIE":
            issues.append(
                RankingValidationIssue(
                    "artificial_preference",
                    f"{comparison.candidate_a} vs {comparison.candidate_b}: both belong to tie "
                    f"group {group_a!r} but the comparison relation is {comparison.relation!r}",
                )
            )


def _check_against_evaluation_run(
    assessments: list[CandidateAssessment],
    evaluation_run_dir: Path,
    issues: list[RankingValidationIssue],
) -> None:
    manifest_path = evaluation_run_dir / "manifest.json"
    if not manifest_path.is_file():
        issues.append(
            RankingValidationIssue(
                "evaluation_run", f"{manifest_path} does not exist; cannot cross-check"
            )
        )
        return
    try:
        evaluation_manifest = EvaluationManifest.from_dict(read_json(manifest_path))
    except (JsonlError, EvaluationModelError) as exc:
        issues.append(RankingValidationIssue("evaluation_run", f"manifest.json: {exc}"))
        return

    try:
        evaluation_repo = EvaluationRepository(evaluation_run_dir)
        results_by_id = {r.candidate_id: r for r in evaluation_repo.load_all()}
        failures_by_id = {f.candidate_id: f for f in evaluation_repo.load_failures()}
    except EvaluationStoreError as exc:
        issues.append(RankingValidationIssue("evaluation_run", str(exc)))
        return

    classifier = CorrectnessClassifier()
    assessed_ids = {a.candidate_id for a in assessments}

    # Spec section 70: every candidate the evaluation run covers must have an assessment.
    for candidate_id in sorted(set(evaluation_manifest.requested_candidate_ids) - assessed_ids):
        issues.append(
            RankingValidationIssue(
                "missing_assessment",
                f"candidate {candidate_id}: evaluated but has no assessment in this ranking run",
            )
        )

    # Spec sections 51, 52: independently recompute from the source evaluation evidence,
    # never trust the stored assessment.
    for assessment in assessments:
        result = results_by_id.get(assessment.candidate_id)
        if result is not None:
            expected_correctness, expected_reason = classifier.classify(result)
            expected_pass_rate = compute_pass_rate(result.tests_passed, result.tests_total)
            expected_counts = (
                result.tests_total,
                result.tests_passed,
                result.tests_failed,
                result.tests_error,
                result.tests_skipped,
            )
        elif assessment.candidate_id in failures_by_id:
            failure = failures_by_id[assessment.candidate_id]
            expected_correctness, expected_reason = classifier.classify_missing(failure.error_type)
            expected_pass_rate = 0.0
            expected_counts = (0, 0, 0, 0, 0)
        else:
            issues.append(
                RankingValidationIssue(
                    "unknown_candidate",
                    f"candidate {assessment.candidate_id}: not present in the evaluation run "
                    f"{evaluation_manifest.evaluation_run_id!r} at all",
                )
            )
            continue

        actual_counts = (
            assessment.tests_total,
            assessment.tests_passed,
            assessment.tests_failed,
            assessment.tests_error,
            assessment.tests_skipped,
        )
        if assessment.correctness != expected_correctness:
            issues.append(
                RankingValidationIssue(
                    "correctness_recompute",
                    f"candidate {assessment.candidate_id}: stored correctness "
                    f"{assessment.correctness!r} does not match recomputed {expected_correctness!r}",
                )
            )
        if abs(assessment.pass_rate - expected_pass_rate) > 1e-9:
            issues.append(
                RankingValidationIssue(
                    "pass_rate_recompute",
                    f"candidate {assessment.candidate_id}: stored pass_rate {assessment.pass_rate} "
                    f"does not match recomputed {expected_pass_rate}",
                )
            )
        if actual_counts != expected_counts:
            issues.append(
                RankingValidationIssue(
                    "counts_recompute",
                    f"candidate {assessment.candidate_id}: stored test counts {actual_counts} do "
                    f"not match the evaluation run's {expected_counts}",
                )
            )


def _check_statistics(
    run_dir: Path,
    manifest: RankingManifest,
    assessments: list[CandidateAssessment],
    rankings: list[RankingResult],
    issues: list[RankingValidationIssue],
) -> None:
    stats_path = run_dir / STATISTICS_FILENAME
    if not stats_path.is_file():
        issues.append(RankingValidationIssue("statistics", f"{stats_path} does not exist"))
        return
    try:
        on_disk = RankingStatistics.from_dict(read_json(stats_path))
    except (JsonlError, RankingModelError) as exc:
        issues.append(RankingValidationIssue("statistics", f"statistics.json: {exc}"))
        return

    recomputed = RankingStatistics.from_records(
        manifest, assessments, rankings, computed_at=on_disk.computed_at
    )
    if recomputed != on_disk:
        issues.append(
            RankingValidationIssue(
                "statistics",
                "statistics.json does not match a fresh recomputation from "
                "assessments.jsonl/rankings.jsonl",
            )
        )


def format_ranking_report(report: RankingValidationReport) -> str:
    if report.valid:
        return "Ranking validation passed.\n"
    lines = ["Ranking validation failed:"]
    lines.extend(f"  {issue.message}" for issue in report.issues)
    return "\n".join(lines) + "\n"


__all__ = [
    "RankingValidationIssue",
    "RankingValidationReport",
    "format_ranking_report",
    "validate_ranking_run",
]
