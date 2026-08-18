"""Text rendering for preference statistics and quality reports (spec sections 54, 75,
80, 81).

:class:`~python_dpo.preferences.models.PreferenceStatistics` and
:class:`~python_dpo.preferences.models.QualityReport` themselves live in ``models.py``;
this module holds only the formatters, following ``ranking/statistics.py``'s (newer)
convention of extracting them here rather than formatting inline in the CLI, since
``preferences stats`` and ``preferences show`` share renderers.
"""

from __future__ import annotations

from .models import PreferencePair, PreferenceStatistics, QualityReport


def format_preference_statistics(stats: PreferenceStatistics) -> str:
    """The spec section 54 counters, plus the spec section 80 train/validation/test
    counts when a split has been computed.
    """
    lines = [
        f"Problems processed: {stats.problems_processed}",
        f"Candidates considered: {stats.candidates_considered}",
        f"Candidate pairs considered: {stats.candidate_pairs_considered}",
        f"Pairs generated: {stats.pairs_generated}",
        f"Pairs rejected: {stats.pairs_rejected}",
        f"  Ties: {stats.ties}",
        f"  Duplicate code: {stats.duplicates}",
        f"  Indeterminate: {stats.indeterminate}",
        f"  Prompt mismatches: {stats.prompt_mismatches}",
        f"  Integrity failures: {stats.integrity_failures}",
        f"Strong pairs: {stats.strong_pairs}",
        f"Medium pairs: {stats.medium_pairs}",
        f"Distinct training records: {stats.training_records}",
    ]
    other_reasons = {
        reason: count
        for reason, count in sorted(stats.rejections_by_reason.items())
        if reason not in {"tie", "identical_code", "indeterminate", "invalid_prompt_match", "integrity_failure"}
    }
    if other_reasons:
        lines.append("")
        lines.append("Policy/other exclusions:")
        for reason, count in other_reasons.items():
            lines.append(f"  {reason}: {count}")
    return "\n".join(lines) + "\n"


def format_quality_report(report: QualityReport) -> str:
    """The spec section 75 quality report."""
    lines = [
        f"Total pairs: {report.total_pairs}",
        f"Strong pairs: {report.strong_pairs}",
        f"Medium pairs: {report.medium_pairs}",
        "",
        f"Problems with pairs ({len(report.problems_with_pairs)}): "
        f"{', '.join(report.problems_with_pairs) or '(none)'}",
        f"Problems without pairs ({len(report.problems_without_pairs)}):",
    ]
    for problem_id in sorted(report.problems_without_pairs):
        lines.append(f"  {problem_id}: {report.problems_without_pairs[problem_id]}")

    if report.score_margin_distribution:
        lines += ["", "Score margin distribution:"]
        for bucket in sorted(report.score_margin_distribution):
            lines.append(f"  {bucket}: {report.score_margin_distribution[bucket]}")

    chosen_strategy = report.strategy_distribution.get("chosen", {})
    if chosen_strategy:
        lines += ["", "Chosen strategy distribution:"]
        for strategy in sorted(chosen_strategy):
            lines.append(f"  {strategy}: {chosen_strategy[strategy]}")

    return "\n".join(lines) + "\n"


def format_pair_table(pairs: list[PreferencePair]) -> str:
    """A one-line-per-pair summary table, used by ``preferences list``/``show``."""
    header = f"{'PREFERENCE_ID':<28}{'PROBLEM':<10}{'MARGIN':<8}{'STRENGTH':<10}{'POLICY'}"
    lines = [header]
    for pair in sorted(pairs, key=lambda p: p.preference_id):
        lines.append(
            f"{pair.preference_id:<28}{pair.problem_id:<10}{pair.score_margin:<8.2f}"
            f"{pair.preference_strength:<10}{pair.selection_policy}"
        )
    return "\n".join(lines) + "\n"


def format_pair_detail(pair: PreferencePair, *, show_code: bool) -> str:
    """The spec section 81 single-pair inspection view."""
    lines = [
        f"preference_id: {pair.preference_id}",
        f"problem_id: {pair.problem_id}",
        f"chosen_candidate_id: {pair.chosen_candidate_id}",
        f"rejected_candidate_id: {pair.rejected_candidate_id}",
        f"chosen_score: {pair.chosen_score}",
        f"rejected_score: {pair.rejected_score}",
        f"score_margin: {pair.score_margin}",
        f"chosen_tests_passed/total: {pair.chosen_tests_passed}/{pair.chosen_tests_total}",
        f"rejected_tests_passed/total: {pair.rejected_tests_passed}/{pair.rejected_tests_total}",
        f"preference_strength: {pair.preference_strength}",
        f"selection_policy: {pair.selection_policy} ({pair.selection_policy_version})",
        f"duplicate_training_record: {pair.duplicate_training_record}",
    ]
    if pair.duplicate_training_record:
        lines.append(f"canonical_preference_id: {pair.canonical_preference_id}")
    if show_code:
        lines += ["", "--- chosen ---", pair.chosen, "", "--- rejected ---", pair.rejected]
    return "\n".join(lines) + "\n"


__all__ = [
    "format_pair_detail",
    "format_pair_table",
    "format_preference_statistics",
    "format_quality_report",
]
