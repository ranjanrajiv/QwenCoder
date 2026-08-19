"""Output diversity and mode-collapse detection (spec 11 sections 26-31, 88).

Diversity is ``unique / total`` over SHA-256 of each sample's extracted code. Section 114
pins the definition with two cases that leave no room for interpretation: ten identical
candidates score 0.1, ten distinct ones score 1.0.

The mode-collapse warning (section 88) fires on a **relative** fall in DPO's diversity
against base, not an absolute one. At ten samples per problem an absolute threshold would
trip on ordinary sampling noise; a relative one asks the question that actually matters --
did alignment make this model repeat itself more than it used to?

A caveat the report is required to carry: low diversity at low temperature is expected and
is not evidence of collapse by itself. Only the base-to-DPO *change* supports that reading.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from .models import DiversityReport


def _digest(code: str | None) -> str | None:
    if code is None:
        return None
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _counts(records: Sequence[Any]) -> tuple[int, int, dict[str, dict[str, float]]]:
    """``(unique, total, per-problem)`` over records that produced code."""
    by_problem: dict[str, list[str]] = {}
    for record in records:
        digest = _digest(getattr(record, "extracted_code", None))
        if digest is None:
            continue
        by_problem.setdefault(record.problem_id, []).append(digest)

    unique = total = 0
    per_problem: dict[str, dict[str, float]] = {}
    for problem_id, digests in sorted(by_problem.items()):
        problem_unique = len(set(digests))
        unique += problem_unique
        total += len(digests)
        per_problem[problem_id] = {
            "unique": problem_unique,
            "total": len(digests),
            "diversity": problem_unique / len(digests) if digests else 0.0,
        }
    return unique, total, per_problem


def build_diversity_report(
    generations: dict[str, Sequence[Any]],
    *,
    mode_collapse_reduction: float = 0.2,
) -> DiversityReport:
    """Sections 26-31 plus section 88's warning."""
    base_unique, base_total, base_per_problem = _counts(generations.get("base", []))
    dpo_unique, dpo_total, dpo_per_problem = _counts(generations.get("dpo", []))

    per_problem: dict[str, dict[str, float]] = {}
    for problem_id in sorted(set(base_per_problem) | set(dpo_per_problem)):
        per_problem[problem_id] = {
            "base": base_per_problem.get(problem_id, {}).get("diversity", 0.0),
            "dpo": dpo_per_problem.get(problem_id, {}).get("diversity", 0.0),
        }

    base_diversity = base_unique / base_total if base_total else 0.0
    dpo_diversity = dpo_unique / dpo_total if dpo_total else 0.0
    warning = False
    if base_diversity > 0:
        relative = (base_diversity - dpo_diversity) / base_diversity
        warning = relative >= mode_collapse_reduction

    return DiversityReport(
        base_unique=base_unique, base_total=base_total,
        dpo_unique=dpo_unique, dpo_total=dpo_total,
        per_problem=per_problem, mode_collapse_warning=warning,
    )


__all__ = ["build_diversity_report"]
