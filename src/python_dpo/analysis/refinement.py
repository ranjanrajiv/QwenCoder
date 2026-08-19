"""Refined datasets and the benchmark-leakage guard (spec 11 sections 59-66, 70-79, 104, 117).

This is the stage's dangerous half, and the danger has a specific shape: the held-out
problems DPO failed are the single most tempting thing to add to training. Doing so would
raise the next evaluation's numbers while destroying their meaning, and -- because the
benchmark is content-hashed but the leak would be upstream in the *training* data -- nothing
downstream would notice.

So :func:`assert_no_benchmark_leakage` runs **before any refined file is written**, and a
hit raises :class:`~python_dpo.analysis.errors.RefinementLeakageError` rather than filtering
the row out silently. A filtered row would let a leak be introduced and quietly corrected,
leaving no evidence it was ever attempted; an exception makes it a visible failure.

Section 77's "never overwrite Stage 8" is guaranteed structurally: this module only ever
opens Stage 8's files for reading, and writes exclusively into the analysis run directory.
The refined dataset is re-versioned ``dpo_preference_v2`` and carries
``parent_preference_run_id`` so its ancestry survives (sections 78, 79).

Hard examples are **pointers**, not training rows (section 64): each references a problem by
id rather than duplicating its definition, so a refined dataset can never become a second,
diverging copy of the problem catalog.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .errors import RefinementLeakageError
from .models import utc_now_iso

REFINED_PREFERENCE_VERSION = "dpo_preference_v2"

# Section 76's per-pair verdicts. Recorded for every pair so a removal is auditable rather
# than a silent drop (CLAUDE.md's Data Integrity rule).
REFINEMENT_VERDICTS = ("retain", "remove", "regenerate")


def assert_no_benchmark_leakage(
    rows: Sequence[dict[str, Any]], benchmark_problem_ids: Sequence[str]
) -> None:
    """Sections 65, 66, 104, 117. Raises rather than filtering.

    Called before every refined-dataset write. ``rows`` may be preference rows or example
    rows; anything carrying a ``problem_id`` is checked.
    """
    benchmark = set(benchmark_problem_ids)
    if not benchmark:
        return
    offenders = sorted({r["problem_id"] for r in rows if r.get("problem_id") in benchmark})
    if offenders:
        raise RefinementLeakageError(
            "refined dataset would contain held-out benchmark problem(s): "
            f"{', '.join(offenders)}. Training on a benchmark problem invalidates every "
            "future evaluation of this model (spec sections 65, 66, 104)"
        )


def build_hard_examples(
    outcomes: Sequence[Any],
    *,
    evaluation_run_id: str,
    benchmark_version: str | None,
) -> list[dict[str, Any]]:
    """Section 59: benchmark problems neither variant solved.

    Section 63's provenance is mandatory on every row, and section 64 forbids duplicating
    the problem definition -- these reference ``problem_id`` only.
    """
    return [
        {
            "problem_id": o.problem_id,
            "reason": "unsolved_by_both_variants",
            "base_best_score": o.base_best_score,
            "dpo_best_score": o.dpo_best_score,
            "source_evaluation_run_id": evaluation_run_id,
            "benchmark_version": benchmark_version,
            "model_variant": "both",
            "created_at": utc_now_iso(),
        }
        for o in outcomes
        if not o.base_solved and not o.dpo_solved
    ]


def build_regression_examples(
    outcomes: Sequence[Any],
    *,
    evaluation_run_id: str,
    benchmark_version: str | None,
) -> list[dict[str, Any]]:
    """Section 61: problems base solved and DPO did not."""
    return [
        {
            "problem_id": o.problem_id,
            "reason": "regressed_under_dpo",
            "outcome": o.outcome,
            "severity": o.severity,
            "base_best_score": o.base_best_score,
            "dpo_best_score": o.dpo_best_score,
            "source_evaluation_run_id": evaluation_run_id,
            "benchmark_version": benchmark_version,
            "model_variant": "base",
            "created_at": utc_now_iso(),
        }
        for o in outcomes
        if o.base_solved and not o.dpo_solved
    ]


def build_successful_dpo_examples(
    outcomes: Sequence[Any],
    *,
    evaluation_run_id: str,
    benchmark_version: str | None,
) -> list[dict[str, Any]]:
    """Section 62: problems DPO solved and base did not."""
    return [
        {
            "problem_id": o.problem_id,
            "reason": "solved_only_under_dpo",
            "outcome": o.outcome,
            "base_best_score": o.base_best_score,
            "dpo_best_score": o.dpo_best_score,
            "source_evaluation_run_id": evaluation_run_id,
            "benchmark_version": benchmark_version,
            "model_variant": "dpo",
            "created_at": utc_now_iso(),
        }
        for o in outcomes
        if o.dpo_solved and not o.base_solved
    ]


def plan_refinement(
    pairs: Sequence[Any],
    *,
    minimum_score_margin: float,
    drop_duplicate_code: bool,
    drop_infrastructure_errors: bool,
    benchmark_problem_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Section 71/76: a ``retain``/``remove``/``regenerate`` verdict for every pair.

    Every pair gets a row, including retained ones, so the plan is a complete audit of what
    refinement did rather than a list of survivors.

    Benchmark exclusion is checked **first** and recorded as an explicit verdict, so a
    held-out problem's removal appears in the audit trail with its reason rather than being
    dropped by a guard. Stage 8 legitimately builds pairs on benchmark problems -- they
    land in the test/validation splits and are never trained on -- so this is ordinary
    filtering, not an anomaly. :func:`assert_no_benchmark_leakage` remains the backstop
    that proves the filtering worked.
    """
    benchmark = set(benchmark_problem_ids)
    plan: list[dict[str, Any]] = []
    for pair in pairs:
        verdict = "retain"
        reason = "meets every refinement criterion"

        margin = getattr(pair, "score_margin", None)
        if pair.problem_id in benchmark:
            plan.append(
                {
                    "preference_id": getattr(pair, "preference_id", None),
                    "problem_id": pair.problem_id,
                    "verdict": "remove",
                    "reason": (
                        "problem is held out in the evaluation benchmark; training on it "
                        "would invalidate every future evaluation (spec sections 65, 66)"
                    ),
                    "score_margin": margin,
                }
            )
            continue

        if margin is not None and margin < minimum_score_margin:
            verdict, reason = "remove", (
                f"score margin {margin:.3f} is below the configured minimum "
                f"{minimum_score_margin:.3f}"
            )
        elif drop_duplicate_code and getattr(pair, "chosen_code_sha256", None) == getattr(
            pair, "rejected_code_sha256", object()
        ):
            verdict, reason = "remove", "chosen and rejected code are identical"
        elif drop_infrastructure_errors and getattr(pair, "chosen_correctness", None) == (
            "indeterminate"
        ):
            verdict, reason = "regenerate", (
                "chosen candidate's correctness is indeterminate, so the preference is "
                "not grounded in a clean execution result"
            )

        plan.append(
            {
                "preference_id": getattr(pair, "preference_id", None),
                "problem_id": pair.problem_id,
                "verdict": verdict,
                "reason": reason,
                "score_margin": margin,
            }
        )
    return plan


def build_refined_preferences(
    pairs: Sequence[Any],
    plan: Sequence[dict[str, Any]],
    *,
    parent_preference_run_id: str,
    benchmark_problem_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Sections 77-79: the retained pairs, re-versioned, with the leakage guard applied.

    The guard runs on the rows about to be written, so a benchmark problem cannot reach the
    file even if it somehow survived the split and the plan.
    """
    verdict_by_id = {row["preference_id"]: row["verdict"] for row in plan}
    rows = [
        {
            "problem_id": pair.problem_id,
            "prompt": pair.prompt,
            "chosen": pair.chosen,
            "rejected": pair.rejected,
            "preference_version": REFINED_PREFERENCE_VERSION,
            "parent_preference_run_id": parent_preference_run_id,
            "parent_preference_id": getattr(pair, "preference_id", None),
            "score_margin": getattr(pair, "score_margin", None),
        }
        for pair in pairs
        if verdict_by_id.get(getattr(pair, "preference_id", None)) == "retain"
    ]
    assert_no_benchmark_leakage(rows, benchmark_problem_ids)
    return rows


__all__ = [
    "REFINED_PREFERENCE_VERSION",
    "REFINEMENT_VERDICTS",
    "assert_no_benchmark_leakage",
    "build_hard_examples",
    "build_refined_preferences",
    "build_regression_examples",
    "build_successful_dpo_examples",
    "plan_refinement",
]
