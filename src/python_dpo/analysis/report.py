"""Rendering the analysis (spec 11 sections 96-103).

Two wording rules are implemented here, not merely intended:

**Section 38 -- no causal claims.** A coverage gap that coincides with a failure is a
*potential data gap*, never "DPO failed because of insufficient data". The distinction
matters because this stage's output decides what gets built next; a causal claim the
evidence cannot support would send the next iteration somewhere arbitrary.
:data:`FORBIDDEN_CAUSAL_PHRASES` is asserted against the rendered text in tests.

**Section 99 -- "Likely failure" only when a subcategory supports it.** If no exception
class dominates, the line is omitted rather than printed with a guess.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .models import SUMMARY_VERSION, utc_now_iso
from .taxonomy import NO_FAILURE

# Section 38: phrases that assert causation. Asserted absent from rendered data-gap
# sections in tests, so the rule cannot rot into a comment nobody enforces.
FORBIDDEN_CAUSAL_PHRASES = (
    "because of",
    "caused by",
    "due to insufficient",
    "resulted from",
    "is the reason",
)


def build_summary(
    *,
    analysis_run_id: str,
    lineage: Any,
    benchmark_version: str | None,
    outcomes: Sequence[Any],
    profiles: dict[str, Any],
    diversity: Any,
    category_gaps: Sequence[Any],
    difficulty_gaps: Sequence[Any],
    decision: dict[str, Any],
    recommendations: Sequence[Any],
    training_curve: dict[str, Any],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    """Section 96's field list."""
    improvements = [o for o in outcomes if o.outcome.endswith("improvement")]
    regressions = [o for o in outcomes if o.outcome.endswith("regression")]
    return {
        "summary_version": SUMMARY_VERSION,
        "analysis_run_id": analysis_run_id,
        "created_at": utc_now_iso(),
        "lineage": lineage.to_dict(),
        "benchmark_version": benchmark_version,
        "problems_analysed": len(outcomes),
        "improvements": len(improvements),
        "regressions": len(regressions),
        "unchanged": len(outcomes) - len(improvements) - len(regressions),
        "error_profiles": {v: p.to_dict() for v, p in profiles.items()},
        "diversity": diversity.to_dict(),
        "category_gaps": [g.to_dict() for g in category_gaps],
        "difficulty_gaps": [g.to_dict() for g in difficulty_gaps],
        "training_curve": training_curve,
        "preference_coverage": coverage,
        "iteration_decision": decision,
        "recommendations": [r.to_dict() for r in recommendations],
    }


def _fmt_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_analysis_md(summary: dict[str, Any], *, test_failures: Sequence[Any] = ()) -> str:
    """Section 97's report."""
    lines: list[str] = []
    add = lines.append

    add(f"# Error Analysis — {summary['analysis_run_id']}")
    add("")
    decision = summary["iteration_decision"]

    add("## Headline")
    add("")
    add(f"**Iteration decision: `{decision['decision']}`**")
    add("")
    for reason in decision.get("reasons", []):
        add(f"- {reason}")
    add("")
    if decision.get("gated"):
        add(
            "> This decision gates every other finding below. The analyses still ran and "
            "their numbers are reported, but the evidence does not meet the configured "
            "minimum, so none of them is offered as a conclusion about model quality."
        )
        add("")

    add("## Lineage")
    add("")
    for key, value in summary["lineage"].items():
        add(f"- `{key}`: {value}")
    add("")

    add("## Outcomes")
    add("")
    add(
        f"{summary['problems_analysed']} problem(s) analysed — "
        f"{summary['improvements']} improved, {summary['regressions']} regressed, "
        f"{summary['unchanged']} unchanged."
    )
    add("")

    add("## Error profiles")
    add("")
    add("| Variant | Samples | Passed | Failure categories |")
    add("|---|---|---|---|")
    for variant, profile in summary["error_profiles"].items():
        nonzero = {k: v for k, v in profile["counts_by_category"].items() if v}
        add(
            f"| {variant} | {profile['total_samples']} | {profile['passed']} | "
            f"{nonzero or '—'} |"
        )
    add("")

    add("## Failure subcategories")
    add("")
    for variant, profile in summary["error_profiles"].items():
        subcats = profile.get("counts_by_subcategory") or {}
        # Section 99: only claim a likely failure mode when a subcategory supports it.
        if subcats:
            dominant = max(subcats.items(), key=lambda kv: (kv[1], kv[0]))
            if dominant[0] != NO_FAILURE:
                add(f"- **{variant}** — likely failure: `{dominant[0]}` ({dominant[1]} occurrence(s))")
        else:
            add(f"- **{variant}** — no failing tests recorded")
    add("")

    add("## Output diversity")
    add("")
    d = summary["diversity"]
    add(f"- base: {d['base_unique']}/{d['base_total']} unique ({d['base_diversity']:.3f})")
    add(f"- dpo: {d['dpo_unique']}/{d['dpo_total']} unique ({d['dpo_diversity']:.3f})")
    if d.get("relative_change") is not None:
        add(f"- relative change: {d['relative_change']:+.1%}")
    add(f"- mode-collapse warning: `{str(d['mode_collapse_warning']).lower()}`")
    add("")
    add(
        "> Low absolute diversity is expected at low sampling temperature and is not "
        "evidence of collapse on its own; only the base-to-DPO change speaks to that."
    )
    add("")

    add("## Test-level failures")
    add("")
    if test_failures:
        add("| Problem | Test | Base | DPO | Hard | DPO-specific | Base-specific |")
        add("|---|---|---|---|---|---|---|")
        for stat in test_failures:
            add(
                f"| {stat.problem_id} | {stat.test_case_id} | "
                f"{stat.base_failure_rate:.0%} | {stat.dpo_failure_rate:.0%} | "
                f"{'yes' if stat.hard_test else ''} | "
                f"{'yes' if stat.dpo_specific else ''} | "
                f"{'yes' if stat.base_specific else ''} |"
            )
    else:
        add("No test-level failures recorded.")
    add("")

    add("## Coverage gaps")
    add("")
    add("| Category | Training share | Benchmark share | Ratio | Verdict |")
    add("|---|---|---|---|---|")
    for gap in summary["category_gaps"]:
        add(
            f"| {gap['name']} | {gap['training_share']:.0%} | {gap['benchmark_share']:.0%} | "
            f"{_fmt_ratio(gap['coverage_ratio'])} | `{gap['verdict']}` |"
        )
    add("")
    add("| Difficulty | Training share | Benchmark share | Ratio | Verdict |")
    add("|---|---|---|---|---|")
    for gap in summary["difficulty_gaps"]:
        add(
            f"| {gap['name']} | {gap['training_share']:.0%} | {gap['benchmark_share']:.0%} | "
            f"{_fmt_ratio(gap['coverage_ratio'])} | `{gap['verdict']}` |"
        )
    add("")
    add(
        "> These shares are arithmetic over a catalog carrying roughly one problem per "
        "category. The arithmetic is correct; at this sample size the distribution it "
        "describes is close to noise, and the table should be read as a structural "
        "observation rather than a measurement."
    )
    add("")

    add("## Training curve")
    add("")
    tc = summary["training_curve"]
    add(f"- verdict: `{tc.get('verdict')}`")
    if tc.get("reason"):
        add(f"- {tc['reason']}")
    add(f"- preference overfitting: `{tc.get('preference_overfitting')}`")
    if tc.get("preference_overfitting_reason"):
        add(f"  - {tc['preference_overfitting_reason']}")
    add("")

    add("## Preference coverage")
    add("")
    cov = summary["preference_coverage"]
    add(f"- pairs in the run: {cov.get('total_pairs')}")
    add(f"- pairs that reached training: {cov.get('trained_pairs')}")
    add(f"- problems with pairs: {', '.join(cov.get('problems_with_pairs') or []) or '—'}")
    add(f"- problems without pairs: {', '.join(cov.get('problems_without_pairs') or []) or '—'}")
    add("")

    add("## Recommendations")
    add("")
    for index, rec in enumerate(summary["recommendations"], start=1):
        add(f"### {index}. `{rec['category']}` (score {rec['recommendation_score']:.3f})")
        add("")
        add(f"**Hypothesis.** {rec['hypothesis']}")
        add("")
        add(f"- confidence: {rec['confidence']}")
        add(f"- evidence: `{rec['evidence']}`")
        add("")

    add("## What this analysis does not establish")
    add("")
    add(
        "Coverage gaps reported above are associations between the training split's "
        "composition and observed failures. They are potential data gaps. This analysis "
        "does not establish that any gap produced any failure, and with roughly one "
        "problem per category it is not capable of establishing that."
    )
    add("")

    return "\n".join(lines) + "\n"


def render_next_experiment(
    summary: dict[str, Any], *, source_evaluation_run_id: str
) -> dict[str, Any]:
    """Section 101's six required fields. Emitted and then **nothing is trained**
    (sections 5, 113) -- this is a proposal, not an action."""
    recommendations = summary.get("recommendations") or []
    top = recommendations[0] if recommendations else None
    return {
        "source_evaluation_run_id": source_evaluation_run_id,
        "analysis_run_id": summary["analysis_run_id"],
        "iteration_decision": summary["iteration_decision"]["decision"],
        "hypothesis": top["hypothesis"] if top else "No change is supported by this evidence.",
        "proposed_changes": [
            {"category": r["category"], "hypothesis": r["hypothesis"], "confidence": r["confidence"]}
            for r in recommendations
        ],
        "success_criteria": {
            "note": (
                "Set before the next run, not after it. Held-out pass@1 must improve by a "
                "margin decided in advance, on a benchmark large enough to detect it."
            ),
        },
    }


__all__ = [
    "FORBIDDEN_CAUSAL_PHRASES",
    "build_summary",
    "render_analysis_md",
    "render_next_experiment",
]
