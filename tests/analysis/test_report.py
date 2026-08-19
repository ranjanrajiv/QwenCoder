"""Tests for the rendered analysis (spec 11 sections 38, 96-103).

Two wording rules are enforced here rather than merely intended: section 38 forbids causal
claims about data gaps, and section 99 forbids printing a likely failure mode without a
subcategory that supports it.
"""

from __future__ import annotations

from python_dpo.analysis.models import (
    CategoryGap,
    DiversityReport,
    ErrorProfile,
    ExperimentLineage,
    ProblemOutcome,
    Recommendation,
)
from python_dpo.analysis.report import (
    FORBIDDEN_CAUSAL_PHRASES,
    build_summary,
    render_analysis_md,
    render_next_experiment,
)

LINEAGE = ExperimentLineage(
    evaluation_run_id="eval_x", training_run_id="dpo_x", preference_run_id="pref_x",
    ranking_run_id="rank_x", candidate_run_id="run_x",
)

SUMMARY_FIELDS = {
    "summary_version", "analysis_run_id", "created_at", "lineage", "benchmark_version",
    "problems_analysed", "improvements", "regressions", "unchanged", "error_profiles",
    "diversity", "category_gaps", "difficulty_gaps", "training_curve",
    "preference_coverage", "iteration_decision", "recommendations",
}


def make_summary(*, subcategories=None, gaps=None, recommendations=None):
    return build_summary(
        analysis_run_id="analysis_x",
        lineage=LINEAGE,
        benchmark_version="v1",
        outcomes=[
            ProblemOutcome(problem_id="p001", outcome="unchanged", base_best_score=1.0,
                           dpo_best_score=1.0, base_solved=True, dpo_solved=True),
        ],
        profiles={
            "base": ErrorProfile(
                model_variant="base", total_samples=10, passed=9,
                counts_by_category={"assertion_failure": 1},
                counts_by_subcategory=subcategories if subcategories is not None else {},
            ),
        },
        diversity=DiversityReport(5, 10, 5, 10),
        category_gaps=gaps if gaps is not None else [
            CategoryGap(name="lists", training_share=0.0, benchmark_share=0.5,
                        coverage_ratio=0.0, verdict="underrepresented"),
        ],
        difficulty_gaps=[
            CategoryGap(name="hard", training_share=0.0, benchmark_share=0.0,
                        coverage_ratio=None, verdict="absent_from_both"),
        ],
        decision={"decision": "insufficient_evidence", "reasons": ["too few problems"],
                  "gated": True},
        recommendations=recommendations if recommendations is not None else [
            Recommendation(
                category="add_data", hypothesis="Adding list problems should help.",
                evidence={"categories": ["lists"]}, confidence="medium",
                expected_impact=0.5, evidence_strength=0.5, implementation_cost=0.5,
            ).scored({"expected_impact": 0.5, "evidence_strength": 0.3,
                      "implementation_cost": 0.2}),
        ],
        training_curve={"verdict": "insufficient_data", "reason": "one step",
                        "preference_overfitting": "not_applicable"},
        coverage={"total_pairs": 5, "trained_pairs": 2, "problems_with_pairs": ["p007"],
                  "problems_without_pairs": ["p001"]},
    )


# ------------------------------------------------------------------------- section 96


def test_summary_carries_every_required_field():
    assert set(make_summary()) == SUMMARY_FIELDS


# ------------------------------------------------------------------------- section 97


def test_report_contains_every_required_section():
    text = render_analysis_md(make_summary())
    for heading in (
        "## Headline", "## Lineage", "## Outcomes", "## Error profiles",
        "## Failure subcategories", "## Output diversity", "## Test-level failures",
        "## Coverage gaps", "## Training curve", "## Preference coverage",
        "## Recommendations", "## What this analysis does not establish",
    ):
        assert heading in text, f"missing section: {heading}"


def test_a_gated_decision_is_stated_up_front():
    text = render_analysis_md(make_summary())
    headline = text.split("## Lineage")[0]
    assert "insufficient_evidence" in headline
    assert "gates every other finding" in headline


# ------------------------------------------------------------------------- section 38


def test_no_causal_phrasing_appears_anywhere_in_the_report():
    """A coverage gap that coincides with a failure is a *potential* data gap. Stating
    causation the evidence cannot support would send the next iteration somewhere
    arbitrary."""
    text = render_analysis_md(make_summary()).lower()
    for phrase in FORBIDDEN_CAUSAL_PHRASES:
        assert phrase not in text, f"causal phrasing in the report: {phrase!r}"


def test_the_report_states_what_it_does_not_establish():
    text = render_analysis_md(make_summary())
    assert "does not establish" in text
    assert "potential data gaps" in text


def test_the_coverage_table_carries_its_sample_size_caveat():
    text = render_analysis_md(make_summary())
    assert "one problem per" in text


# ------------------------------------------------------------------------- section 99


def test_likely_failure_is_printed_when_a_subcategory_supports_it():
    text = render_analysis_md(make_summary(subcategories={"AssertionError": 12}))
    assert "likely failure" in text
    assert "AssertionError" in text


def test_likely_failure_is_omitted_when_no_subcategory_supports_it():
    text = render_analysis_md(make_summary(subcategories={}))
    assert "likely failure" not in text
    assert "no failing tests recorded" in text


# ------------------------------------------------------------------------ section 101


def test_next_experiment_carries_the_required_fields():
    payload = render_next_experiment(make_summary(), source_evaluation_run_id="eval_x")
    for key in (
        "source_evaluation_run_id", "analysis_run_id", "iteration_decision",
        "hypothesis", "proposed_changes", "success_criteria",
    ):
        assert key in payload


def test_next_experiment_says_nothing_is_supported_when_there_are_no_recommendations():
    payload = render_next_experiment(
        make_summary(recommendations=[]), source_evaluation_run_id="eval_x"
    )
    assert "No change is supported" in payload["hypothesis"]


def test_a_ratio_that_is_none_renders_without_crashing():
    text = render_analysis_md(make_summary())
    assert "n/a" in text
