"""Section 118's end-to-end scenario, on synthetic data.

The committed real run is degenerate by construction -- 0 wins, 0 losses, 7 ties, one
training step -- so the improvement, regression and non-degenerate coverage paths never
execute against it. Section 118 specifies a scenario that does exercise them: 20 problems,
base solving 8, DPO solving 11, with failures concentrated in recursion, dynamic
programming and edge cases against a thin preference distribution.

Building it as fixtures rather than committing a fabricated analysis run beside the honest
one is the plan's decision 3.
"""

from __future__ import annotations

from python_dpo.analysis.config import AnalysisConfig
from python_dpo.analysis.coverage import build_gaps, preference_coverage
from python_dpo.analysis.diversity import build_diversity_report
from python_dpo.analysis.outcomes import build_problem_outcomes, partition
from python_dpo.analysis.recommend import build_recommendations, decide_iteration
from python_dpo.analysis.report import build_summary, render_analysis_md, render_next_experiment
from python_dpo.analysis.models import ExperimentLineage

from .conftest import FakeEvaluation, FakeGeneration, FakePair, make_problem

# 20 problems: the first 6 are the categories section 118 says failures concentrate in.
CATEGORIES = (
    ["recursion"] * 3 + ["edge_cases"] * 3 + ["lists"] * 4
    + ["strings"] * 4 + ["dictionaries"] * 3 + ["sorting"] * 3
)
PROBLEMS = {
    f"p{i:03d}": make_problem(
        f"p{i:03d}", category=CATEGORIES[i - 1],
        difficulty="hard" if i <= 6 else "easy",
    )
    for i in range(1, 21)
}

# Base solves 8, DPO solves 11 -- the three extra all outside the weak categories.
BASE_SOLVED = {f"p{i:03d}" for i in range(7, 15)}
DPO_SOLVED = BASE_SOLVED | {"p015", "p016", "p017"}


def build_evaluations():
    evaluations = {"base": [], "dpo": []}
    for problem_id in PROBLEMS:
        for variant, solved_set in (("base", BASE_SOLVED), ("dpo", DPO_SOLVED)):
            solved = problem_id in solved_set
            for sample in range(5):
                evaluations[variant].append(
                    FakeEvaluation(
                        problem_id=problem_id, model_variant=variant, sample_index=sample,
                        tests_total=10, tests_passed=10 if solved else 2,
                        status="passed" if solved else "failed",
                        error_type=None if solved else "assertion_failure",
                    )
                )
    return evaluations


def build_generations():
    generations = {"base": [], "dpo": []}
    for problem_id in PROBLEMS:
        for variant in ("base", "dpo"):
            for sample in range(5):
                generations[variant].append(
                    FakeGeneration(
                        problem_id, variant, sample,
                        extracted_code=f"def f():\n    return {problem_id}_{variant}_{sample}\n",
                    )
                )
    return generations


# Preference pairs cover only the strong categories -- 2%/5%/8% for the weak ones, per
# section 118, approximated here as "no pairs at all on recursion or edge_cases".
PAIRS = [FakePair(f"pref_{i}", f"p{i:03d}") for i in range(7, 15)]
TRAINED = {p.problem_id for p in PAIRS}

LINEAGE = ExperimentLineage(
    evaluation_run_id="eval_x", training_run_id="dpo_x", preference_run_id="pref_x",
    ranking_run_id="rank_x", candidate_run_id="run_x",
)


def run_scenario(config: AnalysisConfig | None = None):
    config = config or AnalysisConfig()
    evaluations = build_evaluations()
    outcomes = build_problem_outcomes(evaluations, PROBLEMS)
    benchmark_ids = sorted(PROBLEMS)
    category_gaps = build_gaps(
        attribute="category", pairs=PAIRS, problems=PROBLEMS,
        benchmark_problem_ids=benchmark_ids, trained_problem_ids=TRAINED,
    )
    difficulty_gaps = build_gaps(
        attribute="difficulty", pairs=PAIRS, problems=PROBLEMS,
        benchmark_problem_ids=benchmark_ids, trained_problem_ids=TRAINED,
    )
    diversity = build_diversity_report(build_generations())
    coverage = preference_coverage(PAIRS, PROBLEMS, TRAINED)
    curve = {"verdict": "healthy", "reason": "loss fell", "preference_overfitting": "not_applicable"}
    decision = decide_iteration(
        config=config, benchmark_problem_count=len(benchmark_ids), paired_ci_width=0.05,
        outcomes=outcomes, category_gaps=category_gaps, diversity=diversity,
    )
    recommendations = build_recommendations(
        config=config, outcomes=outcomes, category_gaps=category_gaps,
        difficulty_gaps=difficulty_gaps, diversity=diversity, error_comparisons=[],
        training_curve=curve, coverage=coverage,
    )
    summary = build_summary(
        analysis_run_id="analysis_x", lineage=LINEAGE, benchmark_version="v1",
        outcomes=outcomes, profiles={}, diversity=diversity, category_gaps=category_gaps,
        difficulty_gaps=difficulty_gaps, decision=decision, recommendations=recommendations,
        training_curve=curve, coverage=coverage,
    )
    return summary, outcomes, category_gaps, decision, recommendations


def test_improvements_are_detected():
    """The three problems DPO solves and base does not."""
    _, outcomes, _, _, _ = run_scenario()
    buckets = partition(outcomes)
    assert [o.problem_id for o in buckets["improvements"]] == ["p015", "p016", "p017"]
    assert buckets["regressions"] == []


def test_the_weak_categories_are_identified_as_data_gaps():
    """Section 118: failures concentrate in recursion and edge_cases, which the preference
    distribution barely covers."""
    _, _, gaps, _, _ = run_scenario()
    by_name = {g.name: g for g in gaps}
    assert by_name["recursion"].verdict == "underrepresented"
    assert by_name["edge_cases"].verdict == "underrepresented"
    assert by_name["recursion"].training_share == 0.0


def test_add_data_is_recommended_and_names_the_weak_categories():
    _, _, _, _, recommendations = run_scenario()
    add_data = next(r for r in recommendations if r.category == "add_data")
    named = set(add_data.evidence["benchmark_categories_unsolved_by_dpo"])
    assert {"recursion", "edge_cases"} <= named


def test_the_evidence_gates_pass_at_this_scale():
    """20 problems is still below the shipped floor of 30, so the decision is gated --
    which is itself the correct behaviour. Lowering the floor lets the real decision show."""
    _, _, _, gated, _ = run_scenario()
    assert gated["decision"] == "insufficient_evidence"

    config = AnalysisConfig.from_mapping({"minimum_evidence": {"benchmark_problems": 10}})
    _, _, _, decision, _ = run_scenario(config)
    assert decision["gated"] is False
    assert decision["decision"] == "refine_data"


def test_the_full_summary_and_report_render():
    summary, _, _, _, _ = run_scenario()
    assert summary["problems_analysed"] == 20
    assert summary["improvements"] == 3
    assert summary["regressions"] == 0
    text = render_analysis_md(summary)
    assert "## Recommendations" in text
    payload = render_next_experiment(summary, source_evaluation_run_id="eval_x")
    assert payload["proposed_changes"]


def test_difficulty_skew_is_detected():
    """All trained problems are easy; the benchmark is 30% hard."""
    _, _, _, _, recommendations = run_scenario()
    assert "increase_problem_difficulty" in [r.category for r in recommendations]
