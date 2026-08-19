"""Tests for refinement and the benchmark-leakage guard (spec 11 sections 59-79, 104, 117).

Section 117 makes the leakage test required, and it is the centrepiece here: feeding a
held-out benchmark problem back into training would raise the next evaluation's numbers
while destroying their meaning, and nothing downstream would detect it.
"""

from __future__ import annotations

import pytest

from python_dpo.analysis.errors import RefinementLeakageError
from python_dpo.analysis.refinement import (
    REFINED_PREFERENCE_VERSION,
    assert_no_benchmark_leakage,
    build_hard_examples,
    build_refined_preferences,
    build_regression_examples,
    build_successful_dpo_examples,
    plan_refinement,
)
from python_dpo.analysis.models import ProblemOutcome

from .conftest import FakePair


def outcome(problem_id, base_solved, dpo_solved, kind="unchanged"):
    return ProblemOutcome(
        problem_id=problem_id, outcome=kind,
        base_best_score=1.0 if base_solved else 0.0,
        dpo_best_score=1.0 if dpo_solved else 0.0,
        base_solved=base_solved, dpo_solved=dpo_solved,
    )


# ------------------------------------------------------ section 117: the leakage guard


def test_a_benchmark_problem_cannot_reach_the_refined_dataset():
    with pytest.raises(RefinementLeakageError, match="p004"):
        assert_no_benchmark_leakage([{"problem_id": "p004"}], ["p004", "p005"])


def test_the_guard_names_every_offender():
    with pytest.raises(RefinementLeakageError) as excinfo:
        assert_no_benchmark_leakage(
            [{"problem_id": "p004"}, {"problem_id": "p005"}], ["p004", "p005"]
        )
    assert "p004" in str(excinfo.value)
    assert "p005" in str(excinfo.value)


def test_the_guard_passes_when_nothing_is_held_out():
    assert_no_benchmark_leakage([{"problem_id": "p007"}], ["p004", "p005"])


def test_build_refined_preferences_applies_the_guard_before_returning():
    """The guard runs on the rows about to be written, so a leak cannot be produced even
    if it somehow survived the plan."""
    pairs = [FakePair("pref_a", "p004")]
    plan = [{"preference_id": "pref_a", "problem_id": "p004", "verdict": "retain"}]
    with pytest.raises(RefinementLeakageError):
        build_refined_preferences(
            pairs, plan, parent_preference_run_id="pref_run", benchmark_problem_ids=["p004"]
        )


def test_benchmark_problems_are_removed_by_the_plan_with_a_stated_reason():
    """Ordinary filtering with an audit trail, not a guard firing: Stage 8 legitimately
    builds pairs on benchmark problems that land in the test/validation splits."""
    plan = plan_refinement(
        [FakePair("pref_a", "p004"), FakePair("pref_b", "p007")],
        minimum_score_margin=0.2, drop_duplicate_code=True, drop_infrastructure_errors=True,
        benchmark_problem_ids=["p004"],
    )
    by_id = {row["preference_id"]: row for row in plan}
    assert by_id["pref_a"]["verdict"] == "remove"
    assert "held out" in by_id["pref_a"]["reason"]
    assert by_id["pref_b"]["verdict"] == "retain"


# ------------------------------------------------------------------- section 71's filters


def test_a_thin_margin_is_removed():
    plan = plan_refinement(
        [FakePair("pref_a", "p007", score_margin=0.05)],
        minimum_score_margin=0.2, drop_duplicate_code=True, drop_infrastructure_errors=True,
    )
    assert plan[0]["verdict"] == "remove"
    assert "margin" in plan[0]["reason"]


def test_identical_code_on_both_sides_is_removed():
    plan = plan_refinement(
        [FakePair("pref_a", "p007", chosen_code_sha256="x" * 64, rejected_code_sha256="x" * 64)],
        minimum_score_margin=0.2, drop_duplicate_code=True, drop_infrastructure_errors=True,
    )
    assert plan[0]["verdict"] == "remove"
    assert "identical" in plan[0]["reason"]


def test_indeterminate_correctness_is_marked_for_regeneration_not_removal():
    plan = plan_refinement(
        [FakePair("pref_a", "p007", chosen_correctness="indeterminate")],
        minimum_score_margin=0.2, drop_duplicate_code=True, drop_infrastructure_errors=True,
    )
    assert plan[0]["verdict"] == "regenerate"


def test_every_pair_appears_in_the_plan_including_retained_ones():
    """Data Integrity: the plan is a complete audit, not a list of survivors."""
    pairs = [
        FakePair("a", "p007"),
        FakePair("b", "p008", score_margin=0.01),
        FakePair("c", "p004"),
    ]
    plan = plan_refinement(
        pairs, minimum_score_margin=0.2, drop_duplicate_code=True,
        drop_infrastructure_errors=True, benchmark_problem_ids=["p004"],
    )
    assert len(plan) == 3
    assert {row["verdict"] for row in plan} == {"retain", "remove"}


# --------------------------------------------------------- sections 63, 64, 77, 78, 79


def test_refined_rows_carry_versioning_and_parent_run_id():
    pairs = [FakePair("pref_a", "p007")]
    plan = [{"preference_id": "pref_a", "problem_id": "p007", "verdict": "retain"}]
    [row] = build_refined_preferences(
        pairs, plan, parent_preference_run_id="pref_run_x", benchmark_problem_ids=["p004"]
    )
    assert row["preference_version"] == REFINED_PREFERENCE_VERSION
    assert row["parent_preference_run_id"] == "pref_run_x"
    assert row["parent_preference_id"] == "pref_a"


def test_example_rows_carry_mandatory_provenance():
    rows = build_hard_examples(
        [outcome("p001", False, False)], evaluation_run_id="eval_x", benchmark_version="v1"
    )
    assert rows[0]["source_evaluation_run_id"] == "eval_x"
    assert rows[0]["benchmark_version"] == "v1"
    assert rows[0]["problem_id"] == "p001"


def test_examples_reference_problems_by_id_and_never_duplicate_the_definition():
    """Section 64: a refined dataset must not become a second, diverging copy of the
    problem catalog."""
    rows = build_hard_examples(
        [outcome("p001", False, False)], evaluation_run_id="eval_x", benchmark_version="v1"
    )
    for forbidden in ("prompt", "tests", "reference_solution", "signature"):
        assert forbidden not in rows[0]


def test_hard_regression_and_success_examples_select_the_right_problems():
    outcomes = [
        outcome("p001", False, False),                          # hard
        outcome("p002", True, False, "complete_regression"),    # regression
        outcome("p003", False, True, "complete_improvement"),   # dpo-only success
        outcome("p004", True, True),                            # neither
    ]
    hard = build_hard_examples(outcomes, evaluation_run_id="e", benchmark_version=None)
    regressed = build_regression_examples(outcomes, evaluation_run_id="e", benchmark_version=None)
    improved = build_successful_dpo_examples(outcomes, evaluation_run_id="e", benchmark_version=None)

    assert [r["problem_id"] for r in hard] == ["p001"]
    assert [r["problem_id"] for r in regressed] == ["p002"]
    assert [r["problem_id"] for r in improved] == ["p003"]
