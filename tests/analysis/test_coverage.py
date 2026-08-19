"""Tests for training-vs-benchmark coverage (spec 11 sections 32-42, 115)."""

from __future__ import annotations

import json

from python_dpo.analysis.coverage import (
    build_gaps,
    correlate_errors_with_coverage,
    preference_coverage,
    training_problem_ids,
)
from python_dpo.analysis.report import FORBIDDEN_CAUSAL_PHRASES

from .conftest import FakePair, FakeSplitManifest, make_problem


def gaps_for(pairs, problems, benchmark_ids, trained, **kwargs):
    return {
        g.name: g
        for g in build_gaps(
            attribute="category", pairs=pairs, problems=problems,
            benchmark_problem_ids=benchmark_ids, trained_problem_ids=trained, **kwargs
        )
    }


# ------------------------------------------------------------- section 115's worked example


def test_worked_example_training_5_percent_benchmark_25_percent_is_underrepresented():
    """Training share 5%, benchmark share 25% -> ratio 0.2 -> underrepresented."""
    problems = {}
    pairs = []
    # 1 pair on `recursion` out of 20 total -> 5% training share.
    problems["r1"] = make_problem("r1", category="recursion")
    pairs.append(FakePair("pref_r1", "r1"))
    for i in range(19):
        pid = f"o{i}"
        problems[pid] = make_problem(pid, category="lists")
        pairs.append(FakePair(f"pref_{pid}", pid))
    # Benchmark: 1 of 4 problems is `recursion` -> 25% benchmark share.
    bench = []
    for i in range(4):
        pid = f"b{i}"
        problems[pid] = make_problem(pid, category="recursion" if i == 0 else "strings")
        bench.append(pid)

    trained = {p.problem_id for p in pairs}
    gaps = gaps_for(pairs, problems, bench, trained)

    assert gaps["recursion"].training_share == 0.05
    assert gaps["recursion"].benchmark_share == 0.25
    assert abs(gaps["recursion"].coverage_ratio - 0.2) < 1e-9
    assert gaps["recursion"].verdict == "underrepresented"


# ------------------------------------------- the two degenerate cases found in the real data


def test_category_absent_from_the_benchmark_has_no_ratio():
    """Division by zero. The verdict carries the meaning instead."""
    problems = {"p1": make_problem("p1", category="exceptions"),
                "b1": make_problem("b1", category="lists")}
    pairs = [FakePair("pref_p1", "p1")]
    gaps = gaps_for(pairs, problems, ["b1"], {"p1"})
    assert gaps["exceptions"].coverage_ratio is None
    assert gaps["exceptions"].verdict == "not_in_benchmark"


def test_category_absent_from_both_has_no_ratio():
    """0/0. Present in the catalog, used by neither side."""
    problems = {"p1": make_problem("p1", category="exceptions"),
                "b1": make_problem("b1", category="lists"),
                "unused": make_problem("unused", category="async")}
    pairs = [FakePair("pref_p1", "p1")]
    gaps = gaps_for(pairs, problems, ["b1"], {"p1"})
    assert gaps["async"].coverage_ratio is None
    assert gaps["async"].verdict == "absent_from_both"


def test_emitted_json_contains_no_infinity_or_nan():
    """Neither is representable in JSON; a downstream reader would fail to parse them."""
    problems = {"p1": make_problem("p1", category="exceptions"),
                "b1": make_problem("b1", category="lists"),
                "unused": make_problem("unused", category="async")}
    pairs = [FakePair("pref_p1", "p1")]
    payload = json.dumps([
        g.to_dict() for g in build_gaps(
            attribute="category", pairs=pairs, problems=problems,
            benchmark_problem_ids=["b1"], trained_problem_ids={"p1"},
        )
    ])
    assert "Infinity" not in payload
    assert "NaN" not in payload


# ----------------------------------------------------------------------- verdict boundaries


def test_balanced_and_overrepresented_verdicts():
    problems = {
        "t1": make_problem("t1", category="lists"),
        "b1": make_problem("b1", category="lists"),
        "b2": make_problem("b2", category="strings"),
    }
    pairs = [FakePair("pref_t1", "t1")]
    # training 100% lists, benchmark 50% lists -> ratio 2.0, at the boundary -> balanced.
    gaps = gaps_for(pairs, problems, ["b1", "b2"], {"t1"}, under=0.5, over=2.0)
    assert gaps["lists"].coverage_ratio == 2.0
    assert gaps["lists"].verdict == "balanced"
    # Tightening the ceiling pushes the same ratio over.
    gaps = gaps_for(pairs, problems, ["b1", "b2"], {"t1"}, under=0.5, over=1.5)
    assert gaps["lists"].verdict == "overrepresented"


# ------------------------------------------------------------- the training population rule


def test_only_train_split_problems_count_as_trained():
    """A pair on a test-split problem was never trained on; counting it would overstate
    coverage of exactly the categories this analysis exists to find holes in."""
    split = FakeSplitManifest(train_problem_ids=["p007"])
    pairs = [FakePair("a", "p007"), FakePair("b", "p004")]
    assert training_problem_ids(split, pairs) == {"p007"}


def test_falls_back_to_pair_bearing_problems_without_a_split_manifest():
    pairs = [FakePair("a", "p007"), FakePair("b", "p004")]
    assert training_problem_ids(None, pairs) == {"p007", "p004"}


def test_untrained_pairs_are_excluded_from_shares():
    problems = {"p007": make_problem("p007", category="exceptions"),
                "p004": make_problem("p004", category="sets"),
                "b1": make_problem("b1", category="sets")}
    pairs = [FakePair("a", "p007"), FakePair("b", "p004")]
    gaps = gaps_for(pairs, problems, ["b1"], {"p007"})
    # `sets` was paired but never trained, so its training share is zero.
    assert gaps["sets"].training_share == 0.0
    assert gaps["sets"].verdict == "underrepresented"


# ------------------------------------------------------------------- section 38's wording


def test_correlation_output_states_a_potential_gap_and_never_causation():
    problems = {"b1": make_problem("b1", category="lists")}
    gaps = build_gaps(
        attribute="category", pairs=[], problems=problems,
        benchmark_problem_ids=["b1"], trained_problem_ids=set(),
    )

    class _Outcome:
        problem_id, category, dpo_solved = "b1", "lists", False

    rows = correlate_errors_with_coverage(gaps, [_Outcome()])
    assert rows, "expected a correlation row for an underrepresented category with failures"
    text = " ".join(r["observation"] for r in rows).lower()
    assert "potential data gap" in text
    for phrase in FORBIDDEN_CAUSAL_PHRASES:
        assert phrase not in text, f"causal phrasing leaked into a data-gap observation: {phrase}"


def test_preference_coverage_reports_problems_without_pairs():
    problems = {"p1": make_problem("p1"), "p2": make_problem("p2")}
    coverage = preference_coverage([FakePair("a", "p1")], problems, {"p1"})
    assert coverage["problems_without_pairs"] == ["p2"]
    assert coverage["trained_pairs"] == 1
