"""Fixtures for the Stage 11 analysis suite.

Everything here is synthetic. The real committed run is degenerate by design (0 wins,
0 losses, 7 ties, one training step), so the improvement, regression, mode-collapse and
non-degenerate coverage paths would never execute against it. These builders exercise
those paths without fabricating an analysis run beside the honest one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from python_dpo.problems.models import Problem, TestCase


@dataclass
class FakeGeneration:
    problem_id: str
    model_variant: str
    sample_index: int
    status: str = "generated"
    extracted_code: str | None = "def f():\n    return 1\n"
    raw_response: str = "```python\ndef f():\n    return 1\n```"


@dataclass
class FakeEvaluation:
    problem_id: str
    model_variant: str
    sample_index: int
    tests_total: int = 10
    tests_passed: int = 10
    status: str = "passed"
    error_type: str | None = None


@dataclass
class FakeTestResult:
    problem_id: str
    candidate_id: str
    test_case_id: str
    status: str = "passed"
    error_type: str | None = None


@dataclass
class FakePair:
    preference_id: str
    problem_id: str
    prompt: str = "solve it"
    chosen: str = "def f():\n    return 1\n"
    rejected: str = "def f():\n    return 0\n"
    score_margin: float = 0.5
    chosen_code_sha256: str = "a" * 64
    rejected_code_sha256: str = "b" * 64
    chosen_correctness: str = "correct"
    chosen_strategy: str = "normal"
    rejected_strategy: str = "alternative"


@dataclass
class FakeSplitManifest:
    train_problem_ids: list[str]


def make_problem(problem_id: str, category: str = "lists", difficulty: str = "easy") -> Problem:
    return Problem(
        id=problem_id,
        prompt=f"Solve {problem_id}",
        signature="def f(x):",
        entry_point="f",
        category=category,
        difficulty=difficulty,
        reference_solution="def f(x):\n    return x\n",
        tests=(TestCase(id=f"{problem_id}_t001", input={"x": 1}, expected=1),),
    )


def samples(
    problem_id: str, variant: str, *, passed: int, total: int = 10, tests_total: int = 10
) -> tuple[list[FakeGeneration], list[FakeEvaluation]]:
    """``passed`` of ``total`` samples solve the problem completely; the rest fail all tests."""
    generations, evaluations = [], []
    for index in range(total):
        solved = index < passed
        generations.append(FakeGeneration(problem_id, variant, index))
        evaluations.append(
            FakeEvaluation(
                problem_id=problem_id, model_variant=variant, sample_index=index,
                tests_total=tests_total, tests_passed=tests_total if solved else 0,
                status="passed" if solved else "failed",
                error_type=None if solved else "assertion_failure",
            )
        )
    return generations, evaluations


def evaluations_for(spec: dict[str, tuple[int, int]]) -> dict[str, list[Any]]:
    """``{problem_id: (base_passed, dpo_passed)}`` -> the evaluations dict the analysis takes."""
    result: dict[str, list[Any]] = {"base": [], "dpo": []}
    for problem_id, (base_passed, dpo_passed) in spec.items():
        _, base_evals = samples(problem_id, "base", passed=base_passed)
        _, dpo_evals = samples(problem_id, "dpo", passed=dpo_passed)
        result["base"].extend(base_evals)
        result["dpo"].extend(dpo_evals)
    return result
