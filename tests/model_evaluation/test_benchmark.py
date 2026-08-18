"""Tests for the held-out benchmark (spec sections 6-10, 91)."""

from __future__ import annotations

import dataclasses

import pytest

from python_dpo.generation.prompt_builder import build_canonical_prompt
from python_dpo.model_evaluation.benchmark import (
    Benchmark,
    build_benchmark,
    check_leakage,
    compute_dataset_hash,
    load_benchmark,
    save_benchmark,
)
from python_dpo.model_evaluation.errors import BenchmarkError, BenchmarkLeakageError
from python_dpo.problems.models import Problem, TestCase


def make_problem(problem_id: str, *, prompt: str = "Do a thing.") -> Problem:
    return Problem(
        id=problem_id,
        prompt=prompt,
        signature="def solve(x):",
        entry_point="solve",
        category="lists",
        difficulty="easy",
        reference_solution="def solve(x):\n    return x",
        tests=(TestCase(id="t1", input={"x": 1}, expected=1),),
    )


def make_problems(n: int) -> list[Problem]:
    return [make_problem(f"p{i:03d}") for i in range(1, n + 1)]


def test_build_benchmark_selects_exactly_the_given_ids():
    problems = make_problems(5)
    manifest = build_benchmark("python_eval_v1", problems, ["p001", "p003"])
    assert manifest.problem_ids == ("p001", "p003")
    assert manifest.problem_count == 2
    assert manifest.benchmark_version == "python_eval_v1"


def test_build_benchmark_rejects_unknown_problem_id():
    problems = make_problems(2)
    with pytest.raises(BenchmarkError):
        build_benchmark("python_eval_v1", problems, ["p999"])


def test_build_benchmark_rejects_empty_selection():
    with pytest.raises(BenchmarkError):
        build_benchmark("python_eval_v1", make_problems(2), [])


def test_dataset_hash_is_stable_across_input_order():
    problems = make_problems(3)
    hash_a = compute_dataset_hash(problems)
    hash_b = compute_dataset_hash(list(reversed(problems)))
    assert hash_a == hash_b


def test_dataset_hash_changes_when_content_changes():
    problems = make_problems(3)
    mutated = [dataclasses.replace(problems[0], prompt="A different problem."), *problems[1:]]
    assert compute_dataset_hash(problems) != compute_dataset_hash(mutated)


def test_save_and_load_round_trip(tmp_path):
    problems = make_problems(4)
    manifest = build_benchmark("python_eval_v1", problems, ["p001", "p002"])
    save_benchmark(tmp_path, manifest)

    loaded = load_benchmark(tmp_path, "python_eval_v1", problems)
    assert loaded.benchmark_version == "python_eval_v1"
    assert {p.id for p in loaded.problems} == {"p001", "p002"}


def test_load_benchmark_detects_drift(tmp_path):
    """Spec section 10: mutating a benchmarked problem must fail validation."""
    problems = make_problems(2)
    manifest = build_benchmark("python_eval_v1", problems, ["p001", "p002"])
    save_benchmark(tmp_path, manifest)

    mutated_problems = [dataclasses.replace(problems[0], prompt="Mutated!"), problems[1]]
    with pytest.raises(BenchmarkError, match="drifted"):
        load_benchmark(tmp_path, "python_eval_v1", mutated_problems)


def test_load_benchmark_missing_manifest(tmp_path):
    with pytest.raises(BenchmarkError):
        load_benchmark(tmp_path, "does_not_exist", make_problems(2))


def test_check_leakage_detects_train_overlap():
    problems = make_problems(3)
    manifest = build_benchmark("python_eval_v1", problems, ["p001", "p002"])
    benchmark = Benchmark(manifest=manifest, problems=tuple(p for p in problems if p.id in manifest.problem_ids))
    split_manifest = {"train_problem_ids": ["p001"], "validation_problem_ids": []}
    with pytest.raises(BenchmarkLeakageError, match="p001"):
        check_leakage(benchmark, split_manifest)


def test_check_leakage_detects_validation_overlap():
    problems = make_problems(3)
    manifest = build_benchmark("python_eval_v1", problems, ["p001", "p002"])
    benchmark = Benchmark(manifest=manifest, problems=tuple(p for p in problems if p.id in manifest.problem_ids))
    split_manifest = {"train_problem_ids": [], "validation_problem_ids": ["p002"]}
    with pytest.raises(BenchmarkLeakageError, match="p002"):
        check_leakage(benchmark, split_manifest)


def test_check_leakage_passes_when_disjoint():
    problems = make_problems(3)
    manifest = build_benchmark("python_eval_v1", problems, ["p001", "p002"])
    benchmark = Benchmark(manifest=manifest, problems=tuple(p for p in problems if p.id in manifest.problem_ids))
    split_manifest = {"train_problem_ids": ["p003"], "validation_problem_ids": []}
    check_leakage(benchmark, split_manifest)  # must not raise


def test_reference_solution_never_reaches_the_canonical_prompt():
    """Spec section 6: the model must not receive reference_solution during inference."""
    problem = make_problem("p001", prompt="Write a function.")
    problem = dataclasses.replace(
        problem, reference_solution="def solve(x):\n    return 'THE_SECRET_SOLUTION'"
    )
    prompt = build_canonical_prompt(problem)
    assert "THE_SECRET_SOLUTION" not in prompt
