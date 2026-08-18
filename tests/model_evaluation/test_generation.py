"""Tests for paired candidate generation (spec sections 20-36, 114-116). No GPU: driven
through a fake runner satisfying the same minimal interface as the real ones."""

from __future__ import annotations

from python_dpo.candidates.hashing import sha256_text
from python_dpo.generation.prompt_builder import build_canonical_prompt
from python_dpo.model_evaluation.benchmark import Benchmark, build_benchmark
from python_dpo.model_evaluation.config import GenerationSettings
from python_dpo.model_evaluation.generation import GenerationDriver, compute_seed
from python_dpo.model_evaluation.runners import Generation
from python_dpo.problems.models import Problem, TestCase


class FakeRunner:
    """The minimal surface :class:`GenerationDriver` needs -- no torch anywhere."""

    def __init__(self, variant: str, response_by_seed: dict[int, str] | None = None) -> None:
        self.variant = variant
        self.loaded = False
        self.calls: list[tuple[str, int]] = []
        self._response_by_seed = response_by_seed or {}

    def ensure_loaded(self) -> None:
        self.loaded = True

    def generate(self, prompt: str, *, seed: int) -> Generation:
        self.calls.append((prompt, seed))
        text = self._response_by_seed.get(seed, "```python\ndef solve(x):\n    return x\n```")
        return Generation(text=text, generation_time_ms=10, generated_tokens=5)


def make_problem(problem_id: str) -> Problem:
    return Problem(
        id=problem_id,
        prompt=f"Solve problem {problem_id}.",
        signature="def solve(x):",
        entry_point="solve",
        category="lists",
        difficulty="easy",
        reference_solution="def solve(x):\n    return x",
        tests=(TestCase(id="t1", input={"x": 1}, expected=1),),
    )


def make_benchmark(problem_ids: list[str]) -> Benchmark:
    problems = [make_problem(pid) for pid in problem_ids]
    manifest = build_benchmark("bench", problems, problem_ids)
    return Benchmark(manifest=manifest, problems=tuple(problems))


def test_seed_schedule_is_deterministic():
    assert compute_seed(1000, 0, 0) == 1000
    assert compute_seed(1000, 0, 1) == 1001
    assert compute_seed(1000, 1, 0) == 2000
    assert compute_seed(1000, 1, 5) == 2005


def test_base_and_dpo_receive_identical_seed_schedules():
    """Spec section 116: base.seed[i] == dpo.seed[i] for every corresponding sample."""
    benchmark = make_benchmark(["p001", "p002"])
    generation = GenerationSettings(num_samples=3, base_seed=1000)

    base_records = GenerationDriver("eval_x").run(FakeRunner("base"), benchmark, generation)
    dpo_records = GenerationDriver("eval_x").run(FakeRunner("dpo"), benchmark, generation)

    base_seeds = {(r.problem_id, r.sample_index): r.seed for r in base_records}
    dpo_seeds = {(r.problem_id, r.sample_index): r.seed for r in dpo_records}
    assert base_seeds == dpo_seeds
    assert len(base_seeds) == 6  # 2 problems x 3 samples


def test_base_and_dpo_receive_identical_prompts():
    """Spec sections 27, 114: the same prompt, verified by hash."""
    benchmark = make_benchmark(["p001"])
    generation = GenerationSettings(num_samples=1, base_seed=1000)

    base_records = GenerationDriver("eval_x").run(FakeRunner("base"), benchmark, generation)
    dpo_records = GenerationDriver("eval_x").run(FakeRunner("dpo"), benchmark, generation)

    expected_hash = sha256_text(build_canonical_prompt(benchmark.problems[0]))
    assert base_records[0].prompt_sha256 == expected_hash
    assert dpo_records[0].prompt_sha256 == expected_hash


def test_extraction_failure_produces_generation_error_with_raw_response_retained():
    """Spec section 35: a failed extraction never becomes a candidate."""
    benchmark = make_benchmark(["p001"])
    generation = GenerationSettings(num_samples=1, base_seed=1000)
    seed = compute_seed(1000, 0, 0)

    runner = FakeRunner("base", response_by_seed={seed: "I refuse to write code today."})
    records = GenerationDriver("eval_x").run(runner, benchmark, generation)

    assert len(records) == 1
    record = records[0]
    assert record.status == "generation_error"
    assert record.extracted_code is None
    assert record.syntax_valid is None
    assert record.raw_response == "I refuse to write code today."
    assert record.error


def test_successful_generation_records_extracted_code_and_syntax_validity():
    benchmark = make_benchmark(["p001"])
    generation = GenerationSettings(num_samples=1, base_seed=1000)
    records = GenerationDriver("eval_x").run(FakeRunner("base"), benchmark, generation)

    assert records[0].status == "generated"
    assert records[0].extracted_code == "def solve(x):\n    return x"
    assert records[0].syntax_valid is True
    assert records[0].candidate_id == "p001_c001"


def test_existing_pairs_are_skipped_for_resume():
    benchmark = make_benchmark(["p001", "p002"])
    generation = GenerationSettings(num_samples=1, base_seed=1000)
    runner = FakeRunner("base")

    records = GenerationDriver("eval_x").run(
        runner, benchmark, generation, existing={("p001", 0)}
    )
    assert {r.problem_id for r in records} == {"p002"}
    assert len(runner.calls) == 1


def test_on_record_callback_fires_per_record():
    benchmark = make_benchmark(["p001", "p002"])
    generation = GenerationSettings(num_samples=1, base_seed=1000)
    seen = []
    GenerationDriver("eval_x").run(
        FakeRunner("base"), benchmark, generation, on_record=seen.append
    )
    assert len(seen) == 2
    assert {r.problem_id for r in seen} == {"p001", "p002"}
