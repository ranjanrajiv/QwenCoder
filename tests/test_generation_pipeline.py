"""End-to-end generation tests driven by the mock model and real run directories
(spec 03 sections 46, 47; spec 04 sections 42, 49, 50).

No test here loads the real Qwen model, and none needs a GPU, weights, or a network.
"""

from __future__ import annotations

import pytest

from python_dpo.candidates import sha256_text
from python_dpo.generation import (
    PROMPT_VERSION,
    STRATEGIES,
    CandidateGenerator,
    resolve_strategies,
)
from python_dpo.models import GenerationConfig, InferenceError, MockModelClient, ModelLoadError
from python_dpo.problems import Problem, TestCase
from python_dpo.runs import RunRepository

GOOD_CODE = "def sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)"
GOOD_OUTPUT = f"```python\n{GOOD_CODE}\n```"
BROKEN_OUTPUT = "```python\ndef sum_even(numbers:\n    return 0\n```"
PROSE_OUTPUT = "I am not able to help with that request."
WRONG_NAME_OUTPUT = "```python\ndef solve(numbers):\n    return 0\n```"

MODEL_CONFIG = {"provider": "mock", "name": "mock/deterministic-coder"}


@pytest.fixture
def problem() -> Problem:
    return Problem(
        id="p001",
        prompt="Return the sum of the even integers in a list.",
        signature="def sum_even(numbers):",
        entry_point="sum_even",
        category="lists",
        difficulty="easy",
        reference_solution="def sum_even(numbers):\n    return 0\n",
        tests=(TestCase(id="t001", input={"numbers": [2]}, expected=2),),
    )


def make_run_repo(tmp_path) -> RunRepository:
    return RunRepository(tmp_path / "runs")


def make_manifest(
    run_repo: RunRepository,
    *,
    problem_ids=("p001",),
    count=5,
    override=None,
    max_attempts=1,
    run_id=None,
):
    strategies = resolve_strategies(STRATEGIES, count, override=override)
    return run_repo.create_run(
        requested_problem_ids=problem_ids,
        requested_candidates_per_problem=count,
        strategies=strategies,
        model_config=MODEL_CONFIG,
        generation_config=GenerationConfig().to_dict(),
        prompt_version=PROMPT_VERSION,
        retry={"max_attempts": max_attempts},
        run_id=run_id,
    )


def make_generator(run_repo, manifest, client=None):
    client = client or MockModelClient()
    repository = run_repo.candidates(manifest.run_id)
    generator = CandidateGenerator(client=client, repository=repository)
    return generator, repository, client


def run(tmp_path, problem, client=None, **manifest_kwargs):
    """One generate() call against a fresh run. Returns (summary, manifest, repository)."""
    run_repo = make_run_repo(tmp_path)
    manifest = make_manifest(run_repo, **manifest_kwargs)
    generator, repository, client = make_generator(run_repo, manifest, client)
    summary = generator.generate([problem], manifest)
    return summary, manifest, repository, run_repo


# ------------------------------------------------------------------- integration (§47)


def test_one_problem_five_candidates(tmp_path, problem):
    summary, manifest, repository, _ = run(tmp_path, problem)

    assert summary.generated == 5
    assert summary.failed == 0

    records = repository.load_all()
    assert len(records) == 5
    assert {r.problem_id for r in records} == {"p001"}
    assert len({r.candidate_id for r in records}) == 5
    assert {r.strategy for r in records} == set(STRATEGIES)
    assert [r.generation_index for r in records] == [1, 2, 3, 4, 5]
    assert all(r.raw_output for r in records)
    assert all(r.code for r in records)
    assert all(r.syntax_valid for r in records)
    assert all(r.function_name_valid for r in records)
    assert all(r.prompt_version == PROMPT_VERSION for r in records)
    assert all(r.run_id == manifest.run_id for r in records)
    assert all(r.generation_config == GenerationConfig().to_dict() for r in records)
    assert all(r.schema_version == "2.0" for r in records)
    assert all(r.code_sha256 == sha256_text(r.code) for r in records)
    assert all(r.attempt == 1 for r in records)
    assert not repository.failures_path.exists()


def test_candidate_ids_follow_the_documented_shape(tmp_path, problem):
    _, _, repository, _ = run(tmp_path, problem)
    assert [r.candidate_id for r in repository.load_all()] == [
        "p001_c001",
        "p001_c002",
        "p001_c003",
        "p001_c004",
        "p001_c005",
    ]


def test_raw_output_is_kept_alongside_extracted_code(tmp_path, problem):
    _, _, repository, _ = run(
        tmp_path, problem, MockModelClient(script=[GOOD_OUTPUT]), count=1
    )

    record = repository.load_all()[0]
    assert record.raw_output == GOOD_OUTPUT
    assert record.code == GOOD_CODE
    assert record.raw_output != record.code
    assert record.extraction_format == "python_fence"
    assert record.raw_output_sha256 == sha256_text(GOOD_OUTPUT)


def test_strategy_override_applies_to_every_candidate(tmp_path, problem):
    _, _, repository, _ = run(tmp_path, problem, count=3, override=["edge_case_focused"])
    assert {r.strategy for r in repository.load_all()} == {"edge_case_focused"}


def test_prompts_are_persisted_before_inference(tmp_path, problem):
    _, manifest, repository, _ = run(tmp_path, problem, count=1)
    prompts = repository.load_prompts()
    assert len(prompts) == 1
    assert prompts[0].problem_id == "p001"
    assert prompts[0].generation_index == 1

    candidate = repository.load_all()[0]
    assert candidate.prompt_sha256 == prompts[0].prompt_sha256


# ------------------------------------------------------------------------------ resume


def test_resuming_the_same_run_regenerates_nothing(tmp_path, problem):
    client = MockModelClient()
    run_repo = make_run_repo(tmp_path)
    manifest = make_manifest(run_repo)
    generator, repository, client = make_generator(run_repo, manifest, client)
    generator.generate([problem], manifest)
    assert client.call_count == 5

    # A second generate() call against the same run directory is resume: nothing new is
    # requested from the model, and every existing record is skipped.
    generator2, repository2, _ = make_generator(run_repo, manifest, client)
    summary = generator2.generate([problem], manifest)
    assert summary.generated == 0
    assert summary.skipped == 5
    assert client.call_count == 5, "resume must not call the model again"
    assert len(repository2.load_all()) == 5


def test_force_creates_a_new_run_and_leaves_the_old_one_untouched(tmp_path, problem):
    run_repo = make_run_repo(tmp_path)
    _, first_manifest, first_repository, _ = run(tmp_path, problem)
    before_bytes = first_repository.candidates_path.read_bytes()

    # --force never overwrites in place (spec 04 section 13): it seeds a brand-new run
    # from the original manifest and regenerates into that run's own empty directory.
    second_manifest = run_repo.create_run_from(first_manifest)
    assert second_manifest.run_id != first_manifest.run_id

    generator, second_repository, _ = make_generator(run_repo, second_manifest)
    summary = generator.generate([problem], second_manifest)
    assert summary.generated == 5
    assert summary.skipped == 0

    assert first_repository.candidates_path.read_bytes() == before_bytes, (
        "the earlier run's file must be byte-for-byte unchanged"
    )
    assert len(second_repository.load_all()) == 5
    assert {c.run_id for c in second_repository.load_all()} == {second_manifest.run_id}


def test_resume_retries_a_generation_that_previously_failed(tmp_path, problem):
    run_repo = make_run_repo(tmp_path)
    manifest = make_manifest(run_repo, count=3)

    # First attempt: candidate 2 produces nothing extractable.
    generator, repository, _ = make_generator(
        run_repo, manifest, MockModelClient(script=[GOOD_OUTPUT, PROSE_OUTPUT])
    )
    generator.generate([problem], manifest)
    assert {r.generation_index for r in repository.load_all()} == {1, 3}

    generator2, repository2, _ = make_generator(run_repo, manifest, MockModelClient())
    summary = generator2.generate([problem], manifest)
    assert summary.generated == 1
    assert summary.skipped == 2
    assert {r.generation_index for r in repository2.load_all()} == {1, 2, 3}


# ---------------------------------------------------------------------- failure paths


def test_empty_response_is_a_failure_not_a_candidate(tmp_path, problem):
    summary, _, repository, _ = run(
        tmp_path, problem, MockModelClient(script=["   "]), count=2
    )

    assert summary.generated == 1
    assert summary.failed == 1
    assert len(repository.load_all()) == 1

    failures = repository.load_failures()
    assert len(failures) == 1
    assert failures[0].error_type == "empty_output"
    assert failures[0].generation_index == 1
    assert failures[0].strategy == "normal"
    assert failures[0].attempt == 1
    assert failures[0].prompt_sha256


def test_unextractable_output_is_a_failure_not_a_candidate(tmp_path, problem):
    summary, _, repository, _ = run(
        tmp_path, problem, MockModelClient(script=[PROSE_OUTPUT]), count=2
    )

    assert summary.generated == 1
    assert summary.failed == 1
    assert repository.load_failures()[0].error_type == "code_extraction"
    assert "p001_c001" not in {r.candidate_id for r in repository.load_all()}


def test_inference_error_is_recorded_and_the_run_continues(tmp_path, problem):
    summary, _, repository, _ = run(
        tmp_path, problem, MockModelClient(script=[InferenceError("CUDA blew up")]), count=5
    )

    assert summary.failed == 1
    assert summary.generated == 4
    failure = repository.load_failures()[0]
    assert failure.error_type == "inference"
    assert "CUDA blew up" in failure.error_message


def test_model_load_failure_aborts_the_run(tmp_path, problem):
    # Section 26.2: no candidate in the run can succeed, so retrying per candidate would
    # only emit one identical failure per generation.
    run_repo = make_run_repo(tmp_path)
    manifest = make_manifest(run_repo, count=5)
    generator, repository, _ = make_generator(
        run_repo, manifest, MockModelClient(script=[ModelLoadError("no weights")])
    )

    with pytest.raises(ModelLoadError):
        generator.generate([problem], manifest)

    failures = repository.load_failures()
    assert len(failures) == 1
    assert failures[0].error_type == "model_load"
    assert repository.load_all() == []


def test_syntax_error_produces_a_candidate_and_no_failure_record(tmp_path, problem):
    # Section 19.1: the record is the model's actual output, malformed or not.
    _, _, repository, _ = run(tmp_path, problem, MockModelClient(script=[BROKEN_OUTPUT]), count=1)
    record = repository.load_all()[0]
    assert record.syntax_valid is False
    assert record.syntax_error
    assert record.function_name_valid is False
    assert record.code == "def sum_even(numbers:\n    return 0"
    assert not repository.failures_path.exists()


def test_wrong_function_name_is_recorded_not_rejected(tmp_path, problem):
    _, _, repository, _ = run(
        tmp_path, problem, MockModelClient(script=[WRONG_NAME_OUTPUT]), count=1
    )
    record = repository.load_all()[0]
    assert record.syntax_valid is True
    assert record.function_name_valid is False


# --------------------------------------------------------------------------- retries


def test_a_retried_infrastructure_failure_still_succeeds(tmp_path, problem):
    run_repo = make_run_repo(tmp_path)
    manifest = make_manifest(run_repo, count=1, max_attempts=2)
    generator, repository, _ = make_generator(
        run_repo, manifest, MockModelClient(script=[InferenceError("transient"), GOOD_OUTPUT])
    )

    summary = generator.generate([problem], manifest)

    assert summary.generated == 1
    assert summary.failed == 0
    assert summary.retries == 1

    candidate = repository.load_all()[0]
    assert candidate.attempt == 2

    failures = repository.load_failures()
    assert len(failures) == 1, "the attempt-1 failure is retained, not overwritten"
    assert failures[0].attempt == 1
    assert failures[0].error_type == "inference"


def test_exhausting_retries_leaves_only_failure_records(tmp_path, problem):
    run_repo = make_run_repo(tmp_path)
    manifest = make_manifest(run_repo, count=1, max_attempts=2)
    generator, repository, _ = make_generator(
        run_repo,
        manifest,
        MockModelClient(script=[InferenceError("one"), InferenceError("two")]),
    )

    summary = generator.generate([problem], manifest)

    assert summary.generated == 0
    assert summary.failed == 1
    failures = repository.load_failures()
    assert [f.attempt for f in failures] == [1, 2]
    assert repository.load_all() == []


def test_candidate_failures_are_never_retried(tmp_path, problem):
    # Empty output / extraction failure is terminal even when attempts remain.
    run_repo = make_run_repo(tmp_path)
    manifest = make_manifest(run_repo, count=1, max_attempts=3)
    generator, repository, client = make_generator(
        run_repo, manifest, MockModelClient(script=["   "])
    )

    generator.generate([problem], manifest)

    assert client.call_count == 1, "a candidate failure must not consume another attempt"
    assert len(repository.load_failures()) == 1


# -------------------------------------------------------------------------- duplicates


def test_exact_duplicates_are_flagged_and_kept(tmp_path, problem):
    summary, _, repository, _ = run(
        tmp_path,
        problem,
        MockModelClient(script=[GOOD_OUTPUT, GOOD_OUTPUT, GOOD_OUTPUT]),
        count=3,
    )

    assert summary.generated == 3
    assert summary.duplicates == 2

    records = repository.load_all()
    assert len(records) == 3, "duplicates are kept for analysis, never deleted"
    assert records[0].duplicate_of is None
    assert records[1].duplicate_of == "p001_c001"
    assert records[2].duplicate_of == "p001_c001"


def test_duplicate_detection_does_not_cross_runs(tmp_path, problem):
    # Spec 04 section 20: duplicates are detected within a run, never auto-rejected
    # across runs. A --force regeneration into a new run gets identical code from the
    # deterministic mock but no duplicate_of link, since each run's index is independent.
    run_repo = make_run_repo(tmp_path)
    _, first_manifest, first_repository, _ = run(tmp_path, problem, count=1)
    first_code_hash = first_repository.load_all()[0].code_sha256

    second_manifest = run_repo.create_run_from(first_manifest)
    generator, second_repository, _ = make_generator(run_repo, second_manifest)
    generator.generate([problem], second_manifest)

    second_candidate = second_repository.load_all()[0]
    assert second_candidate.code_sha256 == first_code_hash
    assert second_candidate.duplicate_of is None

    # Cross-run analysis is still possible via the hash, just not automatic.
    assert second_repository.find_by_hash(first_code_hash) == [second_candidate]


def test_distinct_code_is_not_flagged(tmp_path, problem):
    _, _, repository, _ = run(tmp_path, problem)
    assert all(record.duplicate_of is None for record in repository.load_all())


# ------------------------------------------------------------- §42/§49 mandatory resume


def test_interruption_and_resume_preserve_completed_work(tmp_path):
    problems = [
        Problem(
            id=f"p00{n}",
            prompt=f"Problem {n}.",
            signature="def solve(x):",
            entry_point="solve",
            category="lists",
            difficulty="easy",
            reference_solution="def solve(x):\n    return x\n",
            tests=(TestCase(id="t1", input={"x": 1}, expected=1),),
        )
        for n in (1, 2, 3)
    ]

    run_repo = make_run_repo(tmp_path)
    manifest = make_manifest(run_repo, problem_ids=[p.id for p in problems], count=5)
    run_repo.start_run(manifest.run_id)

    interrupting_client = MockModelClient(script=[GOOD_OUTPUT] * 7 + [KeyboardInterrupt()])
    generator, repository, _ = make_generator(run_repo, manifest, interrupting_client)

    with pytest.raises(KeyboardInterrupt):
        generator.generate(problems, manifest)

    interrupted = run_repo.interrupt_run(manifest.run_id)
    assert interrupted.status == "interrupted"

    before = repository.candidates_path.read_bytes()
    assert len(repository.load_all()) == 7

    resumed_manifest = run_repo.resume_run(manifest.run_id)
    assert resumed_manifest.status == "running"

    resuming_generator, repository2, _ = make_generator(run_repo, resumed_manifest, MockModelClient())
    summary = resuming_generator.generate(problems, resumed_manifest)

    after = repository2.candidates_path.read_bytes()
    assert after.startswith(before), "the first 7 records must be byte-for-byte unchanged"

    all_candidates = repository2.load_all()
    assert len(all_candidates) == 15
    assert summary.generated == 8
    assert summary.skipped == 7

    completed = run_repo.complete_run(manifest.run_id)
    assert completed.status == "completed"


# ---------------------------------------------------------------- §50 reproducibility


def test_mock_generation_is_reproducible_across_runs(tmp_path):
    problems = [
        Problem(
            id=f"p00{n}",
            prompt=f"Problem {n}.",
            signature="def solve(x):",
            entry_point="solve",
            category="lists",
            difficulty="easy",
            reference_solution="def solve(x):\n    return x\n",
            tests=(TestCase(id="t1", input={"x": 1}, expected=1),),
        )
        for n in (1, 2)
    ]

    run_repo = make_run_repo(tmp_path)
    manifest_a = make_manifest(run_repo, problem_ids=[p.id for p in problems], count=3)
    generator_a, repo_a, _ = make_generator(run_repo, manifest_a, MockModelClient())
    generator_a.generate(problems, manifest_a)

    manifest_b = make_manifest(run_repo, problem_ids=[p.id for p in problems], count=3)
    generator_b, repo_b, _ = make_generator(run_repo, manifest_b, MockModelClient())
    generator_b.generate(problems, manifest_b)

    key = lambda c: (c.problem_id, c.generation_index)  # noqa: E731
    a_by_key = {key(c): c for c in repo_a.load_all()}
    b_by_key = {key(c): c for c in repo_b.load_all()}

    assert set(a_by_key) == set(b_by_key)
    for k in a_by_key:
        assert a_by_key[k].code == b_by_key[k].code
        assert a_by_key[k].code_sha256 == b_by_key[k].code_sha256
        assert a_by_key[k].strategy == b_by_key[k].strategy
    # Only the run identity and per-record timestamps differ between the two runs.
    assert {a_by_key[k].run_id for k in a_by_key} == {manifest_a.run_id}
    assert {b_by_key[k].run_id for k in b_by_key} == {manifest_b.run_id}

    # Real-model reproducibility is NOT claimed (spec 04 section 50): GPU kernels,
    # framework versions, sampling implementation, model revision, and hardware can all
    # make a real model's output differ run to run even with an identical seed. Only the
    # deterministic mock is asserted reproducible here.
