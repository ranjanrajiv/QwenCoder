"""Docker integration test: proves Stage 10 reuses the Stage 6 sandbox unmodified (spec
sections 37, 38, 117).

Needs the built ``python-dpo-evaluator:1.0`` image and a running Docker daemon, so this
module is deselected by default. Run explicitly with:

    pytest -q -m integration
"""

from __future__ import annotations

import pytest

from python_dpo.candidates.hashing import sha256_text
from python_dpo.evaluation.config import EvaluationConfig
from python_dpo.evaluation.executor import CandidateEvaluator
from python_dpo.evaluation.pytest_runner import PytestRunner, build_evaluation_sandbox_config
from python_dpo.evaluation.repository import EvaluationRepository
from python_dpo.generation.prompt_builder import build_canonical_prompt
from python_dpo.model_evaluation.evaluation import EvaluationDriver
from python_dpo.model_evaluation.models import GenerationRecord
from python_dpo.problems.models import Problem, TestCase
from python_dpo.sandbox import DockerContainerRuntime, SandboxConfig, SandboxExecutor

pytestmark = pytest.mark.integration

EVALUATOR_IMAGE = EvaluationConfig().image
EVALUATION_RUN_ID = "eval_20260818_140000_a1b2"
TIMEOUT = 15
GRACE = 15

CORRECT_CODE = "def sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)\n"
WRONG_CODE = "def sum_even(numbers):\n    return 0\n"


@pytest.fixture(scope="session")
def runtime() -> DockerContainerRuntime:
    docker = DockerContainerRuntime()
    try:
        docker.check_available()
    except Exception as exc:  # noqa: BLE001 - reported as a clear failure, not a skip
        pytest.fail(
            f"integration tests require a running Docker daemon: {exc}\n"
            "Run the offline suite with `pytest -q` instead."
        )
    if not docker.image_present(EVALUATOR_IMAGE):
        pytest.fail(
            f"the evaluator image {EVALUATOR_IMAGE} is not present. "
            f"Run: docker build -t {EVALUATOR_IMAGE} docker/evaluator/"
        )
    return docker


@pytest.fixture
def evaluator_and_repository(runtime: DockerContainerRuntime, tmp_path):
    sandbox_config = build_evaluation_sandbox_config(
        SandboxConfig(timeout_seconds=TIMEOUT, startup_grace_seconds=GRACE),
        EvaluationConfig(timeout_seconds=TIMEOUT, startup_grace_seconds=GRACE),
    )
    runner = PytestRunner(SandboxExecutor(config=sandbox_config, runtime=runtime))
    repository = EvaluationRepository(tmp_path)
    return CandidateEvaluator(runner=runner, repository=repository), repository


def make_problem() -> Problem:
    return Problem(
        id="p001",
        prompt="Return the sum of the even integers in a list.",
        signature="def sum_even(numbers):",
        entry_point="sum_even",
        category="lists",
        difficulty="easy",
        reference_solution=CORRECT_CODE,
        tests=(
            TestCase(id="t001", input={"numbers": [1, 2, 3, 4]}, expected=6),
            TestCase(id="t002", input={"numbers": []}, expected=0),
        ),
    )


def make_generation_record(
    code: str, problem: Problem, *, model_variant: str = "base", sample_index: int = 0
) -> GenerationRecord:
    return GenerationRecord(
        evaluation_run_id=EVALUATION_RUN_ID,
        problem_id=problem.id,
        model_variant=model_variant,
        sample_index=sample_index,
        seed=1000,
        prompt_sha256=sha256_text(build_canonical_prompt(problem)),
        raw_response=f"```python\n{code}```",
        extraction_format="python_fence",
        generation_time_ms=100,
        generated_tokens=20,
        status="generated",
        extracted_code=code,
        syntax_valid=True,
    )


def test_evaluation_driver_reuses_the_real_sandbox_for_a_correct_candidate(evaluator_and_repository):
    evaluator, repository = evaluator_and_repository
    problem = make_problem()
    driver = EvaluationDriver(
        evaluator=evaluator,
        repository=repository,
        evaluation_run_id=EVALUATION_RUN_ID,
        model_variant="base",
        model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
        model_revision=None,
        generation_config={},
    )

    records = driver.run(
        [make_generation_record(CORRECT_CODE, problem, model_variant="base")], {"p001": problem}
    )

    assert len(records) == 1
    assert records[0].status == "passed"
    assert records[0].correct is True
    assert records[0].tests_total == 2


def test_evaluation_driver_classifies_a_wrong_candidate_as_failed(evaluator_and_repository):
    evaluator, repository = evaluator_and_repository
    problem = make_problem()
    driver = EvaluationDriver(
        evaluator=evaluator,
        repository=repository,
        evaluation_run_id=EVALUATION_RUN_ID,
        model_variant="dpo",
        model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
        model_revision=None,
        generation_config={},
    )

    records = driver.run(
        [make_generation_record(WRONG_CODE, problem, model_variant="dpo")], {"p001": problem}
    )

    assert records[0].status == "failed"
    assert records[0].correct is False
    assert records[0].error_type == "assertion_failure"


def test_generation_errors_never_reach_the_sandbox(evaluator_and_repository):
    """Spec section 35: no candidate is built or evaluated for a failed extraction."""
    evaluator, repository = evaluator_and_repository
    problem = make_problem()
    driver = EvaluationDriver(
        evaluator=evaluator,
        repository=repository,
        evaluation_run_id=EVALUATION_RUN_ID,
        model_variant="base",
        model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
        model_revision=None,
        generation_config={},
    )
    failed_generation = GenerationRecord(
        evaluation_run_id=EVALUATION_RUN_ID,
        problem_id="p001",
        model_variant="base",
        sample_index=0,
        seed=1000,
        prompt_sha256="a" * 64,
        raw_response="I refuse.",
        extraction_format="unknown",
        generation_time_ms=10,
        generated_tokens=2,
        status="generation_error",
        error="no code",
    )
    records = driver.run([failed_generation], {"p001": problem})
    assert records == []
