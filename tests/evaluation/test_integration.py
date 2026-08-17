"""Docker integration tests for the candidate test executor (spec 06 sections 65-90).

Needs the built ``python-dpo-evaluator:1.0`` image
(``docker build -t python-dpo-evaluator:1.0 docker/evaluator/``) and a running Docker
daemon, so this module is deselected by default
(``addopts = "-ra -m 'not integration'"`` in pyproject.toml). Run explicitly with:

    pytest -q -m integration

Mirrors ``tests/sandbox/test_sandbox_integration.py``'s philosophy: fail loudly rather
than skip silently when Docker or the image is unavailable, since these were asked for
explicitly. The default ``pytest -q`` remains offline and zero-skip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from python_dpo.candidates.models import Candidate
from python_dpo.evaluation.config import EvaluationConfig
from python_dpo.evaluation.executor import CandidateEvaluator
from python_dpo.evaluation.pytest_runner import PytestRunner, build_evaluation_sandbox_config
from python_dpo.evaluation.repository import EvaluationRepository
from python_dpo.problems.models import Problem, TestCase
from python_dpo.problems.storage import dataset_path, load_problems
from python_dpo.sandbox import DockerContainerRuntime, SandboxConfig, SandboxExecutor

pytestmark = pytest.mark.integration

EVALUATOR_IMAGE = EvaluationConfig().image
EVALUATION_RUN_ID = "eval_20260817_154500_a12f"
CANDIDATE_RUN_ID = "run_20260817_055411"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Container startup alone is ~2s; long enough that a slow daemon does not cause a false
# timeout, short enough to keep the suite usable.
TIMEOUT = 15
GRACE = 15


@pytest.fixture(scope="session")
def runtime() -> DockerContainerRuntime:
    """Fail fast, and legibly, when the suite is run without a usable daemon or image."""
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
def runner(runtime: DockerContainerRuntime) -> PytestRunner:
    sandbox_config = build_evaluation_sandbox_config(
        SandboxConfig(timeout_seconds=TIMEOUT, startup_grace_seconds=GRACE),
        EvaluationConfig(timeout_seconds=TIMEOUT, startup_grace_seconds=GRACE),
    )
    return PytestRunner(SandboxExecutor(config=sandbox_config, runtime=runtime))


@pytest.fixture(autouse=True)
def no_containers_left_behind(runtime: DockerContainerRuntime):
    """Spec section 76's guarantee, applied here too: no test may leave an abandoned
    sandbox container behind."""
    yield
    lingering = runtime.list_sandbox_containers()
    assert lingering == [], f"sandbox containers left behind: {lingering}"


def make_problem(**overrides: Any) -> Problem:
    fields: dict[str, Any] = dict(
        id="p001",
        prompt="Return the sum of the even integers in a list.",
        signature="def sum_even(numbers):",
        entry_point="sum_even",
        category="lists",
        difficulty="easy",
        reference_solution=(
            "def sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)\n"
        ),
        tests=(
            TestCase(id="t001", input={"numbers": [1, 2, 3, 4]}, expected=6),
            TestCase(id="t002", input={"numbers": []}, expected=0),
            TestCase(id="t003", input={"numbers": [1, 3, 5]}, expected=0),
        ),
    )
    fields.update(overrides)
    return Problem(**fields)


def make_candidate(code: str, **overrides: Any) -> Candidate:
    fields: dict[str, Any] = dict(
        candidate_id="p001_c001",
        problem_id="p001",
        run_id=CANDIDATE_RUN_ID,
        generation_index=1,
        strategy="normal",
        model="mock/deterministic-coder",
        provider="mock",
        prompt_version="v1",
        prompt="Solve it.",
        raw_output=f"```python\n{code}```",
        code=code,
        extraction_format="python_fence",
        syntax_valid=True,
        function_name_valid=True,
        generation_config={},
        created_at="2026-08-17T12:00:00Z",
    )
    fields.update(overrides)
    return Candidate.create(**fields)


def evaluate_one(
    runner: PytestRunner, tmp_path: Path, candidate: Candidate, problem: Problem
) -> tuple[Any, EvaluationRepository]:
    repository = EvaluationRepository(tmp_path)
    evaluator = CandidateEvaluator(runner=runner, repository=repository)
    result = evaluator.evaluate(candidate, problem, evaluation_run_id=EVALUATION_RUN_ID)
    return result, repository


# ---------------------------------------------------------------- spec section 66 fixtures

CORRECT_CODE = "def sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)\n"
WRONG_CODE = "def sum_even(numbers):\n    return 0\n"
SYNTAX_ERROR_CODE = "def sum_even(numbers)\n    return 0\n"
RUNTIME_ERROR_CODE = "def sum_even(numbers):\n    raise ValueError('boom')\n"
INFINITE_LOOP_CODE = "def sum_even(numbers):\n    while True:\n        pass\n"
NETWORK_ATTEMPT_CODE = (
    "import socket\n"
    "def sum_even(numbers):\n"
    "    socket.create_connection(('8.8.8.8', 53), timeout=3)\n"
    "    return 0\n"
)


# -------------------------------------------------------------------- spec section 83: passed


def test_known_good_candidate_passes_every_test(runner, tmp_path):
    problem = make_problem()
    candidate = make_candidate(CORRECT_CODE)

    result, repository = evaluate_one(runner, tmp_path, candidate, problem)

    assert result.status == "passed"
    assert result.tests_total == 3
    assert result.tests_passed == 3
    assert result.tests_failed == 0
    test_results = repository.test_results_for(candidate.candidate_id)
    assert all(t.status == "passed" for t in test_results)


# -------------------------------------------------------------------- spec section 84: wrong


def test_wrong_candidate_fails_with_the_specific_tests_identified(runner, tmp_path):
    problem = make_problem()
    candidate = make_candidate(WRONG_CODE)

    result, repository = evaluate_one(runner, tmp_path, candidate, problem)

    assert result.status == "failed"
    # WRONG_CODE always returns 0: t001 expects 6 (fails), t002/t003 both expect 0 (pass).
    assert result.tests_passed == 2
    by_id = {t.test_case_id: t.status for t in repository.test_results_for(candidate.candidate_id)}
    assert by_id["p001_t001"] == "failed"
    assert by_id["p001_t002"] == "passed"
    assert by_id["p001_t003"] == "passed"


# ------------------------------------------------------------------ spec section 85: timeout


def test_infinite_loop_times_out_with_cleanup(runtime, tmp_path):
    problem = make_problem()
    candidate = make_candidate(INFINITE_LOOP_CODE)

    sandbox_config = build_evaluation_sandbox_config(
        SandboxConfig(timeout_seconds=2, startup_grace_seconds=GRACE),
        EvaluationConfig(timeout_seconds=2, startup_grace_seconds=GRACE),
    )
    fast_runner = PytestRunner(SandboxExecutor(config=sandbox_config, runtime=runtime))

    result, _ = evaluate_one(fast_runner, tmp_path, candidate, problem)

    assert result.status == "timeout"
    assert result.timeout is True
    assert runtime.list_sandbox_containers() == []


# ------------------------------------------------------- spec section 86: docker unavailable


def test_docker_unavailable_produces_infrastructure_error(tmp_path):
    # Injects an unavailable runtime rather than stopping the real daemon.
    from python_dpo.sandbox.errors import DockerUnavailableError

    class BrokenRuntime:
        def check_available(self) -> None:
            raise DockerUnavailableError("the Docker daemon is not reachable")

        def start(self, spec):
            # Execution itself calls start() directly without a prior availability
            # check, so the failure must surface here to be caught as infrastructure
            # trouble (spec sections 79-81) instead of propagating as a raw AttributeError.
            raise DockerUnavailableError("the Docker daemon is not reachable")

    sandbox_config = build_evaluation_sandbox_config(SandboxConfig(), EvaluationConfig())
    broken_runner = PytestRunner(SandboxExecutor(config=sandbox_config, runtime=BrokenRuntime()))

    problem = make_problem()
    candidate = make_candidate(CORRECT_CODE)
    result, _ = evaluate_one(broken_runner, tmp_path, candidate, problem)

    assert result.status == "infrastructure_error"
    assert result.infrastructure_error is True
    assert result.tests_total == 0


# ----------------------------------------------------------------- spec section 87: network


def test_network_attempt_is_still_classified_correctly(runner, tmp_path):
    problem = make_problem()
    candidate = make_candidate(NETWORK_ATTEMPT_CODE)

    result, repository = evaluate_one(runner, tmp_path, candidate, problem)

    # No network access means the candidate's own connection attempt raises inside the
    # test -- a candidate-level failure, never infrastructure trouble.
    assert result.status == "failed"
    assert result.infrastructure_error is False
    assert result.tests_error > 0


# ------------------------------------------------------------- spec section 68: arithmetic


@pytest.mark.parametrize("code", [CORRECT_CODE, WRONG_CODE, SYNTAX_ERROR_CODE, RUNTIME_ERROR_CODE])
def test_result_counts_always_sum_to_the_total(runner, tmp_path, code):
    problem = make_problem()
    candidate = make_candidate(code)
    result, _ = evaluate_one(runner, tmp_path, candidate, problem)
    assert (
        result.tests_passed + result.tests_failed + result.tests_error + result.tests_skipped
        == result.tests_total
    )


# ---------------------------------------------------------------- remaining §66 candidates


def test_syntax_error_candidate_is_classified_as_syntax_error(runner, tmp_path):
    problem = make_problem()
    candidate = make_candidate(SYNTAX_ERROR_CODE)
    result, repository = evaluate_one(runner, tmp_path, candidate, problem)
    assert result.status == "syntax_error"
    assert result.syntax_error is True
    # Spec section 68: every expected test id is still accounted for, synthesized as
    # "error" since pytest never reached any of them -- tests_total is never silently 0.
    assert result.tests_total == len(problem.tests)
    assert result.tests_passed == 0
    test_results = repository.test_results_for(candidate.candidate_id)
    assert all(t.status == "error" and t.error_type == "SyntaxError" for t in test_results)


def test_runtime_exception_candidate_is_a_candidate_failure_not_infrastructure(runner, tmp_path):
    problem = make_problem()
    candidate = make_candidate(RUNTIME_ERROR_CODE)
    result, repository = evaluate_one(runner, tmp_path, candidate, problem)
    assert result.status == "failed"
    assert result.infrastructure_error is False
    assert result.tests_error == result.tests_total
    test_results = repository.test_results_for(candidate.candidate_id)
    assert all(t.error_type == "ValueError" for t in test_results)


# --------------------------------------------------------------- spec section 67: end-to-end


def test_end_to_end_evaluate_many_persists_every_result(runner, tmp_path):
    problem = make_problem()
    candidates = [
        make_candidate(CORRECT_CODE, candidate_id="p001_c001"),
        make_candidate(WRONG_CODE, candidate_id="p001_c002"),
        make_candidate(SYNTAX_ERROR_CODE, candidate_id="p001_c003"),
    ]
    repository = EvaluationRepository(tmp_path)
    evaluator = CandidateEvaluator(runner=runner, repository=repository)

    summary = evaluator.evaluate_many(
        candidates, {problem.id: problem}, evaluation_run_id=EVALUATION_RUN_ID
    )

    assert summary.evaluated == 3
    assert summary.machinery_failed == 0
    statuses = {r.candidate_id: r.status for r in repository.load_all()}
    assert statuses == {
        "p001_c001": "passed",
        "p001_c002": "failed",
        "p001_c003": "syntax_error",
    }


# ------------------------------------------------------------- reference-solution self-check


def test_every_real_problems_generated_suite_passes_against_its_own_reference_solution(
    runner, tmp_path
):
    """The sanctioned validation use of the reference solution (spec section 9): proves the
    generator's comparison semantics still match what the real dataset was validated
    under. Without this, a subtle divergence in ``TestGenerator`` would show up as "the
    model is bad" rather than "the generator is wrong."
    """
    problems = load_problems(dataset_path(PROJECT_ROOT / "data" / "problems"))
    assert problems, "the real problem dataset must not be empty"

    repository = EvaluationRepository(tmp_path)
    evaluator = CandidateEvaluator(runner=runner, repository=repository)

    total_tests = 0
    for problem in problems:
        reference_candidate = make_candidate(
            problem.reference_solution,
            candidate_id=f"{problem.id}_c001",
            problem_id=problem.id,
        )
        result = evaluator.evaluate(
            reference_candidate, problem, evaluation_run_id=EVALUATION_RUN_ID
        )
        assert result.status == "passed", (
            f"{problem.id}: reference solution did not pass its own generated suite "
            f"({result.tests_passed}/{result.tests_total}); the generator's comparison "
            "semantics have diverged from what the dataset was validated under"
        )
        total_tests += result.tests_total

    assert total_tests == sum(len(p.tests) for p in problems)
