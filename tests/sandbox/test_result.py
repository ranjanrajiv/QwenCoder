"""Tests for ExecutionResult and the pure classify() function (spec 05 sections 7, 8)."""

from __future__ import annotations

from typing import Any

import pytest

from python_dpo.sandbox import EXECUTION_STATUSES, ExecutionResult, ExecutionResultError, classify
from python_dpo.sandbox.result import looks_like_compile_error, signal_from_exit_code

# Real CPython output shapes — the classifier keys off these, so the fixtures must be
# faithful rather than approximate.
SYNTAX_STDERR = '''  File "/workspace/candidate.py", line 1
    def broken(:
               ^
SyntaxError: invalid syntax
'''

INDENTATION_STDERR = '''  File "/workspace/candidate.py", line 2
    return 1
    ^
IndentationError: expected an indented block
'''

RUNTIME_STDERR = '''Traceback (most recent call last):
  File "/workspace/candidate.py", line 1, in <module>
    raise ValueError("test error")
ValueError: test error
'''

# A program that compiles fine and then chooses to raise SyntaxError at runtime. The naive
# "does stderr mention SyntaxError" check would misclassify this.
RAISED_SYNTAX_ERROR_STDERR = '''Traceback (most recent call last):
  File "/workspace/candidate.py", line 1, in <module>
    raise SyntaxError("deliberate")
SyntaxError: deliberate
'''


def make_result(**overrides: Any) -> ExecutionResult:
    fields: dict[str, Any] = {
        "status": "success",
        "exit_code": 0,
        "stdout": "hello\n",
        "stderr": "",
        "duration_ms": 37,
    }
    fields.update(overrides)
    return ExecutionResult(**fields)


# ------------------------------------------------------------------------------- schema


def test_status_set_matches_the_specification():
    assert EXECUTION_STATUSES == {
        "success",
        "syntax_error",
        "runtime_error",
        "timeout",
        "resource_exceeded",
        "infrastructure_error",
        "cancelled",
    }


def test_valid_result_round_trips_through_dict():
    result = make_result()
    assert ExecutionResult.from_dict(result.to_dict()) == result


def test_created_at_is_stamped_when_absent():
    assert make_result().created_at


def test_unknown_status_is_rejected():
    with pytest.raises(ExecutionResultError, match="status"):
        make_result(status="exploded")


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"exit_code": "0"}, "exit_code"),
        ({"stdout": None}, "stdout"),
        ({"duration_ms": -1}, "duration_ms"),
        ({"duration_ms": True}, "duration_ms"),
        ({"timed_out": "yes"}, "timed_out"),
        ({"sandbox_config": []}, "sandbox_config"),
    ],
)
def test_invalid_results_are_rejected(overrides, match):
    with pytest.raises(ExecutionResultError, match=match):
        make_result(**overrides)


def test_timed_out_must_agree_with_status():
    # A result claiming it timed out while reporting some other status would be incoherent
    # for the later evaluation stage.
    with pytest.raises(ExecutionResultError, match="timed_out"):
        make_result(status="runtime_error", timed_out=True)
    assert make_result(status="timeout", exit_code=137, timed_out=True).timed_out is True


def test_from_dict_rejects_unknown_and_missing_fields():
    payload = make_result().to_dict()
    with pytest.raises(ExecutionResultError, match="unknown field"):
        ExecutionResult.from_dict({**payload, "gpu": 1})
    del payload["stdout"]
    with pytest.raises(ExecutionResultError, match="missing required field"):
        ExecutionResult.from_dict(payload)


def test_candidate_and_infrastructure_outcomes_are_distinguished():
    # Spec section 81: a candidate must never be judged badly because Docker failed.
    assert make_result(status="runtime_error", exit_code=1).is_candidate_outcome
    assert not make_result(status="runtime_error", exit_code=1).is_infrastructure_failure

    infra = ExecutionResult.infrastructure_failure(
        error_type="DockerUnavailableError", error_message="daemon down"
    )
    assert infra.is_infrastructure_failure
    assert not infra.is_candidate_outcome
    assert infra.exit_code is None


# ------------------------------------------------------------- compile vs runtime errors


def test_compile_error_detection():
    assert looks_like_compile_error(SYNTAX_STDERR)
    assert looks_like_compile_error(INDENTATION_STDERR)
    assert not looks_like_compile_error(RUNTIME_STDERR)
    assert not looks_like_compile_error("")


def test_raised_syntax_error_is_not_a_compile_error():
    # The program compiled fine and then chose to raise. CPython prints the Traceback
    # header for that, and never for a genuine compile failure.
    assert not looks_like_compile_error(RAISED_SYNTAX_ERROR_STDERR)


# ------------------------------------------------------------------------------ classify


def test_zero_exit_is_success():
    assert classify(exit_code=0, stderr="") == "success"


def test_syntax_error_is_classified_from_stderr():
    assert classify(exit_code=1, stderr=SYNTAX_STDERR) == "syntax_error"


def test_runtime_error_is_classified_from_stderr():
    assert classify(exit_code=1, stderr=RUNTIME_STDERR) == "runtime_error"


def test_deliberately_raised_syntax_error_is_a_runtime_error():
    assert classify(exit_code=1, stderr=RAISED_SYNTAX_ERROR_STDERR) == "runtime_error"


def test_timeout_outranks_the_exit_code():
    assert classify(exit_code=137, stderr="", timed_out=True) == "timeout"


def test_oom_is_resource_exceeded():
    assert classify(exit_code=137, stderr="", oom_killed=True) == "resource_exceeded"


def test_output_limit_is_resource_exceeded():
    assert classify(exit_code=0, stderr="", output_limit_exceeded=True) == "resource_exceeded"


def test_resource_limits_outrank_a_timeout():
    # An OOM kill is a more specific, more actionable explanation than "it ran too long".
    assert (
        classify(exit_code=137, stderr="", timed_out=True, oom_killed=True)
        == "resource_exceeded"
    )


@pytest.mark.parametrize("exit_code", [125, 126, 127])
def test_docker_cli_exit_codes_are_infrastructure_failures(exit_code):
    # `docker run` uses these for its own failures; the candidate never ran.
    assert classify(exit_code=exit_code, stderr="") == "infrastructure_error"


def test_container_that_never_started_is_an_infrastructure_failure():
    assert classify(exit_code=None, stderr="", container_started=False) == "infrastructure_error"


def test_missing_exit_code_is_an_infrastructure_failure():
    assert classify(exit_code=None, stderr="") == "infrastructure_error"


def test_arbitrary_nonzero_exit_is_a_runtime_error():
    assert classify(exit_code=3, stderr="something went wrong") == "runtime_error"


# -------------------------------------------------------------------------------- signals


@pytest.mark.parametrize(
    "exit_code, expected",
    [(137, 9), (139, 11), (143, 15), (0, None), (1, None), (128, None), (None, None)],
)
def test_signal_is_recovered_from_the_exit_code(exit_code, expected):
    # Spec section 51: signal termination is recorded, never hidden.
    assert signal_from_exit_code(exit_code) == expected
