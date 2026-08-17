"""Tests for prompt construction, strategies, code extraction, and static validation."""

from __future__ import annotations

import pytest

from python_dpo.generation import (
    PROMPT_VERSION,
    STRATEGIES,
    STRATEGY_INSTRUCTIONS,
    StrategyError,
    build_prompt,
    check_function_name,
    check_syntax,
    extract_code,
    instruction_for,
    resolve_strategies,
)
from python_dpo.problems import Problem, TestCase

CODE = "def foo(x):\n    return x + 1"


@pytest.fixture
def problem() -> Problem:
    return Problem(
        id="p999",
        prompt="Add one to a number.",
        signature="def add_one(value):",
        entry_point="add_one",
        category="lists",
        difficulty="easy",
        reference_solution="def add_one(value):\n    return value + 1\n",
        tests=(TestCase(id="t001", input={"value": 1}, expected=2),),
    )


# -------------------------------------------------------------------------- strategies


def test_five_strategies_exist_in_specification_order():
    assert STRATEGIES == (
        "normal",
        "straightforward",
        "edge_case_focused",
        "alternative",
        "optimized",
    )
    assert set(STRATEGY_INSTRUCTIONS) == set(STRATEGIES)
    assert all(STRATEGY_INSTRUCTIONS[name].strip() for name in STRATEGIES)


def test_each_strategy_has_a_distinct_instruction():
    assert len(set(STRATEGY_INSTRUCTIONS.values())) == len(STRATEGIES)


def test_default_five_candidates_use_one_strategy_each():
    assert resolve_strategies(STRATEGIES, 5) == STRATEGIES


def test_strategy_override_replaces_the_configured_list():
    assert resolve_strategies(STRATEGIES, 3, override=["normal"]) == (
        "normal",
        "normal",
        "normal",
    )


def test_counts_beyond_the_strategy_list_cycle():
    assert resolve_strategies(STRATEGIES, 7)[5:] == ("normal", "straightforward")


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ((STRATEGIES, 0), {}),
        ((STRATEGIES, -1), {}),
        (((), 1), {}),
        ((STRATEGIES, 1), {"override": ["nonsense"]}),
        ((["nonsense"], 1), {}),
    ],
)
def test_resolve_strategies_rejects_bad_input(args, kwargs):
    with pytest.raises(StrategyError):
        resolve_strategies(*args, **kwargs)


def test_instruction_for_rejects_unknown_strategy():
    with pytest.raises(StrategyError, match="unknown strategy"):
        instruction_for("creative")


# ----------------------------------------------------------------------- prompt builder


def test_prompt_contains_problem_signature_strategy_and_output_rules(problem):
    prompt = build_prompt(problem, "edge_case_focused")
    assert problem.prompt in prompt
    assert problem.signature in prompt
    assert STRATEGY_INSTRUCTIONS["edge_case_focused"] in prompt
    assert "Return only the implementation." in prompt
    assert "Do not use eval()." in prompt
    assert "Do not perform network operations." in prompt


def test_prompt_is_deterministic(problem):
    assert build_prompt(problem, "normal") == build_prompt(problem, "normal")


def test_prompt_differs_per_strategy(problem):
    prompts = {build_prompt(problem, name) for name in STRATEGIES}
    assert len(prompts) == len(STRATEGIES)


def test_prompt_version_is_declared():
    assert PROMPT_VERSION == "v1"


# ---------------------------------------------------------------------- code extractor


def test_extracts_python_fence():
    result = extract_code(f"```python\n{CODE}\n```")
    assert result.extracted is True
    assert result.source_format == "python_fence"
    assert result.code == CODE


def test_extracts_generic_fence():
    result = extract_code(f"```\n{CODE}\n```")
    assert result.extracted is True
    assert result.source_format == "generic_fence"
    assert result.code == CODE


def test_extracts_plain_code():
    result = extract_code(CODE)
    assert result.extracted is True
    assert result.source_format == "plain"
    assert result.code == CODE


def test_extracts_from_explanatory_prefix():
    raw = f"Here is the implementation:\n\n```python\n{CODE}\n```\n\nHope this helps!"
    result = extract_code(raw)
    assert result.extracted is True
    assert result.source_format == "python_fence"
    assert result.code == CODE


def test_prefers_the_python_fence_over_a_later_generic_one():
    raw = f"```python\n{CODE}\n```\n\n```\nnot code\n```"
    assert extract_code(raw).code == CODE


def test_preserves_internal_formatting_exactly():
    code = "def foo():\n    if True:\n\n        return    1  # spaced"
    assert extract_code(f"```python\n{code}\n```").code == code


@pytest.mark.parametrize("raw", ["", "   \n\t ", "I am unable to help with that."])
def test_extraction_fails_without_code(raw):
    result = extract_code(raw)
    assert result.extracted is False
    assert result.code is None
    assert result.source_format == "unknown"
    assert result.error == "No Python code detected"


def test_unterminated_fence_is_not_repaired():
    # Guessing where the block ends would mean the stored candidate is no longer what
    # the model produced.
    result = extract_code(f"```python\n{CODE}")
    assert result.extracted is False
    assert result.source_format == "unknown"


def test_generic_fence_without_code_is_rejected():
    assert extract_code("```\njust some prose in a fence\n```").extracted is False


def test_extraction_does_not_require_valid_syntax():
    broken = "def foo(:\n    return 1"
    result = extract_code(f"```python\n{broken}\n```")
    assert result.extracted is True
    assert result.code == broken


# ------------------------------------------------------------------- static validation


def test_valid_python_passes_the_syntax_check():
    check = check_syntax(CODE)
    assert check.valid is True
    assert check.error_message is None


def test_invalid_python_fails_with_a_message():
    check = check_syntax("def foo(:\n    return 1")
    assert check.valid is False
    assert check.error_message
    assert "line" in check.error_message


def test_syntax_check_survives_null_bytes():
    assert check_syntax("def foo():\x00\n    pass").valid is False


def test_function_name_matches():
    assert check_function_name("def add_one(v):\n    return v + 1", "add_one") is True


def test_function_name_mismatch_is_reported():
    assert check_function_name("def solution(v):\n    return v + 1", "add_one") is False


def test_missing_function_is_reported():
    assert check_function_name("value = 1", "add_one") is False


def test_async_function_counts():
    assert check_function_name("async def fetch(x):\n    return x", "fetch") is True


def test_nested_function_counts():
    code = "def outer():\n    def add_one(v):\n        return v + 1\n    return add_one"
    assert check_function_name(code, "add_one") is True


def test_unparseable_code_has_no_function_name():
    assert check_function_name("def foo(:", "foo") is False
