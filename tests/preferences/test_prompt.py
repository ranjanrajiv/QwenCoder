"""Tests for verify_prompt_lineage / build_canonical_prompt (spec 08 decision 1).

Covers the stage's load-bearing assumption: every candidate of a problem was generated
under a strategy-specific prompt, so no two candidates share a ``prompt_sha256``, and the
canonical (strategy-free) prompt must be provably derived from the same template rather
than invented.
"""

from __future__ import annotations

from typing import Any

import pytest

from python_dpo.candidates.hashing import sha256_text
from python_dpo.candidates.models import Candidate
from python_dpo.generation.prompt_builder import PROMPT_VERSION, build_canonical_prompt, build_prompt
from python_dpo.preferences.prompt import PromptLineageError, verify_prompt_lineage
from python_dpo.problems.models import Problem, TestCase

PROBLEM = Problem(
    id="p001",
    prompt="Write a function that returns the sum of all even integers in a list.",
    signature="def sum_even(numbers):",
    entry_point="sum_even",
    category="lists",
    difficulty="easy",
    reference_solution="def sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)",
    tests=(TestCase(id="t1", input={"numbers": [1, 2, 3, 4]}, expected=6),),
)

OTHER_PROBLEM = Problem(
    id="p002",
    prompt="Write a function that reverses a string.",
    signature="def reverse_str(value):",
    entry_point="reverse_str",
    category="strings",
    difficulty="easy",
    reference_solution="def reverse_str(value):\n    return value[::-1]",
    tests=(TestCase(id="t1", input={"value": "ab"}, expected="ba"),),
)


def make_candidate(*, strategy: str = "normal", problem: Problem = PROBLEM, **overrides: Any) -> Candidate:
    prompt = overrides.pop("prompt", None) or build_prompt(problem, strategy)
    code = overrides.pop("code", "def sum_even(numbers):\n    return 0")
    fields: dict[str, Any] = {
        "candidate_id": f"{problem.id}_c001",
        "problem_id": problem.id,
        "run_id": "run_20260817_055411",
        "generation_index": 1,
        "strategy": strategy,
        "model": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "provider": "transformers",
        "prompt_version": PROMPT_VERSION,
        "prompt": prompt,
        "raw_output": f"```python\n{code}\n```",
        "code": code,
        "extraction_format": "python_fence",
        "syntax_valid": True,
        "function_name_valid": True,
        "generation_config": {},
        "created_at": "2026-08-17T05:54:18Z",
    }
    fields.update(overrides)
    return Candidate.create(**fields)


# ------------------------------------------------------------------------- canonical prompt


def test_canonical_prompt_is_identical_across_strategies():
    # build_canonical_prompt takes no strategy argument at all, so by construction it
    # cannot vary by strategy; this asserts it is also stable across repeated calls.
    prompts = {build_canonical_prompt(PROBLEM) for _ in range(3)}
    assert len(prompts) == 1

    from python_dpo.generation.strategies import STRATEGIES

    raw_prompts = {build_prompt(PROBLEM, strategy) for strategy in STRATEGIES}
    assert len(raw_prompts) == len(STRATEGIES)  # every strategy's raw prompt is distinct


def test_canonical_prompt_differs_across_problems():
    assert build_canonical_prompt(PROBLEM) != build_canonical_prompt(OTHER_PROBLEM)


def test_canonical_prompt_has_no_strategy_block_or_label_leakage():
    canonical = build_canonical_prompt(PROBLEM)
    assert "Strategy:" not in canonical
    for label in ("CORRECT", "INCORRECT", "CHOSEN", "REJECTED"):
        assert label not in canonical


# --------------------------------------------------------------------------- lineage checks


def test_verify_prompt_lineage_passes_for_real_generation_prompts():
    candidates = [make_candidate(strategy=s) for s in ("normal", "alternative", "optimized")]
    canonical_prompt, canonical_prompt_sha256 = verify_prompt_lineage(PROBLEM, candidates)
    assert canonical_prompt == build_canonical_prompt(PROBLEM)
    assert canonical_prompt_sha256 == sha256_text(canonical_prompt)


def test_verify_prompt_lineage_empty_candidate_list_still_returns_canonical_prompt():
    canonical_prompt, canonical_prompt_sha256 = verify_prompt_lineage(PROBLEM, [])
    assert canonical_prompt == build_canonical_prompt(PROBLEM)
    assert canonical_prompt_sha256 == sha256_text(canonical_prompt)


def test_verify_prompt_lineage_rejects_a_candidate_from_a_different_problem():
    candidate = make_candidate(problem=OTHER_PROBLEM)
    with pytest.raises(PromptLineageError):
        verify_prompt_lineage(PROBLEM, [candidate])


def test_verify_prompt_lineage_rejects_a_stale_prompt_version():
    candidate = make_candidate(prompt_version="v0")
    with pytest.raises(PromptLineageError):
        verify_prompt_lineage(PROBLEM, [candidate])


def test_verify_prompt_lineage_rejects_a_tampered_prompt_hash():
    # A candidate whose stored prompt text does not match what the current template
    # produces for its own strategy (e.g. after a template change) must fail loudly,
    # never be silently trusted (spec section 42).
    candidate = make_candidate(strategy="normal")
    # Candidate.create recomputes the hash from `prompt`; build one whose `prompt` field
    # was altered post-hoc by constructing with mismatched prompt/hash directly.
    tampered = Candidate(
        candidate_id=candidate.candidate_id,
        problem_id=candidate.problem_id,
        run_id=candidate.run_id,
        generation_index=candidate.generation_index,
        strategy="alternative",  # claims a different strategy than the prompt was built for
        model=candidate.model,
        provider=candidate.provider,
        prompt_version=candidate.prompt_version,
        prompt=candidate.prompt,  # still the "normal"-strategy prompt text
        raw_output=candidate.raw_output,
        code=candidate.code,
        extraction_format=candidate.extraction_format,
        syntax_valid=candidate.syntax_valid,
        function_name_valid=candidate.function_name_valid,
        generation_config=candidate.generation_config,
        created_at=candidate.created_at,
        code_sha256=candidate.code_sha256,
        prompt_sha256=candidate.prompt_sha256,  # sha256 of the "normal" prompt
        raw_output_sha256=candidate.raw_output_sha256,
    )
    with pytest.raises(PromptLineageError):
        verify_prompt_lineage(PROBLEM, [tampered])
