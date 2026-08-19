"""Package verification: load, generate, execute (spec 12 section 38).

Packaging is not allowed to declare success on the strength of a training loss curve
alone -- Stage 9 learned that lesson for the adapter reload (``training.verify``), and
spec section 38 asks for the same discipline one layer up: load the *packaged* artifact,
generate Python for a fixed prompt, and require that code to actually pass tests before
the package is registered. There is no ``--skip-verification`` escape hatch (plan Phase 3
decision) -- a verification failure is always fatal to whoever called this.

Generated code is executed only through :class:`~python_dpo.evaluation.CandidateEvaluator`
over the Stage 5/6 sandbox (CLAUDE.md's Security rule); this module never calls ``exec``
or a shell on model output. It stays torch-free by taking ``generate`` and ``evaluator``
as already-constructed collaborators rather than importing
:class:`~python_dpo.model_evaluation.runners.AdapterModelRunner` itself -- assembly of
existing parts, not new inference or execution code (plan's reuse finding).
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..candidates.models import Candidate
from ..evaluation.executor import CandidateEvaluator
from ..generation.code_extractor import extract_code
from ..problems.models import Problem, TestCase
from .errors import VerificationError

# A fixed, trivial task -- deliberately not one of the dataset's problems (spec section
# 95 forbids reading a training/benchmark problem's result as a quality signal here).
# This only proves the packaged artifact can generate working Python at all.
VERIFICATION_PROBLEM = Problem(
    id="_packaging_verification",
    prompt="Write a Python function `add_two(a, b)` that returns the sum of `a` and `b`.",
    signature="def add_two(a: int, b: int) -> int:",
    entry_point="add_two",
    category="edge_cases",
    difficulty="easy",
    reference_solution="def add_two(a: int, b: int) -> int:\n    return a + b\n",
    tests=(
        TestCase(id="case_1", input={"a": 2, "b": 3}, expected=5),
        TestCase(id="case_2", input={"a": -1, "b": 1}, expected=0),
    ),
    source="manual",
)


def _is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class VerificationResult:
    """The outcome of one packaging verification (spec section 38)."""

    ok: bool
    candidate_id: str
    raw_output: str
    extracted_code: str | None
    tests_passed: int
    tests_total: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "candidate_id": self.candidate_id,
            "raw_output": self.raw_output,
            "extracted_code": self.extracted_code,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "detail": self.detail,
        }


def verify_package(
    *,
    package_id: str,
    model_name: str,
    generate: Callable[[str], str],
    evaluator: CandidateEvaluator,
    evaluation_run_id: str,
    problem: Problem = VERIFICATION_PROBLEM,
    seed: int = 42,
) -> VerificationResult:
    """Generate for ``problem.prompt`` and require the result to pass its tests.

    Raises :class:`VerificationError` -- never returns a failing result -- so a caller
    cannot accidentally register or ship a package that did not actually verify.
    """
    raw_output = generate(problem.prompt)

    extraction = extract_code(raw_output)
    if not extraction.extracted or not extraction.code:
        raise VerificationError(
            f"packaging verification for {package_id!r}: the model produced no "
            f"extractable Python code ({extraction.error})"
        )

    candidate = Candidate.create(
        candidate_id=f"{problem.id}_c{package_id}",
        problem_id=problem.id,
        run_id=evaluation_run_id,
        generation_index=1,
        strategy="packaging_verification",
        model=model_name,
        provider="local",
        prompt_version="v1",
        prompt=problem.prompt,
        raw_output=raw_output,
        code=extraction.code,
        extraction_format=extraction.source_format,
        syntax_valid=_is_syntax_valid(extraction.code),
        function_name_valid=problem.entry_point in extraction.code,
        generation_config={"seed": seed},
        created_at=_utc_now_iso(),
    )

    result = evaluator.evaluate(candidate, problem, evaluation_run_id=evaluation_run_id)

    verification = VerificationResult(
        ok=result.status == "passed",
        candidate_id=candidate.candidate_id,
        raw_output=raw_output,
        extracted_code=extraction.code,
        tests_passed=result.tests_passed,
        tests_total=result.tests_total,
        detail=f"status={result.status}",
    )
    if not verification.ok:
        raise VerificationError(
            f"packaging verification for {package_id!r} failed: {verification.tests_passed}"
            f"/{verification.tests_total} test(s) passed (status={result.status})"
        )
    return verification


__all__ = ["VERIFICATION_PROBLEM", "VerificationResult", "verify_package"]
