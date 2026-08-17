"""Tests for the pytest result parser and its conftest.py counterpart (spec 06 §65).

Fixture stdout strings below are lightly abbreviated but structurally faithful to what a
real pytest run produces via the generated conftest — including the specific bug this
module exists to defend against: pytest's "-q" progress character (".", "F", "s") is
written to the same line as our print(), with no separating newline, so the nonce is not
reliably the first character of the line. See ``test_nonce_survives_progress_dot_smashing``.

A handful of tests run a real, local pytest subprocess (``_run_local_pytest``) against the
rendered conftest to verify the wire format against reality rather than hand-typed
fixtures alone. This runs pytest **on the host**, which is not a CLAUDE.md violation: the
candidate/test sources involved are trusted, hand-authored fixtures reviewed as part of
this file, never Qwen-generated code. The real evaluation path in ``src/`` always executes
generated candidates and tests inside the Stage 5 sandbox, never on the host.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys

import pytest

from python_dpo.evaluation.errors import ResultParseError
from python_dpo.evaluation.result_parser import (
    CONFTEST_FILENAME,
    CONFTEST_TEMPLATE,
    PytestResultParser,
    new_nonce,
    reconcile,
    render_conftest,
)

NONCE = "0bede1ab55ac7e6df8b5e875359f8ea6"


def line(kind: str, **fields) -> str:
    return f"{NONCE} " + json.dumps({"kind": kind, **fields})


def make_test_event(test_case_id: str, status: str, **fields) -> str:
    payload = {
        "duration_ms": 1,
        "error_type": None,
        "error_message": None,
        "stdout": "",
        "stderr": "",
    }
    payload.update(fields)
    return line("test", test_case_id=test_case_id, status=status, **payload)


# ------------------------------------------------------------------------- the template


def test_rendered_conftest_is_syntactically_valid():
    ast.parse(render_conftest(new_nonce()))


def test_rendered_conftest_embeds_the_nonce():
    nonce = new_nonce()
    assert repr(nonce) in render_conftest(nonce)


def test_two_renders_use_different_nonces():
    assert new_nonce() != new_nonce()


def test_template_uses_no_eval_or_exec():
    # Spec section 44: the reporting plugin is trusted, generated code, but it must still
    # never touch eval/exec even so.
    assert "eval(" not in CONFTEST_TEMPLATE
    assert "exec(" not in CONFTEST_TEMPLATE


# ------------------------------------------------------------------------------ §65.1 all pass


def test_all_tests_pass():
    stdout = "\n".join(
        [
            make_test_event("p001_t001", "passed"),
            make_test_event("p001_t002", "passed"),
            line("session", testscollected=2, testsfailed=0, exitstatus=0),
        ]
    )
    parsed = PytestResultParser().parse(stdout, NONCE)
    assert [r.status for r in parsed.test_results] == ["passed", "passed"]
    assert parsed.exitstatus == 0
    assert parsed.collection_error is None


# ------------------------------------------------------------------------ §65.2 partial failure


def test_partial_failure():
    stdout = "\n".join(
        [
            make_test_event("p001_t001", "passed"),
            make_test_event("p001_t002", "failed", error_type="AssertionError", error_message="assert 6 == 999"),
            make_test_event("p001_t003", "passed"),
            line("session", testscollected=3, testsfailed=1, exitstatus=1),
        ]
    )
    parsed = PytestResultParser().parse(stdout, NONCE)
    statuses = {r.test_case_id: r.status for r in parsed.test_results}
    assert statuses == {"p001_t001": "passed", "p001_t002": "failed", "p001_t003": "passed"}
    failed = next(r for r in parsed.test_results if r.status == "failed")
    assert failed.error_type == "AssertionError"


# ------------------------------------------------------------------------- §65.3 runtime error


def test_runtime_error_is_distinguished_from_a_wrong_answer():
    # Spec section 27: a candidate exception during a test is "error", not "failed".
    stdout = "\n".join(
        [
            make_test_event("p001_t001", "error", error_type="ValueError", error_message="boom"),
            line("session", testscollected=1, testsfailed=1, exitstatus=1),
        ]
    )
    parsed = PytestResultParser().parse(stdout, NONCE)
    assert parsed.test_results[0].status == "error"
    assert parsed.test_results[0].error_type == "ValueError"


# -------------------------------------------------------------------------- §65.4 syntax error


def test_collection_error_is_captured():
    stdout = "\n".join(
        [
            line(
                "collect_error",
                nodeid="test_candidate.py",
                message='File "candidate.py", line 1\n    def broken(:\nE   SyntaxError: invalid syntax',
            ),
            line("session", testscollected=0, testsfailed=1, exitstatus=2),
        ]
    )
    parsed = PytestResultParser().parse(stdout, NONCE)
    assert parsed.collection_error is not None
    assert parsed.is_syntax_error is True
    assert parsed.test_results == ()


def test_a_runtime_raised_syntax_error_is_not_a_collection_error():
    # Mirrors the sandbox-level guard: a *candidate exception* named SyntaxError, raised
    # at runtime after the file compiled fine, is a runtime "error", not a collection
    # failure — there is no collect_error event for it at all.
    stdout = "\n".join(
        [
            make_test_event("p001_t001", "error", error_type="SyntaxError", error_message="deliberate"),
            line("session", testscollected=1, testsfailed=1, exitstatus=1),
        ]
    )
    parsed = PytestResultParser().parse(stdout, NONCE)
    assert parsed.collection_error is None
    assert parsed.test_results[0].error_type == "SyntaxError"


# ---------------------------------------------------------------------------- §65.5 timeout


def test_timeout_leaves_a_partial_result_set():
    # A sandbox timeout kills the container mid-run; only a prefix of tests reported.
    stdout = "\n".join(
        [
            make_test_event("p001_t001", "passed"),
            # process killed here — no session event, no t002/t003
        ]
    )
    parsed = PytestResultParser().parse(stdout, NONCE)
    assert [r.test_case_id for r in parsed.test_results] == ["p001_t001"]
    assert parsed.exitstatus is None


# --------------------------------------------------------------------------- §65.6 skipped


def test_skipped_tests():
    stdout = "\n".join(
        [
            *[make_test_event(f"p001_t00{i}", "passed") for i in range(1, 8)],
            make_test_event("p001_t008", "skipped", error_message="Skipped: not applicable"),
            line("session", testscollected=8, testsfailed=0, exitstatus=0),
        ]
    )
    parsed = PytestResultParser().parse(stdout, NONCE)
    statuses = [r.status for r in parsed.test_results]
    assert statuses.count("passed") == 7
    assert statuses.count("skipped") == 1


# --------------------------------------------------- nonce robustness (the real bug found)


def test_nonce_survives_progress_dot_smashing():
    # pytest's own "-q" terminal reporter writes a bare "." / "F" / "s" character with no
    # trailing newline immediately before our print() runs, so real captured stdout looks
    # like this — the nonce is NOT at column 0. This is not a hypothetical: it reproduced
    # against a real pytest run during implementation.
    stdout = (
        f".{NONCE} " + json.dumps({"kind": "test", "test_case_id": "p001_t001", "status": "passed",
                                     "duration_ms": 1, "error_type": None, "error_message": None,
                                     "stdout": "", "stderr": ""}) + "\n"
        f"F{NONCE} " + json.dumps({"kind": "test", "test_case_id": "p001_t002", "status": "failed",
                                     "duration_ms": 1, "error_type": "AssertionError", "error_message": "x",
                                     "stdout": "", "stderr": ""})
    )
    parsed = PytestResultParser().parse(stdout, NONCE)
    assert [r.test_case_id for r in parsed.test_results] == ["p001_t001", "p001_t002"]


def test_ordinary_candidate_output_is_ignored():
    stdout = "\n".join(
        [
            "hello from the candidate",
            "42",
            make_test_event("p001_t001", "passed"),
        ]
    )
    parsed = PytestResultParser().parse(stdout, NONCE)
    assert len(parsed.test_results) == 1


def test_malformed_nonce_line_is_an_error_not_a_silent_skip():
    stdout = f"{NONCE} not valid json"
    with pytest.raises(ResultParseError, match="malformed"):
        PytestResultParser().parse(stdout, NONCE)


def test_nonce_line_missing_kind_is_an_error():
    stdout = f'{NONCE} {{"test_case_id": "p001_t001"}}'
    with pytest.raises(ResultParseError, match="kind"):
        PytestResultParser().parse(stdout, NONCE)


def test_a_different_jobs_nonce_is_not_matched():
    other_nonce = "f" * 32
    stdout = make_test_event("p001_t001", "passed")  # tagged with NONCE
    parsed = PytestResultParser().parse(stdout, other_nonce)
    assert parsed.test_results == ()


# ---------------------------------------------------------------------------- reconcile


def test_reconcile_fills_missing_ids_as_error():
    stdout = make_test_event("p001_t001", "passed")
    parsed = PytestResultParser().parse(stdout, NONCE)

    reconciled = reconcile(
        parsed.test_results,
        ["p001_t001", "p001_t002"],
        evaluation_run_id="eval_x",
        candidate_run_id="run_x",
        candidate_id="p001_c001",
        problem_id="p001",
        missing_error_type="Timeout",
        missing_error_message="sandbox timed out before this test ran",
    )
    assert [r.status for r in reconciled] == ["passed", "error"]
    assert reconciled[1].error_type == "Timeout"


def test_reconcile_stamps_full_provenance():
    stdout = make_test_event("p001_t001", "passed")
    parsed = PytestResultParser().parse(stdout, NONCE)
    reconciled = reconcile(
        parsed.test_results,
        ["p001_t001"],
        evaluation_run_id="eval_x",
        candidate_run_id="run_x",
        candidate_id="p001_c001",
        problem_id="p001",
        missing_error_type="Timeout",
        missing_error_message="x",
    )
    result = reconciled[0]
    assert result.evaluation_run_id == "eval_x"
    assert result.candidate_run_id == "run_x"
    assert result.candidate_id == "p001_c001"
    assert result.problem_id == "p001"


def test_reconcile_preserves_the_problems_declared_test_order():
    # Emitted out of order; reconcile must restore the expected order regardless.
    stdout = "\n".join([make_test_event("p001_t002", "passed"), make_test_event("p001_t001", "passed")])
    parsed = PytestResultParser().parse(stdout, NONCE)
    reconciled = reconcile(
        parsed.test_results,
        ["p001_t001", "p001_t002"],
        evaluation_run_id="e",
        candidate_run_id="r",
        candidate_id="c",
        problem_id="p001",
        missing_error_type="Timeout",
        missing_error_message="x",
    )
    assert [r.test_case_id for r in reconciled] == ["p001_t001", "p001_t002"]


# --------------------------------------------- real pytest, no Docker (local subprocess)


def _run_local_pytest(tmp_path, candidate_source: str, test_source: str) -> tuple[str, str]:
    """Run the rendered conftest against a real, local pytest — no Docker involved, just
    an actual pytest process so the wire format is verified against reality rather than
    hand-typed fixtures. Returns (stdout, nonce).
    """
    nonce = new_nonce()
    (tmp_path / CONFTEST_FILENAME).write_text(render_conftest(nonce), encoding="utf-8")
    (tmp_path / "candidate.py").write_text(candidate_source, encoding="utf-8")
    (tmp_path / "test_candidate.py").write_text(test_source, encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_candidate.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout, nonce


def test_real_pytest_mixed_outcomes_are_classified_correctly(tmp_path):
    candidate_source = "def sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)\n"
    test_source = (
        "import candidate\n"
        "\n"
        "def test_p001_t001():\n"
        "    assert candidate.sum_even([1, 2, 3, 4]) == 6\n"
        "\n"
        "def test_p001_t002():\n"
        "    assert candidate.sum_even([1, 2, 3, 4]) == 999\n"
        "\n"
        "def test_p001_t003():\n"
        "    raise ValueError('boom')\n"
    )
    stdout, nonce = _run_local_pytest(tmp_path, candidate_source, test_source)
    parsed = PytestResultParser().parse(stdout, nonce)
    statuses = {r.test_case_id: r.status for r in parsed.test_results}
    assert statuses == {"p001_t001": "passed", "p001_t002": "failed", "p001_t003": "error"}


def test_real_pytest_did_not_raise_is_classified_as_failed_not_error(tmp_path):
    # The exact bug found during implementation: pytest.raises(...) raises its own
    # "Failed" exception (not AssertionError) when nothing is raised, which the naive
    # "AssertionError means failed, else error" rule would misclassify as a candidate
    # exception rather than a wrong answer.
    candidate_source = "def factorial(n):\n    return 1\n"
    test_source = (
        "import pytest\n"
        "import candidate\n"
        "\n"
        "def test_p006_t001():\n"
        "    with pytest.raises(Exception) as _exc_info:\n"
        "        candidate.factorial(n=-1)\n"
        "    assert type(_exc_info.value).__name__ == 'ValueError'\n"
    )
    stdout, nonce = _run_local_pytest(tmp_path, candidate_source, test_source)
    parsed = PytestResultParser().parse(stdout, nonce)
    assert parsed.test_results[0].status == "failed"


def test_real_pytest_syntax_error_is_a_collection_error(tmp_path):
    stdout, nonce = _run_local_pytest(
        tmp_path, "def broken(:\n", "import candidate\n\ndef test_p001_t001():\n    pass\n"
    )
    parsed = PytestResultParser().parse(stdout, nonce)
    assert parsed.is_syntax_error is True


def test_reconcile_never_drops_a_test_even_when_everything_is_missing():
    # A total collection failure or infrastructure-side "nothing reported" case: every
    # expected id must still appear (spec section 46's cross-check), never silently 0.
    reconciled = reconcile(
        (),
        ["p001_t001", "p001_t002", "p001_t003"],
        evaluation_run_id="e",
        candidate_run_id="r",
        candidate_id="c",
        problem_id="p001",
        missing_error_type="SyntaxError",
        missing_error_message="collection failed",
    )
    assert len(reconciled) == 3
    assert all(r.status == "error" for r in reconciled)
