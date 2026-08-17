"""Turns pytest's structured output into :class:`TestCaseResult` records.

Spec sections 31, 32: a machine-readable pytest reporting mechanism is preferred over
parsing free-form stdout, and the simplest reliable mechanism should be chosen rather than
adding a dependency purely for parsing. The workspace is read-only and the container is
removed immediately after the run, which rules out a report *file* (JUnit XML,
pytest-json-report) without an extra extraction step. Instead, :data:`CONFTEST_TEMPLATE`
— generated per job by :mod:`python_dpo.evaluation.test_generator` — implements pytest's
own hooks and prints one JSON object per event to stdout, prefixed with a per-job random
nonce. Stage 5 already captures and bounds stdout (spec 05 sections 31, 32), so no new
capture mechanism or dependency is needed, and the record schema is ours rather than a
third-party format's.

**Honestly stated: the nonce defends against accidental collision with ordinary candidate
`print()` output, not against a candidate that deliberately reads its own workspace,
finds `conftest.py`, and forges matching lines.** The defence that actually matters against
that is :func:`reconcile`'s count cross-check (spec section 46) — a forged or truncated
result set whose test ids disagree with the problem's declared set is caught, not trusted.
This is consistent with the threat model in ``docs/sandbox-security.md``: the sandbox
reduces risk, it is not a perfect boundary against code that studies its own environment.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Sequence
from dataclasses import dataclass

from ..sandbox.result import looks_like_compile_error
from .errors import ResultParseError
from .models import TestCaseResult

# 16 bytes -> 32 hex characters. Long enough that an accidental substring collision with
# candidate output is not a practical concern.
_NONCE_BYTES = 16

# A sentinel token replaced with the nonce's repr(); avoids needing to escape the many
# literal `{`/`}` dict braces in the generated source with str.format().
_NONCE_TOKEN = "__PYTHON_DPO_NONCE__"

CONFTEST_FILENAME = "conftest.py"

CONFTEST_TEMPLATE = '''\
"""Generated reporting plugin for one evaluation job.

Not hand-written: produced by TestGenerator for exactly this job and deleted with the
rest of the workspace after the sandbox run. Prints one nonce-prefixed JSON object per
pytest event to stdout, which the host parses back into structured results (spec 06
sections 31, 32).
"""
import json

import pytest

_NONCE = __PYTHON_DPO_NONCE__


def _emit(payload):
    print(_NONCE + " " + json.dumps(payload))


def _test_case_id(item):
    # Generated test functions are always named test_<test_case_id>; stripping the fixed
    # prefix recovers the dataset id pytest's own nodeid does not carry directly.
    return item.name[len("test_"):] if item.name.startswith("test_") else item.name


def pytest_collectreport(report):
    if report.failed:
        _emit({
            "kind": "collect_error",
            "nodeid": report.nodeid,
            "message": str(report.longrepr),
        })


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when == "setup":
        if report.failed:
            exc_type = call.excinfo.typename if call.excinfo else None
            exc_message = str(call.excinfo.value) if call.excinfo else None
            _emit({
                "kind": "test",
                "test_case_id": _test_case_id(item),
                "status": "error",
                "duration_ms": int(report.duration * 1000),
                "error_type": exc_type,
                "error_message": exc_message,
                "stdout": getattr(report, "capstdout", ""),
                "stderr": getattr(report, "capstderr", ""),
            })
        return

    if report.when != "call":
        return

    duration_ms = int(report.duration * 1000)
    stdout = getattr(report, "capstdout", "")
    stderr = getattr(report, "capstderr", "")
    test_case_id = _test_case_id(item)

    if report.passed:
        _emit({
            "kind": "test",
            "test_case_id": test_case_id,
            "status": "passed",
            "duration_ms": duration_ms,
            "error_type": None,
            "error_message": None,
            "stdout": stdout,
            "stderr": stderr,
        })
    elif report.skipped:
        reason = str(report.longrepr) if report.longrepr else None
        _emit({
            "kind": "test",
            "test_case_id": test_case_id,
            "status": "skipped",
            "duration_ms": duration_ms,
            "error_type": None,
            "error_message": reason,
            "stdout": stdout,
            "stderr": stderr,
        })
    else:
        exc_type = call.excinfo.typename if call.excinfo else "Unknown"
        # AssertionError is a wrong-answer test failure. "Failed" is pytest's own
        # outcome exception, raised by pytest.raises(...) itself when the candidate does
        # not raise the expected exception at all ("DID NOT RAISE") — that is also a
        # wrong-answer shape, not a candidate exception. Anything else is a genuine
        # candidate exception during the test (spec section 27's failed-vs-error split).
        status = "failed" if exc_type in ("AssertionError", "Failed") else "error"
        exc_message = str(call.excinfo.value) if call.excinfo else None
        _emit({
            "kind": "test",
            "test_case_id": test_case_id,
            "status": status,
            "duration_ms": duration_ms,
            "error_type": exc_type,
            "error_message": exc_message,
            "stdout": stdout,
            "stderr": stderr,
        })


def pytest_sessionfinish(session, exitstatus):
    _emit({
        "kind": "session",
        "testscollected": getattr(session, "testscollected", 0),
        "testsfailed": getattr(session, "testsfailed", 0),
        "exitstatus": int(exitstatus),
    })
'''


def new_nonce() -> str:
    return secrets.token_hex(_NONCE_BYTES)


def render_conftest(nonce: str) -> str:
    """Render :data:`CONFTEST_TEMPLATE` for one job's nonce.

    Token substitution rather than ``str.format`` — the template is full of literal dict
    braces that would otherwise all need escaping.
    """
    return CONFTEST_TEMPLATE.replace(_NONCE_TOKEN, repr(nonce))


@dataclass(frozen=True)
class RawTestEvent:
    """One test's outcome as reported by the conftest plugin, before provenance is known.

    Deliberately not a :class:`~python_dpo.evaluation.models.TestCaseResult` — that type
    requires non-empty ``evaluation_run_id``/``candidate_run_id``/``candidate_id``/
    ``problem_id``, none of which the conftest plugin has any business knowing (it runs
    once per job, for exactly one candidate). :func:`reconcile` is what attaches them.
    """

    test_case_id: str
    status: str
    duration_ms: int
    error_type: str | None
    error_message: str | None
    stdout: str
    stderr: str


@dataclass(frozen=True)
class ParsedPytestRun:
    """The raw facts recovered from one job's stdout, before reconciliation against the
    problem's expected test ids.
    """

    test_results: tuple[RawTestEvent, ...] = ()
    collection_error: str | None = None
    testscollected: int | None = None
    testsfailed: int | None = None
    exitstatus: int | None = None

    @property
    def is_syntax_error(self) -> bool:
        """Whether the collection failure looks like a compile-time error.

        Reuses :func:`python_dpo.sandbox.result.looks_like_compile_error` — the same
        heuristic already applied at the sandbox layer, so "does this look like a
        SyntaxError" is answered identically everywhere it is asked.
        """
        return self.collection_error is not None and looks_like_compile_error(self.collection_error)


class PytestResultParser:
    """Parses nonce-prefixed JSON lines out of a job's stdout."""

    def parse(self, stdout: str, nonce: str) -> ParsedPytestRun:
        prefix = f"{nonce} "
        events: list[dict] = []
        for line in stdout.splitlines():
            # A substring search, not startswith: pytest's own "-q" progress character
            # (".", "F", "s") is written to the same line as our print() with no
            # separating newline, so the nonce is not reliably at column 0. Searching for
            # it as a substring is what makes this robust regardless of pytest's own
            # terminal-reporting behavior, while still requiring the exact nonce to
            # appear — an unrelated candidate print() cannot be mistaken for a result.
            index = line.find(prefix)
            if index == -1:
                continue
            payload_text = line[index + len(prefix) :]
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError as exc:
                raise ResultParseError(
                    f"malformed nonce-tagged line: {payload_text!r}: {exc}"
                ) from exc
            if not isinstance(payload, dict) or "kind" not in payload:
                raise ResultParseError(f"nonce-tagged line missing 'kind': {line!r}")
            events.append(payload)

        collection_error: str | None = None
        testscollected = testsfailed = exitstatus = None
        test_results: list[RawTestEvent] = []

        for event in events:
            kind = event.get("kind")
            if kind == "collect_error":
                collection_error = event.get("message")
            elif kind == "test":
                test_results.append(
                    RawTestEvent(
                        test_case_id=event["test_case_id"],
                        status=event["status"],
                        duration_ms=event["duration_ms"],
                        error_type=event.get("error_type"),
                        error_message=event.get("error_message"),
                        stdout=event.get("stdout") or "",
                        stderr=event.get("stderr") or "",
                    )
                )
            elif kind == "session":
                testscollected = event.get("testscollected")
                testsfailed = event.get("testsfailed")
                exitstatus = event.get("exitstatus")

        return ParsedPytestRun(
            test_results=tuple(test_results),
            collection_error=collection_error,
            testscollected=testscollected,
            testsfailed=testsfailed,
            exitstatus=exitstatus,
        )


def _attach_provenance(
    event: RawTestEvent,
    *,
    evaluation_run_id: str,
    candidate_run_id: str,
    candidate_id: str,
    problem_id: str,
) -> TestCaseResult:
    return TestCaseResult(
        evaluation_run_id=evaluation_run_id,
        candidate_run_id=candidate_run_id,
        candidate_id=candidate_id,
        problem_id=problem_id,
        test_case_id=event.test_case_id,
        status=event.status,
        duration_ms=event.duration_ms,
        error_type=event.error_type,
        error_message=event.error_message,
        stdout=event.stdout,
        stderr=event.stderr,
    )


def reconcile(
    actual: Sequence[RawTestEvent],
    expected_test_case_ids: Sequence[str],
    *,
    evaluation_run_id: str,
    candidate_run_id: str,
    candidate_id: str,
    problem_id: str,
    missing_error_type: str,
    missing_error_message: str,
) -> list[TestCaseResult]:
    """Reconcile parsed results against the problem's full expected test id list.

    Spec sections 46, 68: every expected test id must be accounted for. Any id present in
    ``actual`` is used as-is (with provenance stamped in); any expected id *absent* — because
    pytest never reached it (a collection failure, a timeout cutting the run short, or a
    forged/truncated result set) — is synthesized as an ``error`` result rather than
    silently omitted, so ``tests_total`` always equals ``len(expected_test_case_ids)`` and
    the four counts always partition it exactly (spec section 68's invariant, enforced
    again by :class:`~python_dpo.evaluation.models.EvaluationResult`).

    Returned in the problem's declared test order, not arrival order, so downstream
    listings are stable.
    """
    by_id = {result.test_case_id: result for result in actual}
    reconciled: list[TestCaseResult] = []
    for test_case_id in expected_test_case_ids:
        found = by_id.get(test_case_id)
        if found is not None:
            reconciled.append(
                _attach_provenance(
                    found,
                    evaluation_run_id=evaluation_run_id,
                    candidate_run_id=candidate_run_id,
                    candidate_id=candidate_id,
                    problem_id=problem_id,
                )
            )
        else:
            reconciled.append(
                TestCaseResult(
                    evaluation_run_id=evaluation_run_id,
                    candidate_run_id=candidate_run_id,
                    candidate_id=candidate_id,
                    problem_id=problem_id,
                    test_case_id=test_case_id,
                    status="error",
                    duration_ms=0,
                    error_type=missing_error_type,
                    error_message=missing_error_message,
                )
            )
    return reconciled


__all__ = [
    "CONFTEST_FILENAME",
    "CONFTEST_TEMPLATE",
    "ParsedPytestRun",
    "PytestResultParser",
    "RawTestEvent",
    "new_nonce",
    "reconcile",
    "render_conftest",
]
