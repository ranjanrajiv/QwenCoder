# src/python_dpo/evaluation/

The candidate test executor — turns a persisted candidate into objective evidence of
whether it solved its problem, by generating a deterministic pytest suite from the
problem's declared test cases and running both inside the Stage 5 Docker sandbox.

This package answers **"what happened when this candidate was tested against its
problem's test suite?"** — never **"is this the best candidate?"**. It never produces
`chosen`/`rejected` preference pairs, a reward, or a ranking; `pass_rate` is stored as a
metric, not used as a preference signal. Those transformations belong to a later stage.

Candidate code and generated tests execute only inside the Stage 5 sandbox
(`python_dpo.sandbox`). This package never calls `exec`, `eval`, or a shell on
candidate-derived text, and never runs generated Python on the host.

## Files

### `errors.py`

`EvaluationError` and its subclasses: `EvaluationConfigError`, `ProblemNotFoundError`,
`CandidateNotFoundError`, `InvalidProblemError` (a problem with no test cases),
`TestGenerationError` (the generated job failed structural validation or the candidate
does not belong to the problem), `ResultParseError`. Sandbox errors are never re-raised
here — they arrive as an `ExecutionResult` with `status="infrastructure_error"` and are
translated by `executor.py`, never leaked as an exception.

### `models.py`

The schema, mirroring `python_dpo.candidates.models` and `python_dpo.runs.models`:
frozen dataclasses validating in `__post_init__`, explicit `to_dict()`/`from_dict()`.

Two failure shapes are deliberately kept apart:

- `EvaluationResult` — a job was built and *attempted* in the sandbox. Its status may
  still be `infrastructure_error` (Docker died mid-run), but a job existed and ran.
  Persisted to `evaluations.jsonl`.
- `EvaluationFailure` — no job was ever attempted: the problem could not be found, had no
  test cases, or the generated job failed validation before ever reaching Docker. These
  are deterministic given the same candidate and problem, unlike a transient Docker
  fault, so resume treats them as settled rather than retryable. Persisted to
  `failures.jsonl`.

`EvaluationResult.__post_init__` enforces the counting invariant directly: `passed`
requires `tests_total > 0` and every test passing; the four counts must always sum to
`tests_total`; the boolean flags (`timeout`, `syntax_error`, `runtime_error`,
`infrastructure_error`) must exactly match `status`/`tests_error`. Exit code 0 alone
never produces `passed`.

`TestCaseResult` is the per-test-case record (`passed`/`failed`/`error`/`skipped`, plus
error type/message/stdout/stderr).

`EvaluationManifest` and `EvaluationStatistics` mirror Stage 4's `RunManifest` /
`RunStatistics`, including `from_records(...)` so statistics are always reconstructable
from the persisted results rather than trusted from an in-memory counter.

### `test_generator.py`

`TestGenerator.build(problem, candidate)` produces a `TestJob`: three files
(`candidate.py`, `test_candidate.py`, `conftest.py`) plus a nonce and the expected test
case ids. One `test_<problem>_<case>()` function per declared test case, so a failure is
traceable straight back to the dataset id.

Details that are load-bearing, not incidental:

- **Literals are embedded via `repr()`** of the JSON-native `TestCase.input`/`expected`
  values — never string interpolation, never `eval()` to rebuild data.
- **Calls are `fn(**kwargs)`**, since `TestCase.input` is a keyword-argument mapping, not
  a positional list.
- **Async entry points** get `asyncio.run(...)`, detected from `Problem.signature`. No
  pytest-asyncio dependency.
- **Generator results are materialized** with `list()`, matching
  `InProcessReferenceExecutor._materialize`.
- **`expected_exception` becomes `pytest.raises(...)`.**
- **The comparison helper mirrors `InProcessReferenceExecutor._values_match`** exactly,
  including the guard that stops `True` from satisfying `1` — the dataset's expected
  values were validated under those exact rules, so any divergence would make a correct
  candidate fail. `tests/evaluation/test_integration.py`'s reference-solution self-check
  exists to keep this honest against the real dataset.

`TestGenerator.validate(job, problem)` checks the job structurally with `ast.parse` only
— never executing anything, the same allowance `python_dpo.generation.validation`
already relies on: the generated file count matches `len(problem.tests)`, every expected
test id is present, and the source parses.

### `result_parser.py`

The wire protocol between the sandboxed pytest run and this process: the generated
`conftest.py` (`render_conftest(nonce)`) implements `pytest_runtest_makereport`,
`pytest_collectreport`, and `pytest_sessionfinish`, printing one nonce-prefixed JSON
object per event to stdout — no report file, no extra dependency, works unchanged
against a read-only workspace.

`PytestResultParser.parse(stdout, nonce)` finds lines containing `f"{nonce} "` — a
substring search, not `startswith`, because pytest's own `-q` progress character (`.`,
`F`, `s`) is written to the same line with no separating newline. Returns a
`ParsedPytestRun` of lightweight `RawTestEvent`s (no provenance, no validation).

`reconcile(actual, expected_test_case_ids, ...)` is what turns those into fully-validated
`TestCaseResult`s, stamping in the provenance fields the parser doesn't know. Any
expected id **absent** from `actual` — a collection failure, a timeout cutting the run
short, or a forged/truncated result set — is synthesized as an `error` result rather than
silently omitted, so `tests_total` always equals `len(expected_test_case_ids)` and the
four counts always partition it exactly.

**The nonce defends against accidental collision, not a deliberately adversarial
candidate.** A candidate that reads its own workspace's `conftest.py` could forge result
lines. What actually catches that is `reconcile`'s count cross-check: a forged or
truncated result set whose test ids disagree with the problem's is an error, never a
silent pass. Consistent with the threat model in `docs/sandbox-security.md`.

### `pytest_runner.py`

`build_evaluation_sandbox_config(base, evaluation)` overlays only `image`,
`timeout_seconds`, `startup_grace_seconds`, and `auto_pull` onto the audited Stage 5
`SandboxConfig` via `dataclasses.replace`. Every other isolation setting — network mode,
user, capabilities, resource limits — is inherited unchanged *by construction*, so no
evaluation-specific configuration can weaken it.

`PYTEST_COMMAND` is a fixed argument list:
`python -m pytest -q -p no:cacheprovider test_candidate.py`. `-p no:cacheprovider`
matters because pytest writes `.pytest_cache` by default and the workspace is mounted
read-only; `PYTHONDONTWRITEBYTECODE` (already in the sandbox's base environment) covers
`__pycache__`.

`PytestRunner.run(job, ...)` is a thin wrapper over `SandboxExecutor.execute_job`.

### `probe.py`

`probe_versions(config, executor=None)` runs a trivial program in the evaluation sandbox
and returns the Python and pytest versions actually present in the image — recorded once
per evaluation run in the manifest, never assumed from the Dockerfile.

### `executor.py`

`CandidateEvaluator` — the orchestrator:
`Candidate + Problem -> TestGenerator -> PytestRunner -> PytestResultParser -> classify
-> EvaluationResult + TestCaseResult(s) -> EvaluationRepository`.

`.evaluate(candidate, problem, ...)` evaluates and persists one candidate; raises only
for machinery-level problems that prevented a job from being attempted at all (never for
anything the candidate itself did). `.evaluate_many(...)` turns those into a persisted
`EvaluationFailure` and skips anything already covered by
`EvaluationRepository.evaluated_keys()` (both results *and* failures — unlike Stage 4's
generation failures, an evaluation failure here is structural and deterministic given the
same candidate and problem, so retrying it on resume achieves nothing).

Classification precedence in `_classify`: infrastructure outranks everything (the
candidate never really ran); a collection failure is the candidate's own doing and
becomes `syntax_error` or `failed`; a timeout is distinct from a plain failure;
otherwise every test must have passed for `passed`.

Discrepancy detection compares `candidate.syntax_valid` (Stage 3's static `ast.parse`
check) against what the sandbox actually reported. Stage 3's record is never
overwritten — a disagreement is recorded as `metadata_discrepancy` on the new result.

### `repository.py`

`EvaluationRepository` — durable, run-scoped persistence for one evaluation run's
`evaluations.jsonl`, `test_results.jsonl`, and `failures.jsonl`, built on
`atomic_io.append_jsonl`/`iter_jsonl` exactly as Stage 4's repositories are. Records are
written the moment they exist, never batched, so a killed run leaves a resumable file
behind.

### `run_repository.py`

`EvaluationRunRepository` — the multi-run manager, mirroring
`python_dpo.runs.repository.RunRepository`: mints `eval_YYYYMMDD_HHMMSS_xxxx` ids, owns
`manifest.json`/`statistics.json`, and is the only code that transitions an evaluation
run's status (`created -> running -> completed/failed/interrupted/cancelled`).

`latest_run_for_candidate_run(candidate_run_id)` is the resume lookup the CLI uses:
`evaluate run --run-id R` resumes the most recent evaluation run for `R` by default —
deliberately the *opposite* default from Stage 4's `generate`, since evaluation is
idempotent and cheap to re-run where generation burns GPU time. `--force` always mints a
new one.

## Persistence layout

```
data/evaluations/runs/eval_YYYYMMDD_HHMMSS_xxxx/
├── manifest.json        # config snapshot, probed versions, status, sandbox config
├── evaluations.jsonl    # one EvaluationResult per candidate
├── test_results.jsonl   # one TestCaseResult per test case
├── failures.jsonl       # machinery failures — no job was ever attempted
└── statistics.json      # reconstructable from evaluations.jsonl + failures.jsonl
```

Every record carries `evaluation_run_id`, `candidate_run_id`, `candidate_id`,
`problem_id`, so the full chain back to model and prompt is reconstructable. Historical
evaluation runs are immutable — a re-evaluation with `--force` is a new run, never an
overwrite.

## The evaluation image

`docker/evaluator/Dockerfile` pins `python:3.12-slim` plus `pytest==8.3.4`, tagged
`python-dpo-evaluator:1.0`, built locally rather than pulled — the container has no
network at evaluation time, so pytest must already be present at image-build time. Build
it with:

```bash
docker build -t python-dpo-evaluator:1.0 docker/evaluator/
```
