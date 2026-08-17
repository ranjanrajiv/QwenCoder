# Stage 6 Implementation Details — Candidate Test Executor

How `src/python_dpo/evaluation/` implements the layer specified in
`.claude/specs/06_candidate_test_executor.md`. For usage, see the "Stage 6 — Candidate
Test Executor" section of the root `README.md`. This file is about *how* it is built and
what was learned building it.

## Goal

Stage 5 built the boundary that makes running untrusted code safe; nothing before this
stage could answer whether a candidate actually *worked*.
`data/candidates/runs/20260817_055411/candidates.jsonl` held 50 real Qwen-generated
candidates and no evidence of correctness. Stage 6 closes that gap: generate a
deterministic pytest suite from a problem's declared test cases, run it against a
candidate inside the Stage 5 sandbox, and persist structured, per-test evidence.

Two constraints define the stage, both preserved unchanged from Stage 5's boundary plus
the scope discipline CLAUDE.md and the spec both require:

- **Security.** No `exec`, `eval`, or shell on candidate-derived text; candidate code and
  the tests generated around it execute only inside the sandbox, never on the host. Every
  Stage 5 isolation setting is inherited, not re-specified.
- **Scope.** The output is objective execution evidence — counts, statuses, durations,
  error types — never `chosen`/`rejected`, a reward, or a ranking. `pass_rate` is a
  metric, not a preference signal.

## 1. Test-generation architecture

`TestGenerator.build(problem, candidate)` produces a `TestJob`: three files
(`candidate.py`, `test_candidate.py`, `conftest.py`), a nonce, and the expected test case
ids. One `def test_<problem_id>_<test_case_id>():` function per declared test case, so a
failure is traceable straight back to the dataset id rather than to "this candidate
failed" undifferentiated.

Five details that are load-bearing, not incidental:

- **Literals via `repr()`** of the JSON-native `TestCase.input`/`expected` values — never
  string interpolation, never `eval()` to rebuild data. Stage 2 already enforces
  JSON-nativeness, so `repr()` round-trips exactly.
- **Calls are `fn(**kwargs)`**, not positional — `TestCase.input` is a keyword-argument
  mapping, which the spec's own examples imply is positional. The dataset's actual shape
  wins.
- **Async entry points get `asyncio.run(...)`**, detected from `Problem.signature`
  starting with `async def`. No pytest-asyncio dependency.
- **Generator results are materialized with `list()`**, matching
  `InProcessReferenceExecutor._materialize`, or `p009`'s `chunk_sequence` fails against
  its own dataset.
- **`expected_exception` becomes `pytest.raises(Exception)`.**

The comparison helper deliberately replicates `InProcessReferenceExecutor._values_match`
exactly, including the guard that stops `True` from satisfying `1`. This is load-bearing:
the dataset's expected values were validated under those exact rules, so any divergence
would make a correct candidate fail. The reference-solution self-check (§9 below) exists
to keep this honest against the real dataset, not just fixtures.

`TestGenerator.validate(job, problem)` checks the job structurally with `ast.parse`
only — never executing anything, the same allowance `python_dpo.generation.validation`
already relies on: file presence, generated test count equals `len(problem.tests)`,
every expected test id present, source parses. A mismatch raises `TestGenerationError`
rather than silently shipping a short suite.

## 2. Result-parser design

The generated `conftest.py` implements `pytest_runtest_makereport` (a hookwrapper, so it
sees the actual outcome including exceptions), `pytest_collectreport` (how a candidate
syntax error surfaces), and `pytest_sessionfinish`, printing one nonce-prefixed JSON
object per event to stdout:

```
7f3a91c2e0b4… {"kind": "test", "test_case_id": "p001_t001", "status": "passed", ...}
```

`PytestResultParser.parse(stdout, nonce)` finds lines **containing** `f"{nonce} "` — a
substring search, not `startswith`. This was a real bug found by running against actual
pytest (§13 below): pytest's own `-q` progress character (`.`/`F`/`s`) is written to the
same line with no separating newline before the `print()`, so `startswith` missed 3 of 4
result lines in manual verification.

`reconcile(actual, expected_test_case_ids, ...)` turns the lightweight, unvalidated
`RawTestEvent`s the parser returns into fully-validated `TestCaseResult`s, stamping in
provenance the parser doesn't know. Any expected id **absent** from `actual` — collection
failure, timeout cutting the run short, or a forged/truncated result set — is synthesized
as an `error` result rather than silently omitted, so `tests_total` always equals
`len(expected_test_case_ids)` and the four counts always partition it exactly.

**Stated honestly, twice (README and here): the nonce defends against accidental
collision, not a deliberately adversarial candidate.** A candidate that reads its own
workspace's `conftest.py` could forge result lines. What actually catches that is
`reconcile`'s count cross-check — a forged or truncated result set whose ids disagree
with the problem's is an error, never a silent pass. Consistent with the threat model
already stated in `docs/sandbox-security.md`.

## 3. Evaluation architecture

```
Candidate + Problem -> TestGenerator -> PytestRunner -> PytestResultParser
                     -> classify -> EvaluationResult + TestCaseResult(s)
                     -> EvaluationRepository
```

| Module | Responsibility |
|---|---|
| `errors.py` | The exception hierarchy; sandbox errors never re-raised, only translated |
| `models.py` | `EvaluationResult`, `TestCaseResult`, `EvaluationFailure`, `EvaluationManifest`, `EvaluationStatistics` |
| `test_generator.py` | `TestGenerator` — builds and structurally validates a job, never executing |
| `result_parser.py` | The nonce wire protocol: `render_conftest`, `PytestResultParser`, `reconcile` |
| `config.py` | `EvaluationConfig` — the `evaluation:` section |
| `pytest_runner.py` | `build_evaluation_sandbox_config`, `PytestRunner` — the fixed pytest argv |
| `probe.py` | `probe_versions` — the real Python/pytest versions used, probed once per run |
| `executor.py` | `CandidateEvaluator` — orchestration and classification |
| `repository.py` | `EvaluationRepository` — one evaluation run's results/tests/failures |
| `run_repository.py` | `EvaluationRunRepository` — multi-run manager, mirrors `runs.RunRepository` |

Classification precedence in `CandidateEvaluator._classify`:

| Condition | Status |
|---|---|
| Sandbox returned `infrastructure_error` | `infrastructure_error` — the candidate never really ran |
| Collection failure that looks like a syntax error | `syntax_error` |
| Collection failure that isn't a syntax error | `failed` |
| Sandbox reported a timeout | `timeout` |
| `tests_total > 0` and every test passed | `passed` |
| Otherwise | `failed` |

A candidate exception during a test is a **candidate** failure — test-level `error`,
candidate-level `failed` — never mistaken for infrastructure trouble. The distinction is
carried on the record itself (`infrastructure_error` and `runtime_error` are independent
booleans, validated as mutually exclusive), so a later stage cannot accidentally conflate
them.

Two failure shapes are deliberately kept apart, mirroring how `candidates.models`
separates a `Candidate` from a `GenerationFailure`:

- `EvaluationResult` — a job was built and *attempted*. Persisted to `evaluations.jsonl`,
  even when `status="infrastructure_error"`.
- `EvaluationFailure` — no job was ever attempted (problem not found, empty test suite,
  job validation failure). Deterministic given the same candidate and problem, so resume
  treats these as settled rather than retryable — unlike Stage 4's generation failures,
  which *are* retried because they may have been a transient model call.
  `EvaluationRepository.evaluated_keys()` covers both files for exactly this reason.

## 4. Docker image and versions

`docker/evaluator/Dockerfile`:

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir --root-user-action=ignore pytest==8.3.4
```

Tagged `python-dpo-evaluator:1.0`, built locally rather than pulled — the container has
no network at evaluation time, so pytest must already be present at build time. Local
image id at time of writing:

```
sha256:50d3905373da63602784949ada7f0c3826271da62111a9483f7bde6f24416deb
```

No `RepoDigest` exists because the image is built locally, never pushed to a registry —
`sandbox.image_digest`'s stronger-pin mechanism (Stage 5) applies to *pulled* images and
was left as `null` here for the same reason.

Actual versions are **probed, not assumed**: `probe.py` runs a trivial program inside the
real evaluation sandbox once per evaluation run and records what it reports. The real run
against all 50 candidates recorded `python_version: "3.12.14"`,
`pytest_version: "8.3.4"` — the pinned pytest version, confirming the build; the Python
patch version is whatever `python:3.12-slim` currently resolves to, which is exactly why
it is probed rather than hardcoded.

`evaluate candidate`/`evaluate run` verify the image is present before starting
(`_ensure_evaluation_image` in `cli.py`), failing once with
`docker build -t <image> docker/evaluator/` rather than producing N identical
infrastructure errors.

## 5. Schema

`EvaluationResult.__post_init__` enforces the counting invariant at construction, not by
convention: `passed` requires `tests_total > 0` and every test passing; the four counts
(`tests_passed`, `tests_failed`, `tests_error`, `tests_skipped`) must always sum to
`tests_total`; the boolean flags (`timeout`, `syntax_error`, `runtime_error`,
`infrastructure_error`) must exactly match `status`/`tests_error > 0`; `pass_rate` must
match `compute_pass_rate(tests_passed, tests_total)`. Exit code 0 alone never produces
`passed`.

`EvaluationManifest` mirrors `RunManifest`: `with_status()` enforces
`EVALUATION_RUN_STATUS_TRANSITIONS`, the same closed transition graph shape as Stage 4.
`EvaluationStatistics.from_records(manifest, results, failures)` is always
reconstructable from the persisted JSONL files, never trusted from an in-memory counter.

`metadata_discrepancy`/`discrepancy_reason` compare `candidate.syntax_valid` (Stage 3's
static `ast.parse` check) against what the sandbox actually reported. Stage 3's record is
never overwritten — a disagreement is a new, separately recorded fact on the evaluation
result.

## 6. Resume behavior

`evaluate run --run-id R` **resumes the latest incomplete evaluation run for `R` by
default**, via `EvaluationRunRepository.latest_run_for_candidate_run`. This is
deliberately the *opposite* default from Stage 4's `generate`, which always starts a new
run and needs an explicit `--resume`. Evaluation is idempotent and cheap to re-run where
generation burns GPU time, which is what makes the asymmetry defensible — recorded as a
deviation below, confirmed with the user during planning.

`--force` always mints a new evaluation run instead of resuming. A selection flag
(`--problem-id`/`--limit`/`--candidate-id`) that disagrees with an existing run's
`requested_candidate_ids` is rejected with a message pointing at `--force` — the same
manifest-is-authoritative rule Stage 4 established for `generate --resume`.

On resume, evaluation runs against the manifest's **full** requested candidate set, not
just the current invocation's selection, since `evaluate_many` skips whatever
`evaluated_keys()` already covers. A run is marked `completed` only when every requested
candidate has a settled outcome (a result or a failure); otherwise it is `interrupted`.

## 7. Failure classification (worked example from the real run)

Evaluating candidate `p008_c001` produced a genuine `AttributeError` at test time:

```json
{
  "candidate_id": "p008_c001", "test_case_id": "p008_t007",
  "status": "error", "error_type": "AttributeError",
  "error_message": "'NoneType' object has no attribute 'strip'"
}
```

The candidate-level `EvaluationResult` for `p008_c001` is `status="failed"`,
`infrastructure_error=false`, `runtime_error=true` — a real candidate bug, correctly
distinguished from Docker trouble, both at the individual test level and the candidate
level.

## 8. Security properties preserved

`build_evaluation_sandbox_config(base, evaluation)` overlays **only** `image`,
`timeout_seconds`, `startup_grace_seconds`, `auto_pull` onto the audited Stage 5
`SandboxConfig` via `dataclasses.replace`. `network_mode`, `user`, `drop_capabilities`,
`cpus`, `memory`, `pids_limit`, `read_only_root`, `tmpfs_size` all come from `base`
unchanged — by construction, so no evaluation-specific configuration can weaken them.

`tests/sandbox/test_sandbox_security.py` now asserts the same isolation contract against
the *evaluation* container's argv: `--network none`, `--user 65534:65534`,
`--cap-drop ALL`, `--security-opt no-new-privileges`, `--read-only`, `--pids-limit 64`,
`--cpus 1.0`, `--memory`/`--memory-swap` pinned, the `:ro` workspace mount, and that the
command is the fixed pytest argument list ending in `test_candidate.py` — never a shell.

`SandboxWorkspace.write_file`/`SandboxExecutor.execute_job` (the one real Stage 5
modification, generalizing single-file `write_candidate`/`execute` to a multi-file job)
inherit every existing bounding, timeout, classification, and unconditional-cleanup rule;
`execute()`/`write_candidate()` are now thin wrappers, so Stage 5's own test suite stayed
green throughout.

## 9. Test results

```
pytest -q                  → 632 passed, 46 deselected   (offline, zero skips)
pytest -q -m integration   → 46 passed                    (33 sandbox + 13 evaluation)
docker ps -a --filter name=python-dpo-sandbox → empty
```

**Offline (`tests/evaluation/`, no Docker, 149 tests):** `test_models.py` (33),
`test_config.py` (16), `test_test_generator.py` (22), `test_result_parser.py` (23),
`test_pytest_runner.py` (7), `test_probe.py` (5), `test_executor.py` (14),
`test_repository.py` (15), `test_run_repository.py` (14).

**Docker required (`tests/evaluation/test_integration.py`, 13 tests):** the six §66
candidate fixtures (correct, wrong result, syntax error, runtime exception, infinite
loop, network attempt) run for real; the four mandatory checks — known-good → `passed`
with every test passing, deliberately wrong → `failed` with the *specific* failing test
ids identified, an infinite loop → `timeout` with the container cleaned up (asserted via
an autouse fixture that fails if any sandbox container survives any test), an injected
Docker-unavailable runtime → `infrastructure_error` without stopping the real daemon; a
network attempt still classified as a candidate failure, never infrastructure trouble;
the result-arithmetic invariant (`passed+failed+error+skipped == total`); an end-to-end
`evaluate_many` persisting every result; and **the reference-solution self-check**,
generating the real pytest suite for all 10 committed problems and running each against
its own trusted `reference_solution` — all 74 reference tests pass, confirming
`TestGenerator`'s comparison semantics still match what the dataset was validated under.

**The real 50-candidate run** (`evaluate run --run-id 20260817_055411`, evaluation run
`eval_20260817_115154_dcd4`, committed under `data/evaluations/runs/`):

```
Candidates requested: 50    Candidates evaluated: 50    Evaluation failures: 0
Passed: 30    Failed: 20    Timeouts: 0    Syntax errors: 0    Infrastructure errors: 0
Tests: 322/370 passed (45 failed, 3 error, 0 skipped)
```

No machinery failures, no infrastructure errors, no lingering containers. The three
`error` results were all the same real `AttributeError` in `p008`'s candidates (§7
above) — a genuine candidate bug the tests correctly surfaced, not a harness problem.

## 10. Files created/modified

**Created:**

- `src/python_dpo/evaluation/` — `__init__.py`, `errors.py`, `models.py`,
  `test_generator.py`, `result_parser.py`, `config.py`, `pytest_runner.py`, `probe.py`,
  `executor.py`, `repository.py`, `run_repository.py`, `README.md`
- `tests/evaluation/` — `__init__.py`, `test_models.py`, `test_config.py`,
  `test_test_generator.py`, `test_result_parser.py`, `test_pytest_runner.py`,
  `test_probe.py`, `test_executor.py`, `test_repository.py`, `test_run_repository.py`,
  `test_integration.py`
- `docker/evaluator/Dockerfile`
- `data/evaluations/runs/eval_20260817_115154_dcd4/` — the real evaluation run
- `06_CANDIDATE_TEST_EXECUTOR.md` (this file)

**Modified:**

- `src/python_dpo/sandbox/workspace.py` — generalized `write_candidate` to `write_file`
  with filename validation
- `src/python_dpo/sandbox/executor.py` — generalized `execute` to `execute_job`
  (files + command), `execute` now a thin wrapper
- `tests/sandbox/test_workspace.py`, `tests/sandbox/test_executor_mock.py` — coverage for
  the two generalizations above
- `tests/sandbox/test_sandbox_security.py` — evaluation-container argv assertions (§8)
- `src/python_dpo/config.py` — `evaluation: EvaluationConfig` on `Config`,
  `_parse_evaluation`
- `src/python_dpo/cli.py` — the `evaluate`/`evaluations` command groups;
  `evaluate` removed from `_PLACEHOLDER_STAGES`
- `tests/test_project.py` — `evaluate`/`evaluations` CLI parsing and error-path tests
- `config.yaml` — the `evaluation:` section
- `src/python_dpo/__init__.py` — version `0.5.0` → `0.6.0`
- `CLAUDE.md` — generated tests are untrusted-adjacent, execute only in the sandbox
- `README.md`, `src/python_dpo/README.md`, `data/README.md`, `tests/README.md` — Stage 6
  documentation

## 11. Dependencies added

**None to the host project.** `pytest==8.3.4` is added to the **evaluation image only**
(`docker/evaluator/Dockerfile`) — not to `pyproject.toml`. The core install and the
offline test suite stay exactly as lightweight as before Stage 6; `pytest` itself (the
one the host uses to *run* this test suite) was already a `dev` extra from Stage 1 and is
unrelated to the pinned in-container version.

## 12. Deviations from the plan/spec

- **The spec file ships as `06_candidate_test_executor.py`** with Markdown content;
  renamed to `.md` before implementation began (separate commit).
- **`TestCase.input` is a kwargs mapping**, so generated calls are `fn(**{...})` rather
  than the positional `fn([...])` the spec's own examples imply — the dataset's actual
  shape wins.
- **No `approx`/float comparison mode.** No problem in the real dataset has a float
  expected value, and the comparison helper is a single function, so adding a mode later
  is a contained change if the dataset ever needs one.
- **Results are captured via a generated `conftest.py` emitting nonce-prefixed JSON
  lines**, not JUnit XML or `pytest-json-report` — zero dependencies, no report-file
  extraction step, works unchanged against a read-only workspace.
- **Resume is the default for `evaluate run`, with no `--resume` flag** — deliberately
  the opposite of Stage 4's `generate`, confirmed with the user during planning (see §6).
- **The nonce is anti-collision, not anti-adversary** — stated explicitly in code
  comments, the package README, and here, consistent with the existing sandbox threat
  model.
- **Async handled with `asyncio.run`, not pytest-asyncio** — no new dependency, and it
  mirrors `InProcessReferenceExecutor._call` exactly.
- **Comparison semantics intentionally replicate `InProcessReferenceExecutor._values_match`**
  bit for bit, including the bool-vs-int guard, because the dataset's expected values
  were validated under exactly those rules.
- **`EvaluationRunRepository` is a new module beyond the plan's literal package list** —
  the plan specified `EvaluationRepository` (run-scoped results) but the CLI's
  resume-by-default semantics need a *multi*-run manager (mint ids, own `manifest.json`,
  list/resume/complete runs) exactly analogous to `runs.RunRepository`. Built as a
  separate module rather than folding run-management into `EvaluationRepository`, so the
  run-scoped/multi-run split mirrors Stage 4's `candidates.repository`/`runs.repository`
  split rather than inventing a new shape.
- **`probe.py` is a new module beyond the plan's literal package list** — the plan asked
  for probed-not-assumed pytest/Python versions in the manifest (§74) but didn't name the
  module; implemented as its own small file rather than folding into `pytest_runner.py`,
  since it constructs and runs its own trivial sandbox job independent of the pytest
  command path.

## Known limitations

- **The nonce defends against accidental collision only** (§2, §8). A candidate that
  deliberately reads its own workspace's `conftest.py` could attempt to forge result
  lines; the count cross-check in `reconcile` is what actually catches a forged or
  truncated result set, not secrecy of the nonce.
- **Container-startup overhead dominates evaluation cost**, as Stage 5's own report
  flagged it would (~2–3s per candidate observed in the real run, ~3 minutes for 50
  candidates). Batching a problem's test cases into fewer container runs was considered
  and deliberately not done: one container per candidate keeps the failure/timeout
  attribution unambiguous (exactly one candidate per sandbox execution) and the
  classification logic simple. Worth revisiting if the candidate count grows by an order
  of magnitude.
- **No float/`approx` comparison mode** (§12 above) — would need adding if a future
  problem introduces a float expected value.
- **All Stage 5 known limitations still apply unchanged** (shared kernel, no user-ns
  remap, default seccomp, root daemon) — see `docs/sandbox-security.md`. Stage 6 adds no
  new limitation to the execution boundary itself, only to the result-integrity story
  above.

Stopping here. Not starting Stage 7 (ranking / correctness classification) without
explicit approval.
