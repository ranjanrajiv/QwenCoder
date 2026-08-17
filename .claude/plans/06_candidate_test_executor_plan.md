# Stage 6 — Candidate Test Executor

## Context

Stage 5 built the execution boundary; Stage 6 is what it was built for. Today
`data/candidates/runs/20260817_055411/candidates.jsonl` holds 50 real Qwen candidates
across all 10 problems, and nothing in the repository knows whether a single one of them
*works*. The sandbox can answer "did this program run safely?" — it cannot answer "did it
solve the problem?"

`.claude/specs/06_candidate_test_executor.py` (note: the spec ships with a `.py` extension
despite being Markdown) asks for the layer that closes that gap: take a persisted
candidate, generate a deterministic pytest suite from its problem's declared test cases,
run both inside the Stage 5 sandbox, and persist structured, per-test evidence of what
happened.

Two boundaries define the stage:

- **Security (§78–§80).** Every Stage 5 property is preserved unchanged — no network, no
  host filesystem, non-root, dropped capabilities, resource limits. Adding pytest must not
  relax any of it. No `exec`, no `eval`, no shell, and **no candidate or generated test code
  runs on the host** (§79).
- **Scope (§2, §62, §89).** The output is *objective execution evidence*: counts, statuses,
  durations, error types. Not `correct`/`incorrect`, not `chosen`/`rejected`, not a reward
  or a ranking. `pass_rate` is stored as a metric (§61) and explicitly not used as a
  preference signal. Those transformations belong to Stage 7+.

**Outcome:** `evaluate run --run-id 20260817_055411` produces
`data/evaluations/runs/eval_.../` with per-candidate and per-test records, and the first
real answer to how good this model's output actually is.

### What exploration established

Facts that materially shape the design, confirmed against the real dataset and code:

| Finding | Consequence |
|---|---|
| `TestCase.input` is a **kwargs mapping** (`{"numbers": [1,2,3,4]}`), not the positional list the spec's example implies | Generated calls are `fn(**{...})`, not `fn([...])` |
| **p010 is `async def gather_in_order`** | Generated test must `asyncio.run(...)` — or take a pytest-asyncio dependency |
| **p005 / p006 / p009 have `expected_exception` test cases** | Generator needs a `pytest.raises` path |
| **p009 `chunk_sequence` returns a generator**, materialised with `list()` by the Stage 2 executor | Generated test must materialise identically or it fails spuriously |
| **No float expected values exist anywhere** | §18's `approx` comparison mode has no consumer |
| 74 test cases across 10 problems; ≥5 per problem enforced by Stage 2 | §45's zero-test case can't occur with the real dataset, but the check still ships |
| `ContainerSpec.command` is **already a field** | The container layer needs no change; only workspace and executor grow |
| `SandboxWorkspace` writes exactly one file (`write_candidate`) | Needs generalising to a multi-file job |

The last row is the one real modification to Stage 5 code this stage requires.

### Decisions confirmed with the user

1. **Results come back as nonce-prefixed JSON lines on stdout**, emitted by a generated
   `conftest.py`. Zero dependencies, no report-file extraction step, works unchanged against
   a read-only workspace and root, and the record schema is ours rather than JUnit's.
2. **Resume by default**, following §56/§57 literally: `evaluate run --run-id R` continues
   the latest incomplete evaluation run for `R`; `--force` mints a new one. This is
   deliberately *opposite* to the `generate` semantics chosen in Stage 4 — recorded as a
   divergence, with the note that `evaluate` is idempotent and cheap to re-run where
   `generate` burns GPU time, which is what makes the asymmetry defensible.
3. **Verification evaluates all 50 real candidates** and commits
   `data/evaluations/runs/<eval_id>/` as a tracked artifact.

---

## New package — `src/python_dpo/evaluation/`

House style throughout: frozen dataclasses validating in `__post_init__`, explicit
`to_dict()`/`from_dict()`, per-folder `README.md`.

**`errors.py`** — `EvaluationError` base; `TestGenerationError` (§46/§47 validation
failures), `InvalidProblemError` (§45), `CandidateNotFoundError`, `ProblemNotFoundError`,
`ResultParseError`. Sandbox errors are *not* re-raised — they arrive as an
`ExecutionResult` with `status="infrastructure_error"` and are translated, never leaked.

**`models.py`** — the two record types plus the run-level ones.

`TestCaseResult` (§30): `test_case_id`, `status` (`passed`/`failed`/`error`/`skipped`),
`duration_ms`, `error_type`, `error_message`, `stdout`, `stderr`, optional `traceback`.

`EvaluationResult` (§22): `evaluation_run_id`, `candidate_run_id`, `candidate_id`,
`problem_id`, `status`, `tests_total/passed/failed/error/skipped`, `pass_rate`,
`duration_ms`, plus the recommended fields — `timeout`, `syntax_error`, `runtime_error`,
`infrastructure_error`, `stdout`, `stderr`, `exit_code`, `sandbox_container_id`,
`evaluation_timestamp`, and `metadata_discrepancy` + `discrepancy_reason` (§63/§64).

`EVALUATION_STATUSES = {passed, failed, timeout, syntax_error, infrastructure_error}` (§23).
**`passed` is validated, not assumed** (§24): `__post_init__` rejects a `passed` record
unless `tests_total > 0` and `tests_passed == tests_total` and failed/error/skipped are all
zero. Exit code 0 alone never produces `passed`.

`EvaluationManifest` and `EvaluationStatistics` mirror Stage 4's `RunManifest` /
`RunStatistics`, including `from_records(...)` so statistics are always reconstructable from
`evaluations.jsonl` rather than trusted from memory.

**`test_generator.py`** — `TestGenerator`, `TEST_GENERATOR_VERSION = "v1"`.

Produces a three-file job from `(Problem, Candidate)`:

```
/workspace/
    candidate.py        # the persisted code, byte-identical (§12, §69)
    test_candidate.py   # one test function per test case (§20)
    conftest.py         # the reporting plugin
```

Generated test shape, one function per case so failures are traceable to a dataset id
(§19, §20):

```python
import asyncio, candidate, pytest          # asyncio only when the entry point is async

def _matches(actual, expected):
    # Mirrors InProcessReferenceExecutor._values_match: True must not pass as 1.
    if isinstance(actual, bool) != isinstance(expected, bool):
        return False
    return actual == expected

def test_p001_tc001():
    result = candidate.sum_even(**{'numbers': [1, 2, 3, 4]})
    assert _matches(result, 6)
```

Five details that are not optional:

- **Literals via `repr()` of JSON-native values** (§16). Stage 2 already enforces
  JSON-nativeness (`_require_json_native`), so `repr()` round-trips exactly. No string
  interpolation of untrusted text, no `eval()` to rebuild data (§44).
- **kwargs, not positional** — `TestCase.input` is a mapping of argument names.
- **Async entry points** get `asyncio.run(candidate.fn(**kwargs))`, detected from
  `Problem.signature` starting with `async def`. No pytest-asyncio dependency, and it
  mirrors what `InProcessReferenceExecutor._call` already does.
- **Generator results are materialised** with `list()`, matching
  `InProcessReferenceExecutor._materialize`, or p009 fails against its own dataset.
- **`expected_exception` becomes `pytest.raises`**, matching how Stage 2 validated it.

The comparison semantics deliberately replicate `problems/executor.py`. This is
load-bearing: the dataset's expected values were *validated* under those exact rules, so
any divergence would make a correct candidate fail — the reference-solution self-check
below exists to keep that honest.

`validate(job)` implements §46/§47 **without executing anything**: the file exists, the
generated test count equals `len(problem.tests)`, every test id is present, and the import
name is right. `ast.parse` on the generated source confirms it is syntactically valid —
building a syntax tree runs nothing, the same allowance Stage 3 already relies on. A
mismatch raises `TestGenerationError`, never a silently short test suite.

**`result_parser.py`** — `PytestResultParser`.

The generated `conftest.py` implements `pytest_runtest_logreport` (per-test outcomes),
`pytest_collectreport` (collection failures, which is how a candidate syntax error
surfaces), and `pytest_sessionfinish` (totals and exit status), printing one
nonce-prefixed JSON object per event:

```
7f3a91c2e0b4… {"kind": "test", "test_case_id": "p001_tc001", "status": "passed", ...}
```

The parser reads only lines carrying the job's nonce and ignores everything else, so
ordinary candidate `print()` output cannot be mistaken for a result. **Stated honestly:
the nonce defends against accidental collision, not against a candidate that deliberately
reads `conftest.py` from its own workspace and forges lines.** The defence that matters
against that is §46's count cross-check — a forged or partial result set whose test count
disagrees with the problem's is an error, not a pass. Both facts go in the report and the
package README.

Parser fixtures cover all six §65 cases: all pass, partial failure, runtime error,
collection/syntax error, sandbox timeout, and skipped tests.

**`pytest_runner.py`** — builds the sandbox job and runs it.

Derives the evaluation `SandboxConfig` from the audited Stage 5 one with
`dataclasses.replace(base, image=..., timeout_seconds=...)`. Deriving rather than
constructing is the point: every isolation setting is inherited by construction, so no
evaluation code path can weaken network mode, capabilities, user, or limits (§78).

Fixed command (§42, §43):

```
python -m pytest -q -p no:cacheprovider test_candidate.py
```

`-p no:cacheprovider` matters: pytest writes `.pytest_cache` by default, and the workspace
is mounted read-only. `PYTHONDONTWRITEBYTECODE=1` (already in Stage 5's base environment)
covers `__pycache__`. With no host config mounted, pytest's rootdir resolves to
`/workspace`, satisfying §39/§40 without extra work.

**`executor.py`** — `CandidateEvaluator`, `EVALUATOR_VERSION = "v1"`.

Per candidate: load problem → generate job → validate job → run in sandbox → parse →
classify → build `EvaluationResult`. Classification (§23–§29), in precedence order:

| Condition | Status |
|---|---|
| Sandbox returned `infrastructure_error`, or job validation failed, or problem has no tests | `infrastructure_error` (§29, §45, §46) |
| Sandbox returned `timeout` | `timeout`, `timeout=true` (§28) — never merely `failed` |
| Collection failure identifying a syntax error | `syntax_error` (§26) |
| `tests_total > 0` and all passed | `passed` (§24) |
| Otherwise | `failed` (§25, §27) |

A candidate exception during a test is a **candidate** failure — test-level `error`,
candidate-level `failed` — never an infrastructure error (§27). The distinction is carried
on the record itself so Stage 7 cannot accidentally conflate them (§29).

Discrepancy detection (§63/§64): when `candidate.syntax_valid` from Stage 3 disagrees with
what the sandbox actually reported, set `metadata_discrepancy=true` with a reason. Stage 3's
record is never overwritten (§63).

**`repository.py`** — `EvaluationRepository`, run-scoped, built on `atomic_io`
(`atomic_write_json`, `append_jsonl`, `iter_jsonl`) exactly as Stage 4's repositories are.
The §52 API: `save`, `get`, `list`, `find_by_candidate`, `find_by_problem`, `count`, plus
`evaluated_keys()` — the resume index (§56) — and `append_test_results`.

---

## Evaluation image — `docker/evaluator/Dockerfile`

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir pytest==8.3.4
```

Pinned Python and pytest, nothing else (§33, §34, §35). Tagged
`python-dpo-evaluator:1.0`. The host project is never copied in — candidate and test files
arrive through the isolated workspace only.

Because the container has no network (§36), pytest must be baked in at build time; that is
why a dedicated image exists rather than installing at run time. Non-root execution stays
`--user 65534:65534` on this image too, unchanged from Stage 5.

`evaluate run` / `evaluate candidate` verify the image is present **before** starting, and
fail with a clear "run `docker build -t python-dpo-evaluator:1.0 docker/evaluator/`"
message rather than producing 50 identical infrastructure errors.

The actual Python and pytest versions are probed once per evaluation run (one short
container) and recorded in the manifest — the versions genuinely used, not assumed (§74).

---

## Modifications to existing code

**`src/python_dpo/sandbox/workspace.py`** — generalise to a multi-file job:
`write_file(name, content)` with `write_candidate` kept as a thin wrapper so Stage 5's
tests and callers are untouched. `name` is validated as a bare filename (no separators, no
`..`), since it now comes from a caller rather than being a constant.

**`src/python_dpo/sandbox/executor.py`** — `execute_job(files, command, ...)` accepting a
mapping of filename → content and a command override, with the existing `execute(code)`
reimplemented on top of it. All bounding, timeout, classification and cleanup logic is
shared, so the evaluation path inherits Stage 5's guarantees rather than reimplementing
them. `ContainerSpec.command` is already parameterised — `container.py` needs no change.

**`src/python_dpo/config.py` + `config.yaml`** — an `evaluation:` section: `image`,
`timeout_seconds` (default 30 — pytest startup plus collection needs more headroom than
Stage 5's 5s), `startup_grace_seconds`, `auto_pull`. Parsed by `_parse_evaluation`
mirroring `_parse_sandbox`, raising its own error type translated to `ConfigError` at the
boundary.

**`src/python_dpo/cli.py`** — `evaluate` leaves `_PLACEHOLDER_STAGES` and becomes a real
group, plus an `evaluations` inspection group mirroring `runs`/`candidates`:

| Command | Behavior |
|---|---|
| `evaluate candidate --run-id R --candidate-id C` | Evaluate one candidate (§53) |
| `evaluate run --run-id R [--problem-id P] [--limit N] [--force]` | Evaluate a generation run, resuming by default (§53–§57) |
| `evaluations list EVAL_ID` | Per-candidate status table (§82) |
| `evaluations show EVAL_ID CANDIDATE_ID` | One candidate's result and its failing tests |
| `evaluations stats EVAL_ID` | The §58 counters |

Resume rule: `evaluate run --run-id R` continues the most recent evaluation run for `R`
whose status is not `completed`, skipping candidates with a persisted result. The
manifest's `requested_candidate_ids` is authoritative for what that run covers — a
selection flag that disagrees with it is an error directing the user to `--force`, the same
manifest-is-truth rule Stage 4 established.

**`data/README.md`, `README.md`, `src/python_dpo/README.md`, `tests/README.md`,
`CLAUDE.md`** — Stage 6 documentation; `CLAUDE.md` gains a line that generated *tests* are
also untrusted-adjacent and execute only in the sandbox.

**`src/python_dpo/__init__.py`** — `__version__` → `0.6.0`.

---

## Persistence layout (§48, §88)

```
data/evaluations/runs/eval_20260817_154500_a12f/
├── manifest.json        # config snapshot, versions, status, environment
├── evaluations.jsonl    # one EvaluationResult per candidate
├── test_results.jsonl   # one TestCaseResult per test case
├── failures.jsonl       # infrastructure failures, kept separate from candidate outcomes
└── statistics.json      # reconstructable from evaluations.jsonl
```

Evaluation run ids are `eval_YYYYMMDD_HHMMSS_xxxx` (§49). Every record carries
`evaluation_run_id`, `candidate_run_id`, `candidate_id`, `problem_id`, so the full chain
back to model and prompt is reconstructable (§50). The manifest records `evaluator_version`,
`test_generator_version`, `pytest_version`, `python_version`, image, digest, and the full
sandbox configuration (§51, §72–§75).

Historical evaluation runs are immutable (§71) — a re-evaluation is a new run, never an
overwrite.

---

## Tests

**No Docker required** — `tests/evaluation/`:

- **`test_models.py`** — validation of both record types; the §24 `passed` rule enforced at
  construction (a `passed` record with a failing test is rejected); `pass_rate`; the closed
  status set; dict round-trips; candidate-vs-infrastructure separation.
- **`test_test_generator.py`** — determinism (same inputs → byte-identical source); kwargs
  call shape; `repr()` literals for every JSON-native type including nested; async →
  `asyncio.run`; generator materialisation; `expected_exception` → `pytest.raises`; the bool
  guard (`True` must not satisfy `1`); one test per case with traceable ids; generated
  count equals `len(problem.tests)`; `ast.parse` accepts the output; **the generated source
  contains no `eval`/`exec`**; §47 validation rejects a short or misnamed suite.
- **`test_result_parser.py`** — all six §65 fixtures; nonce filtering; non-nonce candidate
  output ignored; malformed JSON on a nonce line is an error, not a silent skip.
- **`test_executor.py`** — a fake sandbox executor drives every classification path:
  passed, failed, runtime error → candidate failure not infrastructure, timeout, syntax
  error via collection failure, infrastructure error, zero-test problem, generation-
  validation failure. Plus discrepancy detection and **candidate immutability** (§69):
  source in equals source out.
- **`test_repository.py`** — the §52 API, the resume index, statistics reconstructed from
  records, atomic writes, malformed-line rejection with a line number.

**Docker required (`-m integration`)** — `tests/evaluation/test_integration.py`:

The six §66 candidate fixtures (correct, wrong result, syntax error, runtime exception,
infinite loop, network attempt), the §67 end-to-end flow, and the four mandatory tests:
§83 known-good → `passed` with all tests passing; §84 deliberately wrong → `failed` with the
specific failing test cases identified; §85 `while True: pass` → `timeout` with the
container cleaned up; §87 network attempt → still classified correctly with no network
access. §86's Docker-unavailable case is covered by injecting an unavailable runtime rather
than stopping the daemon. §68's arithmetic is asserted:
`passed + failed + error + skipped == total`.

**The reference-solution self-check.** For every problem, generate its suite and run it
against that problem's own `reference_solution` in the sandbox; all 74 tests must pass. This
is the sanctioned validation use of the reference solution under §9, and it is the only
thing that proves the generator's comparison semantics still match the ones the dataset was
validated under. Without it, a subtle divergence would show up as "the model is bad" rather
than "the generator is wrong."

**Extended** — `tests/sandbox/test_sandbox_security.py` gains assertions that the
*evaluation* container's argv carries every Stage 5 flag (§78), and `tests/test_project.py`
covers the new CLI groups.

---

## Execution order

1. Write this plan to `.claude/plans/06_candidate_test_executor_plan.md`.
2. `errors.py`, `models.py` + tests — pure logic, no Docker.
3. `test_generator.py` + tests, including the `ast.parse` and no-`eval` guards.
4. `result_parser.py` + the six fixture tests.
5. Sandbox multi-file extension (`write_file`, `execute_job`), keeping Stage 5 green.
6. `docker/evaluator/Dockerfile`; build; confirm pytest runs under the read-only workspace.
7. `pytest_runner.py`, `executor.py` + `test_executor.py` against a fake sandbox.
8. `repository.py` + tests; config and CLI wiring.
9. Integration suite, **starting with the reference-solution self-check**.
10. Evaluate all 50 real candidates; docs; the §90 report.

---

## Verification

```bash
source .venv/bin/activate

pytest -q                                              # offline, zero skips
docker build -t python-dpo-evaluator:1.0 docker/evaluator/
python -m python_dpo sandbox health
pytest -q -m integration                               # incl. the reference self-check

# one candidate, then the full real run
python -m python_dpo evaluate candidate --run-id 20260817_055411 --candidate-id p001_c001
python -m python_dpo evaluate run --run-id 20260817_055411

python -m python_dpo evaluations list  EVAL_ID
python -m python_dpo evaluations stats EVAL_ID

# resume: re-running must do nothing; --force must mint a new run
python -m python_dpo evaluate run --run-id 20260817_055411
python -m python_dpo evaluate run --run-id 20260817_055411 --force

# nothing was mutated, nothing leaked
git diff --stat data/problems/ data/candidates/          # empty
docker ps -a --filter name=python-dpo-sandbox            # empty
```

Scope containment:

```bash
grep -rnE "\b(exec|eval)\(" src/                    # still only InProcessReferenceExecutor
grep -rn "shell=True" src/                          # none
grep -rn "os.environ" src/python_dpo/evaluation/    # none
grep -rniE "\b(chosen|rejected|reward|preference|ranking|dpo)\b" src/python_dpo/evaluation/
```

Then produce the §90 report in `06_CANDIDATE_TEST_EXECUTOR.md` and **stop — do not start
Stage 7 (ranking / correctness classification) without explicit approval** (§90).

---

## Deviations to record in the report

- **The spec file is `06_candidate_test_executor.py`**, Markdown content with a `.py`
  extension. Left as-is unless you want it renamed.
- **`TestCase.input` is a kwargs mapping**, so generated calls are `fn(**{...})` rather than
  the positional `fn([...])` in §10/§13's examples. The dataset's shape wins.
- **§18's `approx` comparison mode is not implemented.** No problem has a float expected
  value, and §17 says not to expand the schema unnecessarily. The generator's comparison is
  a single helper, so adding a mode later is a contained change.
- **Resume is the default and there is no `--resume` flag**, per §56/§57 — deliberately
  diverging from Stage 4's `generate`, where you chose always-new-run with explicit resume.
  Justified by evaluation being idempotent and cheap where generation burns GPU time.
- **Results are captured via a generated `conftest.py` emitting nonce-prefixed JSON lines**,
  not JUnit XML or pytest-json-report (§32 permits choosing the simplest reliable
  mechanism). No dependency, and no report-file extraction from a read-only container.
- **The nonce is anti-collision, not anti-adversary.** A candidate that reads its own
  workspace can forge result lines; §46's test-count cross-check is what catches a forged
  or truncated result set. Consistent with the threat model already stated in
  `docs/sandbox-security.md`.
- **Async handled with `asyncio.run`, not pytest-asyncio** — no dependency, and it mirrors
  `InProcessReferenceExecutor`.
- **Comparison semantics intentionally replicate `InProcessReferenceExecutor._values_match`**,
  including the guard stopping `True` from satisfying `1`, because the dataset's expected
  values were validated under exactly those rules.
- **`pytest==8.3.4` is added to the evaluation image only** — not to the host project's
  dependencies, which stay PyYAML plus the `dev`/`model` extras.
- **`evaluate` is removed from `_PLACEHOLDER_STAGES`**; `preferences` and `run` remain.
