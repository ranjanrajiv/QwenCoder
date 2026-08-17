# tests/

The project's test suite. The default run (`pytest -q`) is offline and CPU-only — no
network access, no GPU, no Docker, no Qwen model — and nothing is skipped.

Docker tests — the sandbox security suite and the candidate test executor's integration
suite — are marked `@pytest.mark.integration` and **deselected by default**
(`addopts = "-ra -m 'not integration'"` in `pyproject.toml`), which is what preserves the
zero-skip property on every machine. Run them explicitly:

```bash
pytest -q                  # the offline suite
pytest -q -m integration   # Docker: sandbox security + candidate evaluation
pytest -q -m ""            # everything
```

They deliberately **fail** rather than skip when Docker is unreachable: they were asked for
explicitly, so silently passing an unrun security suite would be the worst outcome.

## Files

### `__init__.py`

Empty. Present so `tests` is an importable package, which keeps test discovery and
imports (e.g. `from tests...`) unambiguous; it has no behavior of its own.

### `test_project.py`

One test per requirement in spec §14, plus a couple of guardrails:

- `test_import_succeeds` — `import python_dpo` works (§14.1).
- `test_version_is_a_nonempty_dotted_string` — `__version__` exists, is a non-empty
  `str`, and parses as a dotted numeric version like `0.1.0` (§14.2).
- `test_config_loads_real_config_yaml` — `Config.load()` against the real
  `config.yaml` returns the correct project name and six absolute paths, each rooted
  under the project directory (§14.3, happy path).
- `test_config_load_raises_on_malformed_yaml` — an incomplete YAML file (missing the
  `paths` section) raises `ConfigError` with a message mentioning what's missing
  (§14.3, negative case).
- `test_config_load_raises_on_missing_file` — a nonexistent config path raises
  `ConfigError` with a "not found" message.
- `test_paths_ensure_exists_creates_all_directories` — `Paths.ensure_exists()` creates
  all six directories under a `tmp_path` (§14.4, isolated).
- `test_real_data_directories_exist` — the six real `data/*` directories exist in the
  repo (§14.4, real repo state).
- `test_cli_help_exits_zero` / `test_cli_version_exits_zero_and_prints_version` — run
  `python -m python_dpo --help` / `--version` as a subprocess (with `PYTHONPATH=src`
  injected, so this passes with or without an editable install) and check exit code 0
  (§14.5).
- `test_placeholder_subcommands_parse_and_return_nonzero` — parametrized over the
  subcommands that are still placeholders; each one's handler must return non-zero.
  Guards against a future handler accidentally reporting fake success. Stage 2 narrowed
  this from "all five subcommands" when `problems` became real — a requirement change,
  not a test bent to fit an implementation.
- `test_problems_is_no_longer_a_placeholder`, `test_problems_subcommands_are_wired`,
  `test_bare_problems_prints_help_and_returns_nonzero` — the `problems` command group
  dispatches `build` and `validate`, and bare `problems` exits 1.
- `test_no_subcommand_prints_help_and_returns_nonzero` — running the CLI with no
  arguments prints help and exits 1.
- `test_evaluate_is_not_a_placeholder`, `test_evaluate_subcommands_parse`,
  `test_bare_evaluate_prints_help_and_returns_nonzero`,
  `test_evaluate_candidate_reports_an_unknown_run_id`,
  `test_evaluate_run_reports_an_unknown_run_id` (Stage 6) — `evaluate candidate`/
  `evaluate run` flag parsing, and that an unknown generation run id is rejected
  *before* any Docker work begins, so these stay fast and Docker-free.
- `test_evaluations_subcommands_parse`, `test_bare_evaluations_prints_help_and_returns_nonzero`,
  `test_evaluations_list_reports_an_unknown_eval_id`,
  `test_evaluations_show_reports_an_unknown_eval_id`,
  `test_evaluations_stats_reports_an_unknown_eval_id` — the `evaluations` inspection
  group.

### `test_problems.py`

Unit tests for the Stage 2 dataset layer (spec 02 §32), grouped by concern:

- **Schema** — a valid problem constructs; empty prompt/signature/reference, missing
  required fields, unknown fields, invalid category, invalid difficulty, a signature
  that doesn't declare its entry point, and an empty test list are all rejected with
  `ProblemError`.
- **Test cases** — valid cases round-trip; a non-mapping `input`, a case setting both
  `expected` and `expected_exception`, a malformed record, and duplicate test ids
  within a problem are rejected.
- **Executor** — passing and failing results, unexpected exceptions, matched and missing
  `expected_exception`, a missing entry point, generator materialization, coroutine
  functions, and the strict comparison that stops `True` passing as `1`.
- **Validation** — too few tests, duplicate problem ids, the wrong problem count, and a
  deliberately broken reference solution each make the dataset invalid.
- **Storage** — JSONL round-trip equality, parent-directory creation, one object per
  line, and rejection of malformed JSON (with its line number), invalid records,
  duplicate ids, and a missing file.

### `test_problems_integration.py`

The end-to-end round trip (spec 02 §33): build the ten problems → validate → run every
reference test → write JSONL → reload → validate again → PASS. Also covers determinism
(two builds are byte-identical), the catalog's shape (ten ids, all ten categories, the
5/4/1 difficulty split, ≥5 tests each), that each stored `reference_solution` is
verbatim the code that runs, and the two properties that can't be expressed as
input/expected pairs:

- `chunk_sequence` really is a generator, yielding chunks one at a time.
- `gather_in_order` really runs concurrently — five 0.05s operations finish well under
  the 0.25s a sequential loop would take, with a wide margin so it can't flake — and
  returns input order rather than completion order.

### `test_models.py`

The Stage 3 model layer, with no GPU, weights, or network involved.

- **`GenerationConfig`** — defaults are valid; out-of-range `top_p`, non-positive
  `max_new_tokens`, negative `temperature`/`seed`, and non-numeric values are rejected;
  `bool` is rejected where a number is expected, since `max_new_tokens=True` would
  otherwise silently mean one token; sampling with `temperature=0` is refused while greedy
  decoding at 0 is allowed; dict round-trip.
- **`ModelConfig`** — provider/device/dtype validation, `cuda:1` accepted, defaults
  applied by `from_mapping`, unknown and missing keys rejected, `trust_remote_code`
  defaulting to `False`, and `quantization` refused rather than silently ignored.
- **`MockModelClient`** — satisfies the protocol, is deterministic across separate
  instances, varies by prompt, honors a script in order, raises scripted exceptions, and
  counts calls.
- **`QwenModelClient`** — satisfies the protocol *without loading anything*, refuses a
  non-transformers provider, and turns a missing backend into a `ModelLoadError` naming
  the `[model]` extra (simulated by putting `None` in `sys.modules["torch"]`).
- **Pure helpers** — `resolve_device` across the `auto`/`cpu`/`cuda` matrix including the
  refusal to silently downgrade an explicit `cuda` request to CPU, `resolve_dtype`, and
  `build_generation_kwargs` omitting sampling parameters when decoding greedily.

### `test_generation.py`

- **Strategies** — the five exist in spec order with distinct instructions; five
  candidates get one strategy each; `--strategy` override; counts beyond five cycle;
  unknown names and non-positive counts raise.
- **Prompt builder** — problem text, signature, strategy instruction, and output rules all
  present; deterministic across calls; different per strategy; version declared.
- **Code extractor** — python fence, generic fence, plain code, explanatory prefix, a
  python fence preferred over a later generic one, internal formatting preserved exactly,
  empty and prose-only output rejected, a generic fence without code rejected, an
  **unterminated fence not repaired**, and extraction succeeding on syntactically invalid
  code (extraction and parsing are separate concerns).
- **Validators** — valid and invalid syntax, null bytes, matching and mismatched function
  names, a missing function, `async def`, nested definitions, and unparseable input.

### `test_candidates.py`

- **Schema** — deterministic zero-padded ids; round-trip; a syntax-invalid candidate is a
  *valid record*; each validation rule rejected individually; unknown and missing fields;
  `error_type` closed set, including that `syntax_error` is not a member.
- **Schema versioning (Stage 4)** — `Candidate.create` computes and verifies all three
  SHA-256 hashes; a tampered hash is rejected at construction; a schema-2.0 record
  requires every hash; a legacy 1.0 record has null hashes and still loads; a record
  missing `schema_version` entirely reads as 1.0; a 1.0 record populated with a hash is
  rejected; the same shape for `GenerationFailure.prompt_sha256`.
- **Repository** — save/load round-trip; records readable before a run finishes; failures
  kept in a separate `failures.jsonl`; prompts persisted and loadable; `existing_keys`
  driving resume, run-scoped; a previously failed generation *not* blocking a retry;
  `code_index` keyed on `code_sha256`, scoped per problem; the spec 04 §23 lookup API
  (`get`, `exists`, `list`, `count`, `find_by_problem`, `find_by_hash`); malformed lines
  and a truncated final line each rejected with a line number.

### `test_atomic_io.py`

The durable-write primitives underneath every repository (spec 04 §21).

- `atomic_write_json` — creates, replaces, leaves no `.tmp` behind, creates parent
  directories, and a failed replace leaves the original file untouched.
- `append_jsonl` / `iter_jsonl` — one complete line per append; a missing file yields
  nothing; invalid JSON, a non-object line, and a blank line are each rejected with a line
  number; a final line with no trailing newline is detected as a torn write, distinct from
  a mid-file corruption (which is also rejected, but with its own message).
- `repair_truncated_tail` — removes exactly the torn bytes and nothing else; a no-op on a
  well-formed file; never touches a corrupt line earlier in the file.

### `test_runs.py`

`RunManifest`, `RunStatistics`, and `RunRepository` (spec 04).

- **Manifest** — round-trip; unknown status rejected; `requested_problems` derived;
  strategies-count/candidates-per-problem mismatch rejected; duplicate problem ids
  rejected; missing `retry.max_attempts` rejected; `with_status` enforces the allowed
  transition graph and carries a `RunFailure`.
- **Statistics** — round-trip; every counter matches a hand-count of given records;
  `problems_completed` requires every requested index to have *an outcome* (candidate or
  failure), not that every one succeeded; `retry_attempts` counts only infrastructure
  failures; duplicates counted.
- **Environment** — `capture_environment()` never contains a username, home path, or
  token.
- **Repository** — run-id format and uniqueness; create/get/list (newest first, tie-broken
  by run id); the full status lifecycle (`created → running → completed`,
  `interrupted → running`); resume refuses a `completed` run; `fail_run` records the
  error; `create_run_from` seeds a new run with a fresh id and identical configuration;
  statistics write/read round-trip; `candidates(run_id)` returns a repository scoped to
  that run's directory.

### `test_run_validation.py`

One test per spec 04 §51 corruption, each built by mutating a real, valid, completed run
directory produced by the actual generator and mock model: missing manifest, malformed
JSON, duplicate candidate id, wrong `code_sha256` (names the candidate), missing required
field, mismatched `run_id`, unknown `problem_id`, drifted `statistics.json`, a truncated
tail, a dangling `duplicate_of`, a `completed` run with missing work, and a prompt hash
absent from `prompts.jsonl`. Each must fail loudly; a clean run must still pass.

### `test_migration.py`

Migrating the Stage 3 flat file into run directories (spec 04 §46): hashes are
back-filled and `schema_version` stamped; the source file is byte-identical afterward; the
migrated run passes `validate_run`; migrating twice without `--force` refuses to clobber;
`--force` overwrites cleanly rather than duplicating records; multiple `run_id`s in one
source file produce multiple run directories.

### `test_generation_pipeline.py`

The end-to-end generation tests (spec 03 §46, §47; spec 04 §42, §49, §50), all driven by
`MockModelClient` against real run directories built through `RunRepository`.

- **Integration** — one problem, five candidates: five records, correct problem id, unique
  ids in the documented shape, all five strategies represented, raw output kept alongside
  extracted code, syntax validated, config embedded, hashes verified, all persisted, no
  failures file; prompts persisted before inference and linked by hash to their candidate.
- **Resume** — a second `generate()` call against the *same* run directory generates
  nothing and, critically, `call_count` proves the model was never asked; `--force`
  (`create_run_from`) seeds a brand-new run with the earlier one preserved byte-for-byte;
  a previously failed generation is retried on resume.
- **Retries** — an infrastructure failure followed by success keeps the attempt-1 failure
  record and stamps `attempt=2` on the candidate; exhausting `max_attempts` leaves only
  failure records; a candidate failure (empty output) never consumes a retry attempt.
- **Failures** — empty response, unextractable output, and inference errors each produce a
  failure record and no candidate while the run continues; a model-load failure records
  one failure and aborts; **a syntax error produces a candidate and no failure record**;
  a wrong function name is recorded rather than rejected.
- **Duplicates** — exact duplicates flagged and kept within a run; duplicates are **not**
  auto-linked across runs even when the deterministic mock reproduces identical code
  (spec 04 §20) — `find_by_hash` is the tool for that cross-run analysis instead; distinct
  code left unflagged.
- **§42/§49 mandatory resume test** — 3 problems × 5 candidates; a scripted
  `KeyboardInterrupt` after 7 candidates leaves the run `interrupted`; resuming fills the
  remaining 8; the final 15 candidates' first 7 records are byte-for-byte unchanged; the
  run ends `completed`.
- **§50 reproducibility test** — two runs with identical problems, mock, prompt version,
  generation config, and strategies produce identical `code_sha256` per
  `(problem_id, generation_index)`, differing only in `run_id`. Real-model reproducibility
  is explicitly not claimed.

### `sandbox/`

The Stage 5 sandbox suite, split by what it needs.

**No Docker required:**

- **`test_config.py`** — defaults; every validation rule; unknown keys rejected;
  `:latest` and unpinned images rejected; `network_mode` values other than `none` rejected;
  UID 0 and named users rejected; digest pinning; the spec §52 environment record; and that
  `config.py` wraps `SandboxConfigError` as `ConfigError` at the package boundary.
- **`test_result.py`** — `ExecutionResult` validation and dict round-trip; the closed status
  set; `timed_out` must agree with `status`; candidate vs infrastructure outcomes. Then
  `classify()` driven as a pure function across every branch, including the case that
  motivates the design: a program that does `raise SyntaxError(...)` compiled fine and must
  be reported as a **runtime** error, which is distinguishable because CPython prints
  `Traceback (most recent call last):` for runtime exceptions and never for compile
  failures.
- **`test_workspace.py`** — `candidate.py` written byte-for-byte (compared as bytes, since
  text-mode newline translation would silently rewrite `\r\n`); directory and file modes
  readable by the container's non-root UID; cleanup on success, on exception, and when
  called twice; the workspace holds only the candidate file and lives outside the project
  tree.
- **`test_sandbox_security.py`** — the argv-level guard, and the most important test in the
  stage. Asserts both halves of the contract: every mandatory isolation flag is present
  (`--network none`, `--read-only`, `--user`, `--cap-drop ALL`, `--pids-limit`, `--memory`,
  `--memory-swap`, `--cpus`, a `:ro` workspace mount, `--workdir`), and the dangerous ones
  never appear (`--privileged`, `--pid=host`, any `docker.sock` mount, the project
  directory, any host env var beyond the three passed deliberately). Plus source-level
  scans proving the package contains no `shell=True`, no `exec`/`eval`/`os.system`, and no
  `os.environ` pass-through. Same philosophy as `test_no_heavy_imports.py`: a rule one
  careless edit away from being broken is asserted, not assumed. Extended in Stage 6 with
  the same assertions run against the *evaluation* container's argv (built through
  `build_evaluation_sandbox_config`), proving adding pytest did not weaken any Stage 5
  isolation flag, and that the pytest command is a fixed argument list ending in
  `test_candidate.py`, never a shell string.
- **`test_executor_mock.py`** — a `FakeContainerRuntime` drives the executor through
  success, runtime error, syntax error, timeout (asserting the container is killed), output
  flood (truncated, terminated, and capped in memory), OOM → `resource_exceeded`, and
  Docker-unavailable → `infrastructure_error` returned rather than raised. Cleanup is
  verified on **every** path — container removed and workspace deleted even when the
  runtime fails mid-execution. One test pins the threading rule directly: the bounded
  reader must hold no container callback, because killing from a reader thread while the
  main thread waits on the same process hangs and leaks the container.

**Docker required (`-m integration`):**

- **`test_sandbox_integration.py`** — all ten mandatory security checks against a live
  daemon: normal execution, infinite loop → `timeout`, outbound connections, DNS, HTTP,
  non-root UID, host environment isolation, `/var/run/docker.sock` absent, runtime error,
  syntax error, output flood. Plus memory and PID limits, read-only root and workspace,
  the writable tmpfs, filesystem isolation against a host marker file, container ID
  recording, and the health check. An autouse fixture asserts after **every** test that no
  sandbox container survived it.

  Two tests are worth singling out: `test_candidate_code_never_runs_on_the_host` executes a
  candidate that would create a host file if it escaped, and asserts the host is untouched;
  `test_container_environment_is_only_the_image_plus_our_three_variables` compares the
  container's environment against the bare image's, which is precise where a keyword scan
  for credential-shaped names is not (the stock image legitimately defines `GPG_KEY`).

### `evaluation/`

The Stage 6 candidate test executor suite, split by what it needs — same split as
`sandbox/`.

**No Docker required:**

- **`test_models.py`** — the schema: `EvaluationResult`'s `passed`-requires-every-test
  invariant enforced at construction (exit code 0 alone never produces `passed`); the
  four counts always summing to `tests_total`; the boolean flags matching `status`; the
  closed status sets; dict round-trips; `EvaluationManifest`'s status transition graph;
  `EvaluationStatistics.from_records` against hand-counted fixtures.
- **`test_config.py`** — `EvaluationConfig` defaults, `:latest`/unpinned image rejection,
  unknown keys, and that `config.py` wraps `EvaluationConfigError` as `ConfigError` at
  the package boundary — the same shape as `sandbox/test_config.py`.
- **`test_test_generator.py`** — determinism; the `repr()`-literal, kwargs-call,
  `asyncio.run`, generator-materialization, and `pytest.raises` code-generation rules;
  the bool guard stopping `True` from satisfying `1`; one test function per case with a
  traceable id; `ast.parse` accepts the generated source; the generated source contains
  no `eval`/`exec`; `validate()` rejects a short or misnamed suite.
- **`test_result_parser.py`** — all classification fixtures (all pass, partial failure,
  runtime error, collection/syntax error, timeout, skipped); nonce filtering ignoring
  ordinary candidate `print()` output; malformed JSON on a nonce line rejected, not
  silently skipped. Includes real-pytest-subprocess regression tests for two bugs found
  during manual verification: pytest's `-q` progress character (`.`/`F`/`s`) is written
  on the same line as the nonce with no separating newline, so the parser must search for
  the nonce as a substring, not `startswith`; and `pytest.raises(...)` raising its own
  `Failed` exception on "DID NOT RAISE" must classify as a wrong-answer `failed`, not a
  candidate `error`.
- **`test_pytest_runner.py`** — `build_evaluation_sandbox_config` overlays only the four
  evaluation-specific fields and inherits every isolation setting from the base sandbox
  config unchanged; `PytestRunner.run` delegates to `execute_job` with the fixed pytest
  command and every job file untouched.
- **`test_probe.py`** — `probe_versions` parses a scripted probe result; a failed or
  malformed probe raises `EvaluationError` rather than propagating a raw parse error.
- **`test_executor.py`** — every classification path driven by a fake sandbox runner:
  passed, failed, runtime error (candidate failure, never infrastructure), timeout,
  syntax error via collection failure, infrastructure error, a zero-test problem, and a
  generation-validation failure. Plus discrepancy detection against Stage 3's
  `syntax_valid` and candidate immutability (source in equals source out).
- **`test_repository.py`** — the lookup API, `evaluated_keys()` covering both results and
  failures (the Stage 6 resume index — deliberately unlike Stage 4's generation
  failures, which *are* retried), malformed-line rejection with a line number, a
  truncated final line.
- **`test_run_repository.py`** — evaluation-run-id format and uniqueness; create/get/list
  (newest first); `latest_run_for_candidate_run` (the resume lookup `evaluate run` uses);
  the full status lifecycle and resume-refuses-`completed` rule, mirroring
  `test_runs.py`'s `RunRepository` coverage exactly.

**Docker required (`-m integration`):**

- **`test_integration.py`** — the six candidate fixtures (correct, wrong result, syntax
  error, runtime exception, infinite loop, network attempt) run for real: a known-good
  candidate passes every test; a deliberately wrong one fails with the *specific* test
  ids identified; an infinite loop times out with the container cleaned up; an injected
  Docker-unavailable runtime produces `infrastructure_error` without stopping the real
  daemon; a network attempt is still classified as a candidate failure, never
  infrastructure trouble, because the connection attempt raises inside the sandboxed
  test. Plus the result-arithmetic invariant (`passed+failed+error+skipped == total`) and
  an end-to-end `evaluate_many` persisting every result.

  **The reference-solution self-check** —
  `test_every_real_problems_generated_suite_passes_against_its_own_reference_solution` —
  generates the real pytest suite for every problem in the committed dataset and runs it
  against that problem's own trusted `reference_solution` inside the real sandbox; every
  one must pass. This is the one test that would catch a subtle divergence between
  `TestGenerator`'s comparison semantics and what the dataset was actually validated
  under — without it, such a divergence would show up as "the model is bad" rather than
  "the generator is wrong."

### `test_no_heavy_imports.py`

A subprocess imports every `python_dpo` module and asserts `torch`, `transformers`, and
`accelerate` are absent from `sys.modules`. Spec 03 §7's lazy-loading rule is one stray
top-level `import torch` away from being broken, so it is asserted rather than assumed.
This test only means something when the `[model]` extra is actually installed — which is
precisely when the rule could regress.

No test is skipped, and none loads the real Qwen model — `pytest -q` must fully pass with
zero skips, offline, on CPU. The Docker suite is opt-in via `-m integration` and is where
the sandbox's security guarantees, and the candidate test executor's classification
behavior, are demonstrated against real containers.
