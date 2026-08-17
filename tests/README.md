# tests/

The project's test suite. Everything here is offline and CPU-only — no network access,
no GPU, no Docker, no Qwen model — and nothing is skipped.

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
- **Repository** — append/load round-trip; records readable before a run finishes;
  failures kept in a separate file; `existing_keys` driving resume; a previously failed
  generation *not* blocking a retry; `code_index` reporting the earliest match and scoping
  per problem; `latest_by_candidate_id` preferring the newer run; run-id disambiguation
  within one second (including a failure-only run); malformed lines rejected with a line
  number.

### `test_generation_pipeline.py`

The end-to-end generation tests (spec 03 §46, §47), all driven by `MockModelClient`.

- **Integration** — one problem, five candidates: five records, correct problem id, unique
  ids in the documented shape, all five strategies represented, raw output kept alongside
  extracted code, syntax validated, config embedded, all persisted, no failures file.
- **Resume** — a second run generates nothing and, critically, `call_count` proves the
  model was never asked; `--force` appends a new run with the earlier one preserved
  verbatim; a previously failed generation is retried.
- **Failures** — empty response, unextractable output, and inference errors each produce a
  failure record and no candidate while the run continues; a model-load failure records
  one failure and aborts; **a syntax error produces a candidate and no failure record**;
  a wrong function name is recorded rather than rejected.
- **Duplicates** — exact duplicates flagged and kept, detection spanning runs so a resumed
  run notices it reproduced earlier code, a regenerated candidate not flagged as a
  duplicate of itself, and distinct code left unflagged.

### `test_no_heavy_imports.py`

A subprocess imports every `python_dpo` module and asserts `torch`, `transformers`, and
`accelerate` are absent from `sys.modules`. Spec 03 §7's lazy-loading rule is one stray
top-level `import torch` away from being broken, so it is asserted rather than assumed.
This test only means something when the `[model]` extra is actually installed — which is
precisely when the rule could regress.

No test is skipped, and none loads the real Qwen model — `pytest -q` must fully pass with
zero skips, offline, on CPU.
