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

No test is skipped — `pytest -q` must fully pass with zero skips.
