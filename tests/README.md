# tests/

The test suite for Step 1. Everything here is offline and CPU-only (spec §14) — no
network access, no GPU, no Docker, no Qwen model.

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
- `test_placeholder_subcommands_parse_and_return_nonzero` — parametrized over all five
  subcommands; each one's handler must return non-zero. Guards against a future
  handler accidentally reporting fake success.
- `test_no_subcommand_prints_help_and_returns_nonzero` — running the CLI with no
  arguments prints help and exits 1.

No test is skipped — spec §15 requires `pytest -q` to fully pass with zero skips.
