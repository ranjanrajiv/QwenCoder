# src/python_dpo/

The `python_dpo` package — the installable core of the project. Through Stage 4 (see the
root [README.md](../../README.md)) it holds the foundation — packaging, CLI, logging,
configuration — the problem dataset, the model abstraction, candidate generation, and a
reliable per-run persistence layer. No sandbox, evaluation, ranking, or training code
lives here yet.

## Subpackages

### [`problems/`](problems/)

The ground-truth layer: the problem/test-case schema, the ten curated problems and their
trusted reference solutions, JSONL persistence, the swappable `ReferenceExecutor`, and
dataset validation. See its [README](problems/README.md) for a file-by-file breakdown.

### [`models/`](models/)

The inference seam: the `ModelClient` protocol, `GenerationConfig`/`ModelConfig`, the
lazily loaded `QwenModelClient`, and the deterministic `MockModelClient`. Nothing outside
this package imports `torch` or `transformers`, and importing it loads neither.

### [`generation/`](generation/)

Prompt construction, the five generation strategies, code extraction, static
(non-executing) validation, and the `CandidateGenerator` that orchestrates them. Takes a
`RunManifest` (from `runs/`) rather than loose arguments, so a run's own configuration —
not today's `config.yaml` — always drives generation and resume.

### [`candidates/`](candidates/)

The `Candidate` and `GenerationFailure` schema (schema 2.0: SHA-256 hashes, retry
`attempt`) plus the durable, **run-scoped** JSONL repository that backs resume and
duplicate detection. See its [README](candidates/README.md).

### [`runs/`](runs/)

Introduced in Stage 4. Run manifests, statistics reconstructable from disk, the
`RunRepository` that owns run directories and status transitions, migration of the
Stage 3 flat file, and integrity validation. See its [README](runs/README.md).

## Files

### `__init__.py`

Defines `__version__ = "0.1.0"` and `__all__ = ["__version__"]`. This is the single
source of truth for the package version — `pyproject.toml` reads it dynamically via
`[tool.setuptools.dynamic]` so the two can never drift. Deliberately has no other
imports, so `import python_dpo` stays cheap and has no dependency side effects.

### `__main__.py`

```python
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

Enables `python -m python_dpo ...`. Without this file the package would import fine
but `python -m python_dpo` would fail — this is required by spec §7 but isn't in the
spec's literal file tree.

### `atomic_io.py`

Introduced in Stage 4. `atomic_write_json` (temp file + fsync + `os.replace`, for
`manifest.json`/`statistics.json`) and `append_jsonl`/`iter_jsonl` (fsynced single-line
appends, with `iter_jsonl` treating a final line with no trailing newline as a torn write
rather than silently ignoring it). `repair_truncated_tail` is the one place a torn tail is
fixed, and it is never called automatically — only from `runs validate --repair`.

### `cli.py`

The command-line interface, built on stdlib `argparse` only (no Click/Typer — spec
keeps runtime dependencies minimal).

- `build_parser() -> ArgumentParser` — a factory function (not a module-level parser)
  so tests can construct an isolated parser without touching global state. Registers
  `--version` (via argparse's built-in `action="version"`) and `--log-level`, plus
  `problems`, `generate`, `runs`, `candidates`, and the `evaluate`/`preferences`/`run`
  placeholders.
- `generate` is implemented (Stage 3/4). `_cmd_generate` dispatches to
  `_cmd_generate_fresh` (loads the dataset, narrows it with `--problem-id`/`--limit`,
  resolves strategies, either prints prompts via `--dry-run` or creates a new run and
  calls `CandidateGenerator`) or `_cmd_generate_resume` (rejects any selection flag —
  the target run's manifest is authoritative — and either resumes it or, with `--force`,
  seeds a new run from its manifest). `_execute_run` owns the shared status lifecycle:
  `KeyboardInterrupt` marks the run `interrupted` rather than propagating as a crash,
  `ModelLoadError` marks it `failed` with the error recorded, and on a clean return the
  run is marked `completed` only if every requested candidate has an outcome. Prompts are
  built *before* any client exists, so a dry run cannot load a model even by accident.
- `runs list` / `runs show` / `runs validate [--repair]` inspect and validate one run via
  `RunRepository` / `validate_run`.
- `candidates list` / `candidates show [--show-code] [--show-raw]` / `candidates stats` /
  `candidates migrate [--source] [--force]` inspect one run's candidates or upgrade the
  Stage 3 legacy flat file into a run directory.
- `problems` is implemented (Stage 2) and owns two subcommands:
  - `_cmd_problems_build` — builds the curated catalog, validates it, and writes
    `data/problems/problems.jsonl`. It writes **nothing** unless the whole dataset
    validates, so a failed build can't leave a half-trustworthy artifact behind.
  - `_cmd_problems_validate` — reloads the persisted dataset, re-runs every reference
    test, writes the summary to stdout, and returns non-zero on failure. Strictly
    read-only.
  - Bare `problems` prints help and returns 1, mirroring the top-level behavior.
- The remaining stages in `_PLACEHOLDER_STAGES` are wired to a placeholder handler from
  `_make_placeholder_handler(name)`, which logs `"<Stage> is not implemented yet."` and
  **returns exit code 1** — never a fake success (spec 01 §7).
- Handlers take `(args, config)`, so commands resolve paths from configuration rather
  than hardcoding them.
- `main(argv=None) -> int` — parses arguments, loads `Config` (via `config.py`),
  configures logging, and dispatches to the chosen subcommand handler. `--help` and
  `--version` are handled entirely by argparse before any config loading happens, so
  they always succeed even if `config.yaml` is broken. If `Config.load()` raises
  `ConfigError`, the message is logged and `main()` returns exit code 2. With no
  subcommand given, it prints help and returns 1.

  The validation summary is the one place the application writes to stdout instead of
  the log stream: it is user-facing report output, not diagnostics, and stays pipeable.

### `config.py`

A small, typed configuration abstraction — the "concrete reason" the project depends
on `PyYAML` (spec §6 exception).

- `ConfigError(Exception)` — raised with actionable messages, e.g.
  `"config.yaml: missing required key 'paths.problems'"`.
- `find_project_root(start=None) -> Path` — walks upward from `start` (default: CWD)
  looking for `pyproject.toml`; falls back to a path relative to this file if none is
  found. This is what lets the CLI work without depending on a specific working
  directory (spec §11).
- `Paths` — a frozen dataclass holding the six absolute data directory paths (`raw`,
  `problems`, `candidates`, `evaluations`, `preferences`, `reports`). Its
  `ensure_exists()` method creates all six with `mkdir(parents=True, exist_ok=True)`.
- `GenerationSettings` — the `generation:` and `generation_strategies:` sections:
  `candidates_per_problem`, a validated `GenerationConfig`, the strategy list, and (Stage
  4) `retry: RetrySettings` from `generation.retry.max_attempts`. The typed model objects
  themselves live in `models/base.py`, so the dependency runs one way — configuration
  imports the model layer, never the reverse.
- `Config` — a frozen dataclass holding `project_name`, `paths`, `log_level`,
  `project_root`, `model`, and `generation`. `Config.load(path=None)` reads and validates
  `config.yaml` with
  `yaml.safe_load` (never `yaml.load`), checking every required key is present and of
  the right type, raising `ConfigError` otherwise.

### `logging_config.py`

`configure_logging(level: str = "INFO") -> None` — attaches a single `StreamHandler`
(stderr) to the `python_dpo` logger, using the format
`"%(asctime)s | %(levelname)s | %(name)s | %(message)s"` (matches the example in spec
§8). Idempotent — calling it more than once (e.g. across repeated CLI invocations in
tests) is a no-op after the first call, so output is never duplicated. Raises
`ValueError` on an unrecognized level name.
