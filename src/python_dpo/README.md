# src/python_dpo/

The `python_dpo` package — the installable core of the project. Through Stage 3 (see the
root [README.md](../../README.md)) it holds the foundation — packaging, CLI, logging,
configuration — the problem dataset, the model abstraction, and candidate generation. No
sandbox, evaluation, ranking, or training code lives here yet.

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
(non-executing) validation, and the `CandidateGenerator` that orchestrates them.

### [`candidates/`](candidates/)

The `Candidate` and `GenerationFailure` schema plus the append-only JSONL repository that
backs resume, `--force`, and duplicate detection.

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

### `cli.py`

The command-line interface, built on stdlib `argparse` only (no Click/Typer — spec
keeps runtime dependencies minimal).

- `build_parser() -> ArgumentParser` — a factory function (not a module-level parser)
  so tests can construct an isolated parser without touching global state. Registers
  `--version` (via argparse's built-in `action="version"`) and `--log-level`, plus five
  subcommands: `problems`, `generate`, `evaluate`, `preferences`, `run`.
- `generate` is implemented (Stage 3). `_cmd_generate` loads the dataset, narrows it with
  `--problem-id` / `--limit`, resolves strategies, and either prints prompts (`--dry-run`)
  or builds a client and runs `CandidateGenerator`. Prompts are built *before* any client
  exists, so a dry run cannot load a model even by accident. `--mock-model` swaps in the
  deterministic client for an offline end-to-end run. Generation failures are recorded and
  reported but do **not** make the command fail — they are data, and the run genuinely
  ran.
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
- `GenerationSettings` — the Stage 3 `generation:` and `generation_strategies:` sections:
  `candidates_per_problem`, a validated `GenerationConfig`, and the strategy list. The
  typed model objects themselves live in `models/base.py`, so the dependency runs one way
  — configuration imports the model layer, never the reverse.
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
