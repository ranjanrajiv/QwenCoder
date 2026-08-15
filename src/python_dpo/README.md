# src/python_dpo/

The `python_dpo` package — the installable core of the project. This is Step 1 of the
pipeline (see the root [README.md](../../README.md)): packaging, CLI, logging, and
configuration only. No model, dataset, sandbox, evaluation, or training code lives here
yet.

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
- Each subcommand is wired to a placeholder handler produced by
  `_make_placeholder_handler(name)`. Every handler logs
  `"<Stage> is not implemented yet."` through the module logger and **returns exit code
  1** — never a fake success (spec §7).
- `main(argv=None) -> int` — parses arguments, loads `Config` (via `config.py`),
  configures logging, and dispatches to the chosen subcommand handler. `--help` and
  `--version` are handled entirely by argparse before any config loading happens, so
  they always succeed even if `config.yaml` is broken. If `Config.load()` raises
  `ConfigError`, the message is logged and `main()` returns exit code 2. With no
  subcommand given, it prints help and returns 1.

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
- `Config` — a frozen dataclass holding `project_name`, `paths`, `log_level`, and
  `project_root`. `Config.load(path=None)` reads and validates `config.yaml` with
  `yaml.safe_load` (never `yaml.load`), checking every required key is present and of
  the right type, raising `ConfigError` otherwise.

### `logging_config.py`

`configure_logging(level: str = "INFO") -> None` — attaches a single `StreamHandler`
(stderr) to the `python_dpo` logger, using the format
`"%(asctime)s | %(levelname)s | %(name)s | %(message)s"` (matches the example in spec
§8). Idempotent — calling it more than once (e.g. across repeated CLI invocations in
tests) is a no-op after the first call, so output is never duplicated. Raises
`ValueError` on an unrecognized level name.
