# Step 1 — Project Skeleton

## Context

`.claude/specs/01_project_skeleton.md` defines a 12-step build of a DPO preference-data
generation pipeline for fine-tuning a Qwen Coder model on Python tasks. Step 1 is
foundation only: an installable package, a CLI with honest placeholder commands, logging,
typed config, tests. No model, dataset, sandbox, evaluation, or training code — those are
later steps and the spec explicitly forbids them now.

The repo today contains only `LICENSE`, a one-line `README.md`, a stock Python
`.gitignore`, and the spec itself. Everything below is new.

**Outcome:** `pip install -e ".[dev]"`, `python -m python_dpo --help/--version`, and
`pytest -q` all succeed on a CPU-only machine with no network.

### Decisions confirmed with the user

1. Skeleton lives **at the repo root** (`/home/rajiv/QwenCoder/QwenCoder`), not in a
   nested `python-dpo/`. `python-dpo` is the distribution name in `pyproject.toml`.
2. **PyYAML is the one runtime dependency** — the "concrete reason" exception in spec §6,
   since `config.yaml` needs a parser and a hand-rolled one is not worth its own tests.
3. **`.claude/specs/` and `.claude/plans/` become tracked**; the rest of `.claude` stays
   ignored. Root `specs/` gets a README pointing at the real location.

---

## Files to create

### Packaging

**`pyproject.toml`**
- `[build-system]`: `setuptools>=68`, backend `setuptools.build_meta`.
- `[project]`: name `python-dpo`, `requires-python = ">=3.11"`, `dynamic = ["version"]`.
- `dependencies = ["PyYAML>=6.0"]` — nothing else. No transformers/torch/datasets/trl/
  accelerate/docker/vLLM (spec §6).
- `[project.optional-dependencies] dev = ["pytest>=8.0"]`.
- `[project.scripts] python-dpo = "python_dpo.cli:main"`.
- `[tool.setuptools.dynamic] version = {attr = "python_dpo.__version__"}` — single source
  of truth, so `__init__.py` and the distribution version can never drift.
- `[tool.setuptools.packages.find] where = ["src"]`.
- `[tool.pytest.ini_options]`: `testpaths = ["tests"]`, `pythonpath = ["src"]` (lets
  `pytest -q` pass even before an editable install), `addopts = "-ra"`.

### Package — `src/python_dpo/`

**`__init__.py`** — `__version__: str = "0.1.0"`, `__all__ = ["__version__"]`. Nothing else;
no eager imports, so `import python_dpo` stays cheap and dependency-light.

**`__main__.py`** — *(addition beyond the spec's tree)* required for `python -m python_dpo`
to work at all (spec §7). Three lines: `from .cli import main` + `raise SystemExit(main())`
under `if __name__ == "__main__"`.

**`config.py`** — *(addition beyond the spec's tree)* the "small configuration abstraction"
spec §10 requires. Frozen dataclasses, no global mutable state:
- `class ConfigError(Exception)` — raised with actionable messages
  (`"config.yaml: missing required key 'paths.problems'"`, `"config.yaml not found at <p>"`).
- `find_project_root(start: Path | None = None) -> Path` — walks up from `start` (default
  CWD) looking for `pyproject.toml`; falls back to `Path(__file__).resolve().parents[2]`
  so it also works when invoked from outside the tree. Satisfies §11 — no `open("config.yaml")`.
- `@dataclass(frozen=True) class Paths` — `raw, problems, candidates, evaluations,
  preferences, reports`, all absolute `Path`s resolved against the project root.
  - `ensure_exists() -> None` — `mkdir(parents=True, exist_ok=True)` for each; this is what
    "directories exist **or can be initialized**" (§14.4) hangs on.
- `@dataclass(frozen=True) class Config` — `project_name: str`, `paths: Paths`,
  `log_level: str`, `project_root: Path`.
  - `@classmethod load(cls, path: Path | None = None) -> Config` — resolves the file, parses
    with `yaml.safe_load` (never `yaml.load`), validates every required key is present and of
    the right type, raises `ConfigError` otherwise.

**`logging_config.py`** — `configure_logging(level: str = "INFO") -> None`.
- Format `"%(asctime)s | %(levelname)s | %(name)s | %(message)s"`, datefmt
  `"%Y-%m-%d %H:%M:%S"` → matches the §8 example line.
- Single `StreamHandler` on stderr attached to the `python_dpo` logger; idempotent (returns
  early if already configured) so repeated CLI/test calls don't duplicate output.
- Raises `ValueError` with a clear message on an unknown level name.

**`cli.py`** — stdlib `argparse` only.
- `build_parser() -> argparse.ArgumentParser` — factory (not a module-level parser), so
  tests can build one in isolation. `--version` via `action="version"` using
  `python_dpo.__version__`; a global `--log-level` defaulting to the config value.
- Subcommands, each a small `_cmd_<name>(args) -> int` handler: `problems`, `generate`,
  `evaluate`, `preferences`, `run`.
- Every handler logs `"<Stage> is not implemented yet."` via the module logger and
  **returns exit code 1**. No prints in application code (§8), no fake success (§7).
- `main(argv: Sequence[str] | None = None) -> int` — parses, calls `configure_logging`,
  loads `Config`, dispatches; catches `ConfigError` → logs the message, returns 2.
- No subcommand given → print help, return 1.

### Configuration & docs

**`config.yaml`** — exactly the §9 shape and nothing more: `project.name`, the six
`paths.*` entries, `logging.level: INFO`. No Qwen/GPU/Docker/DPO keys.

**`README.md`** (replaces the stub) — purpose, **Current status: Step 1 — Project Skeleton**,
the planned pipeline diagram, install (`python -m venv .venv` → `pip install -e ".[dev]"`),
`pytest`, and `python -m python_dpo --help`. Placeholder commands documented **as
placeholders** with their non-zero exit code — nothing described as working that isn't.

**`CLAUDE.md`** — the seven §18 sections: Architecture (incremental), Security (generated
code is untrusted, **never execute it on the host** — the §17 rule verbatim), Testing
(every component gets tests), Reproducibility (persist intermediate artifacts, stages
restartable), Scope Control (no future milestones unless asked), Data Integrity (never
silently drop candidates or evaluation failures), Development Workflow (test → review →
fix → re-test → report; never edit tests to force a pass).

### Directories

`scripts/`, `specs/`, and `data/{raw,problems,candidates,evaluations,preferences,reports}`
— each with a `.gitkeep` so git tracks them. `specs/README.md` points at `.claude/specs/`.

### Tests — `tests/__init__.py`, `tests/test_project.py`

One test per §14 requirement, all offline and CPU-only:
1. `import python_dpo` succeeds.
2. `__version__` exists, is a non-empty `str`, and parses as a dotted numeric version.
3. `Config.load()` on the real `config.yaml` returns absolute paths under the project root;
   plus a negative case — malformed/incomplete YAML in `tmp_path` raises `ConfigError` with
   a useful message.
4. `Paths.ensure_exists()` creates all six directories under `tmp_path`, and the six real
   `data/` directories exist in the repo.
5. `python -m python_dpo --help` exits 0 via `subprocess.run([sys.executable, "-m",
   "python_dpo", "--help"])` with `PYTHONPATH=src` injected, so it passes with or without an
   editable install. Also assert `--version` exits 0 and prints the version.

Plus a check that each placeholder subcommand parses and returns non-zero (guards against a
handler accidentally reporting success later). No skips.

### `.gitignore` (modify)

The stock Python template stays. Two edits:
- Replace the trailing `.claude` line with `.claude/*` + `!.claude/specs/` +
  `!.claude/plans/` — git will not descend into a fully-excluded directory, so the
  `dir/*` + negation form is the only one that works here.
- Append a short commented block: `data/` is **not** ignored (generated preference data is
  the project's deliverable, §13); only `data/raw/` contents — third-party datasets
  downloaded in a later step — are excluded, with `!data/**/.gitkeep` to keep the structure.
  The reasoning goes in the comment, per §13's "document that decision".

---

## Verification

Run from the repo root, in order:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
python -c "import python_dpo; print(python_dpo.__version__)"   # -> 0.1.0
python -m python_dpo --help                                    # exit 0, lists 5 subcommands
python -m python_dpo --version                                 # exit 0
python -m python_dpo generate; echo "exit=$?"                  # logs not-implemented, exit=1
pytest -q                                                      # all pass, 0 skipped
git status                                                     # only intended files; no secrets, no .venv
```

Then walk the §20 acceptance checklist and confirm no `exec`/`eval`/`subprocess`-of-generated-code
exists anywhere: `grep -rnE "\b(exec|eval)\(" src/`.

Finally, produce the §22 report: files created/modified, dependencies added, CLI commands,
tests, test results, deviations, decisions needing review. **Stop there — do not start Step 2.**

---

## Deviations & decisions to flag in the report

- `__main__.py` and `config.py` are not in the spec's §4 tree but are required by §7 and §10.
- PyYAML as the single runtime dependency (§6 exception, user-approved).
- Placeholder subcommands exit **1**, not 0 — the honest reading of §7's "do not create fake
  successful results". Easy to flip if you'd rather they exit 0.
- Skeleton at repo root rather than a nested `python-dpo/` (user-approved).

## Note on the plan file location

You asked for the plan at `.claude/plans/01_project_skeleton_plan.md`. Plan mode restricts
edits to this file only, so copying it there will be the first action taken once the plan is
approved.
