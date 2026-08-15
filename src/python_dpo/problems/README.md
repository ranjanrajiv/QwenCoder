# src/python_dpo/problems/

The problem dataset — the **ground-truth layer** of the pipeline. Ten curated Python
problems, each with a trusted reference solution and executable tests, persisted to
`data/problems/problems.jsonl`. Every later stage joins on the ids produced here.

The governing rule (spec 02 section 3): a problem is not valid unless its own reference
solution passes all of its tests.

## Files

### `models.py`

The schema, as frozen dataclasses that validate in `__post_init__` and raise
`ProblemError` with actionable messages. Because construction validates, every entry
point into the dataset yields already-valid records.

- `Problem` — `id`, `prompt`, `signature`, `entry_point`, `category`, `difficulty`,
  `reference_solution`, `tests`, plus optional `description`, `tags`, `source`,
  `metadata`, `dataset_version`.
- `TestCase` — `id`, `input` (a mapping of keyword arguments), and either `expected` or
  `expected_exception`, never both.
- `TestResult` — `test_id`, `passed`, `actual`, `expected`, `error_type`,
  `error_message`.
- `DATASET_VERSION`, `CATEGORIES`, `DIFFICULTIES`, `MIN_TESTS_PER_PROBLEM`,
  `EXPECTED_PROBLEM_COUNT` — the dataset's constants. `DATASET_VERSION` is deliberately
  separate from `python_dpo.__version__` so dataset revisions and releases move apart.

Two fields go beyond the specification's field list because the pipeline needs them:
`entry_point` (how an evaluator invokes a solution, rather than re-parsing `signature`)
and `expected_exception` (so invalid-input behavior like `factorial(-1)` → `ValueError`
is expressible).

`TestCase` and `TestResult` set `__test__ = False` — they are data models, not pytest
test classes, despite the names the spec assigns them.

### `references.py`

The ten reference implementations as ordinary Python functions: deterministic, no I/O,
no network, no third-party imports. This is the **only** code the validator executes.

Each function is stored in the dataset via `inspect.getsource`, so every solution must
stand alone — any import it needs lives inside the function body (see
`gather_in_order`), exactly as a standalone candidate implementation would.

### `catalog.py`

`build_catalog() -> list[Problem]` — the manually defined problem specifications, pure
and deterministic: same source, same dataset, every time. No timestamps, no generated
ids, no randomness.

Prompts spell out the behavior that would otherwise be ambiguous — tie-breaking, output
ordering, invalid-input handling — because Stage 3 candidates are graded against exactly
those rules.

### `executor.py`

The replaceable execution seam.

- `ReferenceExecutor` — a `Protocol` with `run(problem, test_case) -> TestResult`.
  `validation.py` depends only on this, so a Docker-backed executor can be dropped in
  later without touching the validator.
- `InProcessReferenceExecutor` — compiles the stored reference source and runs it in a
  fresh namespace. Handles coroutine functions via `asyncio.run`, materializes returned
  generators to lists, and compares strictly enough that `True` never passes as `1`.
  Exceptions become `TestResult` fields; a case with `expected_exception` passes only if
  that exact exception type is raised.

**Security:** this class executes code on the host. That is sanctioned for trusted,
manually authored reference solutions only (spec 02 section 23). Model-generated
candidates must never be routed through it — see the Security section of `CLAUDE.md`.

### `storage.py`

JSONL persistence. `save_problems` creates parent directories, writes UTF-8, and sorts
keys so an unchanged catalog rebuilds byte-identically. `load_problems` parses line by
line, validates every record, and rejects malformed lines and duplicate ids with the
offending line number — it never silently skips a record.

Also exposes `PROBLEMS_FILENAME` and `dataset_path(problems_dir)` so callers resolve the
dataset location from configuration instead of hardcoding it.

### `validation.py`

`validate_problem` and `validate_dataset` check the dataset's integrity conditions
(exactly ten problems, unique ids, valid categories and difficulties, non-empty
prompt/signature/reference, at least five tests, unique test ids, and every reference
test passing), returning `ProblemReport` / `DatasetReport` rather than raising.

`format_report` renders the summary block. It returns a string, keeping the module pure;
only the CLI decides to write it to stdout.
