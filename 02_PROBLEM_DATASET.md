# Stage 2 Implementation Details — Problem Dataset

This document explains how `src/python_dpo/problems/` implements the ground-truth
layer specified in `.claude/specs/02_problem_dataset.md`. For usage instructions, see
the "Stage 2 — Problem Dataset" section of the root `README.md`; this file is about
*how* it's built, not how to run it.

## Goal

Stage 2 builds the pipeline's **ground-truth layer**: a dataset of 10 Python problems
that every later stage joins against, where a problem doesn't count as valid unless its
own reference solution passes its own tests. Everything lives under
`src/python_dpo/problems/`.

## The data model — `models.py`

Three frozen dataclasses, all validating in `__post_init__` so an invalid object simply
can't be constructed:

- **`TestCase`** (`models.py:88`) — `id`, `input` (a dict of keyword args), and either
  `expected` or `expected_exception`, never both (`models.py:119-122`). Every value is
  passed through `_require_json_native` (`models.py:63-84`), which recursively rejects
  tuples/sets — those serialize fine to JSON but come back as lists, so an unguarded
  tuple would silently break round-trip equality after a save→load cycle.
- **`Problem`** (`models.py:154`) — the required fields (`id`, `prompt`, `signature`,
  `entry_point`, `category`, `difficulty`, `reference_solution`, `tests`) plus optional
  `description`/`tags`/`source`/`metadata`/`dataset_version`. Validation checks:
  non-empty strings, `category`/`difficulty` are in the allowed sets, `entry_point` is a
  valid identifier that actually appears in `signature` (`models.py:194-202`), at least
  one test, and no duplicate test IDs.
- **`TestResult`** (`models.py:282`) — `test_id`, `passed`, `actual`, `expected`,
  `error_type`, `error_message`.

Two fields go beyond a literal reading of the spec: `entry_point` (so the executor knows
what function to call, rather than parsing it back out of the `signature` string) and
`expected_exception` (so `factorial(-1) → ValueError` is expressible as data). Both
`TestCase` and `TestResult` set `__test__ = False` so pytest doesn't try to collect them
as test classes.

`to_dict()`/`from_dict()` are hand-written rather than `dataclasses.asdict()`, so loading
a record always re-runs full validation.

## Reference solutions — `references.py`

Ten ordinary Python functions — `sum_even`, `most_frequent_value_key`,
`first_non_repeating`, `common_elements`, `k_largest`, `factorial`, `safe_get`,
`parse_int`, `chunk_sequence`, `gather_in_order`. These are the only code the validator
ever executes. Each is deterministic, has no I/O, and — because `catalog.py` pulls its
source via `inspect.getsource()` — any import it needs lives inside the function body
(e.g. `gather_in_order` imports `asyncio` internally) so the extracted text stands
alone.

## Catalog — `catalog.py`

`build_catalog()` returns the 10 `Problem` objects in fixed order, pure and
deterministic — no timestamps, no random IDs. Each problem's prompt pins down the exact
semantics the spec left ambiguous: p002 breaks ties by insertion order, p004 orders
results by first appearance in the first argument, p005 keeps duplicates and rejects
negative `k`, p006 raises on negative input, p009's `ValueError` fires on iteration
(it's a real generator), p010 must run concurrently and preserve input order over
completion order.

## Execution seam — `executor.py`

This is the security-sensitive piece. `ReferenceExecutor` is a `Protocol`
(`executor.py:17`) with one method, `run(problem, test_case) -> TestResult`.
`InProcessReferenceExecutor` (`executor.py:23`) is the only implementation right now:

1. `_load_function` compiles the stored source and `exec`s it into a fresh namespace
   (`executor.py:72-73`), then looks up `entry_point`.
2. `_call` runs it — `asyncio.run(...)` for coroutine functions, and `_materialize`
   turns any returned generator into a list so it stays comparable and
   JSON-serializable.
3. Results are compared with `_values_match`, which explicitly rejects `True == 1`
   (`executor.py:124-128`) — booleans and ints would otherwise compare equal in Python.
4. Any exception is caught and turned into a `TestResult` — if the test case set
   `expected_exception`, a match means pass; otherwise any exception is a failure.

The class docstring is explicit: this executes code on the host, which is sanctioned
**only** for trusted, manually authored reference solutions (spec §23). Model-generated
candidates must never go through it — that's what forces Stage 3 to build a sandboxed
executor behind the same `ReferenceExecutor` protocol instead of reusing this one.
`CLAUDE.md`'s Security section states the same rule.

## Persistence — `storage.py`

`save_problems` writes one JSON object per line, UTF-8, with `sort_keys=True` — so
rebuilding an unchanged catalog produces a byte-identical file (verified by hashing the
file before/after a rebuild and getting the same MD5). `load_problems` parses line by
line and never silently drops a bad record: a malformed JSON line, an invalid record, or
a duplicate ID all raise `DatasetError` naming the exact line number.

## Validation — `validation.py`

`validate_problem` checks per-problem invariants (non-empty fields, valid
category/difficulty, ≥5 tests, no duplicate test IDs) and runs every test through the
executor. `validate_dataset` adds dataset-wide checks (exactly 10 problems, unique IDs).
Both return report objects (`ProblemReport`/`DatasetReport`) rather than raising, so a
caller can decide what to do with failures. `format_report` renders the human-readable
summary block as a plain string — it stays pure, and only the CLI decides where that
string goes.

## CLI wiring — `cli.py`

`problems` is no longer a placeholder; it's its own subparser with `build` and
`validate` (`cli.py:91-111`):

- **`problems build`** (`cli.py:50-69`) — builds the catalog, validates it, and writes
  `problems.jsonl` **only if everything passes**. A broken catalog leaves no file
  behind rather than writing a half-trustworthy one.
- **`problems validate`** (`cli.py:72-88`) — strictly read-only: loads the persisted
  file, re-runs every reference test, writes the summary to `stdout` (not the logger —
  this is user-facing report output, not diagnostics), and returns 0/1 based on the
  result.

All handlers now take `(args, config)` instead of just `(args)`, so
`config.paths.problems` resolves the dataset location instead of it being hardcoded
anywhere.

## Tests

`tests/test_problems.py` — unit tests per spec §32: schema validation (valid/invalid
fields, JSON-native rejection), test-case validation, executor behavior
(pass/fail/exception/coroutine/generator/boolean-strictness), dataset-level validation
(duplicate IDs, wrong count, failing reference), and storage (round-trip, malformed JSON
with line number, duplicate IDs, missing file).

`tests/test_problems_integration.py` — the full round trip from spec §33: build →
validate → save → reload → validate again → PASS, plus build-determinism (byte-identical
hashes), the exact 5/4/1 difficulty split, and two properties that can't be expressed as
input/expected pairs: `chunk_sequence` is genuinely a generator
(`inspect.isgeneratorfunction`), and `gather_in_order` genuinely runs concurrently —
5×0.05s operations finish under a 0.15s ceiling instead of the ~0.25s a sequential loop
would take.

**Result:** 65 tests pass, 10 problems, 74 reference tests all green, and the only
`exec(` call anywhere in `src/` is the one line in `executor.py`, deliberately fenced
off from anything the model will ever generate.
