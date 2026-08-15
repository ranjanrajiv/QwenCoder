# Stage 2 — Python Problem Dataset and Ground Truth

## Context

Stage 1 delivered the skeleton: an installable `python_dpo` package, a CLI whose five
subcommands are honest placeholders, logging, and typed config. `.claude/specs/02_problem_dataset.md`
now asks for the **ground-truth layer** — the data every later stage joins against.

The pipeline can't generate candidates (Stage 3) or judge them (Stage 4) without a trusted
answer key. Stage 2 builds exactly that: 10 manually curated Python problems, each with a
prompt, signature, reference solution, and ≥5 executable tests, persisted to
`data/problems/problems.jsonl` and provably valid — a problem does not count as valid unless
its reference solution passes all of its own tests (spec §3).

**Outcome:** `python -m python_dpo problems build` writes a 10-line JSONL dataset;
`python -m python_dpo problems validate` reloads it, re-runs all ~75 reference tests, and
prints `Dataset validation: PASS`; `pytest -q` stays green.

### Decisions confirmed with the user

1. **Frozen dataclasses**, not Pydantic — matches `Config`/`Paths` from Stage 1 and keeps
   PyYAML the only runtime dependency. Spec §4's "prefer Pydantic" is conditional on it
   fitting Stage 1's architecture; it doesn't.
2. **Reference solutions are authored as real Python functions** in `references.py`; the
   JSONL text is derived at build time via `inspect.getsource()`, so executable code and
   stored text cannot drift.
3. **The §31 summary goes to stdout from the CLI layer only** — `validation.py` returns it
   as a string. Stage 1's "no print in application code" rule targets print-as-logging; a
   user-facing report is legitimate CLI output.

---

## New package — `src/python_dpo/problems/`

**`models.py`** — the schema, all frozen dataclasses validating in `__post_init__` and
raising `ProblemError` with actionable messages (`"p003/t002: 'expected_exception' must be a
type name"`).

- `DATASET_VERSION = "0.1.0"` — the dataset version from §35, deliberately separate from
  `python_dpo.__version__`. Stored on every record.
- `CATEGORIES` / `DIFFICULTIES` — frozensets of the exact §7 / §8 names.
- `TestCase` — `id`, `input: dict[str, Any]` (kwargs, per §9's preference for structured
  values), `expected: Any`, `expected_exception: str | None`.
- `Problem` — required `id, prompt, signature, entry_point, category, difficulty,
  reference_solution, tests`; optional `description, tags, source, metadata,
  dataset_version`.
- `TestResult` — `test_id, passed, actual, expected, error_type, error_message` (§25), with
  `test_id` formatted `f"{problem.id}_{case.id}"` → `p001_t003`.
- `to_dict()` / `from_dict()` on `Problem` and `TestCase` — explicit, not `asdict`, so
  loading validates rather than trusting the file.

Two schema additions beyond §5, both load-bearing:
- **`entry_point`** — the function name to call. §10 requires the evaluator to determine how
  to invoke the candidate; parsing it back out of the `signature` string is fragile.
- **`expected_exception`** — §17/§20 demand explicit invalid-input behavior (`factorial(-1)`
  → `ValueError`), which a plain `expected` value cannot express.

**Constraint:** every `input`/`expected` value must be JSON-native (no tuples, no sets), or
the JSONL round-trip won't compare equal. This is why P004 returns a **list**, not a set.

**`references.py`** — the 10 trusted reference implementations as ordinary module-level
functions. Deterministic, no I/O, no network, no third-party imports (§11). Header comment
states these are trusted, manually authored code — the *only* code this package may execute.

**`catalog.py`** — `build_catalog() -> list[Problem]`, the manually defined specifications
§30 refers to. Pure and deterministic (§34): no timestamps, no random IDs, stable ordering
p001→p010. Pulls each `reference_solution` from `inspect.getsource()`.

**`executor.py`** — the replaceable execution seam §23/§24 require.

- `ReferenceExecutor` — a `typing.Protocol` with `run(problem, test_case) -> TestResult`.
  `validation.py` depends only on this, so Stage 3 can drop in a Docker-backed executor
  without touching the validator.
- `InProcessReferenceExecutor` — compiles the stored `reference_solution` and runs it in a
  fresh namespace, looks up `entry_point`, calls `fn(**test_case.input)`.
  - `asyncio.run(...)` when `inspect.iscoroutinefunction(fn)` (P010).
  - Materializes returned generators to a list (P009), so `expected` stays JSON-native.
  - Strict comparison helper rejecting `True == 1` — matters once Stage 3 reuses this.
  - Catches exceptions into `TestResult.error_type`/`error_message`; a case with
    `expected_exception` passes only if that exact exception type name is raised.

> **Security note for the implementation:** this class calls `exec()` on the stored source.
> Stage 1's `grep -rnE "\b(exec|eval)\(" src/` will no longer come back empty. That is
> sanctioned by §23 *for trusted reference solutions only*. The class docstring must say so
> loudly, and `CLAUDE.md` gets a matching clarification (below). Nothing model-generated may
> ever be routed through it.

**`storage.py`** — `save_problems(problems, path)` and `load_problems(path) -> list[Problem]`.
- Writer (§28): creates parent dirs (reusing `Config.paths`), UTF-8, one compact JSON object
  per line, `sort_keys=True` for deterministic byte-identical output, order preserved.
- Loader (§27): parses line by line, validates every record, rejects malformed lines and
  duplicate IDs with the offending line number in the message. **Never silently skips** —
  matches CLAUDE.md's Data Integrity rule.

**`validation.py`** — `validate_problem(problem, executor) -> ProblemReport`,
`validate_dataset(problems, executor) -> DatasetReport`, `format_report(report) -> str`.
Checks all ten §29 conditions (exactly 10 problems, unique IDs, valid categories and
difficulties, non-empty prompt/signature/reference, ≥5 tests, unique test IDs per problem,
reference passes every test). `format_report` renders the §31 block; returning a string keeps
the module pure and testable.

**`README.md`** — per-folder doc, matching the convention already established in `src/python_dpo/`.

---

## The 10 problems

Difficulty lands on **5 easy / 4 medium / 1 hard** exactly as §8 requires. Each row's
"pinned semantics" is the ambiguity §14–§19 insist be decided in the prompt text, not left to
the implementation. 7–8 tests each, ~75 total.

| ID | Category | Diff | Entry point | Pinned semantics |
|----|----------|------|-------------|------------------|
| p001 | lists | easy | `sum_even(numbers)` | Empty → `0`; negatives and duplicates counted; `0` is even |
| p002 | dictionaries | medium | `most_frequent_value_key(mapping)` | **Ties → the key appearing first in insertion order** (§14); empty → `None` |
| p003 | strings | medium | `first_non_repeating(text)` | First char occurring exactly once; case-sensitive; none → `None` |
| p004 | sets | medium | `common_elements(first, second)` | **Returns a list, deduped, ordered by first appearance in `first`** (§15) |
| p005 | sorting | medium | `k_largest(values, k)` | **Duplicates preserved, descending** (§16); `k=0` → `[]`; `k>len` → all; `k<0` → `ValueError` |
| p006 | recursion | easy | `factorial(n)` | `0`/`1` → `1`; **negative → `ValueError`** (§17) |
| p007 | edge_cases | easy | `safe_get(items, index, default=None)` | Negative indices use normal Python semantics; out-of-range **or non-int** index → `default` |
| p008 | exceptions | easy | `parse_int(text, default=0)` | Surrounding whitespace and `+`/`-` signs OK; `"3.5"`, `""`, `None`, junk → `default` |
| p009 | generators | easy | `chunk_sequence(sequence, size)` | Yields lists, last may be short (§18); `size<=0` → `ValueError` **raised when iteration begins**, since a generator body doesn't run until then |
| p010 | async | hard | `gather_in_order(operations)` | `operations` is a list of `[value, delay]`; must run **concurrently** and return values in **input order, not completion order** (§19) |

Each problem's test set deliberately targets the most likely wrong implementation (§21):
p001 includes `0` and negatives; p004 includes a case where source order ≠ sorted order (so
`sorted(set(a) & set(b))` fails); p005 includes `[5,1,5,3], k=3 → [5,5,3]` (so a
`set()`-based answer fails); p007 covers first, last, `-1`, and past-the-end; p010 gives the
first operation the **longest** delay, so a completion-ordered result is wrong.

**Two properties can't be expressed as declarative input/expected pairs** and get dedicated
checks in the repo's own pytest suite instead of new schema fields:
- P009's reference is a true generator — `inspect.isgeneratorfunction`, per §18's "must not
  return all chunks as a list directly".
- P010 actually runs concurrently — 5 operations × 0.05s must complete well under the ~0.25s
  a sequential `await` loop would take, with a generous margin so it can't flake.

P006's "must be recursive" stays a prompt-level requirement (it's what Stage 3 candidates are
judged against); asserting it via source inspection would be brittle for no real gain.

---

## Files to modify

**`src/python_dpo/cli.py`** — `problems` stops being a placeholder.
- Rename `_STAGE_NAMES` → `_PLACEHOLDER_STAGES` and drop `problems` from it; `generate`,
  `evaluate`, `preferences`, `run` keep their exit-1 placeholder behavior untouched.
- Add a `problems` subparser owning its own `build` / `validate` subcommands
  (`dest="problems_command"`). Bare `problems` prints help and returns 1, mirroring the
  existing no-subcommand behavior.
- Handler signature becomes `(args, config) -> int` — the new commands need
  `config.paths.problems` and must not hardcode paths (§11 / spec 01 §16).
- `problems build`: build catalog → validate → run every reference test → **write only if
  everything passes**; on failure log what failed, write nothing, return 1.
- `problems validate`: load JSONL → validate → execute reference tests → write the §31
  summary to stdout → return 0 on PASS, 1 on FAIL. Strictly read-only (§30).

**`tests/test_project.py`** — `test_placeholder_subcommands_parse_and_return_nonzero` is
parametrized over `_STAGE_NAMES`, which currently includes `problems`; it must narrow to the
four remaining placeholders and adapt to the `(args, config)` signature. This is a
requirement change, not a test bent to fit an implementation — `problems` genuinely isn't a
placeholder anymore, and the new behavior gets stronger coverage below.

**`README.md`** — new "Stage 2 — Problem Dataset" section (§39): purpose, schema, categories,
difficulties, reference-solution and test-case concepts, JSONL location, the two CLI commands,
validation procedure. Status line moves to Stage 2. Candidate generation stays documented as
**not** implemented.

**`CLAUDE.md`** — sharpen the Security section: generated code is untrusted and never runs on
the host; *manually authored reference solutions* are trusted and may run in-process during
validation, isolated behind `ReferenceExecutor` so Stage 3 swaps in the sandbox.

**`src/python_dpo/README.md`**, **`tests/README.md`**, **`data/README.md`** — keep the
per-folder docs accurate now that the package has a subpackage, the suite has new files, and
`data/problems/` holds a real artifact.

**`src/python_dpo/__init__.py`** — `__version__` → `0.2.0`. The dataset carries its own
`DATASET_VERSION`; §35 forbids conflating the two, and nothing asserts a specific package
version.

`data/problems/problems.jsonl` is a **tracked deliverable** — `.gitignore` already excludes
only `data/raw/`, and the build is deterministic, so the file commits cleanly.

---

## Tests

**`tests/test_problems.py`** (unit, §32) — valid problem round-trips; missing required field,
invalid category, invalid difficulty, empty prompt/signature/reference each raise
`ProblemError`; malformed test case; duplicate test ID within a problem; duplicate problem ID
across the dataset; wrong problem count; missing category; JSONL write → load → round-trip
equality; malformed JSONL line rejected with its line number; executor returns a passing
`TestResult` for a correct call, a failing one with `error_type` populated for a raise, and
handles the `expected_exception`, async, and generator paths.

**`tests/test_problems_integration.py`** (§33) — the full round trip: build the 10 problems →
validate → run every reference test → write JSONL to `tmp_path` → reload → validate again →
PASS. Plus: byte-identical output across two builds (§34 determinism), the exact 5/4/1
difficulty split and all 10 categories present, and the two behavioral checks (P009 is a real
generator, P010 is genuinely concurrent).

**`tests/test_project.py`** — additionally assert `problems build` and `problems validate`
parse and dispatch, and that bare `problems` returns non-zero.

Everything stays offline and CPU-only. No skips (§15).

---

## Verification

```bash
source .venv/bin/activate
pytest -q                                    # all pass, 0 skipped
python -m python_dpo problems build          # exit 0, writes the dataset
python -m python_dpo problems validate       # exit 0, "Dataset validation: PASS"
wc -l data/problems/problems.jsonl           # -> 10
python -m python_dpo problems build && git diff --stat data/problems/  # rebuild is a no-op
python -m python_dpo --help                  # problems now shows build/validate
python -m python_dpo generate; echo "exit=$?"  # still exit=1, untouched
git status                                   # only intended files, no secrets, no .venv
```

Then confirm scope containment: no transformers/torch/trl/datasets import anywhere, no Docker,
no model loading, and that the only `exec(` hit in `src/` is the documented one in
`executor.py`.

Finally produce the §42 report — files created/modified, categories, problem and test counts,
reference pass rate, CLI commands, test results, deviations, decisions to review.
**Stop there — do not start Stage 3.**

---

## Deviations & decisions to flag in the report

- `entry_point` and `expected_exception` are schema fields beyond §5/§9, required by §10 and
  §17 respectively.
- Frozen dataclasses over Pydantic (§4's preference is conditional; user-approved).
- `exec()` now appears in `src/` — sanctioned by §23 for trusted reference code only, confined
  to `InProcessReferenceExecutor`.
- P009's `ValueError` surfaces when iteration begins, not at call time — inherent to real
  generator functions, and documented in the problem prompt.
- P010's concurrency and P009's generator-ness are verified in the repo's pytest suite rather
  than as declarative test cases, since neither fits an input/expected pair.
- The §31 summary is printed to stdout rather than logged (user-approved).
- Tie-breaking, ordering, and invalid-input rules for p002/p004/p005/p006/p007/p008 were
  chosen by us where §14–§18 said "choose one and document it" — worth a review pass, as
  Stage 3 candidates will be graded against exactly these choices.
