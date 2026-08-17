# Stage 4 Implementation Details — Candidate Persistence, Runs and Reproducibility

This document explains how `src/python_dpo/atomic_io.py`, `src/python_dpo/runs/`, and the
Stage 4 changes to `src/python_dpo/candidates/`, `src/python_dpo/generation/generator.py`,
and `src/python_dpo/cli.py` implement the layer specified in
`.claude/specs/04_candidate_presistence.md`. For usage instructions, see the "Stage 4 —
Candidate Persistence, Runs and Reproducibility" section of the root `README.md`; this
file is about *how* it's built, not how to run it.

## Goal

Stage 3 gave every candidate provenance fields and a `run_id`, but persisted everything to
one flat, append-only `candidates.jsonl`. That file could not answer "is this run
finished?", could not be validated for integrity, and inferred resume across *all* runs at
once rather than the one being resumed. Stage 4's governing requirement (spec §1):

> A candidate-generation run must be safely interruptible and restartable without losing
> already-generated candidates or silently changing existing artifacts.

The approach: every `generate` invocation is a **run** — one self-contained,
independently auditable directory with its own manifest, candidates, failures, statistics,
and prompt artifact. Nothing here executes candidate code; hashing and counting are the
only things this layer does to candidate text (CLAUDE.md Security, reaffirmed by spec §45).

## 1. Run-management architecture

`src/python_dpo/runs/` owns everything about a run's *lifecycle*, as opposed to its
*content* (which stays in `candidates/`):

- **`models.py`** — `RunManifest` (the historical configuration snapshot) and
  `RunStatistics` (always a cache, reconstructable from disk).
- **`environment.py`** — `capture_environment()`, probing `transformers`/`torch`/CUDA
  versions without ever importing them at module load time.
- **`repository.py`** — `RunRepository`, the only code that mints run ids or writes
  `manifest.json`/`statistics.json`.
- **`migration.py`** — upgrades the Stage 3 flat file into run directories.
- **`validation.py`** — `validate_run`, the integrity checker.

`runs` depends on `candidates` (a manifest embeds `ModelConfig`/`GenerationConfig` dicts; a
`RunStatistics` is computed from `Candidate`/`GenerationFailure` records) — **one
direction only**. `candidates` never imports from `runs`. This was not automatic: an
earlier version of `migration.py` lived inside `candidates/` and imported `runs`, which
created a genuine circular import (`candidates/__init__.py` → `migration` → `runs/__init__.py`
→ `candidates.models`, before `candidates/__init__.py` had finished running). Moving
`migration.py` into `runs/` fixed it; both import orders (`import python_dpo.runs` first
or `import python_dpo.candidates` first) are exercised and pass.

### Run identity and directory layout

Run ids are `run_YYYYMMDD_HHMMSS_xxxx` (`repository.py`'s `new_run_id`) — the spec §5
shape, not Stage 3's bare timestamp. The random 4-hex-digit suffix (`secrets.token_hex(2)`)
makes a collision within the same second negligible, so minting costs one existence check
rather than a directory scan.

```
data/candidates/runs/run_20260817_133700_a81f/
├── manifest.json          # RunManifest, atomically rewritten on every status change
├── candidates.jsonl       # schema 2.0, one fsynced line per candidate
├── failures.jsonl         # one fsynced line per generation failure
├── statistics.json        # RunStatistics cache, rewritten after every generate() call
└── prompts/prompts.jsonl  # exact prompt per attempt, written before inference
```

### Status lifecycle

`RUN_STATUSES = {created, running, completed, failed, interrupted, cancelled}`
(`runs/models.py`). Transitions are enforced by an explicit graph
(`RUN_STATUS_TRANSITIONS`), not left to caller discipline: `completed` and `cancelled` are
terminal; `resume_run` refuses a `completed` run (spec §11); `RunManifest.with_status(...)`
raises rather than silently accepting an illegal transition. `_execute_run` in `cli.py`
drives this: `start_run` → `running`; a clean `generate()` return with every requested
`(problem, index)` accounted for → `complete_run`; anything short of that (including a
partial run that simply ran out of requested work without error) → `interrupt_run`;
`KeyboardInterrupt` → `interrupt_run`; `ModelLoadError` → `fail_run` with the error
recorded (spec §10: error type, message, timestamp, affected problem, affected index).

## 2. Candidate persistence architecture

`CandidateRepository` (`candidates/repository.py`) is now **run-scoped**: one instance
owns exactly one run directory. Its constructor signature is unchanged
(`CandidateRepository(directory)`), so the change is conceptual rather than mechanical —
`RunRepository.candidates(run_id)` is the intended way to obtain one.

Because each run is its own directory, `candidate_id` is unique within a repository — no
`(run_id, candidate_id)` compound key is needed, and there is no `--force`-appends-to-the-
same-file behavior to reconcile. `latest_by_candidate_id()` (Stage 3's cross-run
collapsing helper) is gone; it has no meaning when a repository only ever holds one run.

`FAILURES_FILENAME` changed from `generation_failures.jsonl` to `failures.jsonl` (spec §6);
`LEGACY_FAILURES_FILENAME` keeps the old name so migration can still find a Stage-3-era
failures file if one exists (in practice, none does for the real dataset — see §11).

The spec §23 repository API (`save`, `get`, `exists`, `list`, `count`, `find_by_problem`,
`find_by_hash`) is implemented with the `run_id` argument dropped from each method, since
the repository already is scoped to one run — an explicit, documented deviation (see §17).

## 3. Run manifest schema

```json
{
  "manifest_version": "1.0",
  "run_id": "run_20260817_133700_a81f",
  "status": "completed",
  "created_at": "...", "started_at": "...", "completed_at": "...",
  "candidate_schema_version": "2.0",
  "prompt_version": "v1",
  "model": { "provider": "transformers", "name": "Qwen/Qwen2.5-Coder-3B-Instruct", "...": "..." },
  "generation_config": { "temperature": 0.8, "top_p": 0.95, "...": "..." },
  "strategies": ["normal", "straightforward", "edge_case_focused", "alternative", "optimized"],
  "requested_problem_ids": ["p001", "p002"],
  "requested_problems": 2,
  "requested_candidates_per_problem": 5,
  "retry": { "max_attempts": 2 },
  "environment": { "python_version": "3.12.3", "platform": "Linux-6.8.0-...", "...": "..." },
  "error": null,
  "source": "generate"
}
```

`requested_problem_ids` (not just a count) is what makes resume derivable from the
manifest alone (spec §12, §34) — `--resume RUN_ID` rejects any of `--problem-id`,
`--limit`, `--num-candidates`, `--strategy` as conflicting, because the manifest is
already authoritative for all of them. `source` is `"generate"` or `"migrated"`, so a
migrated run is honestly labeled rather than pretending to have gone through the CLI.

## 4. Candidate schema changes (1.0 → 2.0)

No Stage 3 field was removed. Added: `schema_version`, `code_sha256`, `prompt_sha256`,
`raw_output_sha256`, `attempt`. A record with no `schema_version` field reads as `"1.0"`,
and on such a record the three hash fields must be `null` — never invented (spec §46).
`GenerationFailure` gained `schema_version`, `attempt`, `prompt_sha256`, and an optional
`traceback` that is deliberately left unpopulated by default (a Python traceback embeds
absolute filesystem paths, which the environment rule in spec §33 forbids recording).

`ERROR_TYPES` is now split into `INFRASTRUCTURE_ERROR_TYPES` (`model_load`, `tokenizer`,
`inference`, `timeout` — retried) and `CANDIDATE_ERROR_TYPES` (`empty_output`,
`code_extraction` — never retried), encoding spec §28's distinction as a checked
invariant (`assert INFRASTRUCTURE_ERROR_TYPES | CANDIDATE_ERROR_TYPES == ERROR_TYPES` in
`candidates/models.py`) rather than a comment.

## 5. Hashing strategy

One function, `sha256_text` (`candidates/hashing.py`), used for all three hashes so
they're computed identically everywhere. The important design choice is that hashes are
**verified, not just stored**: `Candidate.__post_init__` recomputes `code_sha256`,
`prompt_sha256`, and `raw_output_sha256` from the corresponding text and raises if any
doesn't match. This means a tampered record cannot be *loaded* — `Candidate.from_dict`
calls the same constructor, so `runs validate`'s hash check is really just "did the record
load," with no separate recomputation pass needed. `Candidate.create(...)` is the
classmethod that computes all three hashes for a new record; the constructor itself is
reserved for reconstructing (and thereby re-verifying) an existing one.

## 6. Duplicate-detection behavior

Detection is **run-scoped**, matching spec §19 (detect within a run) and §20 (never
auto-reject across runs) — this is a deliberate behavior change from Stage 3, not an
oversight. `CandidateRepository.code_index()` only ever sees its own run's file. Two
different runs producing byte-identical code (verified with the deterministic mock —
`test_duplicate_detection_does_not_cross_runs`) get no `duplicate_of` link; `find_by_hash`
is the tool for a human or a later stage to do that cross-run comparison deliberately.
Duplicates are still always kept, never deleted.

## 7. Resume behavior

Confirmed design: `generate` always creates a new run; `--resume RUN_ID` is the only way
to continue an existing one. Mechanically, resuming is nothing more than calling
`CandidateGenerator.generate()` a second time against the *same* run directory —
`existing_keys()` (loaded fresh from `candidates.jsonl` at the start of the call) makes
already-persisted `(problem_id, generation_index)` pairs skip unconditionally. There is no
`force` parameter on `generate()` anymore (see §17) — it would be meaningless once every
run is its own empty-or-not directory.

The mandatory spec §42/§49 integration test
(`test_interruption_and_resume_preserve_completed_work` in
`tests/test_generation_pipeline.py`) does the real thing: 3 problems × 5 candidates, a
`MockModelClient` scripted to raise `KeyboardInterrupt` on the 8th call, `run_repo.interrupt_run`,
then a second `generate()` call. Result: 15 total candidates, the first 7 records
byte-for-byte unchanged (`after.startswith(before)` on the raw file bytes), final status
`completed`. A CLI-level version of the same scenario was also run manually (§13).

## 8. Force-regeneration behavior

`--force` alone (no `--resume`) is now a no-op, since `generate` already always starts a
new run — kept only for Stage 3 flag compatibility. `--resume RUN_ID --force` is where it
does something: `RunRepository.create_run_from(manifest)` seeds a **brand-new** run,
copying every configuration field (problems, strategies, model, generation config, retry
policy, prompt version) but minting a fresh run id and directory. The source run is never
opened for writing. Verified with a byte-comparison of the original run's
`candidates.jsonl` before and after (`test_force_creates_a_new_run_and_leaves_the_old_one_untouched`),
and manually via the CLI (§13).

## 9. Failure-handling behavior

Per-attempt retry lives in `CandidateGenerator.generate()`: a `for attempt in
range(1, max_attempts + 1)` loop around the model call. `ModelLoadError` records one
failure and re-raises immediately (Stage 3's rule: no candidate in the run can succeed, so
retrying it per-candidate would just emit duplicate identical failures). Any other
exception from the client records an `inference` failure for that attempt and, if attempts
remain, the loop continues — **the attempt-1 failure record is never overwritten**, each
attempt gets its own line, and the eventual candidate's `attempt` field names which one
succeeded (spec §29, §30). Once a response is obtained, empty-output and
code-extraction failures are recorded once and never retried (spec §28) — they represent a
usable-but-empty model response, not an infrastructure hiccup.

`generation.retry.max_attempts` (default 2) is new in `config.yaml`, parsed by the new
`RetrySettings` dataclass in `config.py` and copied verbatim into every run's manifest —
a resumed run keeps the retry policy it started with, not whatever `config.yaml` says
today.

## 10. Statistics implementation

`RunStatistics.from_records(manifest, candidates, failures)` is the single source of
truth; `statistics.json` is only ever a cache written after it. It correctly separates
`requested` / `generated` / `valid` / `failed` (spec §26): `candidates_requested` comes
from the manifest (problems × per-problem count), `candidates_generated` from counting
actual `Candidate` records, `generation_failures` from counting **distinct**
`(problem_id, generation_index)` pairs whose only outcome is a failure (not total failure
*lines* — a retried-then-succeeded index contributes to `retry_attempts`, not
`generation_failures`), and `syntax_valid`/`syntax_invalid`/`function_name_valid` from the
candidates themselves. `problems_completed` counts a problem as done when **every**
requested index has an outcome — candidate or terminal failure — matching the run
completion rule in spec §9, not "every index succeeded." (My own implementation plan
mis-stated this as "every index has a candidate," which would have contradicted the
spec's own §25 worked example; the code follows the spec, not that earlier phrasing.)

## 11. Migration behavior

`runs/migration.py`'s `migrate_flat_file(source_path, run_repo, force=False)` reads the
legacy file **read-only** (never writes to it — checked with a byte-for-byte comparison in
`test_source_file_is_left_byte_identical`), groups records by their existing `run_id`,
infers a manifest per group (model/provider/prompt_version must agree across every
candidate in the group, or migration raises rather than guessing; strategy-per-index must
be consistent across problems, likewise), upgrades every candidate to schema 2.0 via
`Candidate.create` (back-filling all three hashes), and writes each group through the same
`RunRepository` code path a real `generate` uses — so a migrated run is inspectable and
validatable exactly like any other. It refuses to overwrite an existing run directory
unless `force=True`, in which case it removes the stale directory first (to avoid
appending duplicate lines onto old data) rather than overwriting in place.

Real-data result: `python -m python_dpo candidates migrate` against the repository's real
`data/candidates/candidates.jsonl` (50 Qwen-generated records, one run, `run_id
20260817_055411`) produced `data/candidates/runs/20260817_055411/` with all 50 candidates,
`runs validate` passing, and the source file untouched (confirmed via `git status`
showing no diff). The migrated run's `candidates stats` shows **31 duplicates out of 50**
— the 3B-Instruct model solving several of the easier problems near-identically across
strategies, exactly the risk flagged as worth watching in the Stage 3 report.

There was no legacy `generation_failures.jsonl` to migrate (the real Stage 3 run recorded
zero failures), but the code path for it exists and is tested
(`test_multiple_run_ids_in_the_source_file_produce_multiple_runs` and friends).

## 12. CLI commands added

`generate` gained `--resume RUN_ID`. `_cmd_generate` dispatches to `_cmd_generate_fresh`
(the Stage 3 behavior, now creating a run manifest before generating) or
`_cmd_generate_resume` (rejects any selection flag, resumes or, with `--force`, seeds a
new run). `_execute_run` is the shared status-lifecycle driver both paths call into.

New command groups, mirroring the existing `problems` group's structure:

| Command | Behavior |
|---|---|
| `runs list` | `RUN ID / STATUS / CANDIDATES / FAILURES`, newest first |
| `runs show RUN_ID` | Manifest, generation config, timestamps, error (if any), statistics |
| `runs validate RUN_ID [--repair]` | The full integrity check; `--repair` truncates a torn tail first |
| `candidates list RUN_ID [--problem-id] [--strategy]` | `candidate_id / problem_id / strategy / syntax` |
| `candidates show RUN_ID CANDIDATE_ID [--show-code] [--show-raw]` | Metadata; code/raw output withheld unless asked |
| `candidates stats RUN_ID` | Full statistics, by-strategy breakdown |
| `candidates migrate [--source] [--force]` | Upgrades the legacy flat file |

Tables print to stdout (user-facing output); diagnostics go to the logger — the Stage 2
precedent.

## 13. Manual verification (spec §55, §56)

All run against the mock model (offline) except the migration, which used the real
historical dataset:

```
pytest -q                                          → 306 passed
python -m python_dpo generate --limit 2 --num-candidates 5 --mock-model
python -m python_dpo runs list / show / validate    → all as expected, "Run validation passed."
python -m python_dpo candidates list / stats / show → correct, code withheld by default

# §56 mandatory resume test — a real interruption, not just a unit test
#   (mock generation completes far faster than a keystroke, so the interruption was
#    simulated by truncating a completed run's candidates.jsonl to 7/15 records and
#    force-setting manifest status to "interrupted" directly, then running the real
#    --resume path)
python -m python_dpo generate --resume RUN_ID --mock-model
  → generated=8 skipped=7, final status completed, runs validate passes

python -m python_dpo generate --resume RUN_ID --force --mock-model
  → new run, 15/15 generated, original run's candidates.jsonl byte-identical (md5 checked)

python -m python_dpo candidates migrate             → 50/50 records migrated, source untouched
python -m python_dpo candidates migrate              (again, no --force) → refused, exit 1
python -m python_dpo candidates migrate --force      → overwrote cleanly, still 50 records

# tampering detection
sed-edited a candidate's code_sha256 in a copy of the migrated run →
  "candidate p001_c004: code_sha256 does not match the stored code: expected ..., got ..."
  (plus the correct cascading statistics-drift and completeness findings)

# torn-tail detection and repair
appended a truncated JSON fragment to a copy's candidates.jsonl →
  validate reports "truncated final line ... looks like a torn write" at the right line
  number, plus every candidate downstream of it reported missing (a torn file is not
  trusted at all until repaired — consistent with how CandidateRepository.load_all()
  already treated any malformed line before Stage 4)
  repair_truncated_tail() removed exactly the appended bytes → validate then passed
```

## 14. Statistics implementation — reconstructability spot-check

`RunStatistics.from_records` was cross-checked against hand-counted fixtures in
`tests/test_runs.py` (`test_statistics_matches_hand_counted_records`,
`test_incomplete_problem_is_not_counted_as_completed`,
`test_retry_attempts_count_infrastructure_failures_only`), and against a live run's
`statistics.json` via `runs validate`'s freshness check, which recomputes and compares on
every invocation.

## 15. Tests added

- `tests/test_atomic_io.py` (18 tests) — the durable-write primitives.
- `tests/test_runs.py` (26 tests) — manifest, statistics, and repository.
- `tests/test_run_validation.py` (15 tests) — one per spec §51 corruption, built by
  mutating a real, valid, completed run.
- `tests/test_migration.py` (6 tests) — legacy-file migration.
- `tests/test_candidates.py` — extended with schema-versioning tests (hash
  verification/rejection, legacy 1.0 round-trip, the spec §23 lookup API) alongside the
  existing schema and repository tests.
- `tests/test_generation_pipeline.py` — rewritten against real run directories (a
  spec-driven behavior change: resume and duplicate detection are now run-scoped, so the
  Stage 3 cross-run tests were replaced with the equivalent run-scoped assertions, not
  loosened); added retry tests, the §42/§49 mandatory interruption/resume integration
  test, and the §50 reproducibility test.
- `tests/test_project.py` — extended with `--resume` parsing, the new `runs`/`candidates`
  subcommand parsing, and CLI-level error-path checks (unknown run id, conflicting
  `--resume` flags, missing migration source).

Total suite: **306 tests, all passing, 0 skipped**, fully offline and CPU-only
(`tests/test_no_heavy_imports.py` confirms `runs/` never pulls in `torch`/`transformers`).

## 16. Resume integration-test results

Both the automated integration test and the manual CLI-level check (§13) produced the
required outcome: exactly the requested candidate count after resume, the pre-interruption
records byte-for-byte unchanged, and final status `completed`.

## 17. Reproducibility-test results

`test_mock_generation_is_reproducible_across_runs` runs two independent runs with
identical problems, mock client, prompt version, generation config, and strategies, and
asserts identical `code_sha256` per `(problem_id, generation_index)` — passing. Real-model
reproducibility is **not** claimed (spec §50): GPU kernel non-determinism, framework
version drift, sampling implementation, model revision, and hardware can all make
identical seeded inputs produce different real-model output. This was already the Stage 3
position and is unchanged.

## 18. Files created/modified

**Created:**

- `src/python_dpo/atomic_io.py`
- `src/python_dpo/candidates/hashing.py`
- `src/python_dpo/runs/__init__.py`, `models.py`, `environment.py`, `repository.py`,
  `migration.py`, `validation.py`, `README.md`
- `tests/test_atomic_io.py`, `test_runs.py`, `test_run_validation.py`, `test_migration.py`
- `04_CANDIDATE_PERSISTENCE.md` (this file)

**Modified:**

- `src/python_dpo/candidates/models.py` — schema 2.0 fields, hash verification, error-type
  split.
- `src/python_dpo/candidates/repository.py` — run-scoped, `atomic_io`-backed, spec §23
  API, prompt artifact.
- `src/python_dpo/candidates/__init__.py`, `README.md` — updated exports and docs.
- `src/python_dpo/generation/generator.py` — takes a `RunManifest`, retry loop, prompt
  persistence before inference, hash-based duplicate detection, dropped `force` parameter.
- `src/python_dpo/cli.py` — `--resume`, the `runs` and `candidates` command groups,
  run-lifecycle management in `_execute_run`.
- `src/python_dpo/config.py` — `RetrySettings`, `generation.retry` parsing.
- `src/python_dpo/__init__.py` — version `0.3.0` → `0.4.0`.
- `config.yaml` — `generation.retry.max_attempts: 2`.
- `CLAUDE.md` — Security section extended: hashing is not execution.
- `README.md`, `src/python_dpo/README.md`, `tests/README.md`, `data/README.md`,
  `scripts/README.md` — Stage 4 documentation.
- `scripts/smoke_real_model.sh` — updated to locate the run directory the smoke test just
  created (the flat-file tail it used to print no longer receives new writes) and to
  print `runs show`/`candidates show`/`runs validate` instead.
- `tests/test_candidates.py`, `test_generation_pipeline.py`, `test_project.py` — see §15.

## 19. Dependencies added

None. `atomic_io.py` uses only `json` and `os` from the standard library; hashing uses
`hashlib`; run ids use `secrets`. No new entry in `pyproject.toml`.

## 20. Deviations from the specification

- **`CandidateRepository` methods drop the `run_id` argument** the spec §23 signatures
  show (e.g. `get(run_id, candidate_id)` → `get(candidate_id)`). The repository is
  constructed per run, so the argument would be redundant and unenforceable; spec §24
  explicitly permits the exact API to differ.
- **JSONL append uses fsync + torn-tail detection on read, not copy-and-rename per
  write.** Spec §21's actual requirement — a half-record is never mistaken for valid data
  — is satisfied either way; a full file rewrite per candidate would be quadratic against
  the 2,500-candidate target in spec §53.
- **`--force` alone is a no-op** under the confirmed "`generate` always starts a new run"
  semantics; it only does something combined with `--resume`. Kept for Stage 3 flag
  compatibility.
- **`duplicate_of` is run-scoped**, dropping Stage 3's cross-run linking — required by
  spec §20, not a regression; `find_by_hash` covers the cross-run analysis case instead.
- **`model_load` failures are not retried** — a run-level abort (Stage 3 §26.2 policy,
  carried forward), since retrying it per candidate would emit one identical failure per
  generation rather than one meaningful one.
- **`GenerationFailure.traceback` stays unpopulated** although spec §27 lists it as
  optional: a traceback embeds absolute filesystem paths, which spec §33's environment
  rule forbids recording. The field exists in the schema and stays `null`.
- **`problems_completed`** counts "every requested index has an outcome," not "every
  index succeeded" — this follows the spec's own §25 worked example (10 problems,
  2 failures, `problems_completed: 10`) rather than a looser paraphrase of it.
- The spec file itself is named `04_candidate_presistence.md` (a misspelling in the
  filename, not the content). Left as-is; happy to rename if you'd like.

## Issues for review before Stage 5

1. **Duplication rate.** The migrated real run shows 31/50 candidates as duplicates of
   another candidate for the same problem — the 3B model converging on near-identical
   solutions for several easy problems, exactly the Stage 3 report's stated risk. Worth
   deciding before Stage 5 (evaluation) whether some problems need re-generation with a
   larger model, different strategies, or higher temperature to get usable preference
   pairs.
2. **`statistics.json` can go stale under abrupt termination** (a real crash, not a
   caught `KeyboardInterrupt`) since it's rewritten only after `generate()` returns or is
   caught. `runs validate` always recomputes and compares against the file rather than
   trusting it, so this is self-healing on the next `validate`/`generate --resume` call,
   but it's worth knowing the cached number can briefly lie.
3. **Two scratch mock-model runs** created during manual verification were deleted from
   `data/candidates/runs/` before finishing; only the real migrated run
   (`20260817_055411`) and the legacy flat file remain as new artifacts under `data/`.
   Nothing has been committed — that's your call.

Stopping here. Not starting Stage 5 (Docker sandbox) without explicit approval.
