# Stage 4 — Candidate Persistence, Runs and Reproducibility

## Context

Stage 3 shipped a working generator: `generate` prompts a model five ways per problem and
appends records to a single flat `data/candidates/candidates.jsonl`, with provenance fields
and a `run_id` stamped on every record. That file currently holds **50 real Qwen records**
from run `20260817_055411`.

The flat file is a *log*, not an experiment store. It has no record of what a run was asked
to do, so an interrupted run cannot be distinguished from a finished one; no manifest, so
reproducing a historical run means trusting today's `config.yaml`; no integrity checks, so a
torn write or an edited line is indistinguishable from real data; and resume is inferred
across *all* runs at once rather than scoped to the experiment being resumed.

`.claude/specs/04_candidate_presistence.md` (note: the filename misspells "persistence")
asks for the layer that turns this into a reliable artifact store — per-run directories with
manifests, atomic persistence, SHA-256 hashes, explicit resume, retry policy, statistics
reconstructable from disk, an integrity validator, and CLI commands to inspect it all.

The hard boundary is unchanged and reaffirmed by spec §45: **this stage never executes
candidate code.** Persistence hashes candidate code and counts it; it never `exec`s, `eval`s,
or subprocesses it. `InProcessReferenceExecutor` stays the only `exec()` in `src/`.

**Outcome:** a run is a self-contained, independently auditable directory. Ctrl+C leaves it
resumable; `generate --resume RUN_ID` fills exactly the gaps without touching a byte of what
exists; `runs validate` fails loudly on any corruption; `pytest -q` stays green and offline.

### Decisions confirmed with the user

1. **`generate` always mints a new run; resume is explicit.** No auto-resume — the surprise
   of a re-run silently doing nothing is worse than the cost of an explicit flag. `--force`
   is documented as "never resume"; combined with `--resume R` it seeds a *new* run from
   `R`'s manifest and regenerates everything (spec §13: force must never overwrite in place).
2. **The legacy flat file is migrated, not discarded.** An explicit `candidates migrate`
   command back-fills hashes into a proper run directory. The source file is left untouched.
   Explicit, so nothing is silently reinterpreted (§46).
3. **Run IDs adopt the spec format** `run_YYYYMMDD_HHMMSS_<4 hex>`. Collision-proof without
   a directory scan. Existing IDs are only ever read, never re-minted.

---

## Layout

```
data/candidates/
├── candidates.jsonl                    # Stage 3 legacy file — read-only, left in place
└── runs/
    └── run_20260817_133700_a81f/
        ├── manifest.json               # config snapshot + status + environment
        ├── candidates.jsonl            # schema_version 2.0, one line per candidate
        ├── failures.jsonl              # every generation that produced no candidate
        ├── statistics.json             # cache; always reconstructable from the two above
        └── prompts/prompts.jsonl       # exact prompt per attempt, written pre-inference
```

`prompts.jsonl` is not redundant with `Candidate.prompt`: it is written **before** the model
call, so a generation that fails or is interrupted still has its exact prompt recoverable
(§31). A candidate record carries the prompt too, so a candidate stays interpretable alone.

---

## New module — `src/python_dpo/atomic_io.py`

Shared durable-IO primitives. Generalizes the existing `_read_records` helper in
`candidates/repository.py:32`, which both repositories will now call.

- `atomic_write_json(path, payload)` — write to `path.with_suffix(".tmp")` in the same
  directory, `flush()`, `os.fsync()`, `os.replace()`, then fsync the directory. Used for
  `manifest.json` and `statistics.json`, which are whole-file rewrites.
- `append_jsonl(path, payload)` — serialize the complete line first, then one `write()` of
  `line + "\n"` in `"a"` mode, `flush()`, `os.fsync(fileno)`. The record is durable before
  the generator moves on, which is what makes Ctrl+C recoverable (§43).
- `iter_jsonl(path)` — yields `(line_number, obj)`. Raises `JsonlError` on invalid JSON, a
  non-object, a blank line, **or a final line with no trailing newline** (a torn write).
  Never silently skips — CLAUDE.md Data Integrity, matching `problems/storage.py:44`.
- `repair_truncated_tail(path) -> int` — truncates the file at the last complete newline and
  returns the bytes removed. Called **only** from `runs validate --repair`, never
  automatically, and it reports exactly what it dropped.

Spec §21's "temp write → flush → fsync → atomic rename" applies literally to the JSON files.
For JSONL append, a full copy-and-rename per candidate would be quadratic in a 2,500-record
run (§53); a single fsynced write of a complete line, plus torn-tail detection on read, gives
the same guarantee — a half-record is never mistaken for valid data (§21's actual requirement).

---

## New package — `src/python_dpo/runs/`

**`models.py`** — frozen dataclasses validating in `__post_init__`, raising `RunError`,
matching the `problems/models.py` and `candidates/models.py` house style.

- `RUN_STATUSES = {"created","running","completed","failed","interrupted","cancelled"}` (§8),
  `MANIFEST_VERSION = "1.0"`, `STATISTICS_VERSION = "1.0"`.
- `RunManifest` — `manifest_version`, `run_id`, `status`, `created_at`, `started_at`,
  `completed_at`, `candidate_schema_version`, `prompt_version`, `model` (full
  `ModelConfig.to_dict()`), `generation_config` (`GenerationConfig.to_dict()`), `strategies`,
  `requested_problem_ids`, `requested_problems`, `requested_candidates_per_problem`,
  `retry`, `environment`, `error`, `source` (`"generate"` | `"migrated"`).
  `requested_problem_ids` is what makes resume derivable from disk rather than from the CLI
  flags of the second invocation (§12, §34).
- `RunStatistics` with `from_records(manifest, candidates, failures)` — the single source of
  truth; `statistics.json` is only its cache (§25). Fields per §25/§26/§41, keeping
  `requested` / `generated` / `valid` / `failed` strictly distinct: `problems_requested`,
  `problems_completed`, `candidates_requested`, `candidates_generated`, `generation_failures`,
  `retry_attempts`, `syntax_valid`, `syntax_invalid`, `function_name_valid`, `duplicates`,
  `candidates_by_strategy`, `failures_by_error_type`, `computed_at`.
  `problems_completed` counts problems where every requested index has a candidate.

**`repository.py`** — `RunRepository(runs_root)`, owning all run state (§24).

`create_run(...)`, `create_run_from(manifest)` (for `--resume --force` and migration),
`get_run(run_id)`, `update_status(run_id, status)`, `list_runs()` (newest first),
`resume_run(run_id)`, `complete_run(run_id)`, `fail_run(run_id, error_type, message,
problem_id, generation_index)`, `write_statistics(run_id, stats)`, `run_dir(run_id)`,
`candidates(run_id) -> CandidateRepository`.

`new_run_id()` = `f"run_{now:%Y%m%d_%H%M%S}_{secrets.token_hex(2)}"`. `resume_run` refuses a
run whose status is `completed`, and moves `interrupted`/`created`/`failed` → `running`.

**`environment.py`** — `capture_environment() -> dict`: `python_version`, `platform`
(`platform.system()`/`release()`/`machine()` — **never** `node()`), `transformers_version`,
`torch_version`, `cuda_version`. Each optional dependency probed inside a `try` and recorded
as `null` when absent, so this module never forces a torch import (`test_no_heavy_imports.py`
must stay green). §33: no username, no home path, no token, ever.

**`validation.py`** — `validate_run(run_dir, known_problem_ids) -> RunValidationReport` plus
`format_run_report(report)`, mirroring `problems/validation.py`'s report shape. It reads raw
JSONL lines itself rather than calling `load_all()`, so it can collect *every* issue instead
of stopping at the first. Checks (§22, §38, §51):

| # | Check |
|---|---|
| 1 | `manifest.json` exists, parses, known `manifest_version`, status in the closed set |
| 2 | Every JSONL line is a complete JSON object (torn tail reported with its line number) |
| 3 | Every record passes `Candidate` / `GenerationFailure` schema validation |
| 4 | `candidate_id` unique within the run |
| 5 | `candidate_id == build_candidate_id(problem_id, generation_index)` |
| 6 | `code_sha256` / `prompt_sha256` / `raw_output_sha256` recompute correctly |
| 7 | Every record's `run_id` matches the directory name |
| 8 | Every `problem_id` exists in `data/problems/problems.jsonl` |
| 9 | `duplicate_of` points at a candidate in this run with the same code hash |
| 10 | Every candidate's `prompt_sha256` appears in `prompts.jsonl` |
| 11 | `statistics.json` equals a fresh recomputation from the records |
| 12 | `status == completed` ⟹ every requested `(problem, index)` has a candidate or a failure |

Legacy `schema_version 1.0` records report check 6 as *skipped*, never as passed.

---

## Modified — `src/python_dpo/candidates/`

**`hashing.py`** (new) — `sha256_text(text) -> str`, UTF-8, hexdigest. One function, so
every hash in the system is computed identically.

**`models.py`** — `CANDIDATE_SCHEMA_VERSION = "2.0"`. No Stage 3 field is removed (§15, §46).

New `Candidate` fields: `schema_version`, `code_sha256`, `prompt_sha256`,
`raw_output_sha256`, `attempt` (default `1`, identifying which attempt succeeded — §30).
`__post_init__` **recomputes and compares** each hash, so a tampered record cannot be
constructed or loaded; `Candidate.create(...)` is a classmethod that computes them for the
generator. `from_dict` treats a missing `schema_version` as `"1.0"`, where the hash fields
must be *absent* and are left `None` — reading an old record must not silently invent
provenance it never had (§46).

New `GenerationFailure` fields: `schema_version`, `attempt`, `prompt_sha256` (links a failed
generation to its `prompts.jsonl` entry), and optional `traceback` — left unpopulated by
default, since tracebacks embed absolute home paths that §33 forbids.
`INFRASTRUCTURE_ERROR_TYPES = {"model_load","tokenizer","inference","timeout"}` vs
`CANDIDATE_ERROR_TYPES = {"empty_output","code_extraction"}` encodes §28's distinction; only
the former are retried.

**`repository.py`** — `CandidateRepository` becomes **run-scoped**. Its constructor already
takes a directory (`repository.py:62`), so this is `CandidateRepository(run_dir)` with no
signature change. `FAILURES_FILENAME` becomes `failures.jsonl` per §6/§57;
`LEGACY_FAILURES_FILENAME` keeps the old name for the migration reader.

§23 API, with `run_id` implicit because the repository is run-scoped (§24 permits an API
that differs — flagged as a deviation): `save`, `get`, `exists`, `list`, `count`,
`find_by_problem`, `find_by_hash`, plus `existing_keys()` (now run-scoped — the resume index)
and `code_index()` (now keyed on `code_sha256`). `append_prompt(record)` / `load_prompts()`
for the prompts artifact. A `load_index()` builds all three lookups in one file pass.

Dropped: `new_run_id` (moves to `RunRepository`) and `latest_by_candidate_id` — within a run
`candidate_id` is unique, and check 4 above now enforces it.

**Duplicate detection becomes run-scoped**, which is the spec's rule, not an accident: §19
detects duplicates within a run; §20 explicitly forbids auto-rejecting duplicates *across*
runs and asks only that the hash be recorded for later cross-run analysis. `find_by_hash`
plus `code_sha256` is that mechanism. Duplicates are still always kept (§19).

---

## Modified — `src/python_dpo/generation/generator.py`

Takes a `RunManifest` instead of a bare `run_id`, plus `retry: RetrySettings`. Per candidate
index:

1. Skip if `(problem_id, index)` is already in this run's `existing_keys()` (§12, §44).
2. Build the prompt; append it to `prompts.jsonl` **before** inference (§31).
3. Attempt loop, `1..max_attempts`:
   - `ModelLoadError` → record `model_load` failure, re-raise; the run aborts (Stage 3 §26.2).
   - Other inference exception → record an `inference` failure **for that attempt** and retry
     if attempts remain. A retry never overwrites the original failure record (§29); each
     attempt gets its own line, and the successful candidate carries `attempt=N` (§30).
   - Success → break.
4. Empty output / no extractable code → a **candidate failure**: recorded once, not retried
   (§28), no candidate.
5. Syntax and function-name results are recorded as fields, never as failures (Stage 3 §19.1).
6. Duplicate check against this run's `code_sha256` index → `duplicate_of`; the duplicate is
   kept (§19).
7. `save()` the candidate — one fsynced append (§21).

`KeyboardInterrupt` is a `BaseException`, so the existing `except Exception` never swallows
it; it propagates to the CLI, which marks the run `interrupted`. At the end of `generate`,
statistics are **recomputed from the persisted files**, not from the in-memory counters (§25);
`GenerationSummary` is derived from that recomputation.

---

## Modified — `src/python_dpo/cli.py`

`_cmd_generate` gains `--resume RUN_ID`. Run lifecycle, with status persisted at each edge:

```
run = repo.create_run(...)                    # created  -> manifest.json written
   or run = repo.resume_run(args.resume)      # interrupted -> running
repo.update_status(run_id, "running")
try:      summary = generator.generate(...)
except KeyboardInterrupt:  update_status("interrupted"); write stats; return 130
except ModelError as exc:  fail_run(run_id, exc, problem_id, index); write stats; return 1
except (CandidateStoreError, JsonlError, OSError) as exc: fail_run(...); return 1
else:     complete_run(run_id) if every requested (problem, index) has a candidate or a
          recorded failure, else update_status("interrupted")     # §9, §10
finally:  write statistics.json
```

On `--resume`, the **manifest is the source of truth** (§34): problems, count, strategies,
generation config, prompt version and model all come from it. A conflicting selection flag
(`--problem-id`, `--limit`, `--num-candidates`, `--strategy`) is an error rather than a
silent override. `--resume R --force` calls `create_run_from(R's manifest)` and generates
everything into the new run, leaving `R` untouched (§13).

New command groups, following the existing `_add_problems_parser` pattern (`cli.py:101`),
with tables on **stdout** and diagnostics on the logger (the Stage 2 precedent, `cli.py:96`):

| Command | Behavior |
|---|---|
| `runs list` | `RUN ID / STATUS / CANDIDATES / FAILURES / CREATED`, newest first (§36) |
| `runs show RUN_ID` | Manifest, generation config, times, candidate and failure counts (§37) |
| `runs validate RUN_ID [--repair]` | The 12 checks; exit 0/1; `--repair` truncates only a torn tail (§38) |
| `candidates list RUN_ID [--problem-id P] [--strategy S]` | `candidate_id / problem_id / strategy / syntax` (§39) |
| `candidates show RUN_ID CANDIDATE_ID [--show-code] [--show-raw]` | Metadata; **no raw output by default** (§40) |
| `candidates stats RUN_ID` | §41 counts plus the per-strategy breakdown |
| `candidates migrate [--source PATH] [--force]` | Legacy flat file → run directory |

`candidates migrate` reads `data/candidates/candidates.jsonl` as schema 1.0, groups by
`run_id`, back-fills the three hashes and run-scoped `duplicate_of`, stamps
`schema_version: "2.0"`, synthesizes a manifest with `source: "migrated"`,
`status: "completed"` and `environment` fields it cannot know set to `null`, writes
`statistics.json`, and leaves the source file byte-identical. It refuses to overwrite an
existing run directory unless `--force` is given.

---

## Modified — config, version, docs

- **`config.yaml`** — add `generation.retry.max_attempts: 2` (§29).
- **`config.py`** — `RetrySettings(max_attempts: int)` frozen dataclass, validated `>= 1`
  (`1` meaning "no retry"); add `retry` to `_GENERATION_KEYS` (`config.py:31`) and to
  `GenerationSettings`. Runs root is `config.paths.candidates / "runs"` — no new path key.
- **`src/python_dpo/__init__.py`** — `__version__` → `0.4.0`.
- **`CLAUDE.md`** — one line under Security: the persistence layer hashes candidate code and
  never executes it (§45), keeping `InProcessReferenceExecutor` the sole `exec()`.
- **Docs** — `04_CANDIDATE_PERSISTENCE.md` at the repo root (the §58 report, matching the
  `02_`/`03_` convention); new `src/python_dpo/runs/README.md`; refresh
  `candidates/README.md`, `src/python_dpo/README.md`, `tests/README.md`, `data/README.md`,
  and the root `README.md` roadmap.

---

## Tests

All offline, CPU-only, no skips, `MockModelClient` throughout.

**New `tests/test_atomic_io.py`** — `atomic_write_json` replaces atomically and leaves no
`.tmp` behind; a failed write leaves the original intact; `append_jsonl` writes exactly one
complete line; a hand-truncated tail is detected with its line number; `repair_truncated_tail`
removes exactly the torn bytes and nothing else; a corrupt line *mid-file* is an error, never
repaired.

**New `tests/test_runs.py`** — `RunManifest` round-trip and rejection of an unknown status;
`RunRepository` create / get / update_status / list / resume / complete / fail; run ID format
and uniqueness; `resume_run` refuses a `completed` run; `RunStatistics.from_records` matches
hand-counted records; `capture_environment()` contains no username, home path, or token.

**New `tests/test_run_validation.py`** — one test per §51 corruption, each written by
mutating a good run directory: duplicate candidate ID, malformed JSON, wrong `code_sha256`,
missing required field, mismatched `run_id`, unknown `problem_id`, drifted `statistics.json`,
truncated tail, dangling `duplicate_of`, `completed` status with missing work. Each must fail
loudly and name the offending candidate.

**New `tests/test_migration.py`** — legacy flat records migrate with hashes back-filled and
`schema_version` stamped; the source file is byte-identical afterwards; the migrated run
passes `validate_run`; re-running without `--force` refuses to clobber.

**Extended `tests/test_candidates.py`** — hashes required and verified at 2.0; a legacy 1.0
record loads with hashes `None`; a mismatched hash is rejected; the §23 lookup methods;
`failures.jsonl` naming.

**Extended `tests/test_generation_pipeline.py`** — existing resume/force/duplicate tests are
rewritten against run directories, because the semantics genuinely changed (run-scoped resume
and run-scoped duplicates are what the spec requires); this is a spec-driven behavior change,
not a test bent to fit an implementation. Plus:

- **§42/§49 mandatory integration test** — 3 problems × 5 candidates; the mock's script
  raises `KeyboardInterrupt` after 7 candidates; assert status `interrupted`; resume; assert
  15 candidates, status `completed`, and that the file's first 7 records are **byte-for-byte
  unchanged** (`before == after[:len(before)]` on the raw bytes).
- **§50 reproducibility test** — run A and run B with identical problems, mock, prompt
  version, generation config, seed and strategies produce identical `code_sha256` values,
  differing only in `run_id` and timestamps. The report will state plainly that real-model
  reproducibility is *not* claimed (GPU kernels, framework versions, sampling, revision).
- **Retry** — an `InferenceError` on attempt 1 then success on attempt 2 yields one candidate
  with `attempt=2` **and** a retained failure record for attempt 1; exhausting
  `max_attempts` yields failures only; a candidate failure (empty output) is never retried.
- **Statistics** — every counter matches a hand-count of the persisted records.

**Extended `tests/test_project.py`** — the new subcommands parse; `--resume` parses; `runs
list` against an empty store exits 0; `--dry-run` still writes nothing.

---

## Execution order

1. Write this plan to `.claude/plans/04_candidate_persistence_plan.md` (repo convention).
2. `atomic_io.py` + its tests — everything else depends on it.
3. `candidates/hashing.py`, candidate/failure schema v2.0 + tests.
4. `runs/` package (models, environment, repository) + tests.
5. Run-scoped `CandidateRepository` + tests.
6. Generator: manifest, prompts artifact, retry, disk-recomputed statistics.
7. CLI: `--resume`, `runs`, `candidates`, `migrate`.
8. `runs/validation.py` + corruption tests.
9. Docs, version bump, `CLAUDE.md`.
10. Full suite; fix; re-run; report.

---

## Verification

```bash
source .venv/bin/activate
pytest -q                                         # all pass, 0 skipped, no downloads

# migrate the 50 existing Qwen records, then prove they survived intact
python -m python_dpo candidates migrate
git diff --stat data/candidates/candidates.jsonl  # empty — source untouched
python -m python_dpo runs validate run_20260817_055411

# a fresh offline run end to end
python -m python_dpo generate --limit 2 --num-candidates 5 --mock-model
python -m python_dpo runs list
python -m python_dpo runs show RUN_ID
python -m python_dpo runs validate RUN_ID         # "Run validation passed."
python -m python_dpo candidates list RUN_ID
python -m python_dpo candidates stats RUN_ID
python -m python_dpo candidates show RUN_ID p001_c001            # no raw output
python -m python_dpo candidates show RUN_ID p001_c001 --show-code

# §56 mandatory manual resume test — Ctrl+C partway through
python -m python_dpo generate --limit 3 --num-candidates 5 --mock-model   # ^C after a few
python -m python_dpo runs show RUN_ID                            # status: interrupted
python -m python_dpo generate --resume RUN_ID
python -m python_dpo runs show RUN_ID                            # status: completed, 15
python -m python_dpo runs validate RUN_ID

# integrity checker fails loudly on tampering (on a scratch copy)
cp -r data/candidates/runs/RUN_ID /tmp/tampered && \
  sed -i '3s/return/return  /' /tmp/tampered/candidates.jsonl
python -m python_dpo runs validate ...            # code_sha256 mismatch, named candidate

# real model, one candidate, then resume it
python -m python_dpo generate --problem-id p001 --num-candidates 5
```

Scope containment:

```bash
grep -rnE "\b(exec|eval)\(" src/                  # only InProcessReferenceExecutor
grep -rnE "subprocess" src/python_dpo/{runs,candidates}/      # no hits
grep -rniE "docker|pytest|trl|peft|lora|dpo" src/python_dpo/{runs,candidates,generation}/
grep -rniE "hf_token|hugging.*token" src/ config.yaml data/   # no values anywhere
python -m pytest tests/test_no_heavy_imports.py   # runs/ must not pull in torch
```

Then produce the §58 report in `04_CANDIDATE_PERSISTENCE.md` and **stop — do not start
Stage 5 (Docker sandbox) without explicit approval** (§58).

---

## Deviations to record in the report

- **`CandidateRepository` methods omit the `run_id` argument** §23 lists (`get(run_id,
  candidate_id)`). The repository is constructed per run, so the argument would be redundant
  and unenforceable. `RunRepository.candidates(run_id)` is the entry point. §24 permits this.
- **JSONL append uses fsync + torn-tail detection, not copy-and-rename.** §21 requires that a
  half-record never be treated as valid data, which this achieves; a full rewrite per
  candidate would be quadratic against §53's 2,500-candidate target.
- **`--force` alone is now a no-op** under the confirmed "always a new run" semantics. It is
  retained for compatibility with the Stage 3 flag set and is meaningful with `--resume`.
- **`duplicate_of` becomes run-scoped**, dropping the Stage 3 cross-run behavior. Required by
  §20; cross-run analysis is served by `code_sha256` + `find_by_hash`.
- **Retry does not cover `model_load`** — a run-level failure aborts the run (Stage 3 §26.2),
  and retrying it per candidate would emit one identical failure per generation.
- **Tracebacks are not persisted** although §27 lists them as optional: they embed absolute
  home paths that §33 forbids. The field exists and stays `null`.
- **Real-model reproducibility is not claimed** (§50) — only the mock path is asserted
  deterministic.
- The spec file is named `04_candidate_presistence.md` (misspelled). Left as-is unless you
  want it renamed.
