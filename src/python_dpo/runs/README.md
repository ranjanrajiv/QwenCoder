# src/python_dpo/runs/

Generation runs: manifests, statistics, integrity validation, and the repository that
owns them all. Introduced in Stage 4 to turn Stage 3's flat, single append-only
`candidates.jsonl` into a reliable experiment artifact store — one self-contained,
independently auditable directory per run (spec 04 §3, §6).

`runs` depends on `candidates` (a `RunManifest` embeds `ModelConfig`/`GenerationConfig`
dicts, a `RunStatistics` is computed from `Candidate`/`GenerationFailure` records). That
dependency runs **one way only** — `candidates` never imports from `runs` — so importing
either package alone never triggers a circular import.

## Files

### `models.py`

- **`RunManifest`** — the historical, immutable configuration snapshot for one run:
  status, model, generation config, strategies, requested problems/count, retry policy,
  environment, and (on failure) a `RunFailure`. This is the source of truth for
  reproducing or resuming a run — never today's `config.yaml` (spec 04 §34).
  `with_status(...)` returns a new manifest after checking the status transition is
  allowed (`RUN_STATUS_TRANSITIONS`); `completed`/`cancelled` are terminal.
- **`RunStatistics`** — always a *cache*. `RunStatistics.from_records(manifest,
  candidates, failures)` recomputes every counter from the persisted JSONL files; nothing
  in `statistics.json` is ever trusted from in-memory counters alone (spec 04 §25).
  `problems_completed` counts a problem as done when every requested index has *either* a
  candidate or a terminal failure — matching the run-completion rule in spec 04 §9, not
  "every index succeeded."

### `environment.py`

`capture_environment()` — Python version, platform, and `transformers`/`torch`/CUDA
versions when installed, each probed inside a `try` so importing this module never forces
a heavy import. Never records a username, home directory, or token (spec 04 §33) —
notably, `platform.node()` (hostname) is never called.

### `repository.py`

`RunRepository(runs_root)` — the only code that mints run ids, writes `manifest.json`, or
writes `statistics.json`. Run ids are `run_YYYYMMDD_HHMMSS_xxxx` (spec 04 §5): the random
hex suffix makes a second collision negligible, so minting costs one existence check
rather than a directory scan.

Status lifecycle: `create_run` → `created`; `start_run`/`resume_run` → `running`;
`complete_run` → `completed`; `interrupt_run` → `interrupted`; `fail_run` → `failed` with
a recorded `RunFailure`. `resume_run` refuses a `completed` run (spec 04 §11).
`create_run_from(manifest)` seeds a **new** run from an existing manifest's
configuration — the mechanism behind `--resume RUN_ID --force` and migration; it never
overwrites the source run (spec 04 §13).

`candidates(run_id)` returns the `CandidateRepository` scoped to that run's directory —
the only supported way to reach a run's candidate data.

### `migration.py`

`migrate_flat_file(source_path, run_repo, force=False)` — reads the Stage 3 flat
`candidates.jsonl` (and a sibling legacy `generation_failures.jsonl`, if present),
groups records by their existing `run_id`, and writes each group into its own run
directory through the same `RunRepository` code path a real `generate` uses. Candidates
are upgraded to schema 2.0 (hashes back-filled via `Candidate.create`); the source file is
only ever read, never modified (spec 04 §46). Refuses to overwrite an existing run
directory unless `force=True`.

### `validation.py`

`validate_run(run_dir, known_problem_ids)` — reads the raw JSONL lines directly (not
through `CandidateRepository`) so it can collect *every* problem in one pass instead of
raising on the first. Checks manifest presence/shape, JSONL structural integrity
(including a torn tail), per-record schema validity — which is also where hash
correctness is caught, since `Candidate.__post_init__` already recomputes and compares
every hash — duplicate candidate ids, `candidate_id`/`problem_id`/`generation_index`
consistency, `run_id` consistency, known `problem_id`s, `duplicate_of` targets, prompt
presence in `prompts.jsonl`, `statistics.json` freshness, and (for a `completed` run)
that every requested `(problem, index)` has an outcome. Returns a
`RunValidationReport`; `format_run_report(report)` renders it as the pass/fail text the
CLI prints.

## Directory layout

```
data/candidates/runs/<run_id>/
├── manifest.json          # RunManifest, atomically rewritten on every status change
├── candidates.jsonl       # append-only, one fsynced line per candidate
├── failures.jsonl         # append-only, one fsynced line per generation failure
├── statistics.json        # RunStatistics cache, rewritten after every generate() call
└── prompts/
    └── prompts.jsonl      # exact prompt per attempt, written before inference
```
