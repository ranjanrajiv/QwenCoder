# src/python_dpo/candidates/

Schema and persistence for generated candidates. Two record types, three files, and a
deliberate rule about which record type is which. As of Stage 4, one repository instance
owns exactly one **run directory** — see `../runs/` for the layer that owns runs
themselves.

## Files

### `hashing.py`

One function, `sha256_text(text) -> str`, so every hash in the system — code, prompt, raw
output — is computed identically.

### `models.py`

Frozen dataclasses validating in `__post_init__`, matching the Stage 2 problem schema.
Construction validates, so every record reaching disk is already well-formed.

**`Candidate`** — one generated program with full provenance:

| Field | Notes |
|---|---|
| `candidate_id` | `p001_c001`, deterministic. Unique **per run** |
| `problem_id`, `run_id`, `generation_index`, `strategy` | Where this candidate came from |
| `model`, `model_revision`, `provider` | Which backend produced it |
| `prompt_version`, `prompt` | The exact prompt, so the record survives a template change |
| `raw_output` | The complete model response, kept for debugging extraction |
| `code` | The extracted source |
| `extraction_format` | How the code was found (`python_fence`, `generic_fence`, `plain`) |
| `syntax_valid`, `syntax_error` | Result of `ast.parse` — a property, not a verdict |
| `function_name_valid` | Whether the expected entry point is defined |
| `duplicate_of` | Earliest candidate **in this run** with identical code, or null |
| `generation_config`, `created_at` | Decoding parameters and timestamp |
| `schema_version` | `"1.0"` (Stage 3, no hashes) or `"2.0"` (Stage 4) |
| `code_sha256`, `prompt_sha256`, `raw_output_sha256` | SHA-256 of the three text fields — `null` on a 1.0 record |
| `attempt` | Which retry attempt produced this candidate (spec 04 §30) |

`code` is required and non-empty: a candidate exists only when code was extracted.
`extraction_format` therefore cannot be `unknown` on a stored candidate — that value only
appears in a failed `ExtractionResult`, which produces a failure record instead.

**The three hashes are verified, not just stored.** `__post_init__` recomputes each from
its source text and rejects a mismatch — a tampered record cannot be constructed or
loaded. Use `Candidate.create(...)` rather than the constructor directly when producing a
new record; it computes all three hashes for you and stamps `schema_version="2.0"`.

A record with no `schema_version` field reads as `"1.0"`; on such a record the three hash
fields must be absent (`None`), never invented. This is how the Stage 3 flat file stays
readable without silently reinterpreting old data (spec 04 §46) — see
`../runs/migration.py` for the explicit, one-time upgrade path.

**`GenerationFailure`** — a generation that produced *no candidate*: `run_id`,
`problem_id`, `generation_index`, `strategy`, `error_type`, `error_message`, `timestamp`,
plus (schema 2.0) `attempt` and `prompt_sha256` linking back to the run's prompt artifact.
`error_type` is a closed set, split into `INFRASTRUCTURE_ERROR_TYPES` (`model_load`,
`tokenizer`, `inference`, `timeout` — retried up to `generation.retry.max_attempts`) and
`CANDIDATE_ERROR_TYPES` (`empty_output`, `code_extraction` — never retried, spec 04 §28).

`syntax_error` is deliberately **absent** from that set. Unparseable code is stored as a
`Candidate` with `syntax_valid=false`, never as a failure — one generation produces one
record or the other, never both (spec 03 §19.1, §26.1).

### `repository.py`

`CandidateRepository(run_dir)` — **run-scoped**: one instance reads and appends exactly
one run's `candidates.jsonl`, `failures.jsonl`, and `prompts/prompts.jsonl`. There is no
`run_id` argument on its methods, because the directory already is one run
(`RunRepository.candidates(run_id)` is the entry point — see `../runs/`).

**Durable appends, not batched writes.** Each `save()`/`save_failure()`/`append_prompt()`
is one `fsync`ed line via `python_dpo.atomic_io.append_jsonl` — a job killed halfway
through leaves a usable, resumable file behind (spec 04 §21, §43).

Because each run is its own directory, `candidate_id` is unique **within** a
`CandidateRepository` — no `(run_id, candidate_id)` compound key is needed, unlike Stage
3's flat file.

The §23 lookup API: `save`, `get`, `exists`, `list`, `count`, `find_by_problem`,
`find_by_hash`. Plus the indexes the generator needs:

- `existing_keys()` — the `(problem_id, generation_index)` resume index, scoped to this
  run. A generation that previously *failed* left no candidate behind and so is absent
  here, which is what makes it retryable on resume.
- `code_index()` — `problem_id → {code_sha256: earliest candidate_id}` for exact duplicate
  detection **within this run** (spec 04 §19). Duplicates are never auto-detected across
  runs (§20) — `find_by_hash` plus manual cross-run comparison is the intended tool for
  that instead.
- `load_index()` — builds `existing_keys()`, `code_index()`, and the full candidate list
  in one file pass.

### `migration.py`

Not here — moved to `../runs/migration.py`. `candidates` must never import from `runs`
(the dependency runs one way only), and migration genuinely needs both packages.
