# src/python_dpo/candidates/

Schema and persistence for generated candidates. Two record types, two files, and a
deliberate rule about which is which.

## Files

### `models.py`

Frozen dataclasses validating in `__post_init__`, matching the Stage 2 problem schema.
Construction validates, so every record reaching disk is already well-formed.

**`Candidate`** — one generated program with full provenance:

| Field | Notes |
|---|---|
| `candidate_id` | `p001_c001`, deterministic. Unique **per run**, not per file — see below |
| `problem_id`, `run_id`, `generation_index`, `strategy` | Where this candidate came from |
| `model`, `model_revision`, `provider` | Which backend produced it |
| `prompt_version`, `prompt` | The exact prompt, so the record survives a template change |
| `raw_output` | The complete model response, kept for debugging extraction (§25) |
| `code` | The extracted source |
| `extraction_format` | How the code was found (`python_fence`, `generic_fence`, `plain`) |
| `syntax_valid`, `syntax_error` | Result of `ast.parse` — a property, not a verdict |
| `function_name_valid` | Whether the expected entry point is defined |
| `duplicate_of` | Earliest candidate with identical code, or null |
| `generation_config`, `created_at` | Decoding parameters and timestamp |

`code` is required and non-empty: a candidate exists only when code was extracted.
`extraction_format` therefore cannot be `unknown` on a stored candidate — that value only
appears in a failed `ExtractionResult`, which produces a failure record instead.

**`GenerationFailure`** — a generation that produced *no candidate*: `run_id`,
`problem_id`, `generation_index`, `strategy`, `error_type`, `error_message`, `timestamp`.
`error_type` is a closed set (`model_load`, `tokenizer`, `inference`, `timeout`,
`empty_output`, `code_extraction`) so failures are countable across runs rather than
grouped by free text.

`syntax_error` is deliberately **absent** from that set. Unparseable code is stored as a
`Candidate` with `syntax_valid=false`, never as a failure — one generation produces one
record or the other, never both (spec 03 §19.1, §26.1).

### `repository.py`

`CandidateRepository`, reading and appending both artifacts in one directory.

**Append-only.** Each `append` opens, writes one line, and closes — a few extra syscalls
per candidate in exchange for a file that stays usable if the run is killed halfway
through (spec 03 §24). Nothing is ever rewritten or deleted.

`--force` mints a new `run_id` and appends rather than replacing, so the same
`candidate_id` legitimately appears more than once. **The file-wide key is
`(run_id, candidate_id)`** (spec 03 §21.1). Use `latest_by_candidate_id()` to collapse to
one record per candidate; because the file is append-only, a later line is by construction
a later run, so no timestamp comparison is needed.

Read helpers:

- `load_all()` / `load_failures()` — validate every line and reject a malformed one with
  its line number. **Never silently skips**, matching `problems/storage.py`.
- `existing_keys()` — the `(problem_id, generation_index)` resume index. A generation that
  previously *failed* left no candidate behind and so is absent here, which is what makes
  failures automatically retryable on the next run.
- `code_index()` — `problem_id → {code: earliest candidate_id}` for exact duplicate
  detection. Scoped per problem: identical code for two different problems is a
  coincidence, not a duplicate.
- `new_run_id()` — the `20260817_103000` shape from spec 03 §22, in **UTC**, suffixed
  `_2`, `_3`, … if that second already produced a run. Deliberately not a UUID.

Note that run ids are unique among runs that persisted at least one record. A run that
wrote nothing at all (a pure resume, for instance) leaves no trace and cannot collide with
anything.
