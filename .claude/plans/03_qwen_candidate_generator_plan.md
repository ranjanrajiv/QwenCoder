# Stage 3 — Qwen Candidate Generator

## Context

Stage 2 delivered the ground-truth layer: ten curated `Problem` records with trusted
reference solutions and ~75 executable tests in `data/problems/problems.jsonl`, plus a
`problems build` / `problems validate` CLI. There is still nothing to have an opinion
*about* — no model output exists anywhere in the repo.

`.claude/specs/03_qwen_candidate_generator.md` asks for the **candidate-generation layer**:
take a validated problem, prompt a Qwen Coder model five different ways, extract the Python
out of whatever the model returns, validate it at the syntax level only, and persist every
candidate with full provenance. DPO needs a *spread* of quality per problem — that spread is
what later stages rank into chosen/rejected pairs.

The hard boundary this stage must respect: the generator answers **"what code did Qwen
generate?"** and nothing else. It never executes a candidate, never decides correctness, and
never repairs malformed output. `ast.parse()` is the only thing that ever touches generated
code, and CLAUDE.md's Security rule stays intact — `InProcessReferenceExecutor` remains the
only `exec()` in `src/`, and generated code is never routed to it.

**Outcome:** `python -m python_dpo generate --problem-id p001 --num-candidates 5` writes five
records to `data/candidates/candidates.jsonl`; re-running it regenerates nothing; `--force`
starts a new run; `--dry-run` prints prompts without loading a single model weight; `pytest -q`
stays green and never downloads a model.

### Decisions confirmed with the user

1. **Default model: `Qwen/Qwen2.5-Coder-3B-Instruct`** — ~6 GB in bf16, fits the RTX 3060's
   12 GB with room for KV cache, no quantization needed (spec §9 defers that). Set in
   `config.yaml` only; never in Python source (§6).
2. **`torch`/`transformers` go in an optional `[model]` extra**, not core dependencies. The
   Qwen client imports them lazily inside its load method, so the default install and the
   whole pytest suite stay offline and lightweight (§32).
3. **`--force` appends a new run rather than rewriting.** `candidates.jsonl` is append-only;
   `--force` mints a fresh `run_id` and regenerates, so `p001_c001` can legitimately appear
   twice with different `run_id`s. `(run_id, candidate_id)` is the unique key. Nothing is ever
   deleted — CLAUDE.md Data Integrity, and §22's stated purpose for `run_id`.

---

## New package — `src/python_dpo/models/`

The model abstraction. Nothing outside this package may import `transformers` or `torch`.

**`base.py`** — the seam.

- `GenerationConfig` — frozen dataclass validating in `__post_init__` (matching `Problem` /
  `Config` style): `temperature`, `top_p`, `max_new_tokens`, `do_sample`,
  `repetition_penalty`, `seed`. Range-checks each (`0 < top_p <= 1`, `max_new_tokens > 0`,
  `temperature >= 0`) and raises `ModelError`. Has `to_dict()` so it can be embedded verbatim
  in every candidate record (§45 traceability).
- `RawGeneration` — frozen dataclass: `text`, `prompt_token_count`, `completion_token_count`,
  `finish_reason`. What a client returns; deliberately *not* a candidate.
- `ModelClient` — `typing.Protocol`, `@runtime_checkable`, with
  `generate(prompt: str, generation_config: GenerationConfig) -> RawGeneration` plus
  `name: str` and `revision: str | None` properties (candidates must record which model
  produced them).
- `ModelError` / `ModelLoadError` / `InferenceError` — the exception hierarchy the generator
  catches and turns into structured failure records.

**`qwen.py`** — `QwenModelClient(ModelClient)`.

- **Lazy everything (§7).** `__init__` stores config and does no work. `transformers` and
  `torch` are imported *inside* `_ensure_loaded()`, which runs on the first `generate()` call.
  A missing extra raises `ModelLoadError("transformers is not installed; pip install -e
  '.[model]'")` rather than an `ImportError` traceback.
- **Device resolution (§10).** `auto` → `cuda` when `torch.cuda.is_available()`, else `cpu`
  with a WARNING that generation will be slow; explicit `cuda` without CUDA available is a
  `ModelLoadError`, not a silent CPU fallback. `dtype: auto` → `bfloat16` on CUDA,
  `float32` on CPU. All hardware branching lives in this file (§10).
- **Chat template.** Qwen `-Instruct` models expect one. If
  `tokenizer.chat_template` exists, wrap the built prompt via `apply_chat_template(...,
  add_generation_prompt=True)`; otherwise pass the raw prompt. Which path was taken is logged
  at DEBUG.
- **Decoding.** `transformers.set_seed(config.seed)` before each call; generate with the
  mapped kwargs; slice off the prompt tokens so `raw_output` is the completion only; decode
  with `skip_special_tokens=True`.
- `trust_remote_code` is threaded explicitly from config into both `from_pretrained` calls,
  **defaulting to `False`** (§8). Qwen2.5-Coder does not need it.
- Never logs the token env var, the prompt at INFO, or the completion at INFO (§39).

**`mock.py`** — `MockModelClient(ModelClient)`.

Deterministic, offline, zero dependencies. Returns canned output selected by `(entry_point
parsed from the prompt, strategy)`, with a per-strategy wrapper so the five strategies produce
five *different* strings — that is what makes the strategy-assignment and duplicate-detection
tests meaningful. Constructor takes optional `responses: dict` and `failures: dict` so tests
can inject an empty response, prose with no code, or a syntax error at a chosen generation
index, and an `InferenceError` for the exception path. Lives in `src/` rather than `tests/` so
the CLI can drive it too (`--mock-model`).

---

## New package — `src/python_dpo/generation/`

**`strategies.py`** — `STRATEGIES: tuple[str, ...]` in spec order (`normal`,
`straightforward`, `edge_case_focused`, `alternative`, `optimized`) and
`STRATEGY_INSTRUCTIONS: dict[str, str]` with the §13 texts verbatim.
`resolve_strategies(configured, count, override=None) -> tuple[str, ...]` returns exactly
`count` strategy names — one per strategy for the default 5, cycling when `count` exceeds the
list, and all-one-strategy when `--strategy` is given. Unknown names raise.

**`prompt_builder.py`** — `PROMPT_VERSION = "v1"` and
`build_prompt(problem: Problem, strategy: str) -> str`. Pure and deterministic (§14): same
problem + strategy → byte-identical string, no timestamps, no randomness. Renders the §14
template with `problem.prompt`, `problem.signature`, the strategy instruction, and the fixed
requirements block (implement the function, follow the signature, return only the
implementation, no `eval`/`exec`, no network, no file I/O). `PROMPT_VERSION` is stamped on
every candidate; the module docstring states the §15 rule — **changing the template requires
bumping the version**.

**`code_extractor.py`** — `ExtractionResult(code, extracted, source_format, error)` frozen
dataclass and `extract_code(raw_output) -> ExtractionResult`. Tried in order:

| Order | `source_format` | Rule |
|---|---|---|
| 1 | `python_fence` | First ```` ```python ```` / ```` ```py ```` fence |
| 2 | `generic_fence` | First bare ```` ``` ```` fence whose body contains a `def`/`class`/`import` line |
| 3 | `plain` | No fences, but the text has a top-level `def`/`class`/`import`/`async def` line |
| — | `unknown` | Nothing matched, or output empty/whitespace → `extracted=False`, `error="No Python code detected"` |

Whitespace around the block is stripped; internal formatting is preserved byte-for-byte (§18).
The extractor **never repairs** code and never invents a successful result (§18, §44) —
unterminated fences fall through to `unknown` rather than being guessed at.

**`validation.py`** — syntax and structure only, no execution.

- `check_syntax(code) -> SyntaxCheck(valid, error_message)` — wraps `ast.parse` and captures
  the `SyntaxError` message with line/offset. This is the *only* thing that ever looks at
  generated code (§19).
- `check_function_name(code, entry_point) -> bool` — walks the parsed AST for a `FunctionDef`
  or `AsyncFunctionDef` named `entry_point` (async matters — p010 is an async problem).
  Returns `False` on unparseable code rather than raising. Reuses `Problem.entry_point` from
  Stage 2 rather than re-parsing `signature` (§43).

**`generator.py`** — `CandidateGenerator`, the orchestrator, taking `(model_client,
repository, generation_config, prompt_version)` — it depends on `ModelClient`, never on Qwen.
`generate_for_problem(problem, count, strategies, run_id, force)` per candidate index:

1. Skip if `(problem_id, index)` already persisted and not `force` (§28).
2. Build the prompt; call the client.
3. Empty output → failure record, no candidate (§26).
4. Extract; extraction failure → failure record, **no candidate** (§18, §26).
5. `check_syntax` + `check_function_name` → recorded as fields, **not** as failures. A
   syntax-invalid candidate is still a real candidate (§19, §42, §49).
6. Duplicate check against already-known code for the same `problem_id` → `duplicate_of`
   (§41); the duplicate is kept, never deleted.
7. Append the record immediately, then log (§24).

`ModelError` from any single generation is caught, recorded, and the loop continues (§26).
Returns a `GenerationSummary` (generated / skipped / failed / duplicates) for the CLI.

---

## New package — `src/python_dpo/candidates/`

**`models.py`** — mirrors `problems/models.py`: frozen dataclasses, `__post_init__`
validation raising `CandidateError`, explicit `to_dict()`/`from_dict()` (not `asdict`, so
loading validates).

`Candidate` fields — every §20 field plus what §45 traceability actually requires:

```
candidate_id      "p001_c001"  — deterministic, {problem_id}_c{index:03d} (§21)
problem_id        run_id            generation_index   strategy
model             model_revision    provider           prompt_version
prompt            raw_output        code (str | None)
extraction_format syntax_valid      syntax_error       function_name_valid
duplicate_of      generation_config (dict)             created_at (UTC ISO-8601)
```

Both `raw_output` **and** `code` are stored — the raw text is the debugging record for
extraction failures (§25). `prompt` is stored so a candidate can be reproduced without
re-deriving it from a template that may since have changed version.

`GenerationFailure` — exactly the §27 fields: `run_id`, `problem_id`, `generation_index`,
`strategy`, `error_type`, `error_message`, `timestamp`. `error_type` is a closed set:
`model_load`, `inference`, `empty_output`, `code_extraction`.

**`repository.py`** — `CandidateRepository`, constructed from `config.paths.candidates`.

- `append(candidate)` — opens in `"a"` mode, writes one `json.dumps(..., sort_keys=True)`
  line, flushes. Written the moment the candidate exists, so a killed run is still resumable
  (§24).
- `append_failure(failure)` — same, to `generation_failures.jsonl` (§27).
- `load_all()` — line-by-line with validation and line numbers in error messages; **never
  silently skips a bad line**, exactly like `problems/storage.py`'s loader.
- `existing_keys() -> set[tuple[str, int]]` — the `(problem_id, generation_index)` resume
  index (§28).
- `code_index() -> dict[str, list[tuple[str, str]]]` — per-problem code seen so far, for
  duplicate detection.
- `latest_by_candidate_id()` — collapses multi-run duplicates to the newest `run_id`, so
  Stage 4 has one obvious way to read the file.
- `existing_run_ids()` — used to uniquify a new `run_id`.

`run_id` is `datetime.now(UTC).strftime("%Y%m%d_%H%M%S")`, suffixed `_2`, `_3`, … if that
second already exists in the file. Deterministic to read, unique in practice, not a UUID (§21
forbids UUIDs only for the *candidate* id).

---

## Files to modify

**`src/python_dpo/config.py`** — two new typed sections, same strict-validation style as
`Paths`.

- `ModelConfig` — `provider` (`transformers` | `mock`), `name`, `revision`, `device`, `dtype`,
  `trust_remote_code`, `quantization` (parsed and carried, but only `null` is accepted in
  Stage 3 — §9's forward-compatible slot without the logic).
- `generation` → builds a `models.base.GenerationConfig` plus `candidates_per_problem`.
- `generation_strategies` → validated against `STRATEGIES`.
- Required keys: `model.name`, `model.provider`. The rest default (`revision: null`,
  `device: auto`, `dtype: auto`, `trust_remote_code: false`). No import cycle: `config.py`
  imports from `models/base.py`, which imports nothing from the package.

**`config.yaml`** — add the §37 `model`, `generation`, and `generation_strategies` blocks with
`name: "Qwen/Qwen2.5-Coder-3B-Instruct"`. **No tokens, ever** (§37/§38).

**`src/python_dpo/cli.py`** — `generate` stops being a placeholder (drop it from
`_PLACEHOLDER_STAGES`; `evaluate`, `preferences`, `run` keep their exit-1 behavior).
`_cmd_generate(args, config)` supports:

| Flag | Behavior |
|---|---|
| `--problem-id P` | Generate only for `P`; unknown id → error, exit 1 (§29) |
| `--limit N` | First `N` problems in dataset order (§29) |
| `--num-candidates N` | Overrides `generation.candidates_per_problem` for this run only (§30) |
| `--strategy S` | Repeatable; overrides the configured list (§31) |
| `--force` | New `run_id`, regenerate regardless of what exists (§28) |
| `--dry-run` | Build and print prompts to stdout; **no model load, no inference, no writes** (§36) |
| `--mock-model` | Use `MockModelClient` — an offline end-to-end exercise of the CLI |

Order of operations matters for `--dry-run`: problems load and prompts build *before* any
client is constructed, so the Qwen path is never even reached. Prompts print to stdout
(user-facing output), diagnostics to the log stream — the Stage 2 precedent.

A `_build_model_client(config, args)` factory keeps client selection in one place.

**`pyproject.toml`** — new optional extra:

```toml
model = ["torch>=2.2", "transformers>=4.44", "accelerate>=0.30"]
```

Core `dependencies` stay `PyYAML` only.

**`src/python_dpo/__init__.py`** — `__version__` → `0.3.0`.

**`CLAUDE.md`** — extend Security: generated candidates are inspected with `ast.parse` only
and are never executed, never `exec`'d, and never passed to `InProcessReferenceExecutor`.

**Docs** — root `README.md` gets a Stage 3 section (model config, the five strategies, the
`generate` flags, candidate schema, the `[model]` extra, the `HF_TOKEN` env-var note *without
a value*, and the reproducibility caveat). New per-folder `README.md` in `models/`,
`generation/`, and `candidates/`; refresh `src/python_dpo/README.md`, `tests/README.md`,
`data/README.md`, `scripts/README.md` — the convention every folder already follows.

**`scripts/smoke_real_model.sh`** — the §48/§34 manual procedure: one problem, one candidate,
real model, with a banner stating it downloads weights and is never run by pytest.

---

## Tests

All offline, CPU-only, no skips. `MockModelClient` everywhere.

**`tests/test_models.py`** — `GenerationConfig` validation and `to_dict`; `MockModelClient`
determinism (same prompt+strategy → identical output) and per-strategy variation;
`isinstance(client, ModelClient)` for both clients; `QwenModelClient(...)` **constructs
without importing torch**; a monkeypatched missing-`transformers` import surfaces
`ModelLoadError` with an install hint; device/dtype resolution is unit-tested through a small
pure helper so no CUDA is required.

**`tests/test_generation.py`** — prompt builder (problem text, signature, strategy
instruction, and output rules all present; identical across two calls; differs per strategy;
carries `PROMPT_VERSION`); `resolve_strategies` for default-5, `--strategy` override, count >
5 cycling, unknown name; extractor across all six §46 cases (plain, python fence, generic
fence, explanatory prefix, empty, malformed) plus a prose-only output and an unterminated
fence; `check_syntax` valid/invalid; `check_function_name` correct / wrong name / no function
at all / async function / unparseable input.

**`tests/test_candidates.py`** — `Candidate` and `GenerationFailure` valid construction and
each validation failure; JSONL write → `load_all` round-trip equality; malformed line rejected
with its line number; `existing_keys`; duplicate detection sets `duplicate_of` to the earliest
match and keeps both records; `latest_by_candidate_id` prefers the newer run; append is
durable mid-run (records readable before the run finishes).

**`tests/test_generation_pipeline.py`** (the §47 integration test) — one problem, five
candidates, `tmp_path` repository: five records, correct `problem_id`, unique candidate ids,
all five strategies represented, `raw_output` and `code` both preserved, syntax validated,
`generation_config` embedded, all persisted. Plus: resume (second run generates 0, file
unchanged); `--force` (new `run_id`, 10 records total, both runs intact); an injected empty
response and an injected prose-only response each produce a failure record and **no**
candidate; an injected `InferenceError` is recorded and the remaining candidates still
generate; an injected syntax error produces a candidate with `syntax_valid=False` and **no**
failure record.

**`tests/test_project.py`** — narrow the placeholder parametrization to the three remaining
stages; assert `generate` is no longer a placeholder; assert every §35 flag parses; a
subprocess `generate --problem-id p001 --dry-run` exits 0, prints a prompt, and writes nothing
to `data/candidates/`.

**`tests/test_no_heavy_imports.py`** — a subprocess imports `python_dpo`,
`python_dpo.models`, and `python_dpo.generation`, then asserts `torch` and `transformers` are
absent from `sys.modules`. This is §7's lazy-loading requirement made enforceable rather than
aspirational.

---

## Verification

```bash
source .venv/bin/activate
pytest -q                                          # all pass, 0 skipped, no downloads

# prompts, with no model loaded
python -m python_dpo generate --problem-id p001 --dry-run
git status --short data/candidates/                # empty — dry run wrote nothing

# full offline pipeline exercise
python -m python_dpo generate --problem-id p001 --num-candidates 5 --mock-model
wc -l data/candidates/candidates.jsonl             # -> 5

# real model (downloads ~6 GB on first run)
pip install -e '.[model]'
python -m python_dpo generate --problem-id p001 --num-candidates 1
python -m python_dpo generate --problem-id p001 --num-candidates 5
python -m python_dpo generate --problem-id p001 --num-candidates 5   # resumes: generates 0
python -m python_dpo generate --problem-id p001 --num-candidates 5 --force  # new run_id
python -m python_dpo generate --limit 2 --num-candidates 5

jq -r '[.candidate_id,.run_id,.strategy,.syntax_valid,.function_name_valid]|@tsv' \
   data/candidates/candidates.jsonl
```

Then confirm scope containment:

```bash
grep -rnE "\b(exec|eval)\(" src/          # only the documented InProcessReferenceExecutor hit
grep -rniE "docker|pytest|trl|peft|lora|dpo" src/python_dpo/{models,generation,candidates}/
grep -rn "Qwen/" src/                     # no hits — model id lives in config.yaml only
grep -rniE "hf_token|hugging.*token" src/ config.yaml data/  # no values anywhere
git status                                # no .venv, no weights, no secrets
```

Finally produce the §54 report — model abstraction, model used, loading and generation
config, strategies, prompt format, extraction behavior, candidate schema, persistence, resume
and failure handling, test results, smoke-test result, files and dependencies, deviations.
**Stop there — do not start Stage 4.**

---

## Spec amendments (folded into the spec, v1.0 → v1.1)

Three internal contradictions were resolved **in the spec itself** rather than carried as
deviations. `.claude/specs/03_qwen_candidate_generator.md` now has a Revision History section
recording them, and the acceptance criteria in §51 test for them.

- **§19.1 / §26.1 — syntax errors are candidates, not failures.** §26 listed "syntax error"
  among generation failures while §19/§42/§49 required persisting a candidate with
  `syntax_valid: false`; obeying both would record one generation twice. Resolved in favour of
  §19: extraction failure → failure record with no candidate (there is no code to store);
  syntax failure → candidate with `syntax_valid=false` and no failure record (there is code,
  and it is stored as generated). Preserves §44 and §41.
- **§26.2 — model-load failure aborts the run** with a single `model_load` failure and a
  non-zero exit, instead of emitting one identical failure per candidate. Already-persisted
  candidates are retained and the run stays resumable.
- **§21.1 / §28.1 — `candidate_id` is unique per run, not per file.** The only way to satisfy
  §21's deterministic IDs, §52's "a new generation run is created", and §25/§41's
  no-discarding rule simultaneously. `(run_id, candidate_id)` is the file-wide key; §52's
  verification step now states the expected ten-record outcome explicitly.

§27's `error_type` was also narrowed to a closed set (`model_load`, `tokenizer`, `inference`,
`timeout`, `empty_output`, `code_extraction`) so failures are countable across runs.

## Remaining deviations & decisions to flag in the report

- **`--mock-model` is a CLI flag beyond §35's list.** It makes the whole pipeline runnable
  offline for verification, which the spec's own flag list can't otherwise do.
- **Reproducibility is seeded, not bit-exact** (§23). `set_seed` fixes Python/NumPy/torch RNGs,
  but CUDA sampling kernels and batching make bit-for-bit output non-guaranteed. Documented
  in the README rather than claimed.
- **Extra candidate fields** beyond §20: `run_id`, `provider`, `extraction_format`,
  `syntax_error`, `function_name_valid`, `duplicate_of` — required by §22, §18, §41, §43.
- **`data/candidates/*.jsonl` stays tracked**, consistent with `problems.jsonl` and CLAUDE.md's
  Reproducibility rule. Worth revisiting if real runs grow the file substantially.
- **The default model is a choice with downstream consequences**: 3B-Instruct will solve the
  easier problems nearly every time, so p001/p006/p008 may yield five near-identical correct
  candidates and no usable preference pair. That is a Stage 5 problem, but it is worth
  watching in the first real run.
