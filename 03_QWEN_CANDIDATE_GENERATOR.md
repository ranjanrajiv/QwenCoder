# Stage 3 Implementation Details — Qwen Candidate Generator

This document explains how `src/python_dpo/models/`, `src/python_dpo/generation/`, and
`src/python_dpo/candidates/` implement the layer specified in
`.claude/specs/03_qwen_candidate_generator.md`. For usage instructions, see the
"Stage 3 — Qwen Candidate Generator" section of the root `README.md`; this file is about
*how* it's built, not how to run it.

## Goal

Stage 3 turns a validated `Problem` (Stage 2's ground truth) into candidate Python
implementations, using a Qwen Coder model, and persists them with enough provenance that
any candidate can be traced back to the exact prompt, model, and generation settings that
produced it. The governing constraint (spec §3, §50) is a strict separation of concerns:

> The generator's responsibility is to generate and persist candidate programs. It must
> not decide whether a candidate is correct.

Nothing in this stage executes generated code. Correctness is a later stage's job, decided
by running candidates against the problem's tests in a sandbox that doesn't exist yet.

## The pipeline

```
Problem → PromptBuilder → ModelClient → RawGeneration → CodeExtractor
        → Candidate → CandidateRepository → candidates.jsonl
```

Three packages implement it, each with one responsibility:

- **`models/`** — the inference seam. Defines what a model client looks like and provides
  two implementations: a real one (Qwen via Transformers) and a deterministic mock.
- **`generation/`** — the pipeline logic. Strategies, prompt construction, code
  extraction, static validation, and the orchestrator that ties them together.
- **`candidates/`** — the schema and storage. What a candidate record looks like and how
  it's written to and read from disk.

## Model abstraction — `models/base.py`

The seam everything else depends on. It imports nothing from the rest of the package —
that's deliberate, because it lets `config.py` import *from* this module without creating
a cycle back the other way.

- **`ModelClient`** (`base.py:238`) — a `runtime_checkable` `Protocol`, not an ABC. Both
  implementations satisfy it structurally, so `isinstance(client, ModelClient)` works
  without either class inheriting from anything. One method,
  `generate(prompt, generation_config) -> RawGeneration` (`base.py:253`), plus `name`,
  `revision`, and `provider` properties for candidate provenance.
- **`GenerationConfig`** (`base.py:74`) — frozen dataclass: `temperature`, `top_p`,
  `max_new_tokens`, `do_sample`, `repetition_penalty`, `seed`. `__post_init__`
  (`base.py:88`) range-checks every field and rejects `bool` where a number is expected
  (`_require_number`/`_require_int`, `base.py:54-65`) — `bool` is an `int` subclass in
  Python, so `max_new_tokens=True` would otherwise silently mean "generate one token."
  Sampling with `temperature=0` is refused (`do_sample=True` needs `temperature>0`);
  greedy decoding at `temperature=0` is allowed. `to_dict()`/`from_dict()` round-trip
  through plain dicts so the config can be embedded verbatim in every candidate record.
- **`ModelConfig`** (`base.py:133`) — `provider`, `name`, `revision`, `device`, `dtype`,
  `trust_remote_code`, `quantization`. Validates `device` against `auto`/`cpu`/`cuda`/
  `cuda:N` and `dtype` against a known set. `trust_remote_code` defaults to `False`
  (spec §8's "default to the safer value"). **`quantization` is rejected outright unless
  `None`** (`base.py:172-177`) — accepting a value that has no effect would silently
  mislead the reader; spec §9 reserves the field for a later stage without asking for the
  logic now.
- **`RawGeneration`** (`base.py:224`) — `text`, `prompt_token_count`,
  `completion_token_count`, `finish_reason`. Deliberately not a candidate: a model client
  never decides whether its own output is usable — that judgment happens downstream.
- **`ModelError` / `ModelLoadError` / `InferenceError`** (`base.py:38-52`) — the exception
  hierarchy the generator catches. The split matters operationally:
  `ModelLoadError` means *no* candidate in the run can succeed (bad weights, bad device),
  so the generator records one failure and aborts the whole run. `InferenceError` means
  *this one prompt* failed, so the generator records it and moves on to the next
  candidate. This distinction resolves the contradiction between spec §26 ("failures must
  not crash the job") and the practical fact that retrying a broken model load five times
  per problem is pointless — documented as spec amendment §26.2.

## Qwen client — `models/qwen.py`

`QwenModelClient` (`qwen.py:94`), the Transformers-backed implementation. Two rules govern
every line of this file:

**Nothing heavy is imported at module scope.** `_import_backend()` (`qwen.py:36`) does
`import torch` and `import transformers` inside a function, called only from
`_ensure_loaded()` (`qwen.py:130`), which itself only runs on the first `generate()` call.
Construction (`__init__`, `qwen.py:101`) does nothing but store the config — it doesn't
even validate that `transformers` is installed. If it's missing, the failure surfaces as
`ModelLoadError` naming the fix directly: `"... pip install -e '.[model]'"`
(`qwen.py:30, 43`), not a bare `ImportError` traceback. This whole discipline exists
because `import python_dpo` must cost milliseconds, not several GB (spec §7) —
`tests/test_no_heavy_imports.py` runs this exact scenario in a subprocess and fails the
build if it ever regresses.

**All hardware-specific behavior lives in this one file** (spec §10). Two pure functions
carry the logic so they can be unit-tested without a GPU:

- `resolve_device(requested, cuda_available)` (`qwen.py:46`) — `auto` picks CUDA when
  available, else CPU with a logged warning. An **explicit** `cuda` request on a machine
  without CUDA raises `ModelLoadError` rather than silently falling back to CPU
  (`qwen.py:52-56`) — a run that quietly takes a hundred times longer than expected is a
  worse failure mode than one that stops immediately and says why.
- `resolve_dtype(requested, device)` (`qwen.py:65`) — `auto` → `bfloat16` on CUDA (halves
  memory at negligible quality cost for a coder model), `float32` on CPU (where half
  precision is often unsupported or slower).

Loading (`_ensure_loaded`, `qwen.py:130-177`) resolves device/dtype, logs the model name
and settings, then calls `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM
.from_pretrained` with `revision` and `trust_remote_code` threaded through explicitly.
`_load_model_with_dtype` (`qwen.py:237`) tries the `dtype=` keyword first and falls back
to `torch_dtype=` on `TypeError` — Transformers renamed that argument between major
versions (the installed 5.15.0 uses `dtype`), and probing avoids pinning a narrow version
range for one keyword name.

`_render(prompt)` (`qwen.py:179`) applies the tokenizer's chat template when the tokenizer
declares one. Qwen `-Instruct` checkpoints are trained with a chat template and produce
measurably worse output without it; base checkpoints have none, and the prompt passes
through unchanged.

`generate()` (`qwen.py:196`) calls `transformers.set_seed(config.seed)` before every
generation, tokenizes the rendered prompt, runs `model.generate(**inputs,
**build_generation_kwargs(config), pad_token_id=...)` inside `torch.inference_mode()`, and
then **slices off the prompt tokens** (`qwen.py:226`) before decoding — so `raw_output`
holds the model's completion alone, not the prompt echoed back plus the completion.
`build_generation_kwargs` (`qwen.py:77`) omits `temperature`/`top_p` entirely when
`do_sample=False`, since Transformers otherwise warns that they're being ignored.

Any exception during generation is caught and re-raised as `InferenceError` with the
original type and message preserved (`qwen.py:216-217`), so the generator's failure
handling never has to know which library raised what.

## Mock client — `models/mock.py`

`MockModelClient` (`mock.py:61`), living in `src/` (not `tests/`) precisely so the CLI can
drive it too, via `generate --mock-model`. It's what makes the whole pipeline exercisable
offline, and what backs every automated test in this stage — spec §32 requires that unit
tests never need a GPU, weights, Hugging Face auth, or a network connection.

Two modes:

- **Synthesized (default).** `_synthesize(prompt)` (`mock.py:123`) hashes the prompt with
  SHA-256 — deliberately `hashlib`, not Python's built-in `hash()`, because string hashing
  is randomized per interpreter process and would make "deterministic" output actually
  vary between runs. The digest picks a numeric "variant" and one of four output wrappers
  (`_wrap_python_fence`, `_wrap_prose_fence`, `_wrap_generic_fence`, `_wrap_plain`,
  `mock.py:42-58`), so a single offline `generate` run exercises every extraction format
  the real model might produce. `entry_point_from_prompt` (`mock.py:113`) recovers the
  target function name straight from the "Required function signature:" line of the
  built prompt — keeping the mock a pure `ModelClient` that only ever sees what a real
  model sees.
- **Scripted.** `MockModelClient(script=[...])` returns each item from the list in call
  order, and *raises* any item that's an exception instance. This is how the test suite
  injects an empty response, a prose-only response, broken syntax, or a specific
  `InferenceError`/`ModelLoadError` at exactly one generation index — see
  `tests/test_generation_pipeline.py`.

`call_count` (`mock.py:89`) exposes how many times `generate` was actually invoked, which
is the mechanism that *proves* resume skips work rather than merely re-deriving the same
answer and discarding it.

## Strategies — `generation/strategies.py`

`STRATEGIES` (`strategies.py`) — the five names from spec §13, in spec order: `normal`,
`straightforward`, `edge_case_focused`, `alternative`, `optimized`. Each maps to an
instruction string in `STRATEGY_INSTRUCTIONS`, copied verbatim from the spec.

`resolve_strategies(configured, count, override)` (`strategies.py:54`) is what the CLI and
generator both call to turn "how many candidates, which strategies" into an exact
per-candidate strategy list. With the default 5 strategies and `count=5`, it's one
strategy per candidate. `count` beyond 5 **cycles** rather than erroring — asking for 10
candidates gives two passes over the five strategies. `--strategy` replaces the configured
list entirely rather than adding to it. An unknown strategy name, an empty list, or a
non-positive count all raise `StrategyError` immediately, validated eagerly
(`instruction_for` is called on every name up front, `strategies.py:70-71`) rather than
failing partway through a generation run.

These are explicitly **generation prompts, not correctness labels** (spec §13) — nothing
in the code claims the `optimized` candidate is actually more efficient than the others.
That's for the evaluator to determine, in a stage that doesn't exist yet.

## Prompt builder — `generation/prompt_builder.py`

`build_prompt(problem, strategy)` (`prompt_builder.py:46`) is a pure function: the
`_TEMPLATE` string (spec §14's exact wording) is filled in with `problem.prompt`,
`problem.signature`, and the strategy's instruction text, and nothing else touches the
output. Same problem, same strategy, same string — every time, forever. No timestamp, no
randomness, no reference to model state.

`PROMPT_VERSION = "v1"` is stamped on every candidate. The module docstring states the
rule from spec §15 directly: **changing `_TEMPLATE` — including whitespace — changes what
the model sees, and therefore what the resulting dataset means, so the version must be
bumped alongside any template edit.** Two datasets generated under different templates
must never be silently conflated.

## Code extractor — `generation/code_extractor.py`

`extract_code(raw_output)` (`code_extractor.py:57`) tries three patterns in order and
returns as soon as one matches:

1. **`python_fence`** — a ` ```python ` / ` ```py ` / ` ```python3 ` fence
   (`_PYTHON_FENCE_RE`, `code_extractor.py:23`).
2. **`generic_fence`** — a bare ` ``` ` fence, accepted only if its body contains a
   `def`/`class`/`import`/`from` line (`_looks_like_code`, `code_extractor.py:53`) — a
   fenced shell transcript or example output shouldn't be mistaken for source.
3. **`plain`** — no fences at all, but the raw text itself has such a line, and contains
   no stray ` ``` ` anywhere.

Anything else returns `ExtractionResult(code=None, extracted=False,
source_format="unknown", error="No Python code detected")`.

**This module never repairs anything** (spec §18, §44). `_normalize` (`code_extractor.py:
48`) only strips leading/trailing blank lines and trailing whitespace — internal
formatting, indentation, and blank lines in the body are preserved byte-for-byte. The
sharpest case: an **unterminated fence fails extraction rather than being guessed at**
(tested explicitly in `tests/test_generation.py::test_unterminated_fence_is_not_repaired`)
— if the model opened a ` ```python ` block and never closed it, patching in a closing
fence would mean the stored candidate is no longer literally what the model produced,
which would quietly corrupt the provenance the whole pipeline depends on.

An extraction failure produces **no candidate at all** — there's nothing to store — and
becomes a `GenerationFailure` with `error_type="code_extraction"` instead.

## Static validation — `generation/validation.py`

Two checks, both pure, and both explicitly **not** an execution of the candidate.

- `check_syntax(code)` (`validation.py:34`) wraps `ast.parse(code)` in a try/except,
  capturing the message and line number on `SyntaxError`
  (`_format_syntax_error`, `validation.py:27`), and separately handling `ValueError`
  (raised for source containing null bytes, which isn't a `SyntaxError`). Building a
  syntax tree parses the code's grammar; it does not import anything, does not run any
  module-level statement, and does not evaluate a single expression. That's precisely why
  this file is the one place in `generation/` allowed to touch untrusted code — the
  CLAUDE.md Security section names this module explicitly as the sanctioned exception
  alongside `InProcessReferenceExecutor`.
- `check_function_name(code, entry_point)` (`validation.py:48`) walks the parsed AST for a
  `FunctionDef` or `AsyncFunctionDef` matching `entry_point` — reusing the `entry_point`
  field that Stage 2's `Problem` schema already carries, rather than re-parsing the
  `signature` string. `async def` counts (p010 is an async problem), and so does a nested
  definition (a helper-wrapped implementation is a structural oddity, not a missing
  function). Unparseable code returns `False` rather than raising — the syntax failure is
  already recorded separately by `check_syntax`.

Both checks establish **structure**, never correctness. `def factorial(n): return 123`
passes both and is still wrong; nothing in this file has an opinion about that.

## The orchestrator — `generation/generator.py`

`CandidateGenerator` (`generator.py:61`) is constructed with a `ModelClient`, a
`CandidateRepository`, a `GenerationConfig`, and a `prompt_version` — never with a
concrete Qwen or mock type, so the exact same orchestration logic drives both.

`generate(problems, count, strategies, run_id, force)` (`generator.py:77`) loads the
resume index (`existing_keys()`) and the duplicate-detection index (`code_index()`) once
up front — re-reading the file per candidate would be quadratic in the number of
candidates — then iterates `problem × generation_index` and, per candidate:

1. **Resume check.** Skip if `(problem_id, index)` is already in `existing_keys()` and
   `force` is false.
2. **Build the prompt, call the client.** A `ModelLoadError` here means the whole run is
   doomed; it's recorded as a single `model_load` failure and **re-raised**, which the CLI
   catches to abort the run and return exit code 1 (spec §26.2). Any other exception is
   caught, recorded as an `inference` failure, and the loop **continues** to the next
   candidate (spec §26).
3. **Empty output check.** A response that's empty or whitespace-only is recorded as
   `empty_output` — no candidate — and the loop continues.
4. **Extraction.** A failed extraction is recorded as `code_extraction` — no candidate —
   and the loop continues.
5. **Static validation.** `check_syntax` and `check_function_name` run on the extracted
   code, and their results are recorded **as fields on the candidate**, never as a
   failure. This is the resolution to the spec's internal contradiction between §26
   (which originally listed "syntax error" among failures) and §19/§42/§49 (which require
   persisting a candidate with `syntax_valid: false`) — formalized as spec amendment
   §19.1/§26.1. The reasoning: a candidate that fails to parse is still the model's actual
   output, and it's exactly the kind of low-quality answer a later stage needs on the
   rejected side of a preference pair. Recording it as a failure *and* a candidate would
   double-count one generation; recording it only as a failure would discard real signal.
6. **Duplicate detection.** The extracted code is looked up in `code_index()`, scoped per
   problem (identical code across two *different* problems is a coincidence, not a
   duplicate). If the match is the candidate's own `candidate_id` — which happens when
   `--force` regenerates the same index and the model produces byte-identical code again —
   it is **not** flagged; that's the generation reproducing itself, not two candidates
   colliding (`generator.py:181-184; found and fixed
   during implementation testing — see "Bug found during testing" below).
7. **Persist immediately.** The candidate is appended to the repository the moment it's
   built, before moving to the next index (spec §24) — so a run killed halfway through
   still leaves every candidate generated so far on disk and resumable.

Returns a `GenerationSummary(run_id, generated, skipped, failed, duplicates)` that the CLI
logs and reports; `attempted` is a derived property (`generated + failed`).

## Candidate schema — `candidates/models.py`

**`Candidate`** (`models.py:121`) — 19 fields, matching spec §20 plus what full
traceability (spec §45) actually requires beyond the spec's literal list: `run_id`
(so multiple generations of the same problem stay distinguishable, spec §22),
`provider`, `extraction_format`, `syntax_error`, `function_name_valid`, `duplicate_of`.
`__post_init__` (`models.py:150`) enforces:

- `code` must be non-empty — **a `Candidate` exists only when code was extracted.** There
  is no code-extraction-failed-but-still-a-candidate state; that outcome is a
  `GenerationFailure` instead. This is the structural expression of spec §19.1.
- `syntax_error` must be `None` when `syntax_valid` is `True` (`models.py:180-181`) — a
  passed check can't also carry an error message.
- `candidate_id` must start with `f"{problem_id}_c"` (`models.py:183-186`) — catches a
  transposed argument at construction time rather than downstream.
- `duplicate_of`, if set, must not equal the candidate's own `candidate_id`
  (`models.py:189-191`) — the guard that made the self-duplicate bug (below) visible as a
  hard failure instead of a silently wrong record.

`build_candidate_id(problem_id, generation_index)` (`models.py:85`) produces the
zero-padded `p001_c001` shape from spec §21. `to_dict()`/`from_dict()`
(`models.py:197, 221`) are hand-written, not `dataclasses.asdict()`, so loading a
persisted record always re-runs full validation — matching the Stage 2 convention exactly.

**`GenerationFailure`** (`models.py:235`) — `run_id`, `problem_id`, `generation_index`,
`strategy`, `error_type`, `error_message`, `timestamp`. `error_type` is checked against a
**closed set**, `ERROR_TYPES` (`models.py:24`): `model_load`, `tokenizer`, `inference`,
`timeout`, `empty_output`, `code_extraction`. `syntax_error` is deliberately **absent**
from this set — its omission is enforced by validation, not just convention, so a stray
`error_type="syntax_error"` raises `CandidateError` immediately rather than silently
polluting the failure log with a category that should never appear there.

## Repository — `candidates/repository.py`

`CandidateRepository` (`repository.py:59`) owns two files in one directory:
`candidates.jsonl` and `generation_failures.jsonl`.

**Append-only, flushed per record.** `append()`/`append_failure()`
(`repository.py:73, 76`) each open the file in `"a"` mode, write one
`json.dumps(..., sort_keys=True)` line, and close — a few extra syscalls per candidate,
in exchange for a file that's fully readable and resumable even if the process is killed
mid-run (spec §24). Nothing is ever rewritten or deleted from either file.

**The file-wide key is `(run_id, candidate_id)`, not `candidate_id` alone.** `--force`
mints a new `run_id` via `new_run_id()` (`repository.py:140`) and appends a fresh set of
candidates rather than overwriting the old ones — so `p001_c001` can legitimately appear
more than once, once per run that regenerated it. This is spec amendment §21.1/§28.1,
resolving the original tension between §21's "deterministic, unique candidate IDs" and
§52's "verify that a new generation run is created" (which only makes sense if nothing
from the old run disappears). `latest_by_candidate_id()` (`repository.py:129`) collapses
to one record per candidate by keeping the last one seen in file order — safe because the
file is append-only, so a later line is by construction a later run; no timestamp
comparison is needed.

`new_run_id(now=None)` (`repository.py:140`) formats `%Y%m%d_%H%M%S` in UTC and
disambiguates against `existing_run_ids()` (which checks *both* files, so a run that
only produced failures still "took" its run_id) by appending `_2`, `_3`, … if that
second was already used.

**Read paths never silently skip a bad record.** `_read_records`
(`repository.py:32`) and the `load_all`/`load_failures` methods built on it
(`repository.py:79, 89`) raise `CandidateStoreError` naming the exact line number for a
blank line, invalid JSON, a non-object JSON value, or a record that fails
`Candidate.from_dict`/`GenerationFailure.from_dict` validation — mirroring
`problems/storage.py`'s loader exactly, and matching CLAUDE.md's Data Integrity rule.

`existing_keys()` (`repository.py:98`) is the resume index: every
`(problem_id, generation_index)` pair that has a **candidate** on disk, across all runs. A
generation that previously *failed* left no candidate behind, so it's absent from this set
— which is precisely what makes a failed generation automatically retried on the next
invocation without any special-casing.

`code_index()` (`repository.py:110`) builds `problem_id → {code: earliest candidate_id}`
for duplicate detection, using `setdefault` so the *first* candidate with a given code
string — in file order, i.e. generation order — is the one later duplicates point at.

## CLI — `cli.py`

`generate` moves out of `_PLACEHOLDER_STAGES` and gets its own subparser
(`_add_generate_parser`, `cli.py:272`) with six spec-required flags plus one addition:
`--problem-id`, `--limit`, `--num-candidates`, `--strategy` (repeatable, `choices=
STRATEGIES` so argparse itself rejects an unknown name with exit code 2), `--force`,
`--dry-run`, and `--mock-model` (not in the spec's flag list, added so the entire pipeline
is exercisable without a GPU or downloaded weights — see the Stage 3 plan's documented
deviation).

`_cmd_generate` (`cli.py:173`) sequence:

1. Load and validate the persisted problem dataset (reusing Stage 2's `load_problems`).
2. `_select_problems` (`cli.py:124`) narrows to `--problem-id` or `--limit` (mutually
   exclusive — passing both is a `ValueError`), or the full dataset if neither is given.
   An unknown `--problem-id` reports the available ids and returns exit code 1.
3. Resolve the candidate count and strategies via `resolve_strategies`.
4. **If `--dry-run`: build every prompt and print it to stdout, then return 0 —
   before a repository or model client is even constructed.** This ordering is the
   actual enforcement mechanism behind spec §36's "must NOT load Qwen": the code path
   physically cannot reach model construction on a dry run, so there's no flag or
   conditional guarding against it — the model client object simply doesn't exist yet at
   that point in the function.
5. Otherwise, construct the `CandidateRepository`, mint a `run_id`, and build the model
   client via `_build_model_client` (`cli.py:148`) — `MockModelClient` if `--mock-model`
   or `config.model.provider == "mock"`, else `QwenModelClient(config.model)`
   (construction only; loading still hasn't happened).
6. Run `CandidateGenerator.generate(...)`. A `ModelError` escaping (i.e., the run-aborting
   `ModelLoadError` case) is caught, logged, and turns into exit code 1. Otherwise the run
   summary is logged, and **individual generation failures do not fail the command** — the
   command genuinely ran and the failures are recorded, observable data (spec §26's "must
   not crash the job," carried through to the exit code).

Prompts print to stdout (user-facing output, matching the Stage 2 precedent of the
`problems validate` summary), while every step-by-step log line goes to stderr through the
logger.

## Configuration — `config.py`, `config.yaml`

`Config` gained two typed sections. `_parse_model` (`config.py`) builds a `ModelConfig`
from the `model:` YAML block via `ModelConfig.from_mapping`, translating any
`ModelError` into `ConfigError` so every configuration failure — Stage 1's paths, Stage
2's nothing, or Stage 3's model settings — surfaces through the same exception type.
`_parse_generation` builds a `GenerationSettings` (`candidates_per_problem` plus a
`GenerationConfig` plus the resolved `generation_strategies` list, defaulting to
`STRATEGIES` if the key is absent), validating every strategy name against
`instruction_for` so a typo in `config.yaml` is caught at load time rather than at the
first generation attempt.

`config.yaml` gained `model`, `generation`, and `generation_strategies` blocks (spec §37),
with `model.name: "Qwen/Qwen2.5-Coder-3B-Instruct"` — chosen because it fits the
development machine's 12 GB RTX 3060 comfortably in bfloat16 (~6 GB) without the
quantization work spec §9 explicitly defers. **No credential of any kind appears in this
file** (spec §38) — `HF_TOKEN`, if a gated model is ever configured, is read from the
environment only, and is documented (never valued) in the README and the smoke-test
script.

## Real-model verification

With `pip install -e '.[model]'` (torch 2.13.0+cu130, transformers 5.15.0) on the RTX
3060:

- One candidate for `p001` loaded the model, generated, extracted, and validated
  successfully.
- Resume against an existing candidate correctly skipped it without invoking the model.
- `--force` correctly minted a new `run_id` and appended five new candidates while
  leaving the first run's five untouched.
- A full run across all ten problems (`python -m python_dpo generate`) produced **50
  candidates in 2m14s with zero generation failures** — 50/50 syntactically valid, 50/50
  correctly named.

## Bug found during testing

Two of the mock-driven pipeline tests failed on the first run, both from the same root
cause: with `--force`, a regenerated candidate keeps its deterministic `candidate_id`
(`p001_c001` stays `p001_c001` in the new run), and when the model produced byte-identical
code the duplicate check found that exact `candidate_id` already in the code index and
tried to set `duplicate_of` to itself — which `Candidate.__post_init__`'s own guard
correctly rejected as invalid. Fixed in `generator.py` by treating a self-match as "not a
duplicate" (the generation reproduced itself, which isn't the same claim as "two different
candidates collided"), and corrected the test that had encoded the wrong expectation to
instead cover the realistic case: a resumed run producing code identical to a candidate an
earlier run already stored.

## Tests

**`tests/test_models.py`** — `GenerationConfig`/`ModelConfig` validation (valid defaults,
every rejected combination, `bool`-as-int rejection, round-trip); `MockModelClient`
determinism, prompt-dependent variation, scripted sequencing and exceptions;
`QwenModelClient` protocol conformance *without loading anything*, provider mismatch
rejection, and a simulated missing-backend `ImportError` (via `monkeypatch.setitem
(sys.modules, "torch", None)`) producing a `ModelLoadError` naming the `[model]` extra;
`resolve_device`/`resolve_dtype`/`build_generation_kwargs` as pure, GPU-free unit tests.

**`tests/test_generation.py`** — all five strategies present with distinct instructions;
`resolve_strategies` for the default count, override, cycling, and every rejection case;
prompt content, determinism, per-strategy variation, and version; the extractor across
every spec §46 case (plain, python fence, generic fence, explanatory prefix, empty,
malformed) plus the unterminated-fence-is-not-repaired case and extraction succeeding on
syntactically broken code (extraction and parsing are different concerns); both static
validators across valid/invalid/async/nested/unparseable inputs.

**`tests/test_candidates.py`** — schema round-trip, the syntax-invalid-candidate-is-still-
valid case, every individual validation rejection, the closed `error_type` set explicitly
excluding `syntax_error`; repository append/load round-trip, mid-run durability,
`existing_keys` resume semantics (including that a failed generation doesn't block a
retry), `code_index` duplicate scoping, `latest_by_candidate_id`, run-id disambiguation
within the same second (including a failure-only run), and malformed-line rejection with
line numbers.

**`tests/test_generation_pipeline.py`** — the spec §47 integration test (one problem, five
candidates, every invariant checked) plus resume (proving via `call_count` that the model
is never re-invoked), `--force` (new run appended, old run byte-identical), retry of a
previously failed generation, every failure path (empty response, unextractable output,
inference error continuing the run, model-load error aborting it), the syntax-error-
produces-a-candidate-not-a-failure case, wrong-function-name-is-recorded-not-rejected, and
duplicate detection including the self-duplicate fix.

**`tests/test_no_heavy_imports.py`** — a subprocess imports every `python_dpo` module and
asserts `torch`/`transformers`/`accelerate` never land in `sys.modules`. This test only
means something once the `[model]` extra is actually installed (otherwise the imports
would fail anyway), which is exactly when the lazy-loading rule could silently regress.

**`tests/test_project.py`** — `generate` removed from the placeholder parametrization; all
seven CLI flags parse with the right types and defaults; an unknown `--strategy` is
rejected by argparse itself (exit 2); `--dry-run` prints a real prompt and provably writes
nothing to `data/candidates/`; an unknown `--problem-id` reports the id and exits 1.

**Result:** 209 tests pass, 0 skipped, running fully offline and CPU-only in about 1.3
seconds. The real-model run produced 50 candidates with zero failures, and the only
`exec(` call anywhere in `src/` remains the single documented line in
`problems/executor.py` — `generation/validation.py` touches untrusted code with
`ast.parse` alone.

## Known issue carried into Stage 4 planning

The real-model run surfaced a diversity problem the spec doesn't address: of 50 candidates
across 10 problems, only **19 are textually distinct**. Four problems (p001, p002, p006,
p009) produced the *same* implementation across all five strategies — the 3B model finds
them easy enough that `normal`, `edge_case_focused`, and `optimized` all converge on
identical code. Since a preference pair needs a chosen and a rejected candidate that
actually differ, those four problems currently have zero usable pairs. This is expected
behavior for this stage — the generator's job is only to record what the model produced —
but it's a real constraint on what Stage 5 (preference-pair construction) will have to
work with, and is worth revisiting via temperature, candidate count, or dataset difficulty
before that stage is designed.
