# python-dpo

A preference-data generation pipeline for DPO (Direct Preference Optimization)
fine-tuning of a Qwen Coder model on Python programming tasks.

**Current status: Stage 7 — Candidate Ranking.** The foundation (packaging, CLI,
logging, typed configuration), the ground-truth layer (10 curated Python problems with
trusted reference solutions and executable tests), candidate generation with a Qwen Coder
model, a reliable per-run artifact store (manifests, SHA-256 hashes, atomic persistence,
resume, retry, integrity validation), an isolated Docker execution sandbox, a
pytest-based candidate test executor, and deterministic candidate ranking are in place.
No preference-pair generation or training code has been implemented yet — and
**generated code, and the tests generated for it, are executed only inside the sandbox,
never on the host**; ranking itself never executes code or calls an LLM at all.

## Planned pipeline

```
Python Problems
      ↓
Qwen Candidate Generation
      ↓
Candidate Persistence
      ↓
Docker Sandbox
      ↓
pytest Evaluation
      ↓
Candidate Ranking
      ↓
Preference Pair Generation
      ↓
DPO Dataset
      ↓
QLoRA + DPO Training
```

The first seven stages are implemented. Everything from preference-pair generation
onward is still a placeholder that documents the intended shape of the pipeline.

## Roadmap

The full build is specified as 12 stages (`.claude/specs/`); each stage's own spec
states its position, e.g. "Stage 3 of 12". Seven have been specified and implemented so
far:

| Stage | Delivers | Status |
|-------|----------|--------|
| 1 — Project Skeleton | Installable package, placeholder CLI, logging, typed config | Done |
| 2 — Problem Dataset | 10 curated problems with reference solutions and tests (ground truth) | Done |
| 3 — Qwen Candidate Generator | Model abstraction, 5 strategies, code extraction, candidate persistence | Done |
| 4 — Candidate Persistence | Per-run directories, manifests, SHA-256 hashes, atomic writes, resume, retry, integrity validation | Done |
| 5 — Isolated Docker Sandbox | Locked-down container execution, resource limits, structured results, security test suite | Done |
| 6 — Candidate Test Executor | Deterministic pytest-suite generation, sandboxed evaluation, per-test evidence | Done |
| 7 — Candidate Ranking | Correctness classification, scoring, deterministic per-problem ranking, pairwise comparison | Done |
| 8–12 — Preference pairs → DPO training | Preference-pair generation, QLoRA + DPO training | Not started |

Stages 8–12 aren't specified yet, so the table above intentionally doesn't assign them
individual names — the pipeline diagram lists the phases in order, but the exact stage
boundaries will be set when each spec is written. Nothing in that range is implemented;
see `CLAUDE.md`'s Scope Control rule.

## Repository layout

```
.
├── src/python_dpo/       # the installable package — see its README for file details
│   ├── cli.py             # argparse CLI: problems, generate, runs, candidates, sandbox, evaluate, rank
│   ├── config.py          # typed config.yaml loader
│   ├── atomic_io.py       # Stage 4: durable JSON/JSONL primitives
│   ├── logging_config.py  # stderr logging setup
│   ├── problems/          # Stage 2: schema, catalog, storage, validation
│   ├── models/            # Stage 3: ModelClient protocol, Qwen client, mock client
│   ├── generation/        # Stage 3: strategies, prompts, extraction, orchestration
│   ├── candidates/        # Stage 3/4: candidate schema and run-scoped repository
│   ├── runs/              # Stage 4: run manifests, statistics, migration, validation
│   ├── sandbox/           # Stage 5: isolated Docker execution of untrusted code
│   ├── evaluation/        # Stage 6: pytest-suite generation, sandboxed evaluation
│   └── ranking/           # Stage 7: correctness classification, scoring, ranking
├── tests/                 # pytest suite — see tests/README.md
├── data/                  # pipeline artifacts (tracked; see data/README.md)
│   ├── problems/problems.jsonl     # the Stage 2 dataset
│   ├── candidates/candidates.jsonl # legacy Stage 3 flat file (read-only; see migrate)
│   ├── candidates/runs/            # Stage 4: one directory per generation run
│   ├── evaluations/runs/           # Stage 6: one directory per evaluation run
│   └── rankings/runs/              # Stage 7: one directory per ranking run
├── docker/evaluator/       # Stage 6: the pytest-preinstalled evaluation image
├── docs/                  # sandbox-security.md — threat model and isolation boundaries
├── examples/              # hello.py — a harmless file for exercising the sandbox
├── scripts/               # operational scripts (real-model smoke test)
├── config.yaml            # project name, data paths, logging, model, generation, sandbox, evaluation
└── CLAUDE.md              # engineering rules for this project
```

Every directory that holds real content has its own `README.md` with a file-by-file
breakdown — start there for implementation details this file doesn't repeat.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The inference backend is a **separate, optional extra**, so the core install and the whole
test suite stay lightweight and offline. Install it only when you want to run a real
model:

```bash
pip install -e ".[model]"     # torch, transformers, accelerate (~2.5 GB of wheels)
```

If a gated or private model is configured, export `HF_TOKEN` in your shell. It is read
from the environment and must never be written into `config.yaml`, source code, datasets,
logs, or this README.

## Testing

```bash
pytest -q                  # offline, Docker-free, zero skips
pytest -q -m integration   # Docker: sandbox security + candidate evaluation (needs a daemon + images)
```

Integration tests are deselected by default so the standard run needs no Docker and reports
no skips. They are not optional extras — they are where the sandbox's security guarantees,
and the candidate test executor's classification behavior, are demonstrated against a real
daemon.

## CLI

```bash
python -m python_dpo --help
python -m python_dpo --version
```

Implemented:

```bash
python -m python_dpo problems build      # validate the catalog and write problems.jsonl
python -m python_dpo problems validate   # re-validate the persisted dataset
python -m python_dpo generate            # generate candidates into a new run (see Stage 3/4)
python -m python_dpo generate --resume RUN_ID   # resume an interrupted run

python -m python_dpo runs list                  # all runs, newest first
python -m python_dpo runs show RUN_ID           # manifest + statistics
python -m python_dpo runs validate RUN_ID       # integrity check (see Stage 4)

python -m python_dpo candidates list RUN_ID     # candidates in a run
python -m python_dpo candidates show RUN_ID CANDIDATE_ID
python -m python_dpo candidates stats RUN_ID
python -m python_dpo candidates migrate         # upgrade the legacy flat file into runs/

python -m python_dpo sandbox health              # verify Docker end to end (see Stage 5)
python -m python_dpo sandbox run --file FILE     # run a file inside the sandbox

python -m python_dpo evaluate candidate --run-id RUN_ID --candidate-id CANDIDATE_ID
python -m python_dpo evaluate run --run-id RUN_ID       # resumes by default (see Stage 6)
python -m python_dpo evaluate run --run-id RUN_ID --force  # start a fresh evaluation run

python -m python_dpo evaluations list EVAL_ID    # results in an evaluation run
python -m python_dpo evaluations show EVAL_ID CANDIDATE_ID
python -m python_dpo evaluations stats EVAL_ID

python -m python_dpo rank run --evaluation-run-id EVAL_ID   # always a new ranking run (see Stage 7)
python -m python_dpo rank run --evaluation-run-id EVAL_ID --resume RANK_ID  # continue one

python -m python_dpo rankings list RANK_ID       # per-problem summary
python -m python_dpo rankings show RANK_ID PROBLEM_ID   # the rank table for one problem
python -m python_dpo rankings stats RANK_ID
python -m python_dpo rankings validate RANK_ID
```

The remaining subcommands exist as **placeholders only**. Each logs a "not implemented
yet" message and exits with status `1` — none of them do real work:

- `python -m python_dpo preferences`
- `python -m python_dpo run`

## Configuration

Runtime settings live in `config.yaml` at the project root: project name, data directory
paths, logging level, the `model` / `generation` / `generation_strategies` sections, the
Stage 5 `sandbox` section (image, resource limits, isolation toggles), and the Stage 6
`evaluation` section (evaluator image, timeout, startup grace, auto-pull — every
isolation setting itself is inherited from `sandbox` at run time, never re-specified).
Stage 7 ranking has no configuration section of its own — v1 has no tunable scoring
parameters. See `src/python_dpo/config.py` for how it's loaded and validated.

The model identifier lives in configuration and never in source code, so swapping models
requires no code change. **No credentials belong in this file.**

## Stage 2 — Problem Dataset

### Purpose

The problem dataset is the project's **ground-truth layer**. Every later stage joins on
its problem ids, and candidate solutions are ultimately judged against its tests. A
problem is not considered valid unless its own reference solution passes all of its
tests, so the dataset is self-verifying.

**At a glance:** 10 problems · 10/10 categories covered · 5 easy / 4 medium / 1 hard ·
74 reference tests, all passing.

### Problem schema

Each record carries:

| Field | Meaning |
|-------|---------|
| `id` | Stable, human-readable identifier (`p001`…`p010`) used to join across stages |
| `prompt` | The task as a candidate model will see it, including all disambiguating rules |
| `signature` | The expected function declaration, e.g. `def sum_even(numbers):` |
| `entry_point` | The function name to call — how an evaluator invokes a solution |
| `category` / `difficulty` | Metadata (see below) |
| `reference_solution` | Trusted, manually authored implementation, stored as source text |
| `tests` | Executable test cases (at least 5 per problem) |
| `description`, `tags`, `source`, `metadata` | Optional extras; `source` is `manual` |
| `dataset_version` | Dataset format/content version, independent of the package version |

### Categories and difficulty

Ten categories, one problem each: `lists`, `dictionaries`, `strings`, `sets`, `sorting`,
`recursion`, `edge_cases`, `exceptions`, `generators`, `async`. Difficulty is `easy`,
`medium`, or `hard`, distributed 5 / 4 / 1.

### Reference solutions

Reference solutions are the **correctness oracle, not the preferred DPO answer**. They
are authored by hand as real Python functions in `src/python_dpo/problems/references.py`
and copied into the dataset via `inspect.getsource()`, so the stored text is always the
code that actually runs. They are deterministic and perform no I/O, network access, or
environment lookups.

Because they are trusted, reviewed code, they may be executed in-process during
validation. That happens behind the `ReferenceExecutor` seam, so Stage 3 can substitute a
sandboxed executor for untrusted model output without touching the validator.

### Test cases

Test cases are structured rather than serialized Python: `input` is a mapping of keyword
arguments, and a case asserts either a return value (`expected`) or an exception type
name (`expected_exception`). All values are JSON-native so the dataset round-trips
exactly.

```json
{"id": "t001", "input": {"numbers": [1, 2, 3, 4]}, "expected": 6, "expected_exception": null}
```

### Dataset location

`data/problems/problems.jsonl` — one JSON object per line, UTF-8, keys sorted so a
rebuild of an unchanged catalog is byte-identical.

### Validation

`problems build` refuses to write anything unless the whole dataset validates;
`problems validate` is strictly read-only and never modifies the dataset.

```bash
python -m python_dpo problems build
python -m python_dpo problems validate
```

`validate` prints a summary and exits non-zero if anything fails:

```
Problems:              10
Valid:                 10
Invalid:                0

Categories:
  lists:                1
  ...

Reference tests:
  Total:               74
  Passed:              74
  Failed:               0

Dataset validation: PASS
```

## Stage 3 — Qwen Candidate Generator

### Purpose

Generate multiple independent Python implementations per problem with a Qwen Coder model,
and persist them with full provenance. The generator answers exactly one question —
**"what code did the model generate?"** — and deliberately not "does it work?", which
belongs to a later stage.

Diversity is the point. DPO needs both better and worse answers for the same problem;
five identical candidates carry no preference signal.

**At a glance:** 10 problems × 5 candidates = 50 candidates · 50/50 syntactically valid ·
50/50 define the expected function · 0 generation failures · 19 distinct implementations.

### Model configuration

```yaml
model:
  provider: transformers        # or `mock` for offline runs
  name: "Qwen/Qwen2.5-Coder-3B-Instruct"
  revision: null
  device: auto                  # auto → CUDA when available, else CPU
  dtype: auto                   # auto → bfloat16 on CUDA, float32 on CPU
  trust_remote_code: false
  quantization: null            # reserved; must be null in Stage 3
```

Qwen2.5-Coder-3B-Instruct occupies roughly 6 GB in bfloat16 and runs comfortably on a
12 GB card without quantization. The 7B checkpoint would need 4-bit weights, which this
stage deliberately does not implement.

The model is **loaded lazily** — `import python_dpo` loads no weights, and neither does
`--dry-run`. Loading begins on the first actual generation.

`trust_remote_code` defaults to `false` and is threaded explicitly into both
`from_pretrained` calls. Qwen2.5-Coder does not need it.

### Generation strategies

Five strategies, one candidate each at the default count:

| Strategy | Instruction |
|---|---|
| `normal` | A clear, correct implementation |
| `straightforward` | Simple and easy to understand |
| `edge_case_focused` | Empty inputs, boundaries, duplicates |
| `alternative` | A different reasonable algorithm |
| `optimized` | Efficient for the problem's constraints |

These are **generation prompts, not correctness labels** — nothing claims the `optimized`
candidate is actually faster. Later evaluation decides that.

### Prompt format

Prompts are deterministic given a problem and strategy, and carry a `prompt_version`
(currently `v1`) recorded on every candidate. Changing the template requires bumping the
version, so datasets built under different templates stay distinguishable.

Inspect them without loading a model:

```bash
python -m python_dpo generate --problem-id p001 --dry-run
```

### Code extraction

Models return fenced markdown, prose preambles, or bare code, so extraction handles
` ```python ` fences, generic fences, plain code, and explanatory prefixes, recording
which form it found. It **never repairs code**: an unterminated fence fails extraction
rather than being guessed at, because a patched candidate would no longer be what the
model produced.

### Validation

Static only — **no generated code is ever executed**. `ast.parse` establishes syntax
validity and whether the expected function is defined. Both are recorded properties, not
verdicts: `def factorial(n): return 123` passes both and is still wrong.

### Candidate records

As of Stage 4, candidates are written to a per-run directory rather than one flat file —
see [Stage 4](#stage-4--candidate-persistence-runs-and-reproducibility) below. Both
`raw_output` and the extracted `code` are kept — the raw text is the only way to debug an
extraction that went wrong.

```json
{
  "candidate_id": "p001_c001", "problem_id": "p001", "run_id": "run_20260817_133700_a81f",
  "generation_index": 1, "strategy": "normal",
  "model": "Qwen/Qwen2.5-Coder-3B-Instruct", "provider": "transformers",
  "prompt_version": "v1", "extraction_format": "python_fence",
  "syntax_valid": true, "function_name_valid": true, "duplicate_of": null,
  "schema_version": "2.0", "code_sha256": "…", "attempt": 1
}
```

Failures that produced *no* candidate go to `failures.jsonl` with a closed-set
`error_type`. Code that fails to parse is **not** a failure — it is stored as a candidate
with `syntax_valid: false`, because it is the model's real output and precisely what a
later stage needs on the rejected side of a preference pair.

### CLI options

| Flag | Effect |
|---|---|
| `--problem-id P` | Generate only for problem `P` |
| `--limit N` | Generate for the first `N` problems |
| `--num-candidates N` | Override `candidates_per_problem` for this run |
| `--strategy S` | Use `S` instead of the configured list; repeatable |
| `--resume RUN_ID` | Resume an incomplete run instead of creating a new one |
| `--force` | With `--resume`, seed a new run from that run's manifest instead of resuming it |
| `--dry-run` | Print prompts; load no model, write nothing |
| `--mock-model` | Use the deterministic mock — a full offline pipeline run |

### Reproducibility

`seed` is applied before each generation via `transformers.set_seed`, which fixes the
Python, NumPy, and torch RNGs. This is **seeding, not a bit-for-bit guarantee**: CUDA
kernel non-determinism and batching mean identical inputs can still produce different
tokens on GPU. `problems.jsonl` is byte-reproducible; `candidates.jsonl` is not, and is
committed as a record of what the model produced rather than a rebuildable artifact.

### Real-model smoke test

Real inference is never part of `pytest` — the suite is offline and CPU-only. Request it
explicitly:

```bash
pip install -e '.[model]'
scripts/smoke_real_model.sh p001
```

## Stage 4 — Candidate Persistence, Runs and Reproducibility

### Purpose

Stage 3's flat `candidates.jsonl` was a log: it had no record of what a run was asked to
do, no way to tell an interrupted run from a finished one, and no integrity checks. Stage
4 turns it into a reliable **experiment artifact store** — every generation belongs to a
self-contained, independently auditable run directory that can be inspected, validated,
and safely resumed after an interruption.

### Run directories

```
data/candidates/runs/run_20260817_133700_a81f/
├── manifest.json          # config snapshot, status, environment — the source of truth
├── candidates.jsonl       # schema 2.0: SHA-256 hashes, attempt number
├── failures.jsonl         # every generation that produced no candidate
├── statistics.json        # always reconstructable from the two files above
└── prompts/prompts.jsonl  # the exact prompt for every attempt, written before inference
```

Run ids are `run_YYYYMMDD_HHMMSS_xxxx` (a random hex suffix, not a UUID). A run's status
is one of `created`, `running`, `completed`, `failed`, `interrupted`, `cancelled`, and is
rewritten atomically at every transition.

### Provenance and hashing

Every candidate carries `code_sha256`, `prompt_sha256`, and `raw_output_sha256` —
recomputed and checked on every load, so a tampered record cannot be read as valid.
Duplicate detection uses these hashes and is **scoped to one run**: identical code from
two different runs is recorded but never auto-linked, since cross-run duplication is
often the point of comparing two experiments, not an error.

### Resume, interruption, and `--force`

`generate` always starts a **new** run. To continue an interrupted one:

```bash
python -m python_dpo generate --limit 3 --num-candidates 5   # ^C partway through
python -m python_dpo runs show RUN_ID                         # status: interrupted
python -m python_dpo generate --resume RUN_ID                 # fills exactly the gap
python -m python_dpo runs show RUN_ID                         # status: completed
```

Already-persisted `(problem_id, generation_index)` pairs are never regenerated, and a
resumed run's first N candidate records are byte-for-byte unchanged. `--force` is
meaningful only with `--resume`: it seeds a **new** run from the original run's manifest
and regenerates everything, leaving the original run untouched — Stage 4 never overwrites
a candidate record in place.

### Retry policy

Infrastructure failures (`model_load`, `tokenizer`, `inference`, `timeout`) are retried up
to `generation.retry.max_attempts` (default 2); every failed attempt is kept as its own
record, and the eventual candidate records which `attempt` succeeded. Candidate failures
(`empty_output`, `code_extraction`) are never retried — there is nothing to retry, the
model's response itself was unusable.

### Integrity validation

```bash
python -m python_dpo runs validate RUN_ID
```

Checks the manifest, JSONL structural integrity (including a torn write at the tail),
per-record schema and hash correctness, duplicate candidate ids, `run_id`/`problem_id`
consistency, `duplicate_of` targets, prompt-artifact presence, `statistics.json`
freshness, and — for a run marked `completed` — that every requested candidate has an
outcome. `--repair` truncates a torn tail before validating; it never touches a
structurally sound but semantically wrong line.

### Migrating the Stage 3 flat file

```bash
python -m python_dpo candidates migrate
```

Reads `data/candidates/candidates.jsonl` **read-only**, groups records by their existing
`run_id`, upgrades them to schema 2.0 with hashes back-filled, and writes a proper run
directory per run id found. The source file is never modified.

## Stage 5 — Isolated Docker Sandbox

### Purpose

Every stage before this one deliberately stopped short of running the model's output.
Stage 5 builds the boundary that makes running it safe: a `SandboxExecutor` that accepts
arbitrary Python and executes it inside a locked-down container, returning a structured
result describing *what happened*.

**`status = "success"` means the program exited zero — not that the candidate is correct.**
Deciding correctness requires running the problem's test suite, which is Stage 6's job.

### Isolation at a glance

| Boundary | Mechanism |
|---|---|
| Network | `--network none` — no internet, no localhost, no DNS |
| Filesystem | Only the job workspace, mounted read-only; no project dir, no Docker socket |
| User | `--user 65534:65534` (non-root); UID 0 rejected at config load |
| Capabilities | `--cap-drop ALL` + `--security-opt no-new-privileges`; never `--privileged` |
| Root FS | `--read-only`, with one size-limited `noexec,nosuid` tmpfs at `/tmp` |
| CPU / memory | `--cpus`, `--memory` + `--memory-swap` (pinned, so swap can't double the limit) |
| Processes | `--pids-limit` |
| Time | Host-side termination, with startup overhead budgeted separately |
| Output | Bounded readers; truncation always recorded, never silent |
| Environment | Exactly three variables passed; the host environment is never inherited |

The whole security surface is one method — `ContainerSpec.to_docker_args()` in
`src/python_dpo/sandbox/container.py`. See [`docs/sandbox-security.md`](docs/sandbox-security.md)
for the threat model, the reasoning behind each flag, and the known limitations.

### Execution model

Candidate source is **written to a file**, never interpolated into a command:

```
candidate.py  →  isolated workspace  →  container  →  python /workspace/candidate.py
```

There is no shell in that path (`shell=False` everywhere), so quoting and injection are
absent by construction rather than defended against.

### Result statuses

`success`, `syntax_error`, `runtime_error`, `timeout`, `resource_exceeded`,
`infrastructure_error`, `cancelled`.

The critical distinction: **a candidate is never judged badly because Docker failed.**
`infrastructure_error` says nothing about the candidate, and later stages must treat it
differently from a genuine `runtime_error`.

Syntax and runtime errors are told apart by the fact that CPython always prints
`Traceback (most recent call last):` before a *runtime* exception and never before a
compile failure — so a program that does `raise SyntaxError("x")` is correctly reported as
a runtime error.

### Setup and use

```bash
docker pull python:3.12-slim            # one-time image prep
python -m python_dpo sandbox health      # verify the whole path end to end
python -m python_dpo sandbox run --file examples/hello.py
```

`sandbox run` **copies** the file into an isolated workspace — the path you give it is never
mounted into the container and never executed on the host.

### Timeout budget

`timeout_seconds` is the candidate's own budget; `startup_grace_seconds` separately covers
container creation and interpreter start, which cost ~2s even on a warm image. Without that
split a 5s timeout would really give a candidate under 3s, and a loaded machine could time
out a program that is merely slow rather than wrong.

### Testing

```bash
pytest -q                  # offline, Docker-free, zero skips
pytest -q -m integration   # 33 tests proving the boundaries against a real daemon
```

The security guarantees are asserted at two levels: `test_sandbox_security.py` pins the
generated argv (no Docker needed, runs on every commit) and `test_sandbox_integration.py`
demonstrates the same properties against a live container.

## Stage 6 — Candidate Test Executor

### Purpose

Stage 5 built the boundary that makes running untrusted code safe; Stage 6 is what it
was built for. `CandidateEvaluator` turns a persisted candidate into objective evidence
of whether it actually solved its problem: a deterministic pytest suite is generated
from the problem's declared test cases, run against the candidate inside the sandbox,
and the per-test outcome is persisted.

**This layer answers "what happened when this candidate was tested?" — never "is this
the best candidate?"** It never produces `chosen`/`rejected` pairs, a reward, or a
ranking; `pass_rate` is stored as a metric, not consumed as a preference signal. Those
transformations belong to Stage 7+.

**At a glance (the real Stage 4 run, `run_20260817_055411`, 50 candidates across all 10
problems):** 30/50 candidates passed every test · 20/50 failed at least one · 0 timeouts
· 0 syntax errors · 0 infrastructure errors · 322/370 individual test cases passed.

### How a candidate is evaluated

```
candidate.py + test_candidate.py + conftest.py  →  isolated workspace  →  container
  →  python -m pytest -q -p no:cacheprovider test_candidate.py
  →  nonce-prefixed JSON result lines on stdout  →  parsed  →  classified  →  persisted
```

`test_candidate.py` has one `def test_<problem>_<case>():` function per declared test
case, so a failure is traceable straight back to the dataset id. Values are embedded via
`repr()` of the problem's JSON-native `input`/`expected` fields — never string
interpolation, never `eval()`. The comparison helper deliberately mirrors
`InProcessReferenceExecutor._values_match` (including the guard that stops `True` from
satisfying `1`), because the dataset's expected values were validated under exactly
those rules; `tests/evaluation/test_integration.py`'s reference-solution self-check
proves this stays true by running every real problem's generated suite against its own
trusted reference solution.

Results come back as nonce-prefixed JSON lines printed by a generated `conftest.py` —
not JUnit XML, not a report file — which needs no extra dependency and works unchanged
against a read-only, non-root workspace. The nonce defends against accidental collision
with a candidate's own `print()` output, not a deliberately adversarial candidate; what
actually catches a forged or truncated result set is that every expected test id must be
accounted for, or it is synthesized as an `error` rather than silently dropped.

### Classification

| Condition | Status |
|---|---|
| Sandbox infrastructure failure, or no job could be attempted at all | `infrastructure_error` |
| Sandbox reported a timeout | `timeout` |
| The candidate's file failed to collect (a real syntax error) | `syntax_error` |
| Every declared test passed | `passed` |
| Otherwise | `failed` |

Exit code 0 alone never produces `passed` — every test must actually be accounted for
and pass. A candidate runtime exception during a test is a **candidate** failure (test-
level `error`, candidate-level `failed`), never mistaken for infrastructure trouble; the
two are validated as mutually exclusive at the record level.

### The evaluation image

`docker/evaluator/Dockerfile` pins `python:3.12-slim` plus `pytest==8.3.4`, tagged
`python-dpo-evaluator:1.0`, built locally rather than pulled — the container has no
network at evaluation time, so pytest must already be present at build time:

```bash
docker build -t python-dpo-evaluator:1.0 docker/evaluator/
```

Every Stage 5 isolation setting — network mode, non-root user, dropped capabilities,
resource limits, the read-only workspace — is inherited from the audited `sandbox:`
config unchanged by `build_evaluation_sandbox_config`; only the image and timeout budget
differ for evaluation. Adding pytest weakens nothing.

### Resume semantics

`evaluate run --run-id R` **resumes the latest incomplete evaluation run for `R` by
default** — deliberately the *opposite* default from `generate` (Stage 4 always starts a
new run). Evaluation is idempotent and cheap to re-run where generation burns GPU time,
which is what makes the asymmetry defensible. `--force` always mints a fresh evaluation
run instead.

```bash
python -m python_dpo evaluate run --run-id RUN_ID        # first call: creates eval_...
python -m python_dpo evaluate run --run-id RUN_ID        # second call: already completed, no-op
python -m python_dpo evaluate run --run-id RUN_ID --force # a brand-new evaluation run
```

### Persisted evidence

```
data/evaluations/runs/eval_YYYYMMDD_HHMMSS_xxxx/
├── manifest.json        # candidate run covered, probed pytest/Python versions, sandbox config
├── evaluations.jsonl     # one EvaluationResult per candidate
├── test_results.jsonl    # one TestCaseResult per declared test case per candidate
├── failures.jsonl        # candidates for which no job was ever attempted
└── statistics.json       # reconstructable from evaluations.jsonl + failures.jsonl
```

### Testing

```bash
pytest -q                  # offline, Docker-free, zero skips
pytest -q -m integration   # sandbox security + 13 evaluation tests against real containers,
                            # including the reference-solution self-check across all 10 problems
```

## Stage 7 — Candidate Ranking

### Purpose

Stage 6 produced objective execution evidence — pass/fail counts, nothing more. Stage 7
turns that evidence into a judgement: a correctness classification
(`correct`/`incorrect`/`indeterminate`), a score, and a deterministic ranking of
candidates **within each problem** — laying the groundwork Step 8 will use to build DPO
preference pairs, without generating any pair itself.

**The ranking signal is test-case performance, full stop.** No LLM judge, no random
ordering, no heuristic scoring. Given identical evaluation evidence and configuration,
the output is byte-for-byte deterministic.

**At a glance (the real Stage 6 evaluation run, `eval_20260817_115154_dcd4`, 50
candidates across all 10 problems):** 30 correct · 20 incorrect · 0 indeterminate · 14 tie
groups · 49 of 50 candidates tied with at least one sibling · 22 of 100 candidate pairs
are decisive, the rest are ties.

That last number is the point, not a surprise: **6 of the 10 problems collapse to a
single tie group and contribute zero ordering** — five have all five candidates fully
passing, one has all five failing at an identical rate. Only 4 problems (`p004`, `p007`,
`p008`, `p010`) produce any preference signal at all from this run. Tie handling isn't an
edge case here; it's the majority behavior, and this stage never invents a preference to
paper over it.

### Correctness classification

| Condition | Result |
|---|---|
| No evaluation result at all | `indeterminate` |
| Sandbox/infrastructure failure | `indeterminate` |
| The problem had zero declared tests | `indeterminate` |
| Every test passed, no timeout | `correct` |
| Otherwise — a failure, a candidate exception, a skipped test, or a candidate-caused timeout | `incorrect` |

A candidate-caused timeout (`while True: pass`) is `incorrect`, never `indeterminate` —
only a genuine absence of usable test evidence is indeterminate. `score == pass_rate ==
tests_passed / tests_total`; nothing else (duration, code length, generation strategy,
syntax validity) is allowed to move it.

### Ranking

Candidates are grouped **strictly by problem** — never compared across problems. Within a
problem, ranked candidates get a **competition rank** (1, 1, 3, 4, 5): two fully-correct
candidates are tied at rank 1, and the next rank skips by the tie group's size. Tie
detection compares the **integer** `tests_passed`, never a float, since every ranked
candidate in a problem shares the same declared test suite — asserted, not assumed.
`indeterminate` candidates are recorded with `rank: null`, listed last, never dropped.

```bash
python -m python_dpo rankings show RANK_ID p004
```
```
RANK  CANDIDATE     TESTS     SCORE   STATUS
1     p004_c001     8/8       1.00    correct
1     p004_c002     8/8       1.00    correct
1     p004_c003     8/8       1.00    correct
4     p004_c004     6/8       0.75    incorrect
4     p004_c005     6/8       0.75    incorrect
```

### Persisted evidence

```
data/rankings/runs/rank_YYYYMMDD_HHMMSS_xxxx/
├── manifest.json        # evaluation run covered, algorithm versions, status
├── assessments.jsonl     # one CandidateAssessment per candidate
├── rankings.jsonl        # one RankingResult per candidate
├── comparisons.jsonl     # one ComparisonResult per in-problem pair
└── statistics.json       # reconstructable from assessments.jsonl + rankings.jsonl
```

### Resume semantics

Unlike `evaluate run`'s resume-by-default, `rank run` follows the spec literally: a bare
invocation **always creates a new ranking run** — matching `generate`, not `evaluate` —
because ranking is pure in-memory computation with no GPU or Docker cost to amortize.
`--resume RANKING_RUN_ID` is the only way to continue one; any selection flag combined
with `--resume` is rejected, since the manifest is authoritative.

```bash
python -m python_dpo rank run --evaluation-run-id EVAL_ID   # first call: creates rank_...
python -m python_dpo rank run --evaluation-run-id EVAL_ID   # second call: a different new run
python -m python_dpo rank run --evaluation-run-id EVAL_ID --resume RANK_ID  # continue RANK_ID
```

### Testing

```bash
pytest -q   # offline, zero skips — ranking is pure computation, no Docker involved at all
```
