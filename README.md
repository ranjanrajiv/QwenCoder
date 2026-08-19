# python-dpo

A preference-data generation pipeline for DPO (Direct Preference Optimization)
fine-tuning of a Qwen Coder model on Python programming tasks.

**Current status: Stage 10 — Base vs DPO Model Evaluation.** The foundation (packaging,
CLI, logging, typed configuration), the ground-truth layer (10 curated Python problems
with trusted reference solutions and executable tests), candidate generation with a Qwen
Coder model, a reliable per-run artifact store (manifests, SHA-256 hashes, atomic
persistence, resume, retry, integrity validation), an isolated Docker execution sandbox, a
pytest-based candidate test executor, deterministic candidate ranking, DPO preference
pair generation, QLoRA/DPO training, and base-vs-DPO model evaluation are all in place.
**Generated code, and the tests generated for it, are executed only inside the sandbox,
never on the host**; ranking, preference-pair generation and training never execute
candidate code at all, and Stage 10 reuses the Stage 5/6 sandbox unmodified rather than
adding a second execution path.

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
      ↓
Base vs DPO Model Evaluation
```

The first ten stages are implemented. Stages 11 (error analysis and iteration) and 12
(pipeline orchestration and productionization) are specified and planned but not yet
implemented.

## Roadmap

The full build is specified as 12 stages (`.claude/specs/`); each stage's own spec
states its position, e.g. "Stage 3 of 12". All twelve have been specified and ten
implemented so far:

| Stage | Delivers | Status |
|-------|----------|--------|
| 1 — Project Skeleton | Installable package, placeholder CLI, logging, typed config | Done |
| 2 — Problem Dataset | 10 curated problems with reference solutions and tests (ground truth) | Done |
| 3 — Qwen Candidate Generator | Model abstraction, 5 strategies, code extraction, candidate persistence | Done |
| 4 — Candidate Persistence | Per-run directories, manifests, SHA-256 hashes, atomic writes, resume, retry, integrity validation | Done |
| 5 — Isolated Docker Sandbox | Locked-down container execution, resource limits, structured results, security test suite | Done |
| 6 — Candidate Test Executor | Deterministic pytest-suite generation, sandboxed evaluation, per-test evidence | Done |
| 7 — Candidate Ranking | Correctness classification, scoring, deterministic per-problem ranking, pairwise comparison | Done |
| 8 — Preference Pair Generation | `{prompt, chosen, rejected}` DPO pairs, selection policies, dedup, problem-level splits | Done |
| 9 — DPO/QLoRA Training | 4-bit NF4 QLoRA + TRL DPOTrainer, preflight, metrics, adapter reload | Done |
| 10 — Base vs DPO Model Evaluation | Held-out benchmark, paired generation, pass@k, bootstrap CIs, win/tie/loss | Done |
| 11 — Error Analysis and Iteration | Error taxonomy, improvement/regression classification, data-gap analysis, evidence-backed recommendations | Specified and planned |
| 12 — Pipeline Orchestration and Productionization | Experiment config, stage graph, caching, resume, lineage, model packaging, registry | Specified and planned |

Stage 11 is specified ([`.claude/specs/11_error_analysis_and_iteration.md`](.claude/specs/11_error_analysis_and_iteration.md))
and planned ([`.claude/plans/11_error_analysis_and_iteration_plan.md`](.claude/plans/11_error_analysis_and_iteration_plan.md)),
and Stage 12 likewise
([`.claude/specs/12_pipeline_orchestration_and_productionization.md`](.claude/specs/12_pipeline_orchestration_and_productionization.md),
[`.claude/plans/12_pipeline_orchestration_and_productionization_plan.md`](.claude/plans/12_pipeline_orchestration_and_productionization_plan.md)),
but no Stage 11 or Stage 12 code exists yet. Stage 12's plan depends on Stage 11: it ships
`error_analysis` as a registered-but-disabled stage until `src/python_dpo/analysis/` lands.
See `CLAUDE.md`'s Scope Control rule.

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
│   ├── ranking/           # Stage 7: correctness classification, scoring, ranking
│   ├── preferences/       # Stage 8: {prompt, chosen, rejected} DPO pair generation
│   ├── training/          # Stage 9: 4-bit QLoRA + DPO training of a LoRA adapter
│   └── model_evaluation/  # Stage 10: base-vs-DPO evaluation, pass@k, bootstrap CIs
├── tests/                 # pytest suite — see tests/README.md
├── data/                  # pipeline artifacts (tracked; see data/README.md)
│   ├── problems/problems.jsonl     # the Stage 2 dataset
│   ├── candidates/candidates.jsonl # legacy Stage 3 flat file (read-only; see migrate)
│   ├── candidates/runs/            # Stage 4: one directory per generation run
│   ├── evaluations/runs/           # Stage 6: one directory per evaluation run
│   ├── rankings/runs/              # Stage 7: one directory per ranking run
│   ├── preferences/runs/           # Stage 8: one directory per preference run
│   ├── training/runs/              # Stage 9: one directory per training run
│   └── model_evaluations/runs/     # Stage 10: one directory per evaluation run
├── benchmarks/             # Stage 10: held-out evaluation benchmark manifests
├── docker/evaluator/       # Stage 6: the pytest-preinstalled evaluation image
├── docs/                  # sandbox-security.md — threat model and isolation boundaries
├── examples/              # hello.py — a harmless file for exercising the sandbox
├── scripts/               # operational scripts (real-model smoke test)
├── configs/training/      # Stage 9: DPO/QLoRA experiment configurations
├── configs/evaluation/    # Stage 10: base-vs-DPO evaluation configurations
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
pip install -e ".[training]"  # adds trl, peft, bitsandbytes, datasets (Stage 9; also
                               # what `evaluate-model` needs for quantized/adapter inference)
```

If a gated or private model is configured, export `HF_TOKEN` in your shell. It is read
from the environment and must never be written into `config.yaml`, source code, datasets,
logs, or this README.

## Testing

```bash
pytest -q                  # offline, Docker-free, GPU-free, zero skips
pytest -q -m integration   # Docker: sandbox security + candidate evaluation (needs a daemon + images)
pytest -q -m gpu           # CUDA: 4-bit load, LoRA, a real training step (needs a GPU + the training extra)
```

Integration and GPU tests are deselected by default so the standard run needs neither
Docker nor a GPU and reports no skips. They are not optional extras — they are where the sandbox's security guarantees,
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

python -m python_dpo preferences generate --ranking-run-id RANK_ID --policy strict
python -m python_dpo preferences generate --ranking-run-id RANK_ID --policy margin --margin 0.2
python -m python_dpo preferences generate --ranking-run-id RANK_ID --resume PREF_ID  # continue one

python -m python_dpo preferences list                    # all preference runs, newest first
python -m python_dpo preferences show --preference-run-id PREF_ID --preference-id PREF
python -m python_dpo preferences stats --preference-run-id PREF_ID
python -m python_dpo preferences validate --preference-run-id PREF_ID

python -m python_dpo train hardware-check                # GPU, CUDA, BF16, 4-bit capability
python -m python_dpo train dpo --config configs/training/dpo_qlora.yaml \
    --preference-run-id PREF_ID [--dry-run | --smoke-test]
python -m python_dpo train verify --training-run-id TRAIN_ID     # mandatory adapter reload
python -m python_dpo train inference --training-run-id TRAIN_ID --prompt "..."
python -m python_dpo train list / show --training-run-id TRAIN_ID
```

The remaining subcommand exists as a **placeholder only**. It logs a "not implemented
yet" message and exits with status `1` — it does no real work:

- `python -m python_dpo run`

## Configuration

Runtime settings live in `config.yaml` at the project root: project name, data directory
paths, logging level, the `model` / `generation` / `generation_strategies` sections, the
Stage 5 `sandbox` section (image, resource limits, isolation toggles), and the Stage 6
`evaluation` section (evaluator image, timeout, startup grace, auto-pull — every
isolation setting itself is inherited from `sandbox` at run time, never re-specified).
Stage 7 ranking has no configuration section of its own — v1 has no tunable scoring
parameters. Stage 9 keeps its hyperparameters in a standalone `configs/training/dpo_qlora.yaml`
rather than here, so a second experiment is a second file; only `paths.training` is added
to `config.yaml`. Stage 8 preference generation does have a section here: a `preferences` section (default
selection policy, minimum score margin, max pairs per problem, split ratios/seed) —
every field is overridable per-invocation by the matching `preferences generate` flag.
See `src/python_dpo/config.py` for how it's loaded and validated.

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

## Stage 8 — DPO Preference Pair Generation

### Purpose

Stage 7 produced a neutral ordering — `A_BETTER`/`TIE`/`INDETERMINATE` — and deliberately
stopped short of a preference. Stage 8 owns that vocabulary: it converts the ranking into
`{prompt, chosen, rejected}` DPO training records, backed entirely by execution evidence.

**No model of any kind is ever called.** Every label comes from Stage 6's pytest counts
by way of Stage 7's `CandidateComparator`, re-run here rather than trusted from a
persisted `ComparisonResult`. Candidate code is never touched — `chosen`/`rejected` are
the exact bytes Qwen generated, never reformatted, repaired, or wrapped in fences. Equal
score is never a preference: ties are excluded, full stop.

**At a glance (the real Stage 7 ranking run, `rank_20260817_161726_a84d`, 100 candidate
pairs across 10 problems):** 78 ties excluded, 22 decisive comparisons — the strict policy
(correct vs incorrect, no margin gate) admits 12 of those 22, all "strong", spanning only
2 problems (`p004`, `p008`). Those 12 pairs collapse to **3 distinct
`(prompt, chosen, rejected)` training records**, since several candidates within a problem
share identical code across strategies. The margin policy (`>= 0.2`) admits a different
10 pairs — 6 strong, 4 medium — spanning `p004` and `p010` instead of `p004`/`p008`; the
two policies overlap only on `p004`.

That is the honest headline, not a bug to engineer away: at ten problems, a
high-confidence DPO dataset is necessarily small. Optimizing for pair count over label
confidence would defeat the point of the policy layer.

### Canonical prompt

Every candidate of a problem was generated under a **different**, strategy-specific
prompt (see Stage 3's `Strategy:` instruction block) — so no two candidates share a
`prompt_sha256`, and a literal "chosen and rejected prompts must match" check would
produce zero pairs under every policy. Stage 8 instead builds `prompt` from a canonical,
strategy-free rendering of the problem, and *proves* it is a genuine rendering of the same
template every candidate was actually generated under: `verify_prompt_lineage` re-derives
each candidate's stored prompt hash from `build_prompt(problem, candidate.strategy)` and
requires an exact match before the canonical prompt is ever used. A mismatch is recorded
as an `integrity_failure`, never a silent fallback.

### Selection policies

| Policy | Admits | Default? |
|---|---|---|
| `strict` | `correct` vs `incorrect` only — ignores the margin entirely | Yes |
| `margin` | Any decisive comparison clearing `minimum_score_margin` (default `0.2`) | No |
| `all_better` | Any decisive comparison, however small the margin | No, experimentation only |

The same ranking run can produce several preference datasets from these — `strict_v1`,
`margin_v1`, `margin_v2` — without rerunning Qwen, Docker, or pytest.

### Deduplication

Three separate notions, deliberately not conflated: a directional `(problem, chosen,
rejected)` pair identity (so `A>B` and `B>A` are different keys, never merged); candidate
code identity (gates a single pair, never removes a candidate from the pool — it may
still pair against a third candidate); and the `(prompt, chosen, rejected)` **text**
identity a DPO trainer actually sees. The last is what collapses the real strict run's 12
pairs to 3 training records: `metadata.jsonl` keeps every pair, `preferences.jsonl` keeps
only the first (by `preference_id`) of each duplicate group.

### Splitting

The split unit is `problem_id`, never a pair — every pair from one problem lands in
exactly one of train/validation/test, so the same prompt can never appear in two splits.
The pool being split is the problems that actually produced a training pair, not the
entire dataset (splitting all ten problems when only two produce pairs would spend the
validation/test budget on problems contributing nothing). A floor rule keeps `train`
non-empty whenever the pool is non-empty. Deterministic: `random.Random(seed)` over a
sorted pool, seed and ratios persisted in `split_manifest.json`.

```bash
python -m python_dpo preferences generate --ranking-run-id rank_20260817_161726_a84d --policy strict
python -m python_dpo preferences stats --preference-run-id PREF_ID
```
```
Problems processed: 10
Candidates considered: 50
Candidate pairs considered: 100
Pairs generated: 12
Pairs rejected: 88
  Ties: 78
  Duplicate code: 0
  Indeterminate: 0
  Prompt mismatches: 0
  Integrity failures: 0
Strong pairs: 12
Medium pairs: 0
Distinct training records: 3

Policy/other exclusions:
  not_correct_vs_incorrect: 10

Split (seed=42): train=1 validation=0 test=1
```

### Persisted evidence

```
data/preferences/runs/pref_YYYYMMDD_HHMMSS_xxxx/
├── manifest.json        # policy, versions, margin, split config, upstream run ids, status
├── metadata.jsonl        # one PreferencePair per generated pair, including collapsed ones
├── rejections.jsonl      # one PreferenceRejection per excluded candidate pair, with a reason
├── preferences.jsonl     # {prompt, chosen, rejected} training records, deduped
├── split_manifest.json   # train/validation/test problem-id membership, seed, ratios
├── train.jsonl           # same three-key shape as preferences.jsonl
├── validation.jsonl
├── test.jsonl
├── statistics.json       # reconstructable from metadata.jsonl + rejections.jsonl
└── quality_report.json   # distributions and a reason per pairless problem; reported, not enforced
```

### Resume semantics

Follows `rank run`'s shape, not `evaluate run`'s resume-by-default: a bare invocation
**always creates a new preference run**. `--resume PREFERENCE_RUN_ID` is the only way to
continue one; `--force` mints a new run rather than modifying an existing one, so
`strict_v1` and `margin_v1` from the same ranking run coexist.

```bash
python -m python_dpo preferences generate --ranking-run-id RANK_ID --policy strict   # creates pref_...
python -m python_dpo preferences generate --ranking-run-id RANK_ID --policy strict --force  # a new run
python -m python_dpo preferences generate --ranking-run-id RANK_ID --resume PREF_ID  # continue PREF_ID
```

### Testing

```bash
pytest -q   # offline, zero skips — preference generation is pure computation, no Docker at all
```

## Stage 9 — DPO/QLoRA Training

### Purpose

Stage 8 produced `{prompt, chosen, rejected}` records backed by execution evidence. Stage 9
is the first stage that changes a model: it fine-tunes a **LoRA adapter over a frozen,
4-bit NF4-quantized** Qwen Coder base using TRL's `DPOTrainer`.

Three rules govern it. **The base model stays frozen** — only LoRA parameters train, and
the parameter check *refuses to proceed* if every parameter is trainable, because that
would be a full fine-tune this stage must never perform by accident. **No candidate code
is executed** — `chosen`/`rejected` are validated to be strings and are only tokenized;
execution belongs to the Stage 5 sandbox alone. **No performance claim** — a falling DPO
loss is not evidence of better Python, and this stage produces `base model + adapter` for
Step 10 to evaluate.

**At a glance (the real run on this machine):** RTX 3060, 11.2 GiB free · Qwen2.5-Coder-3B
in 4-bit NF4 · LoRA r=16 over `q,k,v,o` × 36 layers = **7,372,800 trainable parameters,
0.43% of the 1.7B quantized base** · peak 4.66 GiB VRAM · adapter 14.1 MiB · reload
verified.

That last number is the point: 3 training records over 2 problems is a **single optimizer
step**. Stage 9 demonstrates that the QLoRA/DPO stack loads, quantizes, attaches an
adapter, takes a real gradient step, saves and reloads — and nothing whatsoever about
whether the model writes better Python.

### The canonical prompt, and why training applies a chat template

Stage 3 generated every candidate through the tokenizer's chat template. Training applies
the **same** template, converting Stage 8's model-agnostic string prompt into
conversational form so TRL renders it identically. Training on the bare string would train
on a format the candidates were never produced under.

### Preflight

Everything below runs *before* a single gradient is computed, and any failure stops the
run before an artifact is written:

| Check | Fails when |
|---|---|
| Hardware | No CUDA, or free VRAM below the floor (free, not total — a desktop session holds ~0.5 GiB) |
| Dataset | Missing/malformed split, empty field, `chosen == rejected`, non-string response, empty train |
| Leakage | A problem id appears in more than one split |
| Truncation | More than 5% of examples exceed `max_length` |
| Target modules | *None* of the configured LoRA targets exist on the model |
| Parameters | `trainable == total` (full fine-tune) or `trainable == 0` (LoRA never attached) |

```bash
python -m python_dpo train hardware-check
```
```
Hardware check passed.
  CUDA: CUDA 13.0 via torch 2.13.0+cu130, 1 device(s)
  GPU: NVIDIA GeForce RTX 3060 (compute capability 8.6)
  VRAM: 11.2 GiB free of 11.6 GiB total
  BF16: supported
  4-bit quantization: bitsandbytes is available
```

### Dry run, smoke test, real run

All three are the **same code path stopped at different points**, which is the whole value
of the dry run: it exercises the code that will actually train, not a parallel one.

```bash
python -m python_dpo train dpo --config configs/training/dpo_qlora.yaml \
    --preference-run-id PREF_ID --dry-run      # preflight only, no training
python -m python_dpo train dpo ... --smoke-test # a handful of examples, 1-2 steps
python -m python_dpo train dpo ...              # the real run
python -m python_dpo train verify --training-run-id TRAIN_ID
```
```
Trainable:     7,372,800 of 1,706,045,440 (0.4322%)
Final train loss: 0.693147
Final eval loss:  0.676852
Peak GPU memory: 4.66 GiB
Adapter reload: OK
```

An adapter that does not reload is **not** a training artifact, whatever the loss curve
said — `train verify` reloads the saved adapter from disk against a freshly loaded base
model and generates with it, and the run is not successful until it passes.

### Persisted evidence

```
data/training/runs/dpo_YYYYMMDD_HHMMSS_xxxx/
├── manifest.json          # package versions, seeds, hardware, upstream run ids, status
├── config.yaml            # the resolved experiment config actually used
├── dataset_manifest.json  # preference provenance + all three split hashes
├── hardware.json          # GPU/CUDA/BF16/4-bit capability at run start
├── metrics/metrics.jsonl  # per-step loss, DPO reward metrics, GPU memory
├── logs/training.log
├── adapter/               # the trained LoRA adapter    [tracked, 14.1 MiB]
├── checkpoints/           # periodic checkpoints        [gitignored]
└── final_report.json
```

All three split hashes are recorded **including test** — not because test is used (it is
never handed to the trainer) but because reproducing a run means proving the same test
split was held out. Checkpoints, the tokenizer snapshot and TRL's frozen fp32 reference
adapter are gitignored: reproducible bulk, not deliverables.

### Configuration

Hyperparameters live in `configs/training/dpo_qlora.yaml`, separate from the root
`config.yaml`, so a second experiment is a second file rather than an edit. Every field is
overridable per invocation (`--learning-rate`, `--beta`, `--epochs`, `--max-steps`,
`--seed`, `--lora-r`), so changing one never requires editing YAML *or* Python.

### Testing

```bash
pytest -q          # offline, zero skips — no GPU, no model, no heavy imports
pytest -q -m gpu   # the real stack: 4-bit load, LoRA, a training step, adapter reload
```

The GPU suite **fails rather than skips** when CUDA or the training extra is missing, for
the same reason the Docker suites do: a quietly-unrun QLoRA suite is worse than a red one.

## Stage 10 — Base vs DPO Model Evaluation

### Purpose

Every stage so far deferred the question this one finally asks:

> Did DPO actually make Qwen better at Python programming?

Base Qwen and base+adapter are generated against the **identical** held-out benchmark,
prompts, generation config and seeds, evaluated through the **unmodified** Stage 5/6
sandbox, and compared with pass@k, bootstrap confidence intervals, and win/tie/loss. Four
rules govern it: **the benchmark is never used to tune anything** (consulting it to change
beta, LoRA rank, epochs or the preference policy means a new experiment, not a better
number); **correctness comes from execution, not judgement** (no LLM judge anywhere);
**the comparison isolates the adapter** (same prompt, same chat template, same seeds, same
quantization — the only difference is base weights versus base + adapter); and **no
automatic promotion** (the stage produces evidence and a `DPO_SUCCESS` verdict against
configurable criteria, never a production decision).

### The benchmark, and its ceiling

`python_eval_v1` is the 7 problems never assigned to any Stage 8 preference split
(`p001, p002, p003, p004, p005, p006, p009` — `p007`/`p008` trained, `p010` validated).
From the committed Stage 6/7 evidence, 5 of those 7 are already solved by every one of the
base model's 5 Stage 3 candidates — a hard ceiling that bounds what this benchmark can
show. Selecting only the 2 problems with headroom was rejected as contamination: a
benchmark chosen by inspecting base-model results is no longer held out.

```bash
python -m python_dpo benchmark build --name python_eval_v1 \
    --exclude-preference-run-id pref_20260818_074347_5eff
```
```
Benchmark python_eval_v1 built: 7 problem(s)
  problem_ids: p001, p002, p003, p004, p005, p006, p009
  dataset_hash: 600953284286b946508c563b2b47a52a48b1572bfb2d911a4081e5fb8a2d6d31
```

### Running the evaluation

```bash
python -m python_dpo benchmark validate --benchmark python_eval_v1
python -m python_dpo benchmark check-leakage --benchmark python_eval_v1 \
    --preference-run-id pref_20260818_074347_5eff
python -m python_dpo evaluate-model --benchmark python_eval_v1 \
    --training-run-id TRAIN_ID --smoke-test    # 3 problems, 1 sample, both models
python -m python_dpo evaluate-model --benchmark python_eval_v1 \
    --training-run-id TRAIN_ID --num-samples 10  # the real run: 7 problems x 10 samples x 2 models
python -m python_dpo evaluate-model validate --evaluation-run-id EVAL_ID
python -m python_dpo evaluate-model report   --evaluation-run-id EVAL_ID
python -m python_dpo evaluate-model stats    --evaluation-run-id EVAL_ID
```

`--dry-run`-style staging isn't needed here the way it is for training: `--smoke-test`
takes the same code path as a real run (paired generation, sandboxed evaluation, the full
report) and simply stops after 3 problems and 1 sample.

### Persisted evidence

```
benchmarks/python_eval_v1/manifest.json          # ids + hash; problems stay in problems.jsonl

data/model_evaluations/runs/eval_YYYYMMDD_HHMMSS_xxxx/
├── manifest.json, config.yaml, benchmark_manifest.json
├── generations/{base,dpo}.jsonl        # every attempted generation, including failures
├── evaluations/{base,dpo}.jsonl        # curated per-candidate outcomes
├── evaluations/_sandbox/{base,dpo}/    # unmodified Stage 6 output, for forensics
├── metrics/{summary,pass_at_k,bootstrap}.json
├── reports/base_vs_dpo.{md,json}
├── reports/{improvements,regressions,ties}.jsonl
└── reports/failure_analysis.json
```

### What the committed run shows

The committed adapter (`dpo_20260818_081231_a91d`) trained for a **single optimizer
step** over 3 preference pairs — Stage 9's own headline is that this proves the QLoRA/DPO
stack works, not that it adapted the model. Stage 10's job here is symmetric: prove the
*evaluation* apparatus works end to end, honestly report what a one-step adapter produces
on a 7-problem, ceiling-heavy benchmark, and leave the model-performance question open for
a better-trained adapter to actually answer.

```
Benchmark:      python_eval_v1 (7 problems, ceiling on 5 of them)
Base pass@1:    77.1%   (pass@5 85.7%, pass@10 85.7%)
DPO pass@1:     78.6%   (pass@5 85.7%, pass@10 85.7%)
Improvement:    +1.4 pp   (paired bootstrap 95% CI [+0.0, +4.3] pp)
Win/Tie/Loss:   0 / 7 / 0   (problem-level: both solve/fail the same problems overall)
DPO_SUCCESS:    False   (+1.4 pp is below the configured 2 pp minimum-improvement gate)
```

`eval_20260818_155511_1633`, committed alongside this code. A +1.4 percentage-point
pass@1 nudge from a **single optimizer step** is exactly the noise-level, no-measurable-
difference result the plan predicted — the seven candidates that flipped a sample from
wrong to right did not change which *problems* got solved overall, hence 0 wins and 0
losses at the problem level. Per spec section 144: 7 problems is suitable for pipeline
validation, not for reliable model-performance conclusions — the pipeline is what this
run validates, and it produced a paired, bootstrapped, honestly-reported "no" rather than
a manufactured "yes."

### Testing

```bash
pytest -q                  # offline, zero skips — pass@k, bootstrap, comparison, benchmark
pytest -q -m integration   # Docker: EvaluationDriver through the real Stage 6 sandbox
pytest -q -m gpu           # CUDA: adapter isolation, integrity failures, smoke generation
```
