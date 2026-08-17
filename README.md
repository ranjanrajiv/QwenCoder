# python-dpo

A preference-data generation pipeline for DPO (Direct Preference Optimization)
fine-tuning of a Qwen Coder model on Python programming tasks.

**Current status: Stage 3 — Qwen Candidate Generator.** The foundation (packaging, CLI,
logging, typed configuration), the ground-truth layer (10 curated Python problems with
trusted reference solutions and executable tests), and candidate generation with a Qwen
Coder model are in place. No sandbox, evaluation, ranking, preference-pair generation, or
training code has been implemented yet — and **no generated code is ever executed**.

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

The first three stages are implemented. Everything from the Docker sandbox onward is
still a placeholder that documents the intended shape of the pipeline.

## Roadmap

The full build is specified as 12 stages (`.claude/specs/`); each stage's own spec
states its position, e.g. "Stage 3 of 12". Three have been specified and implemented so
far:

| Stage | Delivers | Status |
|-------|----------|--------|
| 1 — Project Skeleton | Installable package, placeholder CLI, logging, typed config | Done |
| 2 — Problem Dataset | 10 curated problems with reference solutions and tests (ground truth) | Done |
| 3 — Qwen Candidate Generator | Model abstraction, 5 strategies, code extraction, candidate persistence | Done |
| 4–12 — Sandboxed evaluation → DPO training | Docker sandbox, pytest evaluation, ranking, preference-pair generation, QLoRA + DPO training | Not started |

Stages 4–12 aren't specified yet, so the table above intentionally doesn't assign them
individual names — the pipeline diagram lists the phases in order, but the exact stage
boundaries will be set when each spec is written. Nothing in that range is implemented;
see `CLAUDE.md`'s Scope Control rule.

## Repository layout

```
.
├── src/python_dpo/       # the installable package — see its README for file details
│   ├── cli.py             # argparse CLI: problems, generate + placeholders
│   ├── config.py          # typed config.yaml loader
│   ├── logging_config.py  # stderr logging setup
│   ├── problems/          # Stage 2: schema, catalog, storage, validation
│   ├── models/            # Stage 3: ModelClient protocol, Qwen client, mock client
│   ├── generation/        # Stage 3: strategies, prompts, extraction, orchestration
│   └── candidates/        # Stage 3: candidate schema and append-only repository
├── tests/                 # pytest suite — see tests/README.md
├── data/                  # pipeline artifacts (tracked; see data/README.md)
│   ├── problems/problems.jsonl     # the Stage 2 dataset
│   └── candidates/candidates.jsonl # the Stage 3 candidates
├── scripts/               # operational scripts (real-model smoke test)
├── config.yaml            # project name, data paths, logging, model, generation
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
pytest -q
```

## CLI

```bash
python -m python_dpo --help
python -m python_dpo --version
```

Implemented:

```bash
python -m python_dpo problems build      # validate the catalog and write problems.jsonl
python -m python_dpo problems validate   # re-validate the persisted dataset
python -m python_dpo generate            # generate candidates (see Stage 3 below)
```

The remaining subcommands exist as **placeholders only**. Each logs a "not implemented
yet" message and exits with status `1` — none of them do real work:

- `python -m python_dpo evaluate`
- `python -m python_dpo preferences`
- `python -m python_dpo run`

## Configuration

Runtime settings live in `config.yaml` at the project root: project name, data directory
paths, logging level, and the Stage 3 `model`, `generation`, and `generation_strategies`
sections. See `src/python_dpo/config.py` for how it's loaded and validated.

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

`data/candidates/candidates.jsonl`, one candidate per line, written the moment it is
generated so a killed run stays resumable. Both `raw_output` and the extracted `code` are
kept — the raw text is the only way to debug an extraction that went wrong.

```json
{
  "candidate_id": "p001_c001", "problem_id": "p001", "run_id": "20260817_055411",
  "generation_index": 1, "strategy": "normal",
  "model": "Qwen/Qwen2.5-Coder-3B-Instruct", "provider": "transformers",
  "prompt_version": "v1", "extraction_format": "python_fence",
  "syntax_valid": true, "function_name_valid": true, "duplicate_of": null
}
```

Failures that produced *no* candidate go to `generation_failures.jsonl` with a closed-set
`error_type`. Code that fails to parse is **not** a failure — it is stored as a candidate
with `syntax_valid: false`, because it is the model's real output and precisely what a
later stage needs on the rejected side of a preference pair.

### Resume and `--force`

`candidates.jsonl` is append-only. A repeated command resumes, skipping
`(problem_id, generation_index)` pairs that already exist. `--force` mints a new `run_id`
and appends, keeping the earlier run intact — so the same `candidate_id` may appear more
than once and the file-wide key is `(run_id, candidate_id)`.

```bash
python -m python_dpo generate --problem-id p001 --num-candidates 5   # generates 5
python -m python_dpo generate --problem-id p001 --num-candidates 5   # generates 0
python -m python_dpo generate --problem-id p001 --num-candidates 5 --force  # new run
```

### CLI options

| Flag | Effect |
|---|---|
| `--problem-id P` | Generate only for problem `P` |
| `--limit N` | Generate for the first `N` problems |
| `--num-candidates N` | Override `candidates_per_problem` for this run |
| `--strategy S` | Use `S` instead of the configured list; repeatable |
| `--force` | Start a new run instead of resuming |
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
