# python-dpo

A preference-data generation pipeline for DPO (Direct Preference Optimization)
fine-tuning of a Qwen Coder model on Python programming tasks.

**Current status: Stage 2 — Problem Dataset.** The foundation (packaging, CLI, logging,
typed configuration) and the ground-truth layer (10 curated Python problems with trusted
reference solutions and executable tests) are in place. No model inference, candidate
generation, sandbox, evaluation, or training code has been implemented yet.

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

Only the first stage — the problem dataset — is implemented. Everything from candidate
generation onward is still a placeholder that documents the intended shape of the
pipeline.

## Roadmap

The full build is specified as 12 stages (`.claude/specs/`); each stage's own spec
states its position, e.g. "Stage 2 of 12". Only the first two have been specified and
implemented so far:

| Stage | Delivers | Status |
|-------|----------|--------|
| 1 — Project Skeleton | Installable package, placeholder CLI, logging, typed config | Done |
| 2 — Problem Dataset | 10 curated problems with reference solutions and tests (ground truth) | Done |
| 3–12 — Candidate generation → DPO training | Qwen inference, sandboxed evaluation, preference-pair generation, QLoRA + DPO training | Not started |

Stages 3–12 aren't specified yet, so the table above intentionally doesn't assign them
individual names — the pipeline diagram lists the phases in order, but the exact stage
boundaries will be set when each spec is written. Nothing in that range is implemented;
see `CLAUDE.md`'s Scope Control rule.

## Repository layout

```
.
├── src/python_dpo/       # the installable package — see its README for file details
│   ├── cli.py             # argparse CLI: problems build/validate + placeholders
│   ├── config.py          # typed config.yaml loader
│   ├── logging_config.py  # stderr logging setup
│   └── problems/          # Stage 2: schema, catalog, storage, validation
├── tests/                 # pytest suite — see tests/README.md
├── data/                  # pipeline artifacts (tracked; see data/README.md)
│   └── problems/problems.jsonl   # the Stage 2 dataset
├── scripts/               # reserved for future operational scripts
├── config.yaml            # project name, data paths, logging level
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
```

The remaining subcommands exist as **placeholders only**. Each logs a "not implemented
yet" message and exits with status `1` — none of them do real work:

- `python -m python_dpo generate`
- `python -m python_dpo evaluate`
- `python -m python_dpo preferences`
- `python -m python_dpo run`

## Configuration

Runtime settings live in `config.yaml` at the project root (project name, data
directory paths, logging level). See `src/python_dpo/config.py` for how it's loaded
and validated.

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
