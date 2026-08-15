# python-dpo

A preference-data generation pipeline for DPO (Direct Preference Optimization)
fine-tuning of a Qwen Coder model on Python programming tasks.

**Current status: Step 1 — Project Skeleton.** Only the software foundation exists —
packaging, CLI, logging, and typed configuration. No model, dataset, sandbox,
evaluation, or training code has been implemented yet.

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

None of the stages above are implemented yet. The CLI commands described below are
placeholders that document the intended shape of the pipeline.

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

The following subcommands exist as **placeholders only**. Each logs a
"not implemented yet" message and exits with status `1` — none of them do real work:

- `python -m python_dpo problems`
- `python -m python_dpo generate`
- `python -m python_dpo evaluate`
- `python -m python_dpo preferences`
- `python -m python_dpo run`

## Configuration

Runtime settings live in `config.yaml` at the project root (project name, data
directory paths, logging level). See `src/python_dpo/config.py` for how it's loaded
and validated.
