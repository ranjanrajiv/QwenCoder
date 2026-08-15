# Python DPO Data Generation Pipeline
## Step 1 — Project Skeleton

**Specification Version:** 1.0  
**Status:** Implementation Specification  
**Step:** 1 of 12

---

## 1. Objective

Create the initial Python project structure for an automated preference-data generation pipeline for DPO fine-tuning of a Qwen Coder model on Python programming tasks.

Step 1 is limited to establishing the software foundation.

The implementation must NOT yet implement:

- Qwen model inference
- Candidate generation
- Docker sandbox execution
- pytest candidate evaluation
- Preference-pair generation
- DPO training
- Dataset downloading
- Performance benchmarking
- LLM-as-a-judge

Those components will be implemented in later steps.

---

## 2. Project Goals

The project must eventually support this pipeline:

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

Step 1 only creates the foundation required to implement this pipeline incrementally.

---

## 3. Technology Requirements

Use:

- Python 3.11+
- Git
- pytest
- Docker will be required later, but Docker execution must NOT be implemented in Step 1
- Standard Python logging
- Type hints
- pathlib for filesystem operations
- dataclasses or Pydantic for structured configuration where appropriate

Use a modern Python packaging configuration with `pyproject.toml`.

Do not introduce unnecessary dependencies.

---

## 4. Required Directory Structure

Create the following structure:

    python-dpo/
    │
    ├── CLAUDE.md
    ├── README.md
    ├── pyproject.toml
    ├── .gitignore
    ├── config.yaml
    │
    ├── src/
    │   └── python_dpo/
    │       ├── __init__.py
    │       ├── cli.py
    │       └── logging_config.py
    │
    ├── tests/
    │   ├── __init__.py
    │   └── test_project.py
    │
    ├── scripts/
    │
    ├── specs/
    │
    └── data/
        ├── raw/
        ├── problems/
        ├── candidates/
        ├── evaluations/
        ├── preferences/
        └── reports/

The directories under `data/` should be created with `.gitkeep` files if necessary.

---

## 5. Python Package

The main package must be:

    src/python_dpo/

It must be installable as a Python package using the project configuration in `pyproject.toml`.

The package must expose a version.

Example:

    python -c "import python_dpo; print(python_dpo.__version__)"

must execute successfully.

The exact version number may be chosen by the implementation agent.

---

## 6. pyproject.toml

Create a valid `pyproject.toml`.

It must define:

- project name
- version
- Python version requirement
- dependencies
- optional development dependencies
- pytest configuration

Keep runtime dependencies minimal.

At this stage, the minimum runtime dependency should ideally be close to zero.

Development dependencies should include pytest.

Do not add:

- transformers
- torch
- datasets
- trl
- accelerate
- docker SDK
- vLLM

unless there is a concrete reason to do so in Step 1.

Those dependencies belong to later milestones.

---

## 7. CLI Foundation

Create:

    src/python_dpo/cli.py

The project must provide a basic CLI.

It should support:

    python -m python_dpo --help

and:

    python -m python_dpo --version

For now, the CLI only needs placeholder commands for the future pipeline:

    python -m python_dpo problems
    python -m python_dpo generate
    python -m python_dpo evaluate
    python -m python_dpo preferences
    python -m python_dpo run

The commands must NOT implement their future functionality yet.

They may print a clear message such as:

    "Candidate generation is not implemented yet."

Do not create fake successful results.

---

## 8. Logging

Create:

    src/python_dpo/logging_config.py

Provide a reusable logging configuration.

Requirements:

- use Python's standard `logging` module
- support INFO level by default
- include timestamp
- include log level
- include logger name
- include message
- avoid print statements in application code
- allow the CLI to initialize logging

Example format:

    2026-08-15 16:30:00 | INFO | python_dpo.cli | Starting application

The exact format may differ slightly as long as the requirements are met.

---

## 9. Configuration

Create:

    config.yaml

For Step 1, configuration should contain only foundational settings.

Example:

    project:
      name: python-dpo

    paths:
      raw_data: data/raw
      problems: data/problems
      candidates: data/candidates
      evaluations: data/evaluations
      preferences: data/preferences
      reports: data/reports

    logging:
      level: INFO

Do not add Qwen, GPU, Docker, DPO, or training configuration yet.

Those will be introduced in later specifications.

---

## 10. Configuration Design

Create a small configuration abstraction that allows the application to load `config.yaml`.

The configuration implementation must:

- validate required configuration fields
- provide sensible error messages
- resolve paths relative to the project root
- avoid hard-coded absolute paths

Do not over-engineer configuration in Step 1.

A simple typed configuration implementation is preferred.

---

## 11. Project Root Handling

The project must not depend on the current working directory being a particular directory.

For example, avoid code that assumes:

    open("config.yaml")

without determining the project root.

Use robust path handling.

The implementation should work when the CLI is launched from the project root.

If full arbitrary-working-directory support is not practical at this stage, document the expected invocation location.

---

## 12. README

Create a useful `README.md`.

It must contain:

### Project purpose

Explain that the project generates Python programming preference data for DPO fine-tuning.

### Current status

Clearly state:

    Step 1 — Project Skeleton

### Planned pipeline

Document:

    Problems
      ↓
    Candidate Generation
      ↓
    Sandbox
      ↓
    Evaluation
      ↓
    Preferences
      ↓
    DPO

### Installation

Include:

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"

### Testing

Include:

    pytest

### CLI

Document:

    python -m python_dpo --help

Do not document features that have not yet been implemented as if they already work.

---

## 13. .gitignore

Create a `.gitignore` appropriate for:

- Python
- virtual environments
- IDE files
- OS files
- pytest caches
- build artifacts
- generated logs
- generated datasets where appropriate

Do NOT automatically ignore the entire `data/` directory.

The generated dataset will become an important project artifact.

If certain large/generated files should eventually be excluded, document that decision.

---

## 14. Testing Requirements

Create at least one meaningful test:

    tests/test_project.py

The test suite must verify:

1. `python_dpo` can be imported.
2. The package exposes a version.
3. Configuration can be loaded.
4. Required data directories exist or can be initialized.
5. CLI help executes successfully.

Tests must not require:

- GPU
- Qwen
- Docker
- internet access

Step 1 must be completely testable on a CPU-only machine.

---

## 15. Test Command

The following must succeed:

    pytest -q

Expected result:

    all tests pass

There must be no skipped tests used merely to avoid implementing required Step 1 functionality.

---

## 16. Code Quality

Follow these principles:

- Python type hints for public functions
- small functions
- meaningful names
- no unnecessary abstraction
- no global mutable state
- no hard-coded machine-specific paths
- no secrets in source code
- no credentials in configuration
- no unnecessary dependencies
- clear error messages

Do not prematurely implement the complete future architecture.

Step 1 should remain small.

---

## 17. Security Requirements

Even though code execution is not implemented yet, establish the following project rule:

> Generated Python code is untrusted and must never be executed directly by the application host.

Add this rule to `CLAUDE.md`.

Future execution must occur inside an isolated sandbox.

Do not implement any mechanism in Step 1 that executes generated Python.

---

## 18. CLAUDE.md Requirements

Create or update `CLAUDE.md`.

It must contain the following project principles:

### Architecture

The project will be implemented incrementally.

### Security

Generated code is untrusted.

Never execute generated code directly on the host.

### Testing

Every new component must have automated tests.

### Reproducibility

Intermediate artifacts must be persisted.

Pipeline stages should eventually be restartable.

### Scope Control

Do not implement future milestones unless explicitly requested.

### Data Integrity

Never silently discard generated candidates or evaluation failures.

### Development Workflow

After implementation:

1. Run tests.
2. Review failures.
3. Fix implementation issues.
4. Run tests again.
5. Report the final test result.

Do not modify tests merely to make an implementation pass.

---

## 19. What NOT to Implement

The implementation agent must NOT create:

### Model code

No:

    transformers
    AutoModel
    AutoTokenizer
    QwenModelClient

### Dataset code

No:

    HumanEval downloader
    MBPP downloader
    Hugging Face dataset integration

### Execution

No:

    exec()
    eval()
    subprocess execution of generated Python
    Docker execution

### Evaluation

No:

    pytest execution against generated candidates
    benchmark system
    Ruff
    mypy
    security scanner

### DPO

No:

    DPOTrainer
    TRL
    LoRA
    QLoRA
    reward model

These belong to later steps.

---

## 20. Acceptance Criteria

Step 1 is complete only when all of the following are true:

- [ ] Project directory structure exists.
- [ ] `pyproject.toml` is valid.
- [ ] Package installs with `pip install -e ".[dev]"`.
- [ ] `python -c "import python_dpo"` succeeds.
- [ ] `python -m python_dpo --help` succeeds.
- [ ] `python -m python_dpo --version` succeeds.
- [ ] Placeholder CLI commands exist.
- [ ] Configuration loads successfully.
- [ ] Logging configuration works.
- [ ] Required data directories exist.
- [ ] README documents the project.
- [ ] CLAUDE.md documents engineering rules.
- [ ] `.gitignore` exists.
- [ ] pytest tests exist.
- [ ] `pytest -q` passes.
- [ ] No generated Python code is executed.
- [ ] No Qwen/LLM/DPO functionality has been implemented.

---

## 21. Verification Commands

After implementation, run:

    python -m pip install -e ".[dev]"

    python -c "import python_dpo; print(python_dpo.__version__)"

    python -m python_dpo --help

    python -m python_dpo --version

    pytest -q

Also inspect:

    git status

and verify that no unintended files or credentials were added.

---

## 22. Final Implementation Report

After completing Step 1, report:

1. Files created.
2. Files modified.
3. Dependencies added.
4. CLI commands implemented.
5. Tests created.
6. Test results.
7. Any deviations from this specification.
8. Any technical decisions requiring review.

Do not proceed to Step 2 automatically.

Wait for explicit approval before implementing the problem dataset.
:::

### How to use this with Claude Code

I would do this in three commands rather than pasting the whole spec into the conversation.

**1. Create the file yourself:**

```bash
mkdir -p specs
nano specs/01_project_skeleton.md