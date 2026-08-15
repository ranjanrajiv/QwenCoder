# CLAUDE.md

Engineering rules for `python-dpo`, a preference-data generation pipeline for DPO
fine-tuning of a Qwen Coder model on Python tasks.

## Architecture

The project is implemented incrementally, one milestone at a time. Do not build ahead
of the current step's requirements.

## Security

Generated Python code is untrusted.

Never execute generated code directly on the host. Execution of generated candidates
must occur inside an isolated sandbox (introduced in a later step).

## Testing

Every new component must have automated tests.

## Reproducibility

Intermediate artifacts must be persisted. Pipeline stages should eventually be
restartable.

## Scope Control

Do not implement future milestones unless explicitly requested.

## Data Integrity

Never silently discard generated candidates or evaluation failures.

## Development Workflow

After implementation:

1. Run tests.
2. Review failures.
3. Fix implementation issues.
4. Run tests again.
5. Report the final test result.

Do not modify tests merely to make an implementation pass.
