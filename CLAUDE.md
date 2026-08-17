# CLAUDE.md

Engineering rules for `python-dpo`, a preference-data generation pipeline for DPO
fine-tuning of a Qwen Coder model on Python tasks.

## Architecture

The project is implemented incrementally, one milestone at a time. Do not build ahead
of the current step's requirements.

## Security

Generated Python code is untrusted.

Never execute generated code directly on the host. Execution of generated candidates must
occur inside the isolated Docker sandbox, `python_dpo.sandbox.SandboxExecutor` — the only
sanctioned path for running untrusted code. It writes the source to a file and runs it via
a fixed argv (`shell=False`) in a container with no network, no host filesystem access, a
non-root user, dropped capabilities, and CPU/memory/PID/output/time limits. See
`docs/sandbox-security.md` for the threat model and known limitations.

Manually authored **reference solutions** are a deliberate exception: they ship with the
repository, are reviewed like any other source file, and may be executed in-process to
validate the problem dataset. That execution is confined to
`InProcessReferenceExecutor`, behind the `ReferenceExecutor` protocol, so untrusted code
can be routed to a sandboxed executor instead. Never pass model-generated code to the
in-process executor.

Generated candidates are inspected with `ast.parse` only. Building a syntax tree does not
import, evaluate, or run anything, which is why `python_dpo.generation.validation` is
allowed to touch untrusted code while nothing else is. `InProcessReferenceExecutor`
remains the sole `exec()` in `src/`.

The persistence layer (`python_dpo.candidates`, `python_dpo.runs`) hashes candidate code
with SHA-256 for duplicate detection and integrity checking. Hashing is not execution —
`hashlib` never imports, evaluates, or runs the candidate. This layer never calls `exec`,
`eval`, or `subprocess` on generated code.

`python_dpo.sandbox` uses `subprocess` to drive the `docker` CLI on the host. That is the
host-side container command, never the candidate: the generated source is written to
`candidate.py` and reaches the interpreter only as a file path in a fixed argument list.
Candidate text is never interpolated into a command, and `shell=True` appears nowhere in
`src/` — `tests/sandbox/test_sandbox_security.py` asserts both properties.

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
