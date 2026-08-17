# .claude/plans/

Approved implementation plans, tracked in git despite the rest of `.claude/` being
ignored (see `.gitignore`'s `.claude/*` + `!.claude/plans/` rule).

## Files

### `01_project_skeleton_plan.md`

The concrete implementation plan for Step 1, derived from
[`.claude/specs/01_project_skeleton.md`](../specs/01_project_skeleton.md) and
confirmed with the user before implementation started. It records the three
decisions made beyond what the spec itself dictates (skeleton at the repo root
rather than a nested `python-dpo/` directory; PyYAML as the sole runtime
dependency; `.claude/specs/` and `.claude/plans/` becoming tracked while the rest of
`.claude/` stays ignored), an exact file-by-file list of what to create and why,
the verification commands to run afterward, and the deviations from the spec's
literal file tree to flag in the final report (`__main__.py` and `config.py`
weren't in the spec's tree but are required for `python -m python_dpo` and the
config abstraction it calls for). This plan has been fully executed — see the root
[`README.md`](../../README.md) for current project status.

### `02_problem_dataset_plan.md`

The implementation plan for Stage 2 — the problem dataset and ground-truth layer —
derived from [`.claude/specs/02_problem_dataset.md`](../specs/02_problem_dataset.md)
and confirmed with the user before implementation started. It covers the new
`src/python_dpo/problems/` subpackage (schema, catalog, reference solutions,
JSONL storage, a swappable `ReferenceExecutor`, and dataset validation), the
`problems build` / `problems validate` CLI commands, and the unit plus integration
test suites. It pins down the three approved design decisions (frozen dataclasses
rather than Pydantic, reference solutions authored as real Python functions with
their JSONL text derived via `inspect.getsource()`, and the validation summary
printed to stdout from the CLI layer), plus the semantics chosen for each of the ten
problems where the spec required an explicit ruling on ties, ordering, and
invalid-input behavior. **Approved but not yet implemented.**

### `04_candidate_persistence_plan.md`

The implementation plan for Stage 4 — candidate persistence, runs and reproducibility —
derived from
[`.claude/specs/04_candidate_presistence.md`](../specs/04_candidate_presistence.md) and
confirmed with the user before implementation started. It turns Stage 3's flat
append-only `candidates.jsonl` into per-run artifact directories
(`data/candidates/runs/<run_id>/` with a manifest, candidates, failures, statistics and
a prompts artifact), adds SHA-256 hashes for code, prompt and raw output, a
`schema_version` on candidate records, atomic/durable persistence with torn-tail
detection, a retry policy for infrastructure failures, statistics reconstructable from
disk, a run integrity validator, and the `runs` / `candidates` CLI command groups. It
records the three approved design decisions (`generate` always mints a new run with
`--resume RUN_ID` as the only resume path; the existing 50-record flat file is migrated
into a run directory by an explicit `candidates migrate` command rather than discarded;
run IDs adopt the spec's `run_YYYYMMDD_HHMMSS_xxxx` format), and lists the deviations to
flag in the final report — chiefly the run-scoped repository API, run-scoped duplicate
detection, and the decision not to persist tracebacks. This plan has been fully executed —
see [`04_CANDIDATE_PERSISTENCE.md`](../../04_CANDIDATE_PERSISTENCE.md) for the
implementation report.

### `05_docker_sandbox.md`

The implementation plan for Stage 5 — the isolated Docker sandbox — derived from
[`.claude/specs/05_docker_sandbox.md`](../specs/05_docker_sandbox.md) and confirmed with
the user before implementation started. It builds the execution boundary every previous
stage deliberately stopped short of: a new `src/python_dpo/sandbox/` package whose
`SandboxExecutor` runs arbitrary Python inside a locked-down container (no network, no
host filesystem, non-root, capabilities dropped, with CPU/memory/PID/output/time limits)
and returns a structured `ExecutionResult` that reports *what happened* without ever
judging correctness. It records the three approved design decisions (Docker CLI subprocess
with a fixed argv rather than the Python SDK, so zero dependencies are added and the whole
isolation posture is one auditable list; stock `python:3.12-slim` with `--user
65534:65534` rather than a custom Dockerfile; integration tests deselected by default so
`pytest -q` keeps its zero-skip, Docker-free property), and lists the deviations to flag
in the final report — chiefly the streaming-oriented `ContainerRuntime` shape, workspaces
in system temp rather than `data/sandbox/jobs/`, and the hardening added beyond the spec's
literal text (`--memory-swap`, `--security-opt no-new-privileges`). This plan has been
fully executed — see [`05_DOCKER_SANDBOX.md`](../../05_DOCKER_SANDBOX.md) for the
implementation report.

### `06_candidate_test_executor_plan.md`

The implementation plan for Stage 6 — the candidate test executor — derived from
[`.claude/specs/06_candidate_test_executor.py`](../specs/06_candidate_test_executor.py)
(the spec ships with a `.py` extension despite being Markdown) and confirmed with the user
before implementation started. It is what the Stage 5 sandbox was built for: a new
`src/python_dpo/evaluation/` package that turns a problem's declared test cases into a
deterministic pytest suite, runs it against a persisted candidate inside the sandbox, and
persists per-candidate and per-test execution evidence — counts, statuses, durations,
error types — without ever deciding `correct`/`incorrect` or producing preference pairs.

It records the three approved design decisions (results returned as nonce-prefixed JSON
lines from a generated `conftest.py`, so no dependency and no report-file extraction from
a read-only container; resume-by-default per the spec, deliberately diverging from Stage
4's explicit-resume `generate`; and verification evaluating all 50 real Qwen candidates
with the artifacts committed). It also captures the dataset findings that drive the
generator — `TestCase.input` is a kwargs mapping rather than the positional list the spec's
examples imply, p010 is `async`, p005/p006/p009 use `expected_exception`, p009 returns a
generator, and no float expected values exist — plus the reference-solution self-check that
proves the generator's comparison semantics still match the ones the dataset was validated
under. **Approved but not yet implemented.**
