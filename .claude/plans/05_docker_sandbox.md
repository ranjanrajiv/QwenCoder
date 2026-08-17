# Stage 5 — Isolated Docker Sandbox

## Context

Stages 1–4 built everything *up to* the point of execution: a curated problem dataset with
trusted reference solutions, a Qwen candidate generator, and a per-run artifact store with
verified hashes and resumable runs. Every one of those stages stopped at the same hard
line — `data/candidates/runs/<run_id>/candidates.jsonl` holds real model-generated Python
that has never been run, because CLAUDE.md forbids executing it on the host and no sandbox
existed.

`.claude/specs/05_docker_sandbox.md` builds that missing boundary. The deliverable is a
`SandboxExecutor` that accepts arbitrary Python source and runs it inside a locked-down
Docker container — no network, no host filesystem, non-root, capability-dropped, with
CPU/memory/PID/output/time limits — returning a structured `ExecutionResult` describing
*what happened*, never whether the candidate was correct.

Two boundaries define the stage:

- **Security (spec §2).** Candidate source is written to a file and executed only by a
  fixed argv inside a container. Never `exec`, `eval`, `os.system`, a shell, or string
  interpolation into a command. `InProcessReferenceExecutor` stays the only host-side
  `exec()` in `src/`.
- **Scope (spec §82).** The sandbox reports `status = success` when a program *exited
  zero*, which is not a claim about correctness. No pytest evaluation, no ranking, no
  preference logic — those are Stage 6+.

**Outcome:** `python -m python_dpo sandbox health` verifies the whole Docker path end to
end; `sandbox run --file examples/hello.py` executes a file in isolation; `pytest -q` stays
offline, Docker-free and zero-skip; `pytest -q -m integration` proves the ten mandatory
security properties in §88 against a real daemon.

### Environment checked on this machine

Docker **28.3.3** is installed at `/usr/local/bin/docker`, the daemon is reachable, and the
user is in the `docker` group. The `docker` Python SDK is **not** installed and
`python:3.12-slim` is **not** yet pulled — image prep is a documented setup step (§11).

### Decisions confirmed with the user

1. **Docker CLI subprocess, not the SDK.** `subprocess.Popen` with a fixed argv list and
   `shell=False`. Adds **zero dependencies** — core install stays PyYAML-only, consistent
   with how torch/transformers were confined to an optional extra. Every isolation flag
   lives in one greppable list that maps 1:1 to the spec's own examples, so the security
   posture is auditable by reading one function. Live pipes also make bounded output
   reading (§31) straightforward.
2. **Stock `python:3.12-slim` + `--user 65534:65534`** (nobody:nogroup, already in Debian).
   §19 explicitly permits a numeric UID; §10 says install nothing extra. Image prep is just
   `docker pull`. Stage 6 will likely need a custom image once pytest must live inside it —
   building one now would be building ahead of the current step.
3. **Integration tests deselected by default.** `pytest -q` stays offline with **zero
   skips**, preserving the norm `tests/README.md` states explicitly; `pytest -q -m
   integration` runs the Docker suite and fails fast (not skips) if the daemon is
   unreachable.

---

## New package — `src/python_dpo/sandbox/`

Follows spec §5's module breakdown and the project's house style: frozen dataclasses
validating in `__post_init__`, explicit `to_dict()`/`from_dict()`, per-folder `README.md`.

**`errors.py`** — the §78 hierarchy: `SandboxError` base, then `SandboxConfigError`,
`DockerUnavailableError`, `ImageUnavailableError`, `ContainerCreationError`,
`ContainerExecutionError`, `SandboxTimeoutError`, `WorkspaceError`. Raw `subprocess` and
Docker failures are translated here and never leak outward.

**`config.py`** — `SandboxConfig`, the §53 section:

```yaml
sandbox:
  image: "python:3.12-slim"    # pinned tag, never :latest (§9)
  image_digest: null           # optional stronger pin (§9); recorded when set
  network_mode: "none"
  cpus: 1.0
  memory: "512m"
  pids_limit: 64
  timeout_seconds: 5
  max_output_bytes: 1000000
  read_only_root: true
  run_as_non_root: true
  drop_capabilities: true
  user: "65534:65534"          # used when run_as_non_root
  tmpfs_size: "64m"            # writable /tmp under a read-only root (§18)
  workspace_root: null         # null → system temp
  auto_pull: true              # health check may pull a missing image (§11)
```

`network_mode` accepts **only** `"none"` and raises otherwise. §12 makes network isolation a
MUST, so silently honouring a config that disables it would mislead — the same reasoning
already applied to `ModelConfig.quantization` in `models/base.py`, which rejects any
non-null value rather than accepting one it ignores. The key stays in the file as a
forward-compatible, honest slot.

`SandboxConfig` raises `SandboxConfigError`, **not** `ConfigError` — `sandbox/` must not
import `python_dpo.config`, or the dependency cycle reappears. `config.py` catches and
wraps, exactly as it already does for `ModelError` (`config.py:117-123`).

**`result.py`** — `ExecutionStatus` (the §8 closed set: `success`, `syntax_error`,
`runtime_error`, `timeout`, `resource_exceeded`, `infrastructure_error`, `cancelled`) and
`ExecutionResult` with the §7 required fields plus the recommended ones:

| Field | Notes |
|---|---|
| `status`, `exit_code`, `stdout`, `stderr` | §7 core |
| `duration_ms`, `timed_out`, `container_id` | §7 core; `duration_ms` feeds §83 |
| `error_type`, `error_message` | Infrastructure detail (§79) |
| `signal` | `exit_code - 128` when signalled (§51) |
| `memory_limit_exceeded`, `process_limit_exceeded`, `network_blocked` | §7 recommended |
| `stdout_truncated`, `stderr_truncated` | §32 — truncation is never silent |
| `workspace_id`, `created_at` | §7 recommended |
| `sandbox_config` | The §52 environment record: image, digest, limits, network mode |

Plus `classify(...)`, a **pure function** unit-tested without Docker, implementing §48/§49/§51:

- exit 0 → `success`
- container was OOM-killed (from `docker inspect .State.OOMKilled`) → `resource_exceeded`,
  `memory_limit_exceeded=True`
- output cap hit → `resource_exceeded` with the truncation flags set
- wall-clock timeout → `timeout`, `timed_out=True`
- non-zero exit whose stderr shows a **compile-time** error — contains
  `SyntaxError:`/`IndentationError:`/`TabError:` **and** lacks `Traceback (most recent call
  last):` → `syntax_error`. Python always prints that `Traceback` header for a *runtime*
  exception, so a program that deliberately `raise SyntaxError(...)` still classifies
  correctly as `runtime_error`.
- any other non-zero exit → `runtime_error`
- the container never started, or `docker` itself failed (exit 125/126/127, missing binary,
  unreachable daemon) → `infrastructure_error` (§79, §80, §81)

**`workspace.py`** — `SandboxWorkspace`, a context manager that creates a job directory,
writes `candidate.py`, and removes the directory in `__exit__` **unconditionally** (§16).
It never executes anything (§41).

One non-obvious detail: `tempfile.mkdtemp` creates the directory `0o700`, which UID 65534
inside the container cannot traverse. The workspace must `chmod` the directory to `0o755`
and `candidate.py` to `0o644` — otherwise every execution fails with a confusing
permission error. Worth a comment in the code, since it looks like a security loosening
and is actually just "let the non-root container user read its own read-only mount."

**`container.py`** — the §39/§40 abstraction.

- `ContainerSpec` — a frozen dataclass holding everything needed for one run
  (image, argv, workspace path, limits, env). Its `to_docker_args()` builds the argv list;
  **this one method is the entire security surface**, which is why it is unit-tested
  directly (see Tests).
- `ContainerRuntime` — a `Protocol` (matching how `ModelClient` and `ReferenceExecutor` are
  already defined) with `check_available()`, `image_present(image)`, `pull(image)`,
  `start(spec) -> StartedContainer`, `inspect(name)`, `remove(name)`.
- `StartedContainer` — `container_id`, live `stdout`/`stderr` pipes, `wait(timeout)`,
  `kill()`.
- `DockerContainerRuntime` — the CLI implementation.

The argv, with each flag's governing section:

```python
subprocess.Popen(
    [
        "docker", "run",
        "--name", name,                              # §37 deterministic-ish, debuggable
        "--network", "none",                         # §12 no network at all
        "--read-only",                               # §18 read-only root
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m", # §18 controlled writable space
        "--user", "65534:65534",                     # §19 non-root
        "--cap-drop", "ALL",                         # §21
        "--security-opt", "no-new-privileges",       # §22 intent
        "--pids-limit", "64",                        # §24
        "--cpus", "1.0",                             # §25
        "--memory", "512m", "--memory-swap", "512m", # §26 (+ no swap escape)
        "--env", "PYTHONUNBUFFERED=1",               # partial output survives a timeout
        "--env", "PYTHONDONTWRITEBYTECODE=1",        # no __pycache__ on a read-only mount
        "--env", "HOME=/tmp",
        "--volume", f"{job_dir}:/workspace:ro",      # §14/§17 only the job dir, read-only
        "--workdir", "/workspace",                   # §44
        image,
        "python", "/workspace/candidate.py",         # §43/§47 fixed argv, never a shell
    ],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
)
```

Never `--privileged` (§22), never `--pid/--network/--ipc/--uts=host` (§23), never a
`/var/run/docker.sock` mount (§35), never `env=os.environ` (§33) — only the three
`PYTHONUNBUFFERED`/`PYTHONDONTWRITEBYTECODE`/`HOME` variables are passed, so no host
credential can reach the container.

Two deliberate choices inside the runtime:

- **No `--rm`.** The container must survive long enough to be inspected for
  `.State.OOMKilled`, `.State.ExitCode` and `.Id` (§38, §65). Removal is an explicit
  `docker rm -f` in a `finally`, which also satisfies §36's lifecycle and §76's
  no-abandoned-containers requirement more directly than relying on auto-removal.
- **Bounded output via two reader threads** (§31, §75). Each thread appends up to
  `max_output_bytes`; on exceeding it, the container is killed immediately and the stream
  marked truncated. A single `capture_output=True` call would accumulate an unbounded
  in-memory string, which §31 forbids in as many words.

**`executor.py`** — `SandboxExecutor(runtime, config)` with
`execute(code, *, job_id=None, run_id=None) -> ExecutionResult`. One candidate at a time
(§77 forbids parallelism in this stage; the signature stays parallel-friendly for later).
Flow:

1. `SandboxWorkspace` writes `candidate.py`.
2. `runtime.start(spec)`; reader threads begin bounding output.
3. `wait(timeout_seconds)`; on `TimeoutExpired` → `kill()`, keep whatever output arrived.
4. `inspect` for exit code, OOM flag, container ID.
5. `classify(...)` → `ExecutionResult`, stamped with the §52 config record.
6. `finally`: remove the container, then the workspace — even on timeout, Docker error, or
   an exception propagating to the caller (§16).

Container names follow §37: `python-dpo-sandbox-<run_id>-<job_id>`, with `job_id` a
timestamp plus short random suffix in the same shape as Stage 4's run ids (§37 rules out
purely random names). Candidate source never appears in a name.

**`health.py`** — the six §54 checks, each reported individually: docker binary present →
daemon reachable → image available (pulled when `auto_pull`) → container starts → a trivial
Python program runs → cleanup succeeds. Failures render as the §55 shape
(`Sandbox health check failed:\n  Docker daemon is not reachable.`), never a raw traceback.

---

## Files to modify

**`src/python_dpo/config.py`** — add `sandbox: SandboxConfig` to `Config` (currently
`project_name`, `paths`, `log_level`, `project_root`, `model`, `generation`), parsed by a
`_parse_sandbox` helper that mirrors `_parse_model` and wraps `SandboxConfigError` in
`ConfigError`. No new `paths.*` key is needed — workspaces live in system temp.

**`config.yaml`** — the `sandbox:` block above, with a comment that `network_mode` only
accepts `none` and that secrets never belong in this file.

**`src/python_dpo/cli.py`** — a `sandbox` command group added via `_add_sandbox_parser`,
following the existing `_add_runs_parser` / `_add_candidates_parser` pattern (`cli.py:591`,
`cli.py:749`) and registered alongside them in `build_parser` (`cli.py:809-812`):

| Command | Behavior |
|---|---|
| `sandbox health` | The §54 checks; exit 0/1; §55 message shape on failure |
| `sandbox run --file PATH [--timeout N] [--show-stderr]` | §56: **copies** the file into an isolated workspace and runs it in the container — the host path is never mounted (§56, §57) |

Result summaries print to stdout (user-facing), diagnostics to the logger — the Stage 2
precedent already followed by `runs`/`candidates`. `sandbox` is a real command group, so
`_PLACEHOLDER_STAGES` (`evaluate`, `preferences`, `run`) is unchanged.

**`pyproject.toml`** — register the marker and deselect it by default (§60):

```toml
[tool.pytest.ini_options]
markers = ["integration: requires a running Docker daemon"]
addopts = "-ra -m 'not integration'"
```

A later `-m integration` on the command line overrides the one in `addopts`, so
`pytest -q -m integration` runs exactly the Docker suite. **No new dependencies.**

**`CLAUDE.md`** — the Security section currently says execution "must occur inside an
isolated sandbox (introduced in a later step)." That step now exists: name
`python_dpo.sandbox.SandboxExecutor` as the sanctioned path for untrusted code, restate that
`InProcessReferenceExecutor` remains the only host-side `exec()`, and note that the sandbox
reaches Docker only through a fixed argv with `shell=False`.

**`src/python_dpo/__init__.py`** — `__version__` → `0.5.0`.

**Docs** — `docs/sandbox-security.md` (new `docs/` directory) covering the nine §84 topics
and §85's known limitations, including the required sentence verbatim: *Docker isolation
reduces risk but should not be treated as a perfect security boundary for arbitrary hostile
code.* Plus `examples/hello.py` (referenced by §87), a new
`src/python_dpo/sandbox/README.md`, and refreshes to the root `README.md`,
`src/python_dpo/README.md`, `tests/README.md`, and `05_DOCKER_SANDBOX.md` (the §89 report,
matching the `02_`/`03_`/`04_` convention).

---

## Tests

Layout per spec §5 — a `tests/sandbox/` package alongside the existing flat test modules.

**No Docker required (§58):**

- **`test_config.py`** — defaults; every validation rule; unknown keys rejected;
  `network_mode` values other than `none` rejected; `ConfigError` wrapping at the
  `config.py` boundary; the §52 config record round-trips.
- **`test_workspace.py`** — `candidate.py` written verbatim; directory and file
  permissions readable by a non-root container UID; cleanup on success, on exception, and
  on timeout; the workspace path is outside the project tree; nothing is executed.
- **`test_result.py`** — `ExecutionResult` validation and dict round-trip; the status set
  is closed; and `classify(...)` driven as a pure function across every branch: exit 0,
  a runtime traceback, a compile-time `SyntaxError`, a `raise SyntaxError(...)` that must
  still be `runtime_error`, OOM-killed, output cap hit, timeout, docker exit 125.
- **`test_executor_mock.py`** — a `FakeContainerRuntime` implementing the protocol drives
  the executor through success, runtime error, syntax error, timeout (asserting the
  container was killed), output flood (truncation flags set and the container killed),
  OOM → `resource_exceeded`, and `DockerUnavailableError` → `infrastructure_error`.
  Critically: **cleanup runs on every path** — container removed and workspace deleted even
  when the runtime raises mid-execution.
- **`test_sandbox_security.py`** — the argv-level guard, in the spirit of
  `test_no_heavy_imports.py`: security asserted rather than assumed, and cheap enough to
  run on every commit. Builds a `ContainerSpec` and asserts `to_docker_args()` **contains**
  `--network none`, `--read-only`, `--user`, `--cap-drop ALL`, `--pids-limit`, `--memory`,
  `--cpus`, a `:ro` workspace mount and `--workdir /workspace`; and **never contains**
  `--privileged`, `--pid=host`, `--network=host`, `--ipc=host`, `--uts=host`, any
  `docker.sock` mount, any host env var beyond the three allowed, or the project directory.
  Plus a source-level check that `src/python_dpo/sandbox/` contains no `shell=True`.

**Docker required, `@pytest.mark.integration` (§59):** `test_sandbox_integration.py` covers
all ten §88 mandatory candidates — normal execution, infinite loop → `timeout`, socket
connect, DNS resolution, `os.getuid()` non-zero, the §70 env-var leak check (host sets
`PYTHON_DPO_SANDBOX_TEST_SECRET`, container must read `None`), `/var/run/docker.sock`
absent, runtime error, syntax error, output flood — plus §65 memory limit, §74 PID limit,
§69 filesystem isolation against a host marker file, and §76 cleanup verified
programmatically by listing containers matching the sandbox name prefix after each test.

**`tests/test_project.py`** — `sandbox health` / `sandbox run --file` parse; `sandbox` is
not a placeholder; bare `sandbox` prints help and exits 1; `sandbox run` with a missing file
reports it and exits 1.

---

## Execution order

1. Write this plan to `.claude/plans/05_docker_sandbox.md` (repo convention).
2. `errors.py`, `config.py`, `result.py` + their unit tests — no Docker, pure logic.
3. `workspace.py` + tests (including the container-readable permissions case).
4. `container.py`: `ContainerSpec.to_docker_args()` first, with
   `test_sandbox_security.py` written against it before the runtime exists.
5. `DockerContainerRuntime` (Popen, bounded reader threads, inspect, remove).
6. `executor.py` + `test_executor_mock.py` against `FakeContainerRuntime`.
7. `health.py`, then the CLI `sandbox` group.
8. `docker pull python:3.12-slim`, then the integration suite.
9. `pyproject.toml` marker, docs, `CLAUDE.md`, version bump, `examples/hello.py`.
10. Full suite; fix; re-run; the §89 report.

---

## Verification

```bash
source .venv/bin/activate

pytest -q                       # offline, Docker-free, zero skips
docker pull python:3.12-slim    # §11 image prep

python -m python_dpo sandbox health          # "Docker sandbox health check passed."
pytest -q -m integration                      # the Docker security suite
python -m python_dpo sandbox run --file examples/hello.py

docker ps -a --filter name=python-dpo-sandbox   # §76: must be empty
```

The §88 mandatory security checks, run individually through `sandbox run`:

```bash
print("hello")                                        → success, stdout "hello\n"
while True: pass                                      → timeout, container gone
socket.create_connection(("8.8.8.8", 53))             → failure (no network)
socket.gethostbyname("example.com")                   → failure (no DNS)
print(os.getuid())                                    → non-zero
print(os.environ.get("PYTHON_DPO_SANDBOX_TEST_SECRET")) → None
print(os.path.exists("/var/run/docker.sock"))         → False
raise RuntimeError("test")                            → runtime_error
def broken(:                                          → syntax_error
while True: print("x" * 10000)                        → controlled termination + truncation
```

Scope containment:

```bash
grep -rnE "\b(exec|eval)\(" src/            # still only InProcessReferenceExecutor
grep -rn "shell=True" src/                  # no hits
grep -rn "os.environ" src/python_dpo/sandbox/   # no wholesale env pass-through
grep -rn "docker.sock\|privileged\|pid=host" src/python_dpo/sandbox/   # no hits
grep -rniE "pytest|trl|peft|lora|dpo|rank|prefer" src/python_dpo/sandbox/  # no eval logic
```

Then produce the §89 report in `05_DOCKER_SANDBOX.md` and **stop — do not start Stage 6
(pytest evaluation) without explicit approval** (§89).

---

## Deviations to record in the report

- **Docker CLI rather than the SDK** (§39 permits either) — keeps core dependencies at
  PyYAML alone and puts the whole isolation posture in one auditable argv.
- **`ContainerRuntime`'s methods differ from §40's literal `create/start/wait/logs/stop/
  remove` list.** An attached `docker run` fuses create+start+logs, which is what makes
  bounded output reading possible; §40 asks only for "operations conceptually equivalent."
- **Workspaces live in system temp, not `data/sandbox/jobs/`** (§15 permits "an equivalent
  temporary directory"). They are always destroyed, so they are not artifacts; keeping them
  out of `data/` avoids any chance of committing them and needs no `.gitignore` change.
- **Non-root via numeric UID 65534 on the stock image**, no Dockerfile (§19 permits a
  numeric UID; §10 says install nothing extra). Stage 6 will likely introduce a custom
  image when pytest must run inside the container.
- **The workspace is mounted read-only** (§17 "prefer read-only mounts"); the candidate's
  writable space is the size-limited `/tmp` tmpfs.
- **`network_mode` accepts only `"none"`** and rejects anything else, rather than honouring
  a config that would violate §12's MUST — the same reasoning as `ModelConfig.quantization`.
- **`--memory-swap` is pinned equal to `--memory`**, beyond §26's literal text: without it a
  container can use roughly twice its memory limit via swap.
- **`--security-opt no-new-privileges`** is added beyond §22's literal wording, enforcing
  its intent.
- **`cancelled` is defined in the status set but never produced in Stage 5** — there is no
  cancellation API yet. Included so the closed set is stable for later stages.
- **Integration tests are deselected by default** so `pytest -q` keeps its zero-skip
  property; `pytest -q -m integration` runs them.
- **Known limitation, per §84/§85:** Docker reduces risk but is not a perfect boundary for
  hostile code; microVMs / gVisor / Firecracker / dedicated hosts would be needed for a
  production multi-tenant service, and are explicitly out of scope here.
