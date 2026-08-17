# Stage 5 Implementation Details — Isolated Docker Sandbox

How `src/python_dpo/sandbox/` implements the layer specified in
`.claude/specs/05_docker_sandbox.md`. For usage, see the "Stage 5 — Isolated Docker
Sandbox" section of the root `README.md`; for the threat model and isolation inventory, see
`docs/sandbox-security.md`. This file is about *how* it is built and what was learned
building it.

## Goal

Stages 1–4 all stopped at the same line: `data/candidates/runs/<run_id>/candidates.jsonl`
holds real model-generated Python that had never been run, because CLAUDE.md forbids
executing it on the host and no sandbox existed. Stage 5 builds that boundary.

Two constraints define the stage. **Security (spec §2):** candidate source is written to a
file and executed only by a fixed argv inside a container — never `exec`, `eval`,
`os.system`, a shell, or string interpolation. **Scope (spec §82):** the sandbox reports
what happened, never whether the candidate was correct.

## 1. Docker image used

`python:3.12-slim`, pinned by tag. Digest at time of writing:

```
sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a
```

`sandbox.image_digest` accepts that digest to pin by exact bytes rather than a mutable tag;
when set, the sandbox runs `python@sha256:...` and records it in every result. A `:latest`
tag, or an image reference with no tag at all, is **rejected at config load** — spec §9's
requirement made unbypassable rather than merely documented.

No Dockerfile is shipped. Image prep is `docker pull python:3.12-slim`, and non-root
execution is achieved with `--user 65534:65534` (nobody:nogroup, already present in
Debian-derived images), which spec §19 explicitly permits. Stage 6 will likely need a custom
image once pytest must live inside the container; building one now would be building ahead
of the current step.

## 2. Runtime architecture

```
candidate code -> SandboxWorkspace -> ContainerSpec -> ContainerRuntime
               -> bounded stdout/stderr -> classify() -> ExecutionResult
```

| Module | Responsibility |
|---|---|
| `errors.py` | The §78 exception hierarchy; raw Docker/subprocess failures never escape |
| `config.py` | `SandboxConfig` — the `sandbox:` section, validated on construction |
| `result.py` | `ExecutionResult` + `classify()`, a pure function over collected facts |
| `workspace.py` | `SandboxWorkspace` — one temporary job dir per execution; never executes |
| `container.py` | `ContainerSpec.to_docker_args()`, the `ContainerRuntime` protocol, `DockerContainerRuntime`, `BoundedReader` |
| `executor.py` | `SandboxExecutor` — orchestration, wait/kill loop, unconditional cleanup |
| `health.py` | The six §54 checks with §55-shaped reporting |

**Docker CLI, not the SDK** (§39 permits either). Zero new dependencies — the core install
stays PyYAML-only, consistent with how torch/transformers were confined to an optional
extra. More importantly, every isolation flag ends up in one greppable list that maps 1:1
to the spec's own examples, so the security posture is auditable by reading one method.

`sandbox/` never imports `python_dpo.config`; it raises `SandboxConfigError` and `config.py`
translates that to `ConfigError`, exactly as it already does for `ModelError` from
`models/base.py`. The dependency runs one way.

## 3. Resource limits

Enforced by the container runtime, not by application-level checks (§27):

| Limit | Mechanism | Default |
|---|---|---|
| CPU | `--cpus` | 1.0 |
| Memory | `--memory` + `--memory-swap` | 512m |
| Processes | `--pids-limit` | 64 |
| Wall clock | host-side kill | 5s + 10s startup grace |
| Output | `BoundedReader` per stream | 1,000,000 bytes |
| `/tmp` | `--tmpfs size=` | 64m |

`--memory-swap` pinned equal to `--memory` goes beyond §26's literal text and is
deliberate: without it a container may use roughly twice its memory limit via swap, quietly
defeating the ceiling.

**The timeout is split into two budgets**, which was not in the plan and came out of
measurement. Container creation and interpreter startup cost ~2.3s consistently, even on a
warm image. Charged against a 5s timeout that leaves a candidate under 3s, and the boundary
moves with machine load — a slow-but-correct candidate could be falsely timed out.
`timeout_seconds` is now the candidate's own budget and `startup_grace_seconds` covers the
overhead. This matters more for Stage 6 (running a test suite per candidate) than it does
here.

## 4. Network isolation

`--network none`. The container has no network interface beyond loopback, so this is not a
rule a candidate could route around. Verified against a live daemon for outbound TCP, DNS
resolution, and HTTP — all three fail.

`sandbox.network_mode` accepts **only** `"none"`; `bridge` or `host` is a config error, not
an honoured setting. §12 states network isolation as a MUST, so accepting a value that would
disable it would let a config file silently revoke the guarantee. Same reasoning
`ModelConfig.quantization` already applies.

## 5. Filesystem isolation

One host path is visible to the container: its own job workspace, mounted **read-only** at
`/workspace` (§17 prefers read-only mounts). Never mounted: the project directory, `/home`,
the host `/tmp`, `.git`, SSH keys, cloud credentials, or `/var/run/docker.sock`.

Root filesystem is `--read-only`, with one `--tmpfs /tmp` carrying `noexec,nosuid` and a size
limit so it cannot be used to stage and run a binary. Each execution gets a fresh workspace
destroyed afterwards, so no state carries between candidates.

## 6. User / UID configuration

`--user 65534:65534`. A config naming UID 0 is rejected at load, and a *named* user is
rejected too when `run_as_non_root` is set — the stock image has no named non-root user, so
a name would fail confusingly at runtime rather than at config time.

Verified live: `os.getuid()` returns 65534, writes to `/etc/passwd` are refused, writes to
`/tmp` succeed.

## 7. Linux capability configuration

`--cap-drop ALL` plus `--security-opt no-new-privileges`. The latter is beyond §22's literal
wording and enforces its intent: even a setuid binary inside the image cannot gain
privileges. Never `--privileged`; never `--pid/--network/--ipc/--uts=host`. All asserted at
the argv level in `test_dangerous_flags_are_never_present`.

## 8. Timeout implementation

`docker run` has no built-in timeout, so termination is host-side. The executor polls
`wait(timeout=0.1)` against a deadline rather than issuing one long blocking wait, because
the output limit can trip at any moment and the container must be killed promptly when it
does. On expiry it calls `docker kill <name>` — killing the local client process alone would
detach and leave the container running, which §29 forbids.

**All container control happens on one thread.** This was the subject of a real bug (see
§17 below).

## 9. Output-limit implementation

Two `BoundedReader` threads, one per stream, each keeping at most `max_output_bytes` and
then setting a shared `Event`; they keep draining and discarding afterwards so the container
never blocks writing into a full pipe while being torn down. The executor's poll loop
watches the event and terminates the container. Truncation is always recorded
(`stdout_truncated`/`stderr_truncated`) — §32 forbids silent truncation.

An output flood is classified `resource_exceeded`, not `timeout`: `classify()` ranks
resource violations above the timeout, and reporting `timed_out` for a flood would
contradict that.

## 10. Container cleanup strategy

`--rm` is deliberately **not** used: the container must survive long enough to be inspected
for `.State.OOMKilled`, `.State.ExitCode` and `.Id` (§38, §65). Removal is an explicit
`docker rm --force` in a `finally`, which also makes §36's lifecycle auditable. `remove()`
never raises, so a cleanup problem cannot mask a real result.

Containers are named `python-dpo-sandbox-<run_id>-<job_id>` (§37) — a timestamp plus short
random suffix, in the same shape as Stage 4's run ids, so a stray container is traceable.
Candidate source never appears in a name.

Every integration test asserts via an autouse fixture that no sandbox container survived it.

## 11. Workspace cleanup strategy

`SandboxWorkspace` is a context manager; `__exit__` removes the directory whether the block
completed, raised, or timed out (§16). `cleanup()` is idempotent and never raises.

Two details worth recording. `tempfile.mkdtemp` creates `0o700`, which the container's
non-root UID cannot traverse — without an explicit chmod to `0o755`/`0o644` every execution
fails with a confusing permission error. And source is written as **bytes**, not via
`write_text`, because text mode applies newline translation and a candidate containing
`\r\n` would otherwise not reach the container byte-for-byte as the model produced it.

## 12. Error classification

`classify()` is a pure function over already-collected facts, so every branch is unit tested
without Docker. Precedence: infrastructure → resource → timeout → candidate exit.

The interesting case is syntax vs runtime. CPython exits 1 for both an uncaught exception
and a `SyntaxError`, so the exit code alone cannot separate them. The reliable signal is
that CPython **always** prints `Traceback (most recent call last):` before a runtime
exception and **never** before a compile failure. So `raise SyntaxError("deliberate")` is
correctly reported as `runtime_error` — it compiled fine and then chose to raise. Both cases
are pinned in unit tests and against the real interpreter.

The distinction §81 calls critical is carried on the result itself:
`is_candidate_outcome` vs `is_infrastructure_failure`. A candidate is never marked bad
because Docker failed. `SandboxExecutor.execute` returns infrastructure failures rather than
raising, so a caller looping over candidates cannot be derailed by a transient fault.

## 13. CLI commands added

| Command | Behavior |
|---|---|
| `sandbox health` | The six §54 checks, stopping at the first failure; §55 message shape |
| `sandbox run --file PATH [--timeout N] [--show-stderr]` | Copies the file into an isolated workspace and executes it in a container |

`sandbox run` exits 0 for a candidate that crashed — the sandbox did its job — and 1 only
for an infrastructure failure. The supplied path is never mounted and never executed on the
host (§57).

## 14. Security tests executed

Two levels, deliberately.

**Argv level, no Docker, runs on every commit** (`test_sandbox_security.py`, 28 tests).
Asserts both halves of the contract: every mandatory flag present, and every dangerous one
absent — `--privileged`, `--pid=host`, any `docker.sock` mount, the project directory, any
host env var beyond the three passed on purpose. Plus source-level scans proving the package
contains no `shell=True`, no `exec`/`eval`/`os.system`, and no `os.environ` pass-through.
Same philosophy as `test_no_heavy_imports.py`.

**Live daemon** (`test_sandbox_integration.py`, 33 tests). All ten mandatory §88 checks plus
memory/PID limits, read-only root and workspace, filesystem isolation against a host marker,
container-ID recording, and the health check.

The §88 checks, run manually as well:

| Check | Result |
|---|---|
| `print("hello")` | `success`, stdout `hello\n`, exit 0 |
| `while True: pass` | `timeout`, signal 9, container removed |
| `socket.create_connection(("8.8.8.8", 53))` | fails — no route to host |
| `socket.gethostbyname("example.com")` | fails — `gaierror`, name resolution |
| `os.getuid()` | `65534` |
| `os.environ.get("PYTHON_DPO_SANDBOX_TEST_SECRET")` | `None` |
| `os.path.exists("/var/run/docker.sock")` | `False` |
| `raise RuntimeError("test")` | `runtime_error`, exit 1 |
| `def broken(:` | `syntax_error` |
| `while True: print("x" * 10000)` | `resource_exceeded`, truncated at 1MB, killed in 3.05s |

Additionally: a 2GB allocation against a 512m cap → `resource_exceeded` with
`memory_limit_exceeded`; 200 threads against a 64 PID cap → blocked at thread 64.

## 15. Integration-test results

```
pytest -q                  → 453 passed, 33 deselected   (offline, zero skips)
pytest -q -m integration   → 33 passed                    (~2 min, real daemon)
docker ps -a --filter name=python-dpo-sandbox → empty
```

## 16. Known limitations

Documented in full in `docs/sandbox-security.md`, and stated plainly there as spec §84
requires: *Docker isolation reduces risk but should not be treated as a perfect security
boundary for arbitrary hostile code.*

The load-bearing ones: containers share the host kernel, so a kernel exploit or container
escape defeats this entirely; there is no user-namespace remapping, so UID 65534 inside is
UID 65534 outside unless the daemon is configured with `userns-remap`; seccomp is Docker's
default profile rather than a tailored one; and the daemon itself runs as root, so the
sandbox protects the host *from candidates*, not from a user who already has Docker access.
A production hostile-code service would want microVMs, gVisor, or dedicated worker hosts —
explicitly out of scope (§85).

## 17. Bugs found and fixed during implementation

Recorded because both were found only by running against a real daemon, and the unit tests
had been passing throughout.

**Reader threads must never touch the container.** The first implementation passed
`container.kill` as an `on_limit` callback to `BoundedReader`, so a flooding candidate was
killed from a reader thread while the main thread sat in `Popen.wait()` on the same process.
Concurrent waits on one `Popen` are unsafe: the output-flood case took **75 seconds**,
raised inside the reader thread, and **left a container running** — a §76 violation. Fixed
by having readers only set an `Event` and moving all container control to the executor's
poll loop. The flood case now terminates in 3.05s with nothing left behind, and
`test_reader_threads_never_touch_the_container` pins the rule.

**Container startup consumed the candidate's timeout.** The health check's own probe timed
out on first run. Measurement showed ~2.3s of consistent startup overhead, which was being
charged to the candidate. Fixed with the separate `startup_grace_seconds` budget described
in §3.

One test also needed correcting rather than the code: a check scanning container environment
variables for credential-shaped *names* failed on `GPG_KEY`, which the stock Python image
defines itself (it is the public key ID used to verify the Python tarball at build time —
not a host leak). Replaced with a precise assertion that the container's environment is the
image's own plus exactly the three variables the sandbox passes, and a separate test that
sets four credential-shaped host variables and proves none reach the container.

## 18. Files created/modified

**Created:**

- `src/python_dpo/sandbox/` — `__init__.py`, `errors.py`, `config.py`, `result.py`,
  `workspace.py`, `container.py`, `executor.py`, `health.py`, `README.md`
- `tests/sandbox/` — `__init__.py`, `test_config.py`, `test_result.py`,
  `test_workspace.py`, `test_sandbox_security.py`, `test_executor_mock.py`,
  `test_sandbox_integration.py`
- `docs/sandbox-security.md`, `examples/hello.py`, `05_DOCKER_SANDBOX.md` (this file)

**Modified:**

- `src/python_dpo/config.py` — `sandbox: SandboxConfig` on `Config`, `_parse_sandbox`
- `src/python_dpo/cli.py` — the `sandbox` command group
- `src/python_dpo/__init__.py` — version `0.4.0` → `0.5.0`
- `config.yaml` — the `sandbox:` section
- `pyproject.toml` — `integration` marker, deselected by default
- `CLAUDE.md` — Security section now names `SandboxExecutor` as the sanctioned execution
  path and explains why `subprocess` in `sandbox/` is the Docker client, not the candidate
- `README.md`, `src/python_dpo/README.md`, `tests/README.md` — Stage 5 documentation
- `tests/test_project.py` — `sandbox` CLI tests

## 19. Dependencies added

**None.** The Docker CLI approach uses only `subprocess`, `threading`, `shutil`, `json`,
`tempfile` and `secrets` from the standard library. `pyproject.toml`'s `dependencies` list
is unchanged.

## 20. Deviations from the specification

- **Docker CLI rather than the SDK** (§39 permits either) — zero dependencies, and the whole
  isolation posture lives in one auditable argv.
- **`ContainerRuntime`'s methods differ from §40's literal `create/start/wait/logs/stop/
  remove` list.** An attached `docker run` fuses create, start and log attachment, which is
  what makes bounded output reading possible; §40 asks for "operations conceptually
  equivalent."
- **Workspaces live in system temp, not `data/sandbox/jobs/`** (§15 permits "an equivalent
  temporary directory"). They are always destroyed, so they are not artifacts; keeping them
  out of `data/` avoids any chance of committing them and needs no `.gitignore` change.
- **Non-root via numeric UID 65534 on the stock image**, no Dockerfile (§19 permits a
  numeric UID; §10 says install nothing extra).
- **The workspace is mounted read-only** (§17 prefers read-only); the writable area is the
  size-limited `/tmp` tmpfs.
- **`network_mode` accepts only `"none"`**, rejecting a config that would violate §12's MUST.
- **`--memory-swap` pinned equal to `--memory`**, beyond §26's literal text — without it the
  memory ceiling is roughly doubled by swap.
- **`--security-opt no-new-privileges`**, beyond §22's literal wording, enforcing its intent.
- **`startup_grace_seconds` is an addition to §53's config list**, for the measured reason in
  §3 above.
- **`cancelled` is defined in the status set but never produced** — there is no cancellation
  API in this stage. Included so the closed set is stable for later stages.
- **Integration tests are deselected by default** so `pytest -q` keeps its zero-skip
  property; `pytest -q -m integration` runs them, failing rather than skipping if Docker is
  unreachable.

## Issues for review before Stage 6

1. **Per-execution cost is ~2.3s of container startup.** Evaluating 50 candidates × N test
   runs will be dominated by it. Stage 6 should consider batching a problem's whole test
   suite into one container run rather than one container per test — the current interface
   supports that without change, but it is a design decision worth making explicitly.
2. **Stage 6 will need pytest inside the container**, which the stock image does not have.
   That is the point at which a custom image (and a `docker/` Dockerfile plus a build step
   in the health check) becomes justified.
3. **The workspace is currently mounted read-only.** If Stage 6 needs the candidate to write
   artifacts — a coverage file, a pytest report — that either moves to the `/tmp` tmpfs and
   gets collected from container stdout, or the mount becomes read-write. Worth deciding
   deliberately rather than discovering it.

Stopping here. Not starting Stage 6 (pytest evaluation) without explicit approval.
