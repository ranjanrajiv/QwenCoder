# Sandbox Security

How `python_dpo.sandbox` isolates untrusted, model-generated Python, what that protects
against, and — just as importantly — what it does not.

> **Docker isolation reduces risk but should not be treated as a perfect security boundary
> for arbitrary hostile code.**

This project is a research and data-generation environment, not a production multi-tenant
code execution service. The distinction matters and is expanded on under
[Known limitations](#known-limitations).

## 1. Threat model

**What we are defending against.** Candidate programs are written by a language model, not
by an attacker who has studied this sandbox. The realistic failure modes are:

| Threat | Realistic form |
|---|---|
| Accidental host damage | A candidate writes to or deletes files it believes are local |
| Resource exhaustion | An infinite loop, a runaway allocation, a fork bomb, an output flood |
| Credential exfiltration | Code that reads `os.environ` or `~/.aws` and prints what it finds |
| Network egress | A candidate that "helpfully" downloads a package or calls an API |
| Cross-contamination | One candidate leaving state that changes another's result |

**What we are not defending against.** A determined human attacker with a container-escape
exploit, a kernel 0-day, or a malicious Docker image. Nothing in Stage 5 claims to stop
that; see [Known limitations](#known-limitations).

**Trust boundary.** Everything in `src/python_dpo/` is trusted, reviewed code. Everything
in a `candidate.py` is untrusted. The boundary between them is the container.

## 2. Isolation boundaries

The single security surface is `ContainerSpec.to_docker_args()` in
`src/python_dpo/sandbox/container.py`. Every guarantee below is one flag in the list that
method builds, which is why `tests/sandbox/test_sandbox_security.py` asserts both what must
be present and what must never appear. To audit the sandbox, read that one method.

Candidate source **never appears in a command**. It is written to `candidate.py` inside a
temporary workspace and executed by a fixed argv:

```
python /workspace/candidate.py
```

There is no shell anywhere in the path — every `subprocess` call in the package passes a
list with `shell=False` — so quoting and injection are not merely handled, they are absent
by construction. The candidate controls the *contents* of one file and nothing else.

## 3. Network isolation

Containers run with `--network none`. The container gets no network interface beyond
loopback, so this is not a firewall rule that a candidate could route around:

- no internet access
- no access to host services or `localhost`
- no DNS resolution
- no outbound connections to any IP

`sandbox.network_mode` accepts **only** `none`. A configuration requesting `bridge` or
`host` is rejected at load time rather than honoured, because silently accepting it would
mean a config file could disable the guarantee.

Verified by `test_outbound_connections_fail`, `test_dns_resolution_fails`, and
`test_http_requests_fail`.

## 4. Filesystem isolation

The container sees exactly one host path: its own job workspace, mounted **read-only** at
`/workspace`. Never mounted:

- the project directory, or any parent of it
- `/`, `/home`, `/tmp` from the host
- `.git`, SSH keys, cloud credentials, `~/.aws`, `~/.config`
- **`/var/run/docker.sock`** — a container with the Docker socket can control the host
  daemon, which would defeat the entire boundary

The root filesystem is mounted `--read-only`. Because CPython wants somewhere to write,
one controlled `tmpfs` is provided at `/tmp` with an explicit size limit and
`noexec,nosuid`, so it cannot be used to stage and run a binary.

Each execution gets a fresh workspace that is destroyed afterwards, so no state carries
between candidates.

Verified by `test_host_project_files_are_not_reachable`,
`test_the_project_directory_is_not_mounted`, `test_the_workspace_is_mounted_read_only`, and
`test_docker_socket_is_not_present`.

## 5. Resource limits

Enforced by the container runtime, not by application-level checks a candidate could
outlast:

| Limit | Flag | Default |
|---|---|---|
| CPU | `--cpus` | 1.0 |
| Memory | `--memory` + `--memory-swap` | 512m |
| Processes/threads | `--pids-limit` | 64 |
| Wall clock | host-side kill | 5s + 10s startup grace |
| Output | bounded reader | 1,000,000 bytes |
| `/tmp` size | `--tmpfs size=` | 64m |

`--memory-swap` is pinned equal to `--memory`. Without it a container may use roughly twice
its memory limit via swap, quietly defeating the ceiling — this goes beyond the
specification's literal text and is deliberate.

**Timeout.** The candidate's budget (`timeout_seconds`) is separate from
`startup_grace_seconds`, which covers container creation and interpreter start. Those cost
~2s even on a warm image and are not the candidate's doing; without a separate allowance a
5s timeout would really give a candidate under 3s, and a loaded machine could time out a
program that is merely slow rather than wrong.

**Output.** Reader threads keep at most `max_output_bytes` per stream and then flag the
limit; the main loop terminates the container. Truncation is always recorded
(`stdout_truncated`/`stderr_truncated`), never silent. The readers deliberately never touch
the container themselves — calling `wait`/`kill` from a reader thread while the main thread
waits on the same process hangs and leaks the container, which is a bug this project shipped
briefly and fixed.

Verified by `test_memory_limit_is_enforced`, `test_process_limit_is_enforced`,
`test_infinite_loop_times_out_and_is_terminated`, and
`test_output_flood_is_bounded_and_terminated`.

## 6. Non-root execution

Containers run as `--user 65534:65534` (`nobody:nogroup`, already present in Debian-derived
images), so no custom image is needed. A configuration naming UID 0 is rejected at load
time.

Verified by `test_candidate_does_not_run_as_root` and
`test_candidate_cannot_write_to_the_read_only_root_filesystem`.

## 7. Capability restrictions

- `--cap-drop ALL` — generated Python needs no Linux capabilities whatsoever.
- `--security-opt no-new-privileges` — even a setuid binary inside the image cannot gain
  privileges. This goes beyond the specification's literal text and enforces its intent.
- **Never** `--privileged`.
- **Never** `--pid=host`, `--network=host`, `--ipc=host`, or `--uts=host`; the container
  uses isolated namespaces.

Asserted at the argv level in `test_dangerous_flags_are_never_present`.

## 8. Environment isolation

The host environment is **never** passed through. The container receives exactly three
variables, all set deliberately:

| Variable | Why |
|---|---|
| `PYTHONUNBUFFERED=1` | Without it CPython block-buffers a piped stdout, so a candidate killed by the timeout would lose everything it printed |
| `PYTHONDONTWRITEBYTECODE=1` | The workspace is read-only; otherwise CPython noisily fails writing `__pycache__` |
| `HOME=/tmp` | UID 65534's home is `/nonexistent`; point it at the writable tmpfs |

No cloud credential, API key, Hugging Face token, SSH credential, or database password can
reach candidate code, because none of them are passed. `PYTHONPATH` is not propagated
either, so the container uses its own Python environment rather than the host's — which
also matters for reproducibility.

Verified by `test_container_environment_is_only_the_image_plus_our_three_variables` and
`test_host_credential_variables_never_reach_the_container`.

## 9. Container and workspace cleanup

Every execution follows create → start → monitor → collect → stop → remove. The container is
removed in a `finally` block and the workspace in a context manager's `__exit__`, so both
happen whether the candidate succeeded, crashed, timed out, flooded its output, or Docker
itself failed.

Containers are named `python-dpo-sandbox-<run_id>-<job_id>` — traceable for debugging, and
never containing candidate source. Every integration test asserts, via an autouse fixture,
that no sandbox container survives it.

Note that `--rm` is deliberately **not** used: the container must survive long enough to be
inspected for `OOMKilled`, its exit code, and its ID. Removal is an explicit step instead,
which also makes the lifecycle auditable.

## 10. What a result does and does not mean

`status = "success"` means **the program exited zero**. It is not a claim that the candidate
is correct — deciding that requires running the problem's test suite, which is a later
stage's job.

The sandbox also distinguishes two categories that must never be conflated:

- **Candidate outcomes** — `syntax_error`, `runtime_error`, `timeout`, `resource_exceeded`.
  The candidate ran, and this is what it did.
- **Infrastructure outcomes** — `infrastructure_error`. Docker was unavailable, the image
  was missing, the container could not be created. **A candidate is never judged badly
  because Docker failed.**

## Known limitations

Docker is a strong boundary against accidents and a moderate one against hostility. It is
not a security boundary equivalent to a virtual machine.

- **Shared kernel.** Containers share the host kernel. A kernel exploit or a container
  escape defeats this isolation entirely.
- **No user namespace remapping.** UID 65534 inside the container is UID 65534 on the host
  unless the daemon is configured with `userns-remap`. Enabling that is a host-level
  decision outside this project's scope.
- **No seccomp hardening beyond Docker's default profile.** The default profile is
  meaningful but not tailored to this workload.
- **The daemon runs as root.** Anyone who can reach the Docker socket on the host can
  control it; the sandbox protects the host *from candidates*, not from a user who already
  has Docker access.
- **Side channels are not addressed.** Timing, CPU cache, and shared-filesystem side
  channels are out of scope.
- **Resource limits are not perfectly precise.** Memory accounting, OOM timing, and CPU
  quota enforcement vary by kernel, cgroup version, and load.

For a production service executing genuinely hostile code, stronger isolation would be
required — microVMs (Firecracker), gVisor, dedicated single-tenant worker hosts, or a
hardened container runtime. **None of those are implemented here**, and adopting one would
be a deliberate future decision rather than a tweak to this stage.

## Auditing this yourself

```bash
# The whole security surface, one method:
sed -n '/def to_docker_args/,/return args/p' src/python_dpo/sandbox/container.py

# The guarantees, asserted (no Docker needed):
pytest -q tests/sandbox/test_sandbox_security.py

# The guarantees, demonstrated against a real daemon:
pytest -q -m integration

# No shell anywhere, and no host-side execution of candidate code:
grep -rn "shell=True" src/
grep -rn "os.environ" src/python_dpo/sandbox/
grep -rnE "\b(exec|eval)\(" src/          # only InProcessReferenceExecutor, for reference solutions
```
