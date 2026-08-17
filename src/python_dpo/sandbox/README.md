# src/python_dpo/sandbox/

The isolated Docker sandbox — the execution boundary for untrusted candidate code, and the
thing every earlier stage deliberately stopped short of.

This package answers **"what happened when this program ran?"** and nothing else. It never
decides whether a candidate is *correct* — `status="success"` means the program exited
zero, which a later stage will interpret by running the problem's test suite against it.

For the threat model, the full isolation inventory, and the known limitations, see
[`docs/sandbox-security.md`](../../../docs/sandbox-security.md).

## Files

### `errors.py`

The exception hierarchy: `SandboxError` and its subclasses (`SandboxConfigError`,
`DockerUnavailableError`, `ImageUnavailableError`, `ContainerCreationError`,
`ContainerExecutionError`, `SandboxTimeoutError`, `WorkspaceError`). Raw `subprocess` and
Docker failures are translated here and never propagate outward, so callers can handle
"the sandbox could not run this" without importing anything Docker-specific.

`SandboxConfigError` is deliberately **not** `python_dpo.config.ConfigError`: this package
must not import the configuration layer. `config.py` catches and re-raises, exactly as it
already does for `ModelError` from `models/base.py`, keeping the dependency one-way.

### `config.py`

`SandboxConfig` — the `sandbox:` section, validated on construction. Two validation rules
are worth calling out because they reject rather than accommodate:

- **`network_mode` accepts only `"none"`.** Network isolation is mandatory, so a config
  requesting `bridge` or `host` is an error rather than something silently honoured. Same
  reasoning as `ModelConfig.quantization` rejecting a non-null value it would have ignored.
- **`image` must be pinned** and must not be `:latest`, which can change underneath us and
  would silently change what every recorded result means. `image_digest` gives a stronger
  pin still.
- **`user` must be a non-root numeric UID** when `run_as_non_root` is set. UID 0 is refused.

`to_dict()` is the spec §52 environment record stamped onto every `ExecutionResult`.

### `result.py`

`ExecutionResult` plus `classify()`, a **pure function** over already-collected facts, so
every classification branch is unit tested without Docker.

The status set is closed: `success`, `syntax_error`, `runtime_error`, `timeout`,
`resource_exceeded`, `infrastructure_error`, `cancelled`. (`cancelled` has no producer yet —
there is no cancellation API in this stage — but is present so the set is stable for the
stages that will add one.)

Two distinctions the module exists to preserve:

- **Candidate vs infrastructure.** `is_candidate_outcome` / `is_infrastructure_failure`. A
  candidate must never be marked bad because Docker failed.
- **Compile-time vs runtime.** `looks_like_compile_error()` keys off the fact that CPython
  always prints `Traceback (most recent call last):` before a *runtime* exception and never
  before a compile failure. So a program that does `raise SyntaxError("x")` correctly
  classifies as `runtime_error` — it compiled fine and then chose to raise.

### `workspace.py`

`SandboxWorkspace` — a context manager creating one temporary job directory per execution,
writing `candidate.py`, and removing the directory in `__exit__` regardless of how the block
was left. It never executes anything.

One non-obvious detail: `tempfile.mkdtemp` creates `0o700`, which the container's non-root
UID cannot traverse, so the workspace explicitly chmods to `0o755`/`0o644`. That looks like a
security loosening and isn't — the directory holds only the candidate's own source, lives
under a private temporary root, is mounted read-only, and is destroyed after the run.

Source is written as **bytes**, not via `write_text`, because text mode applies newline
translation and a candidate containing `\r\n` would otherwise not reach the container
byte-for-byte as the model produced it.

### `container.py`

`ContainerSpec.to_docker_args()` is **the entire security surface of this project**. Every
isolation guarantee is one flag in the list it builds, each annotated with the specification
section requiring it. `tests/sandbox/test_sandbox_security.py` asserts both halves of the
contract — what must be present, and what must never appear.

`ContainerRuntime` is a `Protocol` (matching `ModelClient` and `ReferenceExecutor`), so the
executor is driven by a fake in unit tests. `DockerContainerRuntime` is the only code in the
project that talks to Docker, via `subprocess` with a fixed argv and `shell=False`.

`BoundedReader` caps each stream at `max_output_bytes` and then sets an `Event`. It
deliberately **never touches the container**: terminating from a reader thread would mean
calling `wait`/`kill` concurrently with the main thread's `wait` on the same process, which
hangs and leaks the container. The reader only raises a flag; the executor does the killing.

`--rm` is deliberately not used: the container must survive long enough to be inspected for
`OOMKilled`, its exit code, and its ID.

### `executor.py`

`SandboxExecutor.execute(code)` — workspace → spec → runtime → bounded output → classify →
`ExecutionResult`. Never raises for a candidate-caused failure (a crash or timeout is a
*result*), and returns infrastructure problems as `status="infrastructure_error"` rather
than propagating, so a caller looping over candidates cannot be derailed by a transient
Docker fault.

`_await_exit` polls rather than issuing one long blocking wait, because the output limit can
be tripped at any moment and the container must be killed promptly when it is. All container
control happens on that one thread.

Cleanup is unconditional: container removed in a `finally`, workspace removed by the context
manager.

### `health.py`

`check_sandbox_health()` runs the six documented checks — docker binary, daemon reachable,
image available (pulled when permitted), container starts, Python actually runs, cleanup
succeeded — stopping at the first failure, because four more messages saying "the daemon is
down" help nobody. `format_health_report()` renders the result as sentences, never a
traceback.
