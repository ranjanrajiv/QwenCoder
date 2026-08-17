"""The sandbox's primary interface (spec 05 sections 6, 36).

    candidate code -> SandboxWorkspace -> ContainerSpec -> ContainerRuntime
                   -> bounded stdout/stderr -> classify -> ExecutionResult

The caller hands over source text and receives a structured
:class:`~python_dpo.sandbox.result.ExecutionResult`. It never executes the candidate
itself, never sees a shell, and never learns anything Docker-specific — an
infrastructure failure arrives as ``status="infrastructure_error"``, not as a Docker
exception (spec sections 78, 81).

Cleanup is unconditional (spec section 16): the container is removed and the workspace
deleted whether the candidate succeeded, crashed, timed out, flooded its output, or Docker
failed outright. One candidate runs at a time (spec section 77 defers parallelism).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .config import SandboxConfig
from .container import (
    BASE_ENVIRONMENT,
    BoundedReader,
    ContainerRuntime,
    ContainerSpec,
    DockerContainerRuntime,
    container_name,
)
from .errors import SandboxError
from .result import ExecutionResult, classify, signal_from_exit_code
from .workspace import CANDIDATE_FILENAME, CONTAINER_WORKSPACE, SandboxWorkspace, new_job_id

logger = logging.getLogger("python_dpo.sandbox")

# Spec sections 43/47: a fixed command. The candidate controls the contents of
# candidate.py and nothing about how it is invoked.
EXECUTION_COMMAND: tuple[str, ...] = ("python", f"{CONTAINER_WORKSPACE}/{CANDIDATE_FILENAME}")

# Grace period for the reader threads to drain after the container exits.
_DRAIN_TIMEOUT = 5.0
# How often the wait loop re-checks the output-limit flag and the deadline. Short enough
# that a flooding candidate is terminated promptly, long enough to cost nothing.
_POLL_INTERVAL = 0.1


class SandboxExecutor:
    """Executes untrusted Python inside an isolated container."""

    def __init__(
        self,
        *,
        config: SandboxConfig | None = None,
        runtime: ContainerRuntime | None = None,
    ) -> None:
        self._config = config or SandboxConfig()
        self._runtime = runtime if runtime is not None else DockerContainerRuntime()

    @property
    def config(self) -> SandboxConfig:
        return self._config

    def execute(
        self,
        code: str,
        *,
        job_id: str | None = None,
        run_id: str | None = None,
        timeout_seconds: int | None = None,
    ) -> ExecutionResult:
        """Run ``code`` in the sandbox and report what happened.

        Never raises for a candidate-caused failure — a crash, a timeout, or a resource
        violation is a *result*, not an exception. Infrastructure problems are likewise
        returned as ``status="infrastructure_error"`` rather than propagated, so a caller
        looping over candidates cannot be derailed by a transient Docker fault.
        """
        config = self._config
        timeout = timeout_seconds if timeout_seconds is not None else config.timeout_seconds
        job_id = job_id or new_job_id()
        name = container_name(job_id, run_id)
        config_record = config.to_dict()
        started = time.monotonic()

        try:
            with SandboxWorkspace(job_id=job_id, root=config.workspace_root) as workspace:
                workspace.write_candidate(code)
                return self._run(
                    workspace=workspace,
                    name=name,
                    timeout=timeout,
                    started=started,
                    config_record=config_record,
                )
        except SandboxError as exc:
            # Docker unavailable, image missing, workspace failure: the candidate never
            # really ran, so it must not be judged on this (spec sections 79, 80, 81).
            logger.error("Sandbox infrastructure failure for job %s: %s", job_id, exc)
            return ExecutionResult.infrastructure_failure(
                error_type=type(exc).__name__,
                error_message=str(exc),
                duration_ms=_elapsed_ms(started),
                sandbox_config=config_record,
                workspace_id=job_id,
            )

    def _run(
        self,
        *,
        workspace: SandboxWorkspace,
        name: str,
        timeout: int,
        started: float,
        config_record: dict[str, Any],
    ) -> ExecutionResult:
        config = self._config
        spec = ContainerSpec(
            name=name,
            image=config.image_reference,
            command=EXECUTION_COMMAND,
            workspace_path=workspace.path,
            config=config,
            environment=dict(BASE_ENVIRONMENT),
        )

        container = None
        timed_out = False
        exit_code: int | None = None

        try:
            container = self._runtime.start(spec)

            # Bounded from the first byte: an output flood is capped rather than accumulated
            # (spec sections 31, 32, 75). Both streams share one event so either can trip
            # the limit; the readers only raise the flag, and this thread does the killing.
            limit_reached = threading.Event()
            stdout_reader = BoundedReader(
                container.stdout, config.max_output_bytes, limit_reached=limit_reached
            )
            stderr_reader = BoundedReader(
                container.stderr, config.max_output_bytes, limit_reached=limit_reached
            )
            stdout_reader.start()
            stderr_reader.start()

            # The candidate gets `timeout`; the grace period covers container creation and
            # interpreter startup, which cost ~2s even on a warm image and are not the
            # candidate's doing. Without it a 5s timeout would really give a candidate under
            # 3s, and a loaded machine could time out a merely slow program.
            wait_budget = timeout + config.startup_grace_seconds
            exit_code, timed_out = self._await_exit(
                container, wait_budget=wait_budget, limit_reached=limit_reached
            )
            if timed_out:
                logger.warning(
                    "Container %s exceeded %ss (%ss candidate timeout + %ss startup grace); "
                    "terminating",
                    name,
                    wait_budget,
                    timeout,
                    config.startup_grace_seconds,
                )
            elif limit_reached.is_set():
                logger.warning(
                    "Container %s exceeded %d bytes of output; terminating",
                    name,
                    config.max_output_bytes,
                )

            stdout_reader.join(timeout=_DRAIN_TIMEOUT)
            stderr_reader.join(timeout=_DRAIN_TIMEOUT)

            state = self._runtime.inspect(name)
            container_id = state.get("id")
            oom_killed = bool(state.get("oom_killed", False))
            # The container's own exit code is more truthful than the docker client's when
            # the container was killed: the client may report its own signal instead.
            inspected_exit = state.get("exit_code")
            if isinstance(inspected_exit, int) and (timed_out or exit_code is None):
                exit_code = inspected_exit

            output_limit_exceeded = stdout_reader.truncated or stderr_reader.truncated
            status = classify(
                exit_code=exit_code,
                stderr=stderr_reader.text,
                timed_out=timed_out,
                oom_killed=oom_killed,
                output_limit_exceeded=output_limit_exceeded,
                container_started=True,
            )

            return ExecutionResult(
                status=status,
                exit_code=exit_code,
                stdout=stdout_reader.text,
                stderr=stderr_reader.text,
                duration_ms=_elapsed_ms(started),
                timed_out=timed_out,
                container_id=container_id,
                signal=signal_from_exit_code(exit_code),
                memory_limit_exceeded=oom_killed,
                stdout_truncated=stdout_reader.truncated,
                stderr_truncated=stderr_reader.truncated,
                network_blocked=config.network_mode == "none",
                workspace_id=workspace.job_id,
                sandbox_config=config_record,
            )
        finally:
            # Spec sections 16, 36, 76: the container is removed on every path out of this
            # method, so `docker ps -a` never accumulates abandoned sandbox containers.
            # The workspace is removed by the context manager in execute().
            if container is not None:
                self._runtime.remove(name)


    def _await_exit(
        self,
        container: Any,
        *,
        wait_budget: float,
        limit_reached: threading.Event,
    ) -> tuple[int | None, bool]:
        """Wait for the container, terminating it on a timeout or an output flood.

        Polls rather than issuing one long blocking wait, because the output limit can be
        tripped at any moment by a reader thread and the container must be killed promptly
        when it is. **All container control happens here, on one thread** — calling
        ``wait``/``kill`` from a reader thread as well would mean concurrent waits on the
        same process, which hangs and leaks the container.

        Returns ``(exit_code, timed_out)``.
        """
        deadline = time.monotonic() + wait_budget

        while True:
            try:
                return container.wait(timeout=_POLL_INTERVAL), False
            except Exception as exc:  # subprocess.TimeoutExpired and friends
                if type(exc).__name__ != "TimeoutExpired":
                    raise

            hit_output_limit = limit_reached.is_set()
            expired = time.monotonic() >= deadline
            if not hit_output_limit and not expired:
                continue

            # Spec section 29: terminate the container, keep whatever output arrived, and
            # never leave it running.
            container.kill()
            try:
                exit_code = container.wait(timeout=_DRAIN_TIMEOUT)
            except Exception:  # noqa: BLE001 - already terminating; inspect fills the gap
                exit_code = None
            # An output flood is a resource violation, not a timeout: classify() ranks
            # resource_exceeded above timeout, and reporting timed_out here would contradict
            # that. Only a genuine deadline expiry sets timed_out.
            return exit_code, expired and not hit_output_limit


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


__all__ = ["EXECUTION_COMMAND", "SandboxExecutor"]
