"""The container runtime seam and its Docker CLI implementation (spec 05 sections 39, 40).

:meth:`ContainerSpec.to_docker_args` is **the entire security surface of this project**.
Every isolation guarantee the sandbox makes — no network, no host filesystem, non-root,
no capabilities, bounded CPU/memory/PIDs — is one flag in the list that method builds, and
``tests/sandbox/test_sandbox_security.py`` asserts both what must be present and what must
never appear. Reading that one method tells you exactly how isolated a candidate is.

Candidate source is never part of the command. It is written to ``candidate.py`` in the
workspace and executed by a fixed argv (spec sections 42, 43, 47), so there is no shell,
no quoting, and no injection surface. Every ``subprocess`` call here passes ``shell=False``
and a list — never a string.

:class:`ContainerRuntime` is a ``Protocol``, matching how ``ModelClient`` and
``ReferenceExecutor`` are already defined, so the executor can be driven by a fake in unit
tests without Docker (spec section 58).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Protocol

from .config import SandboxConfig
from .errors import (
    ContainerCreationError,
    ContainerExecutionError,
    DockerUnavailableError,
    ImageUnavailableError,
)
from .workspace import CONTAINER_WORKSPACE

DOCKER_BINARY = "docker"
CONTAINER_NAME_PREFIX = "python-dpo-sandbox"

# The only environment the container receives (spec section 33). The host environment is
# never passed through, so no cloud credential, API key, or Hugging Face token can reach
# candidate code.
#
#   PYTHONUNBUFFERED     - without it CPython block-buffers stdout when it is a pipe, and a
#                          candidate killed by the timeout would lose everything it printed.
#                          Spec section 29 requires capturing available output on timeout.
#   PYTHONDONTWRITEBYTECODE - the workspace is mounted read-only; without this CPython
#                          tries (and noisily fails) to write __pycache__ next to the source.
#   HOME                 - UID 65534's home is /nonexistent; point it at the writable tmpfs.
BASE_ENVIRONMENT: dict[str, str] = {
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "HOME": "/tmp",
}

# Timeouts for the host-side docker CLI calls themselves — not for candidate code.
_DOCKER_CALL_TIMEOUT = 30
_DOCKER_PULL_TIMEOUT = 600


def container_name(job_id: str, run_id: str | None = None) -> str:
    """``python-dpo-sandbox-<run_id>-<job_id>`` (spec section 37).

    Deterministic enough to trace a stray container back to the execution that made it,
    and never contains candidate source.
    """
    parts = [CONTAINER_NAME_PREFIX]
    if run_id:
        parts.append(run_id)
    parts.append(job_id)
    return "-".join(parts)


@dataclass(frozen=True)
class ContainerSpec:
    """Everything needed to run one container, and nothing else."""

    name: str
    image: str
    command: tuple[str, ...]
    workspace_path: Path
    config: SandboxConfig
    environment: dict[str, str] = field(default_factory=lambda: dict(BASE_ENVIRONMENT))

    def to_docker_args(self) -> list[str]:
        """Build the full ``docker run`` argv.

        Each flag is annotated with the specification section that requires it. Nothing
        here is optional decoration — removing any one of them weakens the boundary.
        """
        config = self.config
        args = [
            DOCKER_BINARY,
            "run",
            # Attached, so stdout/stderr arrive on live pipes we can bound as they stream
            # (spec section 31). Deliberately NOT --rm: the container must survive long
            # enough to be inspected for OOMKilled/ExitCode/Id (spec sections 38, 65), and
            # removal is an explicit step in the caller's finally block instead.
            "--name",
            self.name,
            # Section 12: no network at all. Not a firewall rule, not an application-level
            # check — the container simply has no network stack beyond loopback.
            "--network",
            config.network_mode,
            # Section 24: a candidate cannot fork-bomb the host.
            "--pids-limit",
            str(config.pids_limit),
            # Section 25/26: CPU and memory ceilings enforced by the container runtime, not
            # by a Python-level check. --memory-swap pinned equal to --memory stops the
            # container from using roughly twice its limit via swap.
            "--cpus",
            str(config.cpus),
            "--memory",
            config.memory,
            "--memory-swap",
            config.memory,
        ]

        if config.drop_capabilities:
            # Section 21: generated Python needs no Linux capabilities whatsoever.
            args += ["--cap-drop", "ALL"]

        # Section 22's intent: even a setuid binary inside the image cannot gain privileges.
        args += ["--security-opt", "no-new-privileges"]

        if config.run_as_non_root:
            # Section 19/20: candidate code must not be uid 0.
            args += ["--user", config.user]

        if config.read_only_root:
            # Section 18: the container's root filesystem is immutable, with a single
            # size-limited writable tmpfs for the temporary files CPython may want.
            # noexec/nosuid stop that writable space being used to stage a binary.
            args += [
                "--read-only",
                "--tmpfs",
                f"/tmp:rw,noexec,nosuid,size={config.tmpfs_size}",
            ]

        for key, value in sorted(self.environment.items()):
            args += ["--env", f"{key}={value}"]

        # Sections 14/17: the ONLY host path the container sees is this one job directory,
        # mounted read-only. The project directory, home directory, /tmp, .git, SSH keys and
        # cloud credentials are never mounted, and neither is /var/run/docker.sock (§35).
        args += [
            "--volume",
            f"{self.workspace_path}:{CONTAINER_WORKSPACE}:ro",
            # Section 44: start in the controlled workspace, never a host project directory.
            "--workdir",
            CONTAINER_WORKSPACE,
            self.image,
        ]

        # Section 43/47: a fixed executable and argument list. The candidate controls only
        # the *contents* of candidate.py, never the command that runs it.
        args += list(self.command)
        return args


class StartedContainer(Protocol):
    """A container that is running, with live output streams."""

    @property
    def name(self) -> str: ...

    @property
    def stdout(self) -> IO[bytes] | None: ...

    @property
    def stderr(self) -> IO[bytes] | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


class ContainerRuntime(Protocol):
    """The interface the sandbox executor programs against.

    Deliberately shaped around what an attached, streaming execution needs rather than
    spec section 40's literal ``create/start/wait/logs/stop/remove`` list — an attached
    ``docker run`` fuses create, start and log attachment into one call, which is what makes
    bounded output reading possible. Section 40 asks for "operations conceptually
    equivalent", which this is.
    """

    def check_available(self) -> None:
        """Raise :class:`DockerUnavailableError` unless Docker is usable."""

    def image_present(self, image: str) -> bool: ...

    def pull(self, image: str) -> None: ...

    def start(self, spec: ContainerSpec) -> StartedContainer: ...

    def inspect(self, name: str) -> dict[str, Any]: ...

    def remove(self, name: str) -> None: ...


class _DockerProcess:
    """A ``docker run`` subprocess, presented as a StartedContainer."""

    def __init__(self, name: str, process: subprocess.Popen[bytes]) -> None:
        self._name = name
        self._process = process

    @property
    def name(self) -> str:
        return self._name

    @property
    def stdout(self) -> IO[bytes] | None:
        return self._process.stdout

    @property
    def stderr(self) -> IO[bytes] | None:
        return self._process.stderr

    def wait(self, timeout: float | None = None) -> int:
        return self._process.wait(timeout=timeout)

    def kill(self) -> None:
        """Stop the container, then the client process attached to it.

        ``docker kill`` targets the container itself — killing only the local ``docker run``
        process would detach the client and leave the container running, which spec
        section 29 forbids ("Do not leave orphan containers").
        """
        _run_docker(["kill", self._name], check=False)
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()


def _run_docker(
    args: list[str],
    *,
    check: bool = True,
    timeout: int = _DOCKER_CALL_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    """Run a host-side ``docker`` command with a fixed argument list.

    ``shell=False`` (the default) is load-bearing and never overridden anywhere in this
    package: no string is ever handed to a shell for interpretation.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, shell=False, no candidate input
            [DOCKER_BINARY, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise DockerUnavailableError(
            "the 'docker' command was not found on PATH; install Docker to use the sandbox"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DockerUnavailableError(
            f"docker {' '.join(args)} did not respond within {timeout}s"
        ) from exc

    if check and completed.returncode != 0:
        raise ContainerExecutionError(
            f"docker {' '.join(args)} failed ({completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed


class DockerContainerRuntime:
    """Drives the Docker CLI. The only place in the project that talks to Docker."""

    def check_available(self) -> None:
        if shutil.which(DOCKER_BINARY) is None:
            raise DockerUnavailableError(
                "the 'docker' command was not found on PATH; install Docker to use the sandbox"
            )
        completed = _run_docker(["info", "--format", "{{.ServerVersion}}"], check=False)
        if completed.returncode != 0:
            raise DockerUnavailableError(
                "the Docker daemon is not reachable; start Docker and ensure your user "
                "can access it"
            )

    def server_version(self) -> str:
        return _run_docker(["info", "--format", "{{.ServerVersion}}"]).stdout.strip()

    def image_present(self, image: str) -> bool:
        completed = _run_docker(["image", "inspect", image], check=False)
        return completed.returncode == 0

    def image_digest(self, image: str) -> str | None:
        completed = _run_docker(
            ["image", "inspect", "--format", "{{index .RepoDigests 0}}", image], check=False
        )
        if completed.returncode != 0:
            return None
        digest = completed.stdout.strip()
        return digest.split("@", 1)[1] if "@" in digest else None

    def pull(self, image: str) -> None:
        completed = _run_docker(["pull", image], check=False, timeout=_DOCKER_PULL_TIMEOUT)
        if completed.returncode != 0:
            raise ImageUnavailableError(
                f"could not pull image {image!r}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )

    def start(self, spec: ContainerSpec) -> StartedContainer:
        args = spec.to_docker_args()
        try:
            process = subprocess.Popen(  # noqa: S603 - fixed argv, shell=False; see module docstring
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
            )
        except FileNotFoundError as exc:
            raise DockerUnavailableError(
                "the 'docker' command was not found on PATH; install Docker to use the sandbox"
            ) from exc
        except OSError as exc:
            raise ContainerCreationError(f"could not start container {spec.name}: {exc}") from exc
        return _DockerProcess(spec.name, process)

    def inspect(self, name: str) -> dict[str, Any]:
        """Container state after it has exited: exit code, OOM flag, and full id.

        Returns ``{}`` when the container no longer exists, which is not an error — an
        earlier cleanup may legitimately have removed it.
        """
        completed = _run_docker(["inspect", name], check=False)
        if completed.returncode != 0:
            return {}
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, list) or not payload:
            return {}
        entry = payload[0]
        state = entry.get("State", {}) if isinstance(entry, dict) else {}
        return {
            "id": entry.get("Id"),
            "exit_code": state.get("ExitCode"),
            "oom_killed": bool(state.get("OOMKilled", False)),
            "status": state.get("Status"),
        }

    def remove(self, name: str) -> None:
        """Force-remove the container. Never raises — cleanup must not mask a real result."""
        _run_docker(["rm", "--force", name], check=False)

    def list_sandbox_containers(self) -> list[str]:
        """Names of any lingering sandbox containers — used by cleanup verification."""
        completed = _run_docker(
            ["ps", "--all", "--filter", f"name={CONTAINER_NAME_PREFIX}", "--format", "{{.Names}}"],
            check=False,
        )
        if completed.returncode != 0:
            return []
        return [line for line in completed.stdout.splitlines() if line.strip()]


class BoundedReader:
    """Reads a stream in a thread, keeping at most ``limit`` bytes (spec sections 31, 32).

    A candidate that prints forever must not grow an unbounded in-memory string, which spec
    section 31 forbids in as many words. Once the limit is hit the reader stops storing,
    flags truncation, and sets ``limit_reached``; it keeps draining and discarding so the
    container never blocks writing into a full pipe while it is being torn down.

    The reader deliberately **never touches the container**. Terminating it from this thread
    would mean calling ``Popen.wait``/``docker kill`` concurrently with the main thread's own
    ``wait`` on the same process, which is not safe and in practice hangs and leaks the
    container. The reader only raises a flag; the executor's main loop watches
    ``limit_reached`` and does all the killing.
    """

    def __init__(
        self,
        stream: IO[bytes] | None,
        limit: int,
        limit_reached: threading.Event | None = None,
    ) -> None:
        self._stream = stream
        self._limit = limit
        self.limit_reached = limit_reached if limit_reached is not None else threading.Event()
        self._chunks: list[bytes] = []
        self._size = 0
        self.truncated = False
        self._thread = threading.Thread(target=self._read, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    @property
    def text(self) -> str:
        return b"".join(self._chunks).decode("utf-8", errors="replace")

    def _read(self) -> None:
        if self._stream is None:
            return
        try:
            while True:
                chunk = self._stream.read(8192)
                if not chunk:
                    break
                if self._size < self._limit:
                    remaining = self._limit - self._size
                    self._chunks.append(chunk[:remaining])
                    self._size += min(len(chunk), remaining)
                    if self._size >= self._limit and not self.truncated:
                        self.truncated = True
                        self.limit_reached.set()
                # Past the limit we keep reading and discarding so the writer is never
                # blocked on a full pipe while the container is being torn down.
        except (OSError, ValueError):
            # The pipe was closed underneath us, which is normal when the container is
            # killed mid-read.
            return


__all__ = [
    "BASE_ENVIRONMENT",
    "CONTAINER_NAME_PREFIX",
    "DOCKER_BINARY",
    "BoundedReader",
    "ContainerRuntime",
    "ContainerSpec",
    "DockerContainerRuntime",
    "StartedContainer",
    "container_name",
]
