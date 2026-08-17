"""Executor tests driven by a fake ContainerRuntime — no Docker required (spec 05 §58).

Everything about the executor's behaviour that does not need a real daemon is pinned here:
status classification end to end, timeout handling, output bounding, and — most
importantly — that cleanup happens on *every* path out of an execution (spec section 16).
"""

from __future__ import annotations

import io
import subprocess
import time
from typing import Any

import pytest

from python_dpo.sandbox import (
    ContainerSpec,
    DockerUnavailableError,
    ExecutionResult,
    SandboxConfig,
    SandboxExecutor,
)
from python_dpo.sandbox.errors import ContainerCreationError

SYNTAX_STDERR = b'  File "/workspace/candidate.py", line 1\n    def broken(:\nSyntaxError: invalid syntax\n'
RUNTIME_STDERR = b'Traceback (most recent call last):\n  File "/workspace/candidate.py", line 1\nValueError: boom\n'


class FakeContainer:
    """A StartedContainer whose behaviour each test scripts directly."""

    def __init__(
        self,
        name: str,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        exit_code: int = 0,
        hang: bool = False,
    ) -> None:
        self._name = name
        self._stdout = io.BytesIO(stdout)
        self._stderr = io.BytesIO(stderr)
        self._exit_code = exit_code
        self._hang = hang
        self.killed = False
        self.wait_calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def stdout(self):
        return self._stdout

    @property
    def stderr(self):
        return self._stderr

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        # A hanging container keeps reporting a timeout until it is killed, at which point
        # it reports its exit code. The sleep mirrors a real blocking wait: without it the
        # executor's poll loop would spin, burning the whole budget on CPU instead of time.
        if self._hang and not self.killed:
            time.sleep(min(timeout or 0, 0.05))
            raise subprocess.TimeoutExpired(cmd="docker run", timeout=timeout or 0)
        return self._exit_code

    def kill(self) -> None:
        self.killed = True


class FakeContainerRuntime:
    """Records what the executor asked Docker to do."""

    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        exit_code: int = 0,
        hang: bool = False,
        oom_killed: bool = False,
        inspect_exit_code: int | None = None,
        start_error: Exception | None = None,
        available_error: Exception | None = None,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._exit_code = exit_code
        self._hang = hang
        self._oom_killed = oom_killed
        self._inspect_exit_code = inspect_exit_code
        self._start_error = start_error
        self._available_error = available_error

        self.started_specs: list[ContainerSpec] = []
        self.removed: list[str] = []
        self.containers: list[FakeContainer] = []
        self.pulled: list[str] = []
        # Snapshotted inside start(), since the workspace is deleted before execute()
        # returns — checking directory contents afterward would always see nothing.
        self.workspace_files_at_start: list[list[str]] = []

    def check_available(self) -> None:
        if self._available_error is not None:
            raise self._available_error

    def image_present(self, image: str) -> bool:
        return True

    def pull(self, image: str) -> None:
        self.pulled.append(image)

    def start(self, spec: ContainerSpec) -> FakeContainer:
        if self._start_error is not None:
            raise self._start_error
        self.workspace_files_at_start.append(sorted(p.name for p in spec.workspace_path.iterdir()))
        self.started_specs.append(spec)
        container = FakeContainer(
            spec.name,
            stdout=self._stdout,
            stderr=self._stderr,
            exit_code=self._exit_code,
            hang=self._hang,
        )
        self.containers.append(container)
        return container

    def inspect(self, name: str) -> dict[str, Any]:
        exit_code = (
            self._inspect_exit_code if self._inspect_exit_code is not None else self._exit_code
        )
        return {
            "id": f"container-id-for-{name}",
            "exit_code": exit_code,
            "oom_killed": self._oom_killed,
            "status": "exited",
        }

    def remove(self, name: str) -> None:
        self.removed.append(name)


def make_executor(**runtime_kwargs: Any) -> tuple[SandboxExecutor, FakeContainerRuntime]:
    runtime = FakeContainerRuntime(**runtime_kwargs)
    executor = SandboxExecutor(
        config=SandboxConfig(timeout_seconds=1, startup_grace_seconds=1), runtime=runtime
    )
    return executor, runtime


# ------------------------------------------------------------------------ happy path


def test_successful_execution(tmp_path):
    executor, runtime = make_executor(stdout=b"hello\n", exit_code=0)
    result = executor.execute('print("hello")')

    assert result.status == "success"
    assert result.stdout == "hello\n"
    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.container_id
    assert result.duration_ms >= 0
    assert result.workspace_id


def test_result_records_the_sandbox_environment():
    # Spec section 52: a later stage must be able to tell what environment produced a result.
    executor, _ = make_executor(stdout=b"ok\n")
    record = executor.execute("print('ok')").sandbox_config
    for key in ("image", "memory", "cpus", "timeout_seconds", "pids_limit", "network_mode"):
        assert key in record


def test_network_blocked_is_reported():
    executor, _ = make_executor()
    assert executor.execute("pass").network_blocked is True


def test_the_workspace_is_mounted_read_only():
    executor, runtime = make_executor()
    executor.execute("pass")
    spec = runtime.started_specs[0]
    assert f"{spec.workspace_path}:/workspace:ro" in spec.to_docker_args()


def test_candidate_code_reaches_the_container_as_a_file_not_a_command():
    # Spec sections 42, 43: the source is never part of the argv.
    executor, runtime = make_executor()
    executor.execute("print('unique-marker-12345')")
    args = runtime.started_specs[0].to_docker_args()
    assert "unique-marker-12345" not in " ".join(args)
    assert args[-2:] == ["python", "/workspace/candidate.py"]


# ------------------------------------------------------------------ candidate failures


def test_runtime_error(tmp_path):
    executor, _ = make_executor(stderr=RUNTIME_STDERR, exit_code=1)
    result = executor.execute('raise ValueError("boom")')
    assert result.status == "runtime_error"
    assert result.exit_code == 1
    assert "ValueError" in result.stderr


def test_syntax_error(tmp_path):
    executor, _ = make_executor(stderr=SYNTAX_STDERR, exit_code=1)
    result = executor.execute("def broken(:")
    assert result.status == "syntax_error"
    assert "SyntaxError" in result.stderr


def test_timeout_kills_the_container_and_reports_partial_output():
    # Spec sections 29, 30: terminate, capture what arrived, never leave it running.
    executor, runtime = make_executor(stdout=b"before hanging\n", hang=True, exit_code=137)
    result = executor.execute("while True: pass")

    assert result.status == "timeout"
    assert result.timed_out is True
    assert "before hanging" in result.stdout
    assert runtime.containers[0].killed, "the container must be terminated on timeout"


def test_oom_is_a_resource_violation_not_a_runtime_error():
    executor, _ = make_executor(exit_code=137, oom_killed=True)
    result = executor.execute("x = bytearray(10**10)")
    assert result.status == "resource_exceeded"
    assert result.memory_limit_exceeded is True
    assert result.signal == 9


def test_output_flood_is_bounded_truncated_and_terminated():
    # Spec sections 31, 32, 75: output must not accumulate without bound, truncation must
    # be recorded, and the candidate is terminated.
    runtime = FakeContainerRuntime(stdout=b"x" * 50_000, exit_code=137, hang=True)
    executor = SandboxExecutor(
        config=SandboxConfig(
            timeout_seconds=1, startup_grace_seconds=1, max_output_bytes=1000
        ),
        runtime=runtime,
    )
    result = executor.execute('while True: print("x" * 10000)')

    assert result.status == "resource_exceeded"
    assert result.stdout_truncated is True
    assert result.truncated is True
    assert len(result.stdout) <= 1000, "captured output must respect max_output_bytes"
    assert runtime.containers[0].killed


def test_a_container_that_floods_then_exits_needs_no_kill():
    # The limit can trip on output a container already finished writing. It is still a
    # resource violation, but there is nothing left to terminate — killing a dead container
    # would be a pointless (and on a real daemon, error-logging) call.
    runtime = FakeContainerRuntime(stdout=b"z" * 50_000, exit_code=0)
    executor = SandboxExecutor(
        config=SandboxConfig(
            timeout_seconds=1, startup_grace_seconds=1, max_output_bytes=1000
        ),
        runtime=runtime,
    )
    result = executor.execute("print('z' * 50000)")

    assert result.status == "resource_exceeded"
    assert result.stdout_truncated is True
    assert runtime.containers[0].killed is False
    assert len(runtime.removed) == 1, "it must still be removed"


def test_reader_threads_never_touch_the_container():
    # Concurrent wait()/kill() on one process from a reader thread and the main thread
    # hangs and leaks the container on a real daemon. The reader only raises a flag.
    import threading

    from python_dpo.sandbox.container import BoundedReader

    event = threading.Event()
    reader = BoundedReader(io.BytesIO(b"q" * 500), limit=100, limit_reached=event)
    reader.start()
    reader.join(timeout=5)

    assert reader.truncated is True
    assert event.is_set()
    assert reader.text == "q" * 100
    assert not hasattr(reader, "_on_limit"), "the reader must hold no container callback"


def test_truncation_is_never_silent():
    runtime = FakeContainerRuntime(stdout=b"y" * 5000, exit_code=0)
    executor = SandboxExecutor(
        config=SandboxConfig(
            timeout_seconds=1, startup_grace_seconds=1, max_output_bytes=100
        ),
        runtime=runtime,
    )
    result = executor.execute("print('y' * 5000)")
    assert result.stdout_truncated is True
    assert result.status == "resource_exceeded"


# ------------------------------------------------------------- infrastructure failures


def test_docker_unavailable_is_an_infrastructure_failure_not_a_candidate_failure():
    # Spec sections 79, 81: a candidate is never marked bad because Docker failed.
    executor, _ = make_executor(start_error=DockerUnavailableError("daemon down"))
    result = executor.execute('print("hello")')

    assert result.status == "infrastructure_error"
    assert result.is_infrastructure_failure
    assert not result.is_candidate_outcome
    assert result.error_type == "DockerUnavailableError"
    assert "daemon down" in result.error_message


def test_container_creation_failure_is_an_infrastructure_failure():
    executor, _ = make_executor(start_error=ContainerCreationError("no space left"))
    result = executor.execute("pass")
    assert result.status == "infrastructure_error"
    assert result.error_type == "ContainerCreationError"


def test_infrastructure_failure_does_not_raise_to_the_caller():
    # A caller looping over candidates must not be derailed by a transient Docker fault.
    executor, _ = make_executor(start_error=DockerUnavailableError("gone"))
    assert isinstance(executor.execute("pass"), ExecutionResult)


@pytest.mark.parametrize("exit_code", [125, 126, 127])
def test_docker_cli_exit_codes_are_infrastructure_failures(exit_code):
    executor, _ = make_executor(exit_code=exit_code)
    assert executor.execute("pass").status == "infrastructure_error"


# ------------------------------------------------------------------------- cleanup (§16)


def test_container_is_removed_after_a_successful_run():
    executor, runtime = make_executor(stdout=b"ok\n")
    result = executor.execute("print('ok')")
    assert runtime.removed == [f"python-dpo-sandbox-{result.workspace_id}"]


def test_container_is_removed_after_a_timeout():
    executor, runtime = make_executor(hang=True, exit_code=137)
    executor.execute("while True: pass")
    assert len(runtime.removed) == 1


def test_container_is_removed_after_a_crash():
    executor, runtime = make_executor(stderr=RUNTIME_STDERR, exit_code=1)
    executor.execute("raise ValueError('x')")
    assert len(runtime.removed) == 1


def test_workspace_is_removed_after_every_run(tmp_path):
    config = SandboxConfig(
        timeout_seconds=1, startup_grace_seconds=1, workspace_root=str(tmp_path)
    )
    runtime = FakeContainerRuntime(stdout=b"ok\n")
    executor = SandboxExecutor(config=config, runtime=runtime)
    executor.execute("print('ok')")
    assert list(tmp_path.iterdir()) == [], "the job workspace must not survive the run"


def test_workspace_is_removed_even_when_the_runtime_fails(tmp_path):
    config = SandboxConfig(
        timeout_seconds=1, startup_grace_seconds=1, workspace_root=str(tmp_path)
    )
    runtime = FakeContainerRuntime(start_error=DockerUnavailableError("gone"))
    executor = SandboxExecutor(config=config, runtime=runtime)
    executor.execute("print('ok')")
    assert list(tmp_path.iterdir()) == []


def test_workspace_is_removed_after_a_timeout(tmp_path):
    config = SandboxConfig(
        timeout_seconds=1, startup_grace_seconds=1, workspace_root=str(tmp_path)
    )
    runtime = FakeContainerRuntime(hang=True, exit_code=137)
    executor = SandboxExecutor(config=config, runtime=runtime)
    executor.execute("while True: pass")
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- naming (§37)


def test_container_name_includes_the_run_id_when_supplied():
    executor, runtime = make_executor()
    executor.execute("pass", job_id="20260817_150000_ab12", run_id="run_x")
    assert runtime.started_specs[0].name == "python-dpo-sandbox-run_x-20260817_150000_ab12"


def test_per_execution_timeout_overrides_the_configured_default():
    executor, runtime = make_executor(stdout=b"ok\n")
    result = executor.execute("print('ok')", timeout_seconds=30)
    assert result.status == "success"


# ------------------------------------------------------------------------- execute_job


def test_execute_job_writes_every_file_into_the_workspace():
    # Stage 6's evaluation job needs multiple files, not just candidate.py.
    executor, runtime = make_executor(stdout=b"ok\n")
    executor.execute_job(
        files={
            "candidate.py": "print('a')\n",
            "test_candidate.py": "def test_x(): pass\n",
            "conftest.py": "# plugin\n",
        },
        command=("python", "-m", "pytest", "-q", "test_candidate.py"),
    )
    assert runtime.workspace_files_at_start[0] == ["candidate.py", "conftest.py", "test_candidate.py"]


def test_execute_job_uses_the_supplied_command_not_the_default():
    executor, runtime = make_executor(stdout=b"ok\n")
    command = ("python", "-m", "pytest", "-q", "test_candidate.py")
    executor.execute_job(files={"candidate.py": "pass\n"}, command=command)
    args = runtime.started_specs[0].to_docker_args()
    assert args[-5:] == list(command)


def test_execute_is_equivalent_to_execute_job_with_one_file():
    executor, runtime = make_executor(stdout=b"ok\n")
    executor.execute("print('ok')")
    spec = runtime.started_specs[0]
    assert spec.command == ("python", "/workspace/candidate.py")


def test_execute_job_shares_cleanup_with_execute(tmp_path):
    # The whole point of building execute() on top of execute_job(): identical
    # unconditional-cleanup guarantees for both.
    config = SandboxConfig(timeout_seconds=1, workspace_root=str(tmp_path))
    runtime = FakeContainerRuntime(stdout=b"ok\n")
    executor = SandboxExecutor(config=config, runtime=runtime)
    executor.execute_job(
        files={"candidate.py": "pass\n", "extra.py": "pass\n"},
        command=("python", "/workspace/candidate.py"),
    )
    assert list(tmp_path.iterdir()) == []
    assert len(runtime.removed) == 1


def test_execute_job_classifies_a_test_failure_like_any_other_run():
    executor, runtime = make_executor(stderr=RUNTIME_STDERR, exit_code=1)
    result = executor.execute_job(
        files={"candidate.py": "pass\n"}, command=("python", "/workspace/candidate.py")
    )
    assert result.status == "runtime_error"
