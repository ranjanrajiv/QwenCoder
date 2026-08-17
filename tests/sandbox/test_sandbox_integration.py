"""Docker integration tests — the real isolation boundary (spec 05 sections 59, 88).

These need a running Docker daemon and the configured image, so they are deselected by
default (``addopts = "-ra -m 'not integration'"`` in pyproject.toml). Run them with:

    pytest -q -m integration

They deliberately **fail** rather than skip when Docker is unreachable: they were asked for
explicitly, so silently passing an unrun security suite would be the worst outcome. The
default ``pytest -q`` remains offline and zero-skip.

Every one of the ten mandatory checks in spec section 88 is covered here, plus the resource
and cleanup verifications from sections 65, 69, 74 and 76.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from python_dpo.sandbox import (
    DockerContainerRuntime,
    SandboxConfig,
    SandboxExecutor,
    check_sandbox_health,
)
from python_dpo.sandbox.container import BASE_ENVIRONMENT

pytestmark = pytest.mark.integration

# Long enough that a slow daemon does not cause a false timeout, short enough to keep the
# suite usable. Container startup alone is ~2s, hence the separate grace period.
TIMEOUT = 10
GRACE = 15


@pytest.fixture(scope="session")
def runtime() -> DockerContainerRuntime:
    """Fail fast, and legibly, when the suite is run without a usable daemon."""
    docker = DockerContainerRuntime()
    try:
        docker.check_available()
    except Exception as exc:  # noqa: BLE001 - reported as a clear failure, not a skip
        pytest.fail(
            f"integration tests require a running Docker daemon: {exc}\n"
            "Run the offline suite with `pytest -q` instead."
        )
    if not docker.image_present(SandboxConfig().image):
        pytest.fail(
            f"the sandbox image {SandboxConfig().image} is not present. "
            f"Run: docker pull {SandboxConfig().image}"
        )
    return docker


@pytest.fixture
def executor(runtime: DockerContainerRuntime) -> SandboxExecutor:
    return SandboxExecutor(
        config=SandboxConfig(timeout_seconds=TIMEOUT, startup_grace_seconds=GRACE),
        runtime=runtime,
    )


@pytest.fixture(autouse=True)
def no_containers_left_behind(runtime: DockerContainerRuntime):
    """Spec section 76: no test may leave an abandoned sandbox container behind."""
    yield
    lingering = runtime.list_sandbox_containers()
    assert lingering == [], f"sandbox containers left behind: {lingering}"


# ------------------------------------------------------------------ §88.1 normal execution


def test_normal_execution(executor):
    result = executor.execute('print("hello")')
    assert result.status == "success"
    assert result.stdout == "hello\n"
    assert result.exit_code == 0
    assert result.timed_out is False


def test_stdout_and_stderr_are_captured_separately(executor):
    result = executor.execute(
        'import sys\nprint("to stdout")\nprint("to stderr", file=sys.stderr)\n'
    )
    assert result.stdout == "to stdout\n"
    assert "to stderr" in result.stderr


# ---------------------------------------------------------------------- §88.2 infinite loop


def test_infinite_loop_times_out_and_is_terminated(runtime):
    # Spec sections 28, 30, 64: a runaway candidate is terminated within a bounded period
    # and leaves no running container.
    executor = SandboxExecutor(
        config=SandboxConfig(timeout_seconds=2, startup_grace_seconds=GRACE), runtime=runtime
    )
    result = executor.execute("while True:\n    pass\n")

    assert result.status == "timeout"
    assert result.timed_out is True
    assert runtime.list_sandbox_containers() == []


def test_partial_output_survives_a_timeout(runtime):
    # Spec section 29.3: whatever the candidate managed to print must still be captured.
    executor = SandboxExecutor(
        config=SandboxConfig(timeout_seconds=2, startup_grace_seconds=GRACE), runtime=runtime
    )
    result = executor.execute('print("printed before hanging")\nwhile True:\n    pass\n')
    assert result.status == "timeout"
    assert "printed before hanging" in result.stdout


# ------------------------------------------------------------------ §88.3/§88.4 networking


def test_outbound_connections_fail(executor):
    # Spec sections 12, 13, 67: not a firewall rule the candidate could route around —
    # the container has no network stack at all.
    result = executor.execute(
        'import socket\n'
        'socket.create_connection(("8.8.8.8", 53), timeout=3)\n'
        'print("CONNECTED")\n'
    )
    assert result.status == "runtime_error"
    assert "CONNECTED" not in result.stdout


def test_dns_resolution_fails(executor):
    # Spec section 68: verifies isolation is not merely blocking one port.
    result = executor.execute(
        'import socket\nprint(socket.gethostbyname("example.com"))\n'
    )
    assert result.status == "runtime_error"
    assert "gaierror" in result.stderr or "Name or service not known" in result.stderr


def test_http_requests_fail(executor):
    # Spec section 13's second case.
    result = executor.execute(
        "import urllib.request\n"
        'urllib.request.urlopen("https://example.com", timeout=3)\n'
        'print("FETCHED")\n'
    )
    assert result.status == "runtime_error"
    assert "FETCHED" not in result.stdout


def test_result_reports_the_network_as_blocked(executor):
    assert executor.execute("pass").network_blocked is True


# --------------------------------------------------------------------------- §88.5 non-root


def test_candidate_does_not_run_as_root(executor):
    # Spec sections 19, 20, 72.
    result = executor.execute("import os\nprint(os.getuid())\n")
    assert result.status == "success"
    assert int(result.stdout.strip()) != 0


def test_candidate_cannot_write_to_the_read_only_root_filesystem(executor):
    # Spec section 18.
    result = executor.execute(
        "try:\n"
        "    open('/etc/passwd', 'a').write('x')\n"
        "    print('WROTE')\n"
        "except OSError as exc:\n"
        "    print('blocked:', type(exc).__name__)\n"
    )
    assert result.status == "success"
    assert "WROTE" not in result.stdout
    assert "blocked:" in result.stdout


def test_candidate_may_use_the_writable_tmpfs(executor):
    # Spec section 18: a controlled writable location is provided, since CPython wants one.
    result = executor.execute(
        "p = '/tmp/scratch.txt'\n"
        "open(p, 'w').write('ok')\n"
        "print(open(p).read())\n"
    )
    assert result.status == "success"
    assert result.stdout.strip() == "ok"


# ------------------------------------------------------------------ §88.6 environment leaks


def test_host_environment_is_not_inherited(executor, monkeypatch):
    # Spec sections 33, 70: the host env is never passed wholesale, so no credential leaks.
    monkeypatch.setenv("PYTHON_DPO_SANDBOX_TEST_SECRET", "host-only-value")
    result = executor.execute(
        'import os\nprint(os.environ.get("PYTHON_DPO_SANDBOX_TEST_SECRET"))\n'
    )
    assert result.status == "success"
    assert result.stdout.strip() == "None"


def test_container_environment_is_only_the_image_plus_our_three_variables(executor):
    """Nothing reaches the container that the image did not already provide.

    Stated precisely rather than by keyword heuristic: the stock image itself defines
    GPG_KEY, PYTHON_VERSION and friends, so scanning for credential-shaped *names* produces
    false positives. The property that actually matters is that the container's environment
    is exactly the image's, plus the three variables the sandbox passes on purpose.
    """
    baseline = subprocess.run(
        ["docker", "run", "--rm", SandboxConfig().image, "env"],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    image_keys = {line.split("=", 1)[0] for line in baseline.stdout.splitlines() if "=" in line}

    result = executor.execute("import os\nprint('\\n'.join(sorted(os.environ)))\n")
    assert result.status == "success"
    container_keys = set(result.stdout.split())

    unexpected = container_keys - image_keys - set(BASE_ENVIRONMENT)
    assert unexpected == set(), f"variables leaked into the container: {sorted(unexpected)}"


def test_host_credential_variables_never_reach_the_container(executor, monkeypatch):
    # The direct statement of spec sections 33/34: credential-shaped host variables are
    # invisible inside the sandbox.
    secrets = {
        "HF_TOKEN": "hf-should-not-leak",
        "AWS_SECRET_ACCESS_KEY": "aws-should-not-leak",
        "OPENAI_API_KEY": "oai-should-not-leak",
        "DATABASE_PASSWORD": "db-should-not-leak",
    }
    for key, value in secrets.items():
        monkeypatch.setenv(key, value)

    result = executor.execute(
        "import os\n"
        f"names = {sorted(secrets)!r}\n"
        "print([n for n in names if n in os.environ])\n"
        "print([v for v in os.environ.values() if 'should-not-leak' in v])\n"
    )
    assert result.status == "success"
    assert result.stdout.strip().splitlines() == ["[]", "[]"]


# --------------------------------------------------------------------- §88.7 docker socket


def test_docker_socket_is_not_present(executor):
    # Spec sections 35, 71: access to the socket would mean control of the host daemon.
    result = executor.execute(
        'import os\nprint(os.path.exists("/var/run/docker.sock"))\n'
    )
    assert result.status == "success"
    assert result.stdout.strip() == "False"


# ------------------------------------------------------------ §88.8/§88.9 candidate errors


def test_runtime_error(executor):
    result = executor.execute('raise RuntimeError("test")')
    assert result.status == "runtime_error"
    assert result.exit_code != 0
    assert "RuntimeError" in result.stderr


def test_value_error_is_a_runtime_error(executor):
    result = executor.execute('raise ValueError("test error")')
    assert result.status == "runtime_error"
    assert "ValueError" in result.stderr


def test_syntax_error(executor):
    result = executor.execute("def broken(:\n")
    assert result.status == "syntax_error"
    assert "SyntaxError" in result.stderr


def test_a_deliberately_raised_syntax_error_is_a_runtime_error(executor):
    # The file compiled; it then chose to raise. Classifying this as syntax_error would
    # misreport what actually happened.
    result = executor.execute('raise SyntaxError("deliberate")')
    assert result.status == "runtime_error"


# ---------------------------------------------------------------------- §88.10 output flood


def test_output_flood_is_bounded_and_terminated(runtime):
    # Spec sections 31, 32, 75: output must not grow without bound, truncation is recorded,
    # and the candidate is terminated.
    executor = SandboxExecutor(
        config=SandboxConfig(
            timeout_seconds=TIMEOUT, startup_grace_seconds=GRACE, max_output_bytes=50_000
        ),
        runtime=runtime,
    )
    result = executor.execute('while True:\n    print("x" * 10000)\n')

    assert result.status == "resource_exceeded"
    assert result.stdout_truncated is True
    assert len(result.stdout) <= 50_000
    assert runtime.list_sandbox_containers() == []


# ----------------------------------------------------------------- §65/§74 resource limits


def test_memory_limit_is_enforced(runtime):
    # Spec section 65: deliberately not asserting an exact OS behaviour — only that the
    # allocation does not succeed and the host stays healthy.
    executor = SandboxExecutor(
        config=SandboxConfig(
            timeout_seconds=TIMEOUT, startup_grace_seconds=GRACE, memory="256m"
        ),
        runtime=runtime,
    )
    result = executor.execute(
        "buf = bytearray(2 * 1024 * 1024 * 1024)\nprint('ALLOCATED')\n"
    )
    assert result.status in {"resource_exceeded", "runtime_error"}
    assert "ALLOCATED" not in result.stdout


def test_process_limit_is_enforced(runtime):
    # Spec section 74: a controlled attempt, not thousands of processes.
    executor = SandboxExecutor(
        config=SandboxConfig(
            timeout_seconds=TIMEOUT, startup_grace_seconds=GRACE, pids_limit=32
        ),
        runtime=runtime,
    )
    result = executor.execute(
        "import threading, time\n"
        "started = 0\n"
        "try:\n"
        "    for _ in range(200):\n"
        "        threading.Thread(target=time.sleep, args=(30,), daemon=True).start()\n"
        "        started += 1\n"
        "except RuntimeError:\n"
        "    pass\n"
        "print(started)\n"
    )
    assert result.status in {"success", "resource_exceeded", "runtime_error"}
    if result.status == "success":
        assert int(result.stdout.strip()) < 200, "the PID limit must stop unbounded spawning"


# ------------------------------------------------------------- §69 filesystem isolation


def test_host_project_files_are_not_reachable(executor, tmp_path):
    # Spec section 69: a marker outside the workspace must be invisible. Deliberately a
    # throwaway file, never anything sensitive.
    marker = tmp_path / "host-marker.txt"
    marker.write_text("host-only", encoding="utf-8")

    result = executor.execute(
        f"import os\nprint(os.path.exists({str(marker)!r}))\n"
    )
    assert result.status == "success"
    assert result.stdout.strip() == "False"


def test_the_project_directory_is_not_mounted(executor):
    result = executor.execute(
        "import os\n"
        "entries = sorted(os.listdir('/'))\n"
        "print('workspace' in entries, [e for e in entries if 'python-dpo' in e])\n"
    )
    assert result.status == "success"
    assert "True []" in result.stdout


def test_the_workspace_holds_only_the_candidate(executor):
    result = executor.execute(
        "import os\nprint(sorted(os.listdir('/workspace')))\n"
    )
    assert result.status == "success"
    assert result.stdout.strip() == "['candidate.py']"


def test_the_workspace_is_mounted_read_only(executor):
    # Spec section 17: prefer read-only mounts.
    result = executor.execute(
        "try:\n"
        "    open('/workspace/evil.py', 'w').write('x')\n"
        "    print('WROTE')\n"
        "except OSError as exc:\n"
        "    print('blocked:', type(exc).__name__)\n"
    )
    assert result.status == "success"
    assert "WROTE" not in result.stdout


# ---------------------------------------------------------------------------- §76 cleanup


def test_workspace_is_removed_after_a_real_execution(runtime, tmp_path):
    executor = SandboxExecutor(
        config=SandboxConfig(
            timeout_seconds=TIMEOUT,
            startup_grace_seconds=GRACE,
            workspace_root=str(tmp_path),
        ),
        runtime=runtime,
    )
    executor.execute('print("hello")')
    assert list(tmp_path.iterdir()) == []


def test_containers_are_removed_after_repeated_executions(executor, runtime):
    for _ in range(3):
        executor.execute('print("ok")')
    assert runtime.list_sandbox_containers() == []


def test_container_id_is_recorded(executor):
    # Spec section 38: useful for debugging infrastructure failures.
    result = executor.execute('print("hello")')
    assert result.container_id
    assert len(result.container_id) >= 12


# ------------------------------------------------------------------------- §54 health check


def test_health_check_passes(runtime):
    report = check_sandbox_health(SandboxConfig(), runtime)
    assert report.passed, [c.detail for c in report.failures]
    assert len(report.checks) == 5


def test_health_check_reports_an_unreachable_daemon_clearly():
    # Spec section 55: a clear sentence, never a raw traceback.
    from python_dpo.sandbox import format_health_report
    from python_dpo.sandbox.errors import DockerUnavailableError

    class BrokenRuntime:
        def check_available(self):
            raise DockerUnavailableError("the Docker daemon is not reachable")

    report = check_sandbox_health(SandboxConfig(), BrokenRuntime())
    rendered = format_health_report(report)
    assert not report.passed
    assert rendered.startswith("Sandbox health check failed:")
    assert "not reachable" in rendered
    assert "Traceback" not in rendered


# ------------------------------------------------------ the sandbox is the only executor


def test_candidate_code_never_runs_on_the_host(executor, tmp_path):
    # The strongest available end-to-end assertion of spec section 2: a candidate that
    # would create a host file if executed locally must leave the host untouched.
    marker = tmp_path / "escaped.txt"
    result = executor.execute(f"open({str(marker)!r}, 'w').write('escaped')\n")

    assert not marker.exists(), "candidate code must never execute on the host"
    assert result.status == "runtime_error"  # the path does not exist in the container


def test_sandbox_run_cli_executes_in_a_container(tmp_path):
    # Spec section 57: the CLI must never execute the file on the host.
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = tmp_path / "probe.py"
    marker = tmp_path / "host-side.txt"
    script.write_text(
        f"import os\nprint(os.getuid())\n"
        f"print(os.path.exists({str(marker)!r}))\n",
        encoding="utf-8",
    )

    env = dict(os.environ, PYTHONPATH=os.path.join(project_root, "src"))
    completed = subprocess.run(
        ["python", "-m", "python_dpo", "sandbox", "run", "--file", str(script)],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert "status: success" in completed.stdout
    # Non-root inside the container, and the host-side path is invisible.
    assert "65534" in completed.stdout
    assert "False" in completed.stdout
