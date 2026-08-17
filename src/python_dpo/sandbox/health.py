"""Sandbox health check (spec 05 sections 11, 54, 55).

Verifies the whole Docker path end to end before anything depends on it, and reports each
step individually so a failure names the actual problem. Spec section 55 is explicit that a
user must not be handed a raw Python traceback as the primary error — every check catches
its own failure and turns it into a sentence.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import SandboxConfig
from .container import ContainerRuntime, DockerContainerRuntime
from .errors import SandboxError
from .executor import SandboxExecutor

# A trivial program whose output proves the interpreter actually ran (spec section 54.5).
_PROBE_CODE = 'print("sandbox-ok")\n'
_PROBE_EXPECTED = "sandbox-ok"


@dataclass(frozen=True)
class HealthCheck:
    """One step of the health check."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class HealthReport:
    checks: tuple[HealthCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[HealthCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


def check_sandbox_health(
    config: SandboxConfig | None = None,
    runtime: ContainerRuntime | None = None,
) -> HealthReport:
    """Run the six spec section 54 checks, stopping at the first that fails.

    Stopping early is deliberate: if the daemon is unreachable there is no value in
    reporting four more failures that all say the same thing.
    """
    config = config or SandboxConfig()
    runtime = runtime if runtime is not None else DockerContainerRuntime()
    checks: list[HealthCheck] = []

    # 1 + 2: the docker client exists and the daemon answers.
    try:
        runtime.check_available()
    except SandboxError as exc:
        checks.append(HealthCheck("Docker client and daemon", False, str(exc)))
        return HealthReport(tuple(checks))
    version = getattr(runtime, "server_version", lambda: "")()
    checks.append(
        HealthCheck(
            "Docker client and daemon",
            True,
            f"daemon version {version}" if version else "reachable",
        )
    )

    # 3: the configured image is available, pulled when permitted.
    image = config.image_reference
    try:
        if runtime.image_present(image):
            checks.append(HealthCheck("Sandbox image", True, f"{image} is present"))
        elif config.auto_pull:
            runtime.pull(image)
            checks.append(HealthCheck("Sandbox image", True, f"{image} pulled"))
        else:
            checks.append(
                HealthCheck(
                    "Sandbox image",
                    False,
                    f"{image} is not present and sandbox.auto_pull is false; "
                    f"run: docker pull {image}",
                )
            )
            return HealthReport(tuple(checks))
    except SandboxError as exc:
        checks.append(HealthCheck("Sandbox image", False, str(exc)))
        return HealthReport(tuple(checks))

    # 4 + 5: a container starts and a Python program actually runs inside it.
    # 6: cleanup is implicit — the executor removes the container and workspace, and this
    # check would surface any failure to do so as an infrastructure_error.
    executor = SandboxExecutor(config=config, runtime=runtime)
    result = executor.execute(_PROBE_CODE)

    if result.status == "infrastructure_error":
        checks.append(
            HealthCheck(
                "Container start",
                False,
                result.error_message or "the sandbox container could not be started",
            )
        )
        return HealthReport(tuple(checks))
    checks.append(HealthCheck("Container start", True, "a sandbox container started"))

    if result.status == "success" and _PROBE_EXPECTED in result.stdout:
        checks.append(HealthCheck("Python execution", True, "a probe program ran and exited 0"))
    else:
        checks.append(
            HealthCheck(
                "Python execution",
                False,
                f"probe returned status={result.status} exit_code={result.exit_code} "
                f"stderr={result.stderr.strip()[:200]!r}",
            )
        )
        return HealthReport(tuple(checks))

    lingering = getattr(runtime, "list_sandbox_containers", lambda: [])()
    if lingering:
        checks.append(
            HealthCheck(
                "Cleanup",
                False,
                f"{len(lingering)} sandbox container(s) left behind: {', '.join(lingering)}",
            )
        )
    else:
        checks.append(HealthCheck("Cleanup", True, "no sandbox containers left behind"))

    return HealthReport(tuple(checks))


def format_health_report(report: HealthReport) -> str:
    """Render the report in the spec section 54/55 shape."""
    lines = [f"  {check.name}: {check.detail}" for check in report.checks if check.passed]

    if report.passed:
        body = "\n".join(lines)
        return f"Docker sandbox health check passed.\n{body}\n" if body else (
            "Docker sandbox health check passed.\n"
        )

    out = ["Sandbox health check failed:"]
    out += [f"  {check.name}: {check.detail}" for check in report.failures]
    return "\n".join(out) + "\n"


__all__ = ["HealthCheck", "HealthReport", "check_sandbox_health", "format_health_report"]
