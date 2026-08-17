"""The isolated Docker sandbox — the execution boundary for untrusted candidate code.

This package answers "what happened when this program ran?" and nothing else. It never
decides whether a candidate is *correct* (spec 05 section 82): ``status="success"`` means
the program exited zero, which a later stage will interpret by running the problem's test
suite against it.

Nothing here executes candidate code on the host. Source is written to a file and run by a
fixed argv inside a container with no network, no host filesystem, a non-root user, no
Linux capabilities, and CPU/memory/PID/output/time ceilings. See
``docs/sandbox-security.md`` for the threat model and known limitations.
"""

from .config import DEFAULT_IMAGE, DEFAULT_USER, NETWORK_MODES, SandboxConfig
from .container import (
    CONTAINER_NAME_PREFIX,
    ContainerRuntime,
    ContainerSpec,
    DockerContainerRuntime,
    container_name,
)
from .errors import (
    ContainerCreationError,
    ContainerExecutionError,
    DockerUnavailableError,
    ImageUnavailableError,
    SandboxConfigError,
    SandboxError,
    SandboxTimeoutError,
    WorkspaceError,
)
from .executor import EXECUTION_COMMAND, SandboxExecutor
from .health import HealthCheck, HealthReport, check_sandbox_health, format_health_report
from .result import (
    CANDIDATE_STATUSES,
    EXECUTION_STATUSES,
    INFRASTRUCTURE_STATUSES,
    ExecutionResult,
    ExecutionResultError,
    classify,
)
from .workspace import CANDIDATE_FILENAME, CONTAINER_WORKSPACE, SandboxWorkspace, new_job_id

__all__ = [
    "CANDIDATE_FILENAME",
    "CANDIDATE_STATUSES",
    "CONTAINER_NAME_PREFIX",
    "CONTAINER_WORKSPACE",
    "DEFAULT_IMAGE",
    "DEFAULT_USER",
    "EXECUTION_COMMAND",
    "EXECUTION_STATUSES",
    "INFRASTRUCTURE_STATUSES",
    "NETWORK_MODES",
    "ContainerCreationError",
    "ContainerExecutionError",
    "ContainerRuntime",
    "ContainerSpec",
    "DockerContainerRuntime",
    "DockerUnavailableError",
    "ExecutionResult",
    "ExecutionResultError",
    "HealthCheck",
    "HealthReport",
    "ImageUnavailableError",
    "SandboxConfig",
    "SandboxConfigError",
    "SandboxError",
    "SandboxExecutor",
    "SandboxTimeoutError",
    "SandboxWorkspace",
    "WorkspaceError",
    "check_sandbox_health",
    "classify",
    "container_name",
    "format_health_report",
    "new_job_id",
]
