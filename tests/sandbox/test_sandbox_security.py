"""Security guarantees asserted at the argv level, with no Docker required.

``ContainerSpec.to_docker_args()`` is the entire security surface of this project: every
isolation guarantee is one flag in the list it builds. These tests pin both halves of that
contract — what must always be present, and what must never appear — so a regression that
silently weakens the sandbox fails the suite on every commit rather than being discovered
by an escaped candidate.

Same philosophy as ``tests/test_no_heavy_imports.py``: a rule that is one careless edit away
from being broken is asserted, not assumed. These run without Docker and cost milliseconds,
so there is no reason to defer them to the integration suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from python_dpo.sandbox import (
    EXECUTION_COMMAND,
    ContainerSpec,
    SandboxConfig,
    container_name,
)
from python_dpo.sandbox.container import BASE_ENVIRONMENT

SANDBOX_SOURCE = Path(__file__).resolve().parents[2] / "src" / "python_dpo" / "sandbox"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_args(config: SandboxConfig | None = None, workspace="/tmp/job-dir") -> list[str]:
    config = config or SandboxConfig()
    spec = ContainerSpec(
        name=container_name("20260817_150000_ab12"),
        image=config.image_reference,
        command=EXECUTION_COMMAND,
        workspace_path=Path(workspace),
        config=config,
    )
    return spec.to_docker_args()


def joined(args: list[str]) -> str:
    return " ".join(args)


def pair_present(args: list[str], flag: str, value: str) -> bool:
    """Whether ``flag`` appears immediately followed by ``value``."""
    return any(args[i] == flag and args[i + 1] == value for i in range(len(args) - 1))


# --------------------------------------------------------- mandatory isolation is present


def test_network_is_disabled():
    # Spec sections 12, 67, 68: no internet, no localhost, no DNS. Enforced by Docker
    # itself, not by an application-level check the candidate could route around.
    assert pair_present(build_args(), "--network", "none")


def test_container_runs_as_a_non_root_user():
    # Spec sections 19, 20, 72.
    args = build_args()
    assert pair_present(args, "--user", "65534:65534")
    assert not pair_present(args, "--user", "0")
    assert not pair_present(args, "--user", "0:0")


def test_all_capabilities_are_dropped():
    # Spec section 21: generated Python needs no Linux capabilities.
    assert pair_present(build_args(), "--cap-drop", "ALL")


def test_privilege_escalation_is_blocked():
    # Spec section 22's intent: not even a setuid binary in the image can gain privileges.
    assert pair_present(build_args(), "--security-opt", "no-new-privileges")


def test_root_filesystem_is_read_only_with_a_bounded_tmpfs():
    # Spec section 18.
    args = build_args()
    assert "--read-only" in args
    tmpfs = args[args.index("--tmpfs") + 1]
    assert tmpfs.startswith("/tmp:")
    assert "size=" in tmpfs
    # A writable area that cannot execute or gain privileges.
    assert "noexec" in tmpfs
    assert "nosuid" in tmpfs


def test_resource_limits_are_enforced_by_the_container_runtime():
    # Spec sections 24, 25, 26, 27: limits belong to the runtime, not to a Python check.
    args = build_args()
    assert pair_present(args, "--pids-limit", "64")
    assert pair_present(args, "--cpus", "1.0")
    assert pair_present(args, "--memory", "512m")


def test_memory_swap_is_pinned_to_the_memory_limit():
    # Without this a container may use roughly twice its memory limit via swap, quietly
    # defeating the section 26 ceiling.
    args = build_args()
    assert pair_present(args, "--memory-swap", "512m")


def test_only_the_job_workspace_is_mounted_and_it_is_read_only():
    # Spec sections 14, 17: one dedicated directory, read-only, and nothing else.
    args = build_args(workspace="/tmp/job-dir")
    mounts = [args[i + 1] for i, a in enumerate(args) if a == "--volume"]
    assert mounts == ["/tmp/job-dir:/workspace:ro"]


def test_working_directory_is_the_controlled_workspace():
    # Spec section 44.
    assert pair_present(build_args(), "--workdir", "/workspace")


def test_command_is_a_fixed_argument_list():
    # Spec sections 42, 43, 47: the candidate controls candidate.py's contents, never the
    # command. No shell, no interpolation, no quoting surface.
    args = build_args()
    assert args[-2:] == ["python", "/workspace/candidate.py"]
    assert "sh" not in args
    assert "bash" not in args
    assert "-c" not in args


def test_container_name_is_traceable_and_carries_no_source():
    # Spec section 37.
    name = container_name("20260817_150000_ab12", run_id="run_20260817_133700_a81f")
    assert name.startswith("python-dpo-sandbox-")
    assert "run_20260817_133700_a81f" in name
    assert "20260817_150000_ab12" in name


# ----------------------------------------------------- forbidden configuration is absent


@pytest.mark.parametrize(
    "forbidden",
    [
        "--privileged",          # spec section 22
        "--pid=host",            # spec section 23
        "--network=host",        # spec section 23
        "--ipc=host",            # spec section 23
        "--uts=host",            # spec section 23
        "--cap-add",             # spec section 21
    ],
)
def test_dangerous_flags_are_never_present(forbidden):
    assert forbidden not in joined(build_args())


def test_docker_socket_is_never_mounted():
    # Spec sections 35, 71: a container with the Docker socket can control the host daemon,
    # which would defeat the entire boundary.
    assert "docker.sock" not in joined(build_args())


def test_sensitive_host_paths_are_never_mounted():
    # Spec sections 14, 34.
    args = joined(build_args())
    for path in ("/var/run/docker.sock", "~/.ssh", "~/.aws", "~/.config", "/etc/passwd"):
        assert path not in args
    assert str(PROJECT_ROOT) not in args, "the project directory must never be mounted"


def test_only_the_three_explicit_environment_variables_are_passed():
    # Spec sections 33, 70: no host environment wholesale, so no cloud credential, API key,
    # or Hugging Face token can reach candidate code.
    args = build_args()
    passed = {args[i + 1].split("=", 1)[0] for i, a in enumerate(args) if a == "--env"}
    assert passed == set(BASE_ENVIRONMENT)
    assert passed == {"PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE", "HOME"}


def test_pythonpath_is_not_propagated():
    # Spec sections 45, 46: the container uses its own Python environment, never the host's.
    assert "PYTHONPATH" not in joined(build_args())


def test_image_is_version_pinned_in_the_argv():
    args = build_args()
    assert "python:3.12-slim" in args
    assert "python:latest" not in args


# ------------------------------------------------------------------- source-level guards


def test_sandbox_package_never_uses_a_shell():
    # Spec sections 2, 39, 43. shell=True anywhere in this package would reintroduce the
    # injection surface that writing candidate.py to a file exists to eliminate.
    offenders = [
        path.name
        for path in SANDBOX_SOURCE.rglob("*.py")
        if "shell=True" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_sandbox_package_never_executes_candidate_code_on_the_host():
    # Spec section 2: exec/eval/os.system on candidate source is categorically prohibited.
    banned = ("os.system(", "eval(", "exec(")
    offenders = []
    for path in SANDBOX_SOURCE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            if token in text:
                offenders.append(f"{path.name}: {token}")
    assert offenders == []


def test_sandbox_package_never_passes_the_host_environment():
    # Spec section 33 names `env=os.environ` as the specific antipattern.
    offenders = [
        path.name
        for path in SANDBOX_SOURCE.rglob("*.py")
        if "os.environ" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


# --------------------------------------------------------------- configuration can't relax


def test_a_config_cannot_request_a_networked_container():
    # The guarantee is enforced at construction, so no config file can produce an argv
    # without `--network none`.
    from python_dpo.sandbox import SandboxConfigError

    with pytest.raises(SandboxConfigError):
        SandboxConfig(network_mode="bridge")


def test_a_config_cannot_request_root():
    from python_dpo.sandbox import SandboxConfigError

    with pytest.raises(SandboxConfigError):
        SandboxConfig(user="0:0")


def test_disabling_hardening_is_visible_in_the_argv():
    # The toggles exist for debugging; this pins what they actually change so a future
    # default flip cannot silently drop protection.
    relaxed = SandboxConfig(read_only_root=False, drop_capabilities=False)
    args = build_args(relaxed)
    assert "--read-only" not in args
    assert "--cap-drop" not in args
    # These are never negotiable, whatever else is toggled.
    assert pair_present(args, "--network", "none")
    assert pair_present(args, "--user", "65534:65534")
    assert pair_present(args, "--security-opt", "no-new-privileges")
