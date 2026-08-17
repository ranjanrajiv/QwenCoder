"""Runs one evaluation job inside the Stage 5 sandbox (spec 06 sections 33-43, 78).

The sandbox config used for evaluation is *derived* from the audited Stage 5
``sandbox:`` config, never constructed independently — this is what makes spec section
78's requirement ("adding pytest must not weaken network isolation, capabilities,
resource limits...") true by construction rather than by discipline. Only the image,
timeout, startup grace, and auto-pull setting are evaluation-specific; everything else
is copied unchanged.
"""

from __future__ import annotations

import dataclasses

from ..sandbox import SandboxConfig, SandboxExecutor
from ..sandbox.result import ExecutionResult
from .config import EvaluationConfig
from .test_generator import TestJob

# Spec sections 42, 43: a fixed argument list, never a shell string. "-p no:cacheprovider"
# matters because pytest writes .pytest_cache by default and the workspace is read-only;
# PYTHONDONTWRITEBYTECODE (already in the sandbox's base environment) covers __pycache__.
PYTEST_COMMAND: tuple[str, ...] = (
    "python",
    "-m",
    "pytest",
    "-q",
    "-p",
    "no:cacheprovider",
    "test_candidate.py",
)


def build_evaluation_sandbox_config(
    base: SandboxConfig, evaluation: EvaluationConfig
) -> SandboxConfig:
    """Overlay evaluation-specific settings onto the Stage 5 sandbox config.

    Only ``image``, ``timeout_seconds``, ``startup_grace_seconds``, and ``auto_pull``
    differ; ``network_mode``, ``user``, ``drop_capabilities``, ``cpus``, ``memory``,
    ``pids_limit``, ``read_only_root``, and ``tmpfs_size`` all come from ``base``
    unchanged (spec section 78).
    """
    return dataclasses.replace(
        base,
        image=evaluation.image,
        timeout_seconds=evaluation.timeout_seconds,
        startup_grace_seconds=evaluation.startup_grace_seconds,
        auto_pull=evaluation.auto_pull,
    )


class PytestRunner:
    """Runs one :class:`~python_dpo.evaluation.test_generator.TestJob` in the sandbox."""

    def __init__(self, executor: SandboxExecutor) -> None:
        self._executor = executor

    @property
    def config(self) -> SandboxConfig:
        return self._executor.config

    def run(
        self,
        job: TestJob,
        *,
        job_id: str | None = None,
        run_id: str | None = None,
    ) -> ExecutionResult:
        """Execute the job's files with the fixed pytest command. Never raises for a
        candidate-caused failure — see :meth:`SandboxExecutor.execute_job`.
        """
        return self._executor.execute_job(
            files=job.files,
            command=PYTEST_COMMAND,
            job_id=job_id,
            run_id=run_id,
        )


__all__ = ["PYTEST_COMMAND", "PytestRunner", "build_evaluation_sandbox_config"]
