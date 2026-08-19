"""Preflight validation before an experiment starts (spec 12 sections 60, 61).

Eight checks, run independently (unlike the sandbox health check, a failure here does not
stop the rest -- the point of preflight is to report everything wrong at once, not the
first thing). Every check degrades gracefully when its subject is simply not configured
for this experiment (no training stage enabled, no benchmark named yet) rather than
failing the whole preflight over something the experiment never asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..problems import DatasetError, dataset_path, load_problems
from ..sandbox import DockerContainerRuntime, SandboxError
from .config import ExperimentConfig


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    checks: tuple[PreflightCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[PreflightCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


def _check_gpu() -> PreflightCheck:
    try:
        import torch
    except ImportError:
        return PreflightCheck("GPU", False, "torch is not installed (pip install -e '.[model]')")
    if not torch.cuda.is_available():
        return PreflightCheck("GPU", False, "no CUDA-capable GPU visible to torch")
    name = torch.cuda.get_device_name(0)
    return PreflightCheck("GPU", True, name)


def _check_cuda() -> PreflightCheck:
    try:
        import torch
    except ImportError:
        return PreflightCheck("CUDA", False, "torch is not installed")
    if not torch.cuda.is_available():
        return PreflightCheck("CUDA", False, "CUDA is not available")
    return PreflightCheck("CUDA", True, str(torch.version.cuda))


def _check_docker(runtime: Any = None) -> PreflightCheck:
    runtime = runtime if runtime is not None else DockerContainerRuntime()
    try:
        runtime.check_available()
    except SandboxError as exc:
        return PreflightCheck("Docker", False, str(exc))
    version = getattr(runtime, "server_version", lambda: "")()
    return PreflightCheck("Docker", True, version or "available")


def _check_model(config: Any) -> PreflightCheck:
    name = config.model.name
    if not name:
        return PreflightCheck("Model", False, "model.name is not set in config.yaml")
    return PreflightCheck("Model", True, name)


def _check_dataset(config: Any) -> PreflightCheck:
    path = dataset_path(config.paths.problems)
    if not path.is_file():
        return PreflightCheck("Dataset", False, f"no problem dataset at {path}; run 'problems build'")
    try:
        problems = load_problems(path)
    except DatasetError as exc:
        return PreflightCheck("Dataset", False, str(exc))
    return PreflightCheck("Dataset", True, f"{len(problems)} problem(s) at {path}")


def _check_benchmark(config: Any, experiment: ExperimentConfig) -> PreflightCheck:
    benchmark_name = experiment.stage("model_evaluation").get("benchmark")
    if not experiment.stage("model_evaluation").enabled or not benchmark_name:
        return PreflightCheck("Benchmark", True, "model_evaluation disabled or no benchmark configured")
    from ..model_evaluation import BenchmarkError, load_benchmark

    root = config.project_root / "benchmarks"
    try:
        problems = load_problems(dataset_path(config.paths.problems))
        benchmark = load_benchmark(root, benchmark_name, problems)
    except (DatasetError, BenchmarkError) as exc:
        return PreflightCheck("Benchmark", False, str(exc))
    return PreflightCheck("Benchmark", True, f"{benchmark.benchmark_version} ({len(benchmark)} problem(s))")


def _check_training_config(experiment: ExperimentConfig) -> PreflightCheck:
    if not experiment.stage("dpo_training").enabled:
        return PreflightCheck("Training configuration", True, "dpo_training disabled")
    raw_path = experiment.stage("dpo_training").get("config")
    from ..training.config import DEFAULT_CONFIG_PATH

    path = Path(raw_path) if raw_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        return PreflightCheck("Training configuration", False, f"no training config at {path}")
    return PreflightCheck("Training configuration", True, str(path))


def _check_evaluation_config(experiment: ExperimentConfig) -> PreflightCheck:
    if not experiment.stage("model_evaluation").enabled:
        return PreflightCheck("Evaluation configuration", True, "model_evaluation disabled")
    raw_path = experiment.stage("model_evaluation").get("config")
    from ..model_evaluation.config import DEFAULT_CONFIG_PATH as EVAL_DEFAULT_CONFIG_PATH

    path = Path(raw_path) if raw_path else EVAL_DEFAULT_CONFIG_PATH
    if not path.is_file():
        return PreflightCheck("Evaluation configuration", False, f"no evaluation config at {path}")
    return PreflightCheck("Evaluation configuration", True, str(path))


def run_preflight(config: Any, experiment: ExperimentConfig) -> PreflightReport:
    """Spec section 61's eight checks."""
    checks = (
        _check_gpu(),
        _check_cuda(),
        _check_docker(),
        _check_model(config),
        _check_dataset(config),
        _check_benchmark(config, experiment),
        _check_training_config(experiment),
        _check_evaluation_config(experiment),
    )
    return PreflightReport(checks)


def format_preflight_report(report: PreflightReport) -> str:
    lines = [f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}" for c in report.checks]
    lines.append("")
    lines.append("All preflight checks passed." if report.passed else "Preflight checks failed.")
    return "\n".join(lines) + "\n"


__all__ = [
    "PreflightCheck",
    "PreflightReport",
    "format_preflight_report",
    "run_preflight",
]
