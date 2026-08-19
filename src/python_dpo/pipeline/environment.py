"""Full environment capture for the experiment manifest (spec 12 section 30).

Extends, rather than replaces, the two capture functions that already exist:
:func:`python_dpo.runs.environment.capture_environment` (Stage 4's lighter capture) and
:func:`python_dpo.training.versions.capture_environment` (Stage 9's package/driver
capture, which already covers torch, transformers, trl, peft, bitsandbytes, accelerate,
datasets, safetensors, and the NVIDIA driver version). This module adds the remaining
section 30 items -- OS, CUDA version, GPU name/VRAM, pytest, Docker -- without duplicating
what those two already do correctly.

Every probe is wrapped so a missing tool (no GPU, no Docker, no `nvidia-smi`) degrades to
``None``/``[]`` rather than raising, and nothing here imports torch or transformers at
module scope -- ``tests/test_no_heavy_imports.py`` must stay green. Never calls
``platform.node()`` (the machine's hostname) or reads a username or token, matching the
house rule already documented in ``runs/environment.py`` (spec sections 76-78).
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Any

from ..training.versions import capture_environment as _capture_training_environment
from ..training.versions import package_version


def cuda_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.version.cuda


def gpu_info() -> list[dict[str, str]]:
    """One ``{"name", "memory_total"}`` entry per visible GPU, via ``nvidia-smi``."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    gpus: list[dict[str, str]] = []
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and all(parts):
            gpus.append({"name": parts[0], "memory_total": parts[1]})
    return gpus


def docker_version() -> str | None:
    if shutil.which("docker") is None:
        return None
    try:
        completed = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output or None


def capture_environment() -> dict[str, Any]:
    """The full section 30 block written into ``environment.json``."""
    base = _capture_training_environment()
    return {
        **base,
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "cuda_version": cuda_version(),
        "gpus": gpu_info(),
        "pytest_version": package_version("pytest"),
        "docker_version": docker_version(),
    }


__all__ = ["capture_environment", "cuda_version", "docker_version", "gpu_info"]
