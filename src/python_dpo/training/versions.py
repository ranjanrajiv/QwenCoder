"""Package and driver version capture (spec 09 sections 61, 71).

Follows the principle :mod:`python_dpo.evaluation.probe` established: record what
*genuinely ran*, not what the dependency pins asked for. Versions are read from the
installed distributions at run start, never from ``pyproject.toml``.

Unlike the evaluation probe, a missing package here is recorded as ``None`` rather than
raising. Some of these are genuinely optional at capture time — ``bitsandbytes`` is not
needed to inspect a dataset — and a `train hardware-check` that crashed because one
optional extra was absent would defeat its own purpose. The training path enforces what it
actually needs separately, in :func:`python_dpo.training.loader.import_backend`.

Nothing here imports torch: :func:`importlib.metadata.version` reads distribution
metadata, not the module.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from importlib import metadata

# Spec section 71's list. Order is presentation order in the report.
TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "bitsandbytes",
    "accelerate",
    "datasets",
    "safetensors",
)


def package_version(name: str) -> str | None:
    """The installed version of ``name``, or ``None`` when it is not installed."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def capture_package_versions() -> dict[str, str | None]:
    """Every spec section 71 package, missing ones recorded as ``None``."""
    return {name: package_version(name) for name in TRACKED_PACKAGES}


def nvidia_driver_version() -> str | None:
    """The NVIDIA driver version via ``nvidia-smi``, or ``None`` when unavailable.

    Read from the driver rather than torch because it is a property of the machine, not
    of the Python environment — and a training run's provenance should record the box it
    ran on (spec sections 61, 71).
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    first = completed.stdout.strip().splitlines()
    return first[0].strip() if first else None


def capture_environment() -> dict[str, object]:
    """The full provenance block written into the training manifest (sections 61, 71)."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": capture_package_versions(),
        "nvidia_driver_version": nvidia_driver_version(),
    }


__all__ = [
    "TRACKED_PACKAGES",
    "capture_environment",
    "capture_package_versions",
    "nvidia_driver_version",
    "package_version",
]
