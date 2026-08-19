"""Resource capture for the experiment report (spec 12 section 51).

Per-stage duration is exact -- it is just the already-recorded ``start_time``/``end_time``
on each stage's manifest. GPU/CPU/RAM are a best-effort **snapshot of the host at report
time**, not a per-stage trace: sampling GPU utilization continuously through every stage
would need a background poller this stage does not build (CLAUDE.md's Scope Control), so
this module answers "what did this machine look like right after the run finished" rather
than "how loaded was the GPU during training specifically". Every field is ``None`` when
its source is unavailable, never a fabricated zero.

``nvidia-smi`` is invoked with a fixed argv (no shell), matching the house rule already
enforced for ``git`` in :mod:`python_dpo.pipeline.gitinfo` and for the sandbox.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifest import StageManifest

_MEMINFO_PATH = Path("/proc/meminfo")


def stage_durations(stage_runs: dict[str, StageManifest]) -> dict[str, float]:
    """Wall-clock seconds per stage, from each manifest's own ``start_time``/``end_time``.
    A stage missing either timestamp (skipped, blocked, cancelled before it started) is
    simply absent from the result -- never reported as a zero-second run.
    """
    durations: dict[str, float] = {}
    for name, manifest in stage_runs.items():
        if manifest.start_time is None or manifest.end_time is None:
            continue
        from datetime import datetime

        try:
            start = datetime.fromisoformat(manifest.start_time.replace("Z", "+00:00"))
            end = datetime.fromisoformat(manifest.end_time.replace("Z", "+00:00"))
        except ValueError:
            continue
        durations[name] = max(0.0, (end - start).total_seconds())
    return durations


def _read_ram_bytes() -> tuple[int | None, int | None]:
    """``(used_bytes, total_bytes)`` from ``/proc/meminfo``, or ``(None, None)``."""
    try:
        text = _MEMINFO_PATH.read_text(encoding="utf-8")
    except OSError:
        return None, None

    values: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        rest = rest.strip()
        if not rest.endswith("kB"):
            continue
        try:
            values[key] = int(rest[:-2].strip()) * 1024
        except ValueError:
            continue

    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None:
        return None, total
    return total - available, total


def _read_cpu_percent() -> float | None:
    """1-minute load average as a percentage of CPU count -- a cheap proxy, not a precise
    measurement (that would need a new dependency such as psutil). Good enough for "was
    this run GPU-bound or CPU-bound", which is all the report needs it for."""
    cpu_count = os.cpu_count()
    if not cpu_count:
        return None
    try:
        load1, _, _ = os.getloadavg()
    except OSError:
        return None
    return round((load1 / cpu_count) * 100, 1)


def _run_nvidia_smi() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if completed.returncode != 0 or not completed.stdout.strip():
        return {}

    parts = [p.strip() for p in completed.stdout.strip().splitlines()[0].split(",")]
    if len(parts) != 4:
        return {}
    name, util, mem_used, mem_total = parts
    try:
        return {
            "gpu_name": name,
            "gpu_utilization_percent": float(util),
            "gpu_memory_used_bytes": int(float(mem_used) * 1024 * 1024),
            "gpu_memory_total_bytes": int(float(mem_total) * 1024 * 1024),
        }
    except ValueError:
        return {}


@dataclass(frozen=True)
class ResourceSnapshot:
    """A best-effort host resource snapshot (spec section 51)."""

    cpu_percent: float | None = None
    ram_used_bytes: int | None = None
    ram_total_bytes: int | None = None
    gpu_name: str | None = None
    gpu_utilization_percent: float | None = None
    gpu_memory_used_bytes: int | None = None
    gpu_memory_total_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "ram_used_bytes": self.ram_used_bytes,
            "ram_total_bytes": self.ram_total_bytes,
            "gpu_name": self.gpu_name,
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "gpu_memory_used_bytes": self.gpu_memory_used_bytes,
            "gpu_memory_total_bytes": self.gpu_memory_total_bytes,
        }


def capture_resource_snapshot() -> ResourceSnapshot:
    cpu_percent = _read_cpu_percent()
    ram_used, ram_total = _read_ram_bytes()
    gpu = _run_nvidia_smi()
    return ResourceSnapshot(
        cpu_percent=cpu_percent,
        ram_used_bytes=ram_used,
        ram_total_bytes=ram_total,
        gpu_name=gpu.get("gpu_name"),
        gpu_utilization_percent=gpu.get("gpu_utilization_percent"),
        gpu_memory_used_bytes=gpu.get("gpu_memory_used_bytes"),
        gpu_memory_total_bytes=gpu.get("gpu_memory_total_bytes"),
    )


__all__ = ["ResourceSnapshot", "capture_resource_snapshot", "stage_durations"]
